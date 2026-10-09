import copy
import json

from datasets import Dataset, DatasetDict
import pytest

from tasksource import load_task, list_tasks
from scripts.repackage_dataset.mind2web import mind2web_record, trajectory_splits


def trajectory(identity='trajectory'):
    def candidate(node, **flags):
        return dict(backend_node_id=node, tag='button', attributes=json.dumps({'title': node}), **flags)
    action = dict(action_uid='action', cleaned_html='<html>' + ''.join(
        f'<button backend_node_id="{i}">{i}</button>' for i in ['gold', 'alternative', 'n1', 'n2', 'n3']) + '</html>',
        operation={'op': 'TYPE', 'value': 'SECRET_VALUE'},
        pos_candidates=[candidate('gold', is_original_target=True), candidate('alternative')],
        neg_candidates=[candidate(i) for i in ['alternative', 'n1', 'n2', 'n3']])
    return dict(annotation_id=identity, website='site', domain='domain', confirmed_task='Enter the search query.',
                action_reprs=['SECRET_CURRENT_ACTION', 'SECRET_FUTURE_ACTION'], actions=[action])


def test_candidate_supervision_no_leakage_and_determinism():
    source = trajectory()
    row, reason = mind2web_record(source, 0, elements=True)
    assert reason is None
    assert row == mind2web_record(source, 0, elements=True)[0]
    meta = json.loads(row['metadata'])
    assert len(set(row['choices_list'])) == 4
    assert meta['candidate_ids'][row['labels']] == 'gold'
    assert 'alternative' not in meta['candidate_ids']
    assert all('SECRET' not in v for v in [row['inputs'], *row['choices_list']])
    assert 'is_original_target' not in str(row['choices_list'])
    assert meta['action_uid'] == 'action' and meta['trajectory_id'] == 'trajectory'
    source['actions'][0]['pos_candidates'] = []
    assert mind2web_record(source, 0, elements=True)[1] == 'ambiguous_or_missing_original_target'
    # Missing MC supervision doesn't discard the genuine action label.
    assert mind2web_record(source, 0)[0]['labels'] == 'TYPE'


def test_trajectory_and_shared_page_grouping():
    a, b = trajectory('a'), trajectory('b')
    c = copy.deepcopy(a)
    c['annotation_id'] = 'c'
    c['actions'][0]['cleaned_html'] += '<p>Different page</p>'
    splits, groups = trajectory_splits([a, b, c])
    assert splits['a'] == splits['b'] and groups['a'] == groups['b']
    assert groups['a'] != groups['c']
    assert (splits, groups) == trajectory_splits([c, b, a])


@pytest.mark.parametrize('task,elements', [('mind2web/action', False), ('mind2web/dom-element', True)])
@pytest.mark.parametrize('recast', [None, 'jev', 'instruct'])
def test_text_catalog_loader_and_recasts(monkeypatch, task, elements, recast):
    raw = mind2web_record(trajectory(), 0, elements)[0]
    data = DatasetDict({split: Dataset.from_list([raw]) for split in ['train', 'validation', 'test']})
    monkeypatch.setattr('tasksource.access.load_dataset', lambda *a, **kw: data)
    result = load_task(task, recast=recast)
    assert set(result) == set(data)
    row = result['train'][0]
    if recast == 'jev':
        assert row['answer'] == raw['choices_list'][raw['labels']] if elements else row['answer'] == 'TYPE'
    assert task in set(list_tasks().id)
    assert 'images' not in result['train'].features
