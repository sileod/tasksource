import json

from datasets import Dataset, DatasetDict
import pytest

from tasksource import list_tasks, load_task
from tasksource.preprocess import Classification
from tasksource.tasks import weblinx_record, weblinx_rows


def transition(**kwargs):
    return dict(demo='demo', turn=4, action='click(uid="second")',
                action_history='load(url="https://example.org")', utterances='Click Save.',
                clean_html='<button>Cancel</button><button>Save</button>', viewport='800h x 600w',
                candidates='(uid = first) [[tag]] button [[text]] Cancel\n'
                           '(uid = second) [[tag]] button [[text]] Save', **kwargs)


def test_native_candidates_and_no_action_arguments():
    row = transition()
    row['action'] = 'text_input(text="SECRET_TARGET_ARGUMENT", uid="second")'
    record, reason = weblinx_record(row, elements=True)
    assert reason is None
    assert 'SECRET_TARGET_ARGUMENT' not in record['inputs']
    assert record['choices_list'][record['labels']].endswith('Save')
    assert json.loads(record['metadata'])['candidate_ids'] == ['first', 'second']
    assert 'Operation: text_input' in record['inputs']


@pytest.mark.parametrize('action,reason', [
    ('click(uid=None)', 'unresolved_target'), ('click(uid="absent")', 'unresolved_target'),
    ('scroll(x=0, y=20)', 'non_element_action'), ('future(uid="second")', 'invalid_action'),
])
def test_no_gold_insertion(action, reason):
    row = transition()
    row['action'] = action
    assert weblinx_record(row, elements=True) == (None, reason)


def test_duplicate_uids_rejected():
    row = transition()
    row['candidates'] += '\n(uid = second) Duplicate'
    assert weblinx_record(row, elements=True)[0] is None


def test_missing_context_rejected():
    row = transition()
    row.update(action_history=None, utterances='N o   i n s t r u c t o r   u t t e r a n c e ;')
    assert weblinx_record(row) == (None, 'missing_context')


def test_native_split_opt_out():
    rows = Dataset.from_list([{'inputs': 'one', 'labels': 'yes'}, {'inputs': 'two', 'labels': 'no'}])
    dataset = Classification('inputs', labels='labels', complete_splits=False)(
        DatasetDict(train=rows, test_web=rows.select([0])))
    assert set(dataset) == {'train', 'test_web'}
    assert len(dataset['train']) == 2


@pytest.mark.parametrize('task', ['weblinx/action', 'weblinx/dom-element'])
def test_browser_load_and_jev(monkeypatch, task):
    rows = Dataset.from_list([transition(), transition()])
    source = DatasetDict(train=rows, validation=rows, test=rows, test_iid=rows,
                         test_web=rows)
    monkeypatch.setattr('tasksource.access.load_dataset', lambda *a, **kw: DatasetDict(source))
    native = load_task(task)
    assert set(native) == {'train', 'validation', 'test', 'test_web'}
    if task.endswith('dom-element'):
        assert native['train'][0]['choice0'].endswith('Save')
    result = load_task(task, recast='jev')
    assert all(len(result[split]) == 2 for split in native)
    assert result['train'][0].get('kind', 'choice') == 'choice'
    assert json.loads(result['train'][0]['metadata'])['input_view'] == 'dom'
    assert task in set(list_tasks().id)
    assert task not in set(list_tasks(vision=True).id)


def test_alias_removed_only_once():
    rows = Dataset.from_list([transition()])
    assert set(weblinx_rows(DatasetDict(train=rows, test=rows, test_iid=rows))) == {'train', 'test'}


def test_websrc_native_span_and_boolean():
    from scripts.repackage_dataset.browser import NativeDOM, websrc_record
    html = '<html tid="0"><body tid="1"><p tid="2">Age: <b tid="3">33</b></p><p tid="4">Other</p></body></html>'
    nodes = NativeDOM(html).nodes
    qa = {'id': 'test', 'question': 'How old?', 'element_id': '3', 'answer_start': '0', 'answer': '33'}
    (view, row), reason = websrc_record(qa, html, nodes, {})
    assert reason is None and view == 'element'
    assert row['choices_list'][row['labels']] == 'tid=3 <b> 33'
    assert '33' not in row['inputs'].split('Question: ')[1]
    qa.update(element_id='2', answer_start='5')
    assert websrc_record(qa, html, nodes, {}) == (None, 'non_deepest_target')
    qa.update(element_id='3', answer_start='1')
    assert websrc_record(qa, html, nodes, {}) == (None, 'span_mismatch')
    qa.update(element_id='-1', answer_start='1', answer='yes')
    assert websrc_record(qa, html, nodes, {})[0][1]['labels'] == 'yes'
    qa.update(answer_start='0')
    assert websrc_record(qa, html, nodes, {}) == (None, 'invalid_boolean')


def test_websrc_candidates_independent_of_answer():
    from scripts.repackage_dataset.browser import NativeDOM, websrc_record
    html = '<html tid="0"><p tid="1">33</p><p tid="2">44</p></html>'
    nodes = NativeDOM(html).nodes
    qa = {'id': 'test', 'question': 'How old?', 'element_id': '1', 'answer_start': '0', 'answer': '33'}
    first = websrc_record(qa, html, nodes, {})[0][1]
    second = websrc_record({**qa, 'element_id': '2', 'answer': '44'}, html, nodes, {})[0][1]
    assert first['choices_list'] == second['choices_list']
    assert websrc_record(qa, html, nodes, {}, max_candidates=2) == (None, 'candidate_count')


@pytest.mark.parametrize('html', ['<p tid="1"><b tid="2">Oops</p>', '<p tid="1"/><p tid="1"/>', '<p>No ID</p>'])
def test_malformed_native_dom_rejected(html):
    from scripts.repackage_dataset.browser import NativeDOM
    with pytest.raises(ValueError):
        NativeDOM(html)
