"""Task questions must identify the native decision, including after permutation."""
import pytest
from datasets import Dataset, DatasetDict
from tasksource import eval_only, list_tasks, load_task
from tasksource.tasks import _missing_item_pair, _quarel_parts
from scripts.build_jev_dataset import exclude_publish_sources


@pytest.mark.parametrize('recast', ['jev', 'instruct'])
def test_questions_and_gold_survive_shared_loader(monkeypatch, recast):
    cases = [
        ('missing-item-prediction/contrastive', [
            {'prompt': 'α, β, γ. Is "δ" in the previous list? \nProvide no explanation, answer Yes or No.', 'y': 'No.'},
            {'prompt': 'α, β, γ. Is "β" in the previous list? \nProvide no explanation, answer Yes or No.', 'y': 'Yes.'},
        ], 'contain'),
        ('goal-step-wikihow/order', [
            {'sent2': 'How to bake a cake', 'ending0': 'Mix the ingredients.', 'ending1': 'Bake the batter.', 'label': 0},
            {'sent2': 'How to put on shoes', 'ending0': 'Put on socks.', 'ending1': 'Put on shoes.', 'label': 0},
        ], 'first'),
        ('quarel', [
            {'question': 'Which surface is smoother? (A) ice (B) snow', 'answer_index': 0},
            {'question': 'The glass is (A) more flexible (B) less flexible', 'answer_index': 1},
        ], 'answers or completes'),
    ]
    for task, raw, question in cases:
        data = Dataset.from_list(raw)
        monkeypatch.setattr('tasksource.access.load_dataset', lambda *a, **kw: DatasetDict(
            {split: data for split in ('train', 'validation', 'test')}))
        result = load_task(task, recast=recast)
        rows = result['train']
        for row in rows:
            assert question in row['instructions' if recast == 'jev' else 'inputs']
        if recast == 'jev':
            answers = {row['answer'] for row in rows}
            if task == 'quarel':
                assert answers == {'ice', 'less flexible'}
                assert all(not ({'A', 'B'} & set(row['criteria'])) for row in rows)
                assert all('(A)' not in row['state'] for row in rows)
            elif task.startswith('goal-step'):
                assert answers == {'Mix the ingredients.', 'Put on socks.'}
            else:
                assert answers == {'yes', 'no'}
                assert all('text_A:' in row['state'] and 'text_B:' in row['state'] for row in rows)
                assert all('Provide no explanation' not in row['state'] for row in rows)


def test_native_question_parsers_fail_informatively():
    with pytest.raises(ValueError, match='QuaRel'):
        _quarel_parts('Question without native options')
    with pytest.raises(ValueError, match='membership'):
        _missing_item_pair({'prompt': 'α, β, γ'})
    assert _missing_item_pair({'prompt': 'a.b, "c". Is "a.b" in the previous list?'}) == {
        'items': 'a.b, "c"', 'queried_item': 'a.b'}


def test_cladder_is_evaluation_only_and_cached_exports_exclude_it():
    assert 'cladder' not in set(list_tasks().id)
    assert eval_only.cladder.dataset_name == 'tasksource/cladder'
    assert 'cladder' in eval_only.REASONS
    rows = Dataset.from_dict({'source': ['cladder', 'cladder/variant', 'quarel']})
    assert exclude_publish_sources(rows)['source'] == ['quarel']
