"""Explicit question context preserves inputs and supervision across recasts."""
import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

from tasksource.preprocess import add_question, classification_prompt
from tasksource.recast import recast_instruct, recast_jev
from tasksource.tasks import _eurlex_descriptors, lex_glue___eurlex
from tasksource.metadata.eurovoc import EUROVOC
from scripts.build_jev_dataset import add_example_ids, example_key


def source():
    return DatasetDict(train=Dataset.from_dict({
        'sentence1': ['Marked document', 'Marked document'],
        'sentence2': ['Event type: Award', 'Event type: Attack'], 'labels': [0, 1],
    }, features=Features({'sentence1': Value('string'), 'sentence2': Value('string'),
                          'labels': ClassLabel(names=['Agent', 'Patient'])})))


def test_context_moves_to_question_without_changing_gold_or_raw_data():
    dataset = source()
    question = 'Which role does the marked argument play?\n{sentence2}'
    rows = recast_jev(dataset, question=question)['train']
    assert rows['state'] == ['Marked document'] * 2
    assert rows['instructions'] == [question.replace('{sentence2}', x) for x in dataset['train']['sentence2']]
    assert rows['answer'] == ['Agent', 'Patient']
    assert dataset['train'][0]['sentence2'] == 'Event type: Award'
    instruct = recast_instruct(dataset, question=question)['train'][0]
    assert 'Event type: Award' in instruct['inputs']
    assert 'text_B' not in instruct['inputs']
    prompted = add_question(dataset, question)['train'][0]
    assert prompted['sentence2'] == rows[0]['instructions']
    assert '{sentence2}' not in prompted['sentence2']


def test_empty_context_fails_and_genuine_pairs_remain_pairs():
    with pytest.raises(ValueError, match='nonempty sentence2'):
        classification_prompt({'sentence1': 'A'}, '{sentence2}')
    assert classification_prompt({'sentence1': 'A', 'sentence2': 'B'}, 'Compare.') == ('text_A: A\ntext_B: B', 'Compare.')


def test_per_question_context_keeps_example_ids_distinct():
    rows = Dataset.from_list([dict(id=f's:{i}', variant='direct', source='hallmarks-of-cancer/yesno',
        state='Same sentence', options=['no', 'yes'], target=[0., 1.], question=q)
        for i, q in enumerate(['Does this describe angiogenesis?', 'Does this describe metastasis?'])])
    ids = add_example_ids(rows)['example_id']
    assert ids[0] != ids[1]
    assert example_key('ordinary', 'A', ['yes', 'no'], [1., 0.]) == example_key('ordinary', 'A', ['yes', 'no'], [1., 0.], 'Choose.')


def test_eurlex_names_are_mapped_by_code_not_position_and_multilabel_is_preserved():
    dataset = DatasetDict(train=Dataset.from_dict({'text': ['Document'], 'labels': [[1, 0]]},
        features=Features({'text': Value('string'), 'labels': Sequence(ClassLabel(names=['100237', '100222']))})))
    result = _eurlex_descriptors(dataset)
    assert result['train'][0]['labels'] == [1, 0]
    assert result['train'].features['labels'].feature.names == ['transport policy', 'information and information processing']
    assert len(EUROVOC) == len(set(EUROVOC.values())) == 100
    assert 'all applicable' in lex_glue___eurlex.question
    # Multi-label EURLEX must not silently become single-choice supervision.
    canonical = DatasetDict(train=result['train'].rename_column('text', 'sentence1'))
    with pytest.raises(TypeError, match='ClassLabel'):
        recast_jev(canonical, question=lex_glue___eurlex.question)


def test_parked_tasks_leave_catalog_and_probability_words_stays():
    from tasksource import list_tasks, parked
    from tasksource.tasks import probability_words_nli
    tasks = set(list_tasks().id)
    assert not tasks & {'wouldyourather', 'puzzte', 'linguisticprobing/sentence_length'}
    assert 'probability_words_nli/usnli' in tasks
    assert all(name in parked.PARKED for name in ['wouldyourather', 'puzzte', 'linguisticprobing___sentence_length'])
    state, question = classification_prompt({'sentence1': 'Probably A.', 'sentence2': 'A.'}, probability_words_nli.question)
    assert state == 'Probably A.' and question.endswith('\nA.')
    assert 'text_A' not in question and 'text_B' not in question


def test_soft_question_context_preserves_vote_distribution():
    from tasksource.jev.recast import _recast_soft
    dataset = DatasetDict(train=Dataset.from_dict({'sentence1': ['Evidence'],
        'sentence2': ['Per-example query'], 'options': [['no', 'yes']], 'labels': [[.25, .75]]}))
    row = _recast_soft(dataset, 'demo', 'Assess this query: {sentence2}', 'choice', False, 'demo')['train'][0]
    assert row['state'] == 'Evidence'
    assert row['instructions'] == 'Assess this query: Per-example query'
    assert row['target'] == [.25, .75]


@pytest.mark.parametrize('recast', ['jev', 'instruct'])
def test_no_history_intent_has_no_empty_field_or_dangling_variable(recast):
    from tasksource.tasks import multidogo__intent
    source = DatasetDict(train=Dataset.from_dict({'sentence1': ['', 'Customer: Hello.'],
        'sentence2': ['Order pizza.', 'Order pizza.'], 'labels': [0, 0]},
        features=Features({'sentence1': Value('string'), 'sentence2': Value('string'),
                           'labels': ClassLabel(names=['order', 'greet'])})))
    result = (recast_jev if recast == 'jev' else recast_instruct)(source, question=multidogo__intent.question)['train']
    first = result[0]['state' if recast == 'jev' else 'inputs']
    second = result[1]['state' if recast == 'jev' else 'inputs']
    assert 'Order pizza.' in first and 'text_A:' not in first and 'text_B:' not in first
    assert 'Customer: Hello.' in second and 'Order pizza.' in second
    assert 'text_A' not in multidogo__intent.question and 'text_B' not in multidogo__intent.question
    if recast == 'jev':
        assert result['answer'] == ['order', 'order']
