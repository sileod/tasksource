"""Failures after filtering should explain the missing training data."""
import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

from tasksource.access import _declaration_order
from tasksource.preprocess import Classification, MultipleChoice, fix_labels, fix_splits


def rows(labels=('yes', 'no')):
    return Dataset.from_dict({'sentence1': ['question'] * len(labels), 'labels': list(labels)},
                             features=Features({'sentence1': Value('string'), 'labels': Value('string')}))


def test_declaration_order_uses_python_assignments(tmp_path):
    source = tmp_path / 'catalog.py'
    source.write_text('''first = (
    task()
)
second: Task = task()
third = fourth = task()
def helper():
    nested = task()
"fake = task()"
''')
    assert list(_declaration_order(source)) == ['first', 'second', 'third', 'fourth']


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('stage', ['pre_process', 'post_process'])
def test_empty_training_after_processing(explicit, stage):
    task = Classification(complete_splits=False,
                          label_values={'yes': 'Yes', 'no': 'No'} if explicit else None,
                          **{stage: lambda ds: DatasetDict({k: v.select([]) for k, v in ds.items()})})
    with pytest.raises(ValueError, match='nonempty train split'):
        task(DatasetDict(train=rows()))


def test_fix_labels_empty_training():
    with pytest.raises(ValueError, match='fix_labels: requires a nonempty train split'):
        fix_labels(DatasetDict(train=rows(()), validation=rows()))


def test_empty_evaluation_preserves_schema():
    result = Classification(complete_splits=False)(DatasetDict(train=rows(), validation=rows(())))
    assert len(result['train']) == 2 and len(result['validation']) == 0
    assert result['train'].features == result['validation'].features
    assert isinstance(result['train'].features['labels'], ClassLabel)


@pytest.mark.parametrize('gold_first', [False, True])
def test_empty_mc_training_after_option_filtering(gold_first):
    features = Features({'inputs': Value('string'), 'labels': Value('int64'),
                         'choices_list': Sequence(Value('string'))})
    train = Dataset.from_dict({'inputs': ['q', 'q', 'q'], 'labels': [0, 0, 0],
                               'choices_list': [[], ['only'], None]}, features=features)
    evaluation = Dataset.from_dict({'inputs': ['q'], 'labels': [0],
                                    'choices_list': [['gold', 'other']]}, features=features)
    with pytest.raises(ValueError, match='multiple-choice filtering.*nonempty train split'):
        MultipleChoice(inputs='inputs', choices_list='choices_list', complete_splits=False)(
            DatasetDict(train=train, validation=evaluation), gold_first=gold_first, max_options=None)


@pytest.mark.parametrize('complete', [False, True])
def test_fix_splits_all_labels_hidden(complete):
    train = Dataset.from_dict({'labels': [None]}, features=Features({'labels': ClassLabel(names=['a', 'b'])}))
    with pytest.raises(ValueError, match='fix_splits: requires a nonempty train split'):
        fix_splits(DatasetDict(train=train), complete=complete)


def test_fix_splits_keeps_empty_native_evaluation():
    ds = DatasetDict(train=rows(), validation=rows(()))
    assert fix_splits(ds) is ds
    assert len(ds['validation']) == 0
