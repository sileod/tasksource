from datasets import Dataset, DatasetDict
from tasksource.tasks import soul, checkthat2024__check_worthiness


def test_soul_keeps_reviews_together_and_drops_contradictory_decisions():
    dataset = DatasetDict({
        'train': Dataset.from_list([
            {'review_text': 'Shared review', 'statement': 'Good', 'label': 'true'},
            {'review_text': 'Other review', 'statement': 'Good', 'label': 'true'},
            {'review_text': 'Other review', 'statement': 'Good', 'label': 'false'},
            {'review_text': 'Unique review', 'statement': 'Good', 'label': 'true'}]),
        'test': Dataset.from_list([
            {'review_text': 'Shared review', 'statement': 'Different question', 'label': 'false'}])})
    result = soul.pre_process(dataset)
    assert result['train']['review_text'] == ['Unique review']
    assert len(result['test']) == 1


def test_checkthat_excludes_hidden_and_derived_labels():
    dataset = DatasetDict({s: Dataset.from_list([{'text': s, 'label_ct': label}])
                           for s, label in [('train', 1), ('dev', 0), ('dev_test', 1),
                                            ('test', 9), ('train_hq', 0), ('train_rus', 1)]})
    result = checkthat2024__check_worthiness.pre_process(dataset)
    assert set(result) == {'train', 'dev', 'dev_test'}
    assert result['train']['label_ct'] == [1]
