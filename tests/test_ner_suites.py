import json

import pytest
from datasets import Dataset, DatasetDict

from scripts.build_jev_extraction import NER
from tasksource import eval_only, list_tasks, task_licenses
from tasksource.metadata.source_revisions import SOURCE_REVISIONS
from tasksource.tasks import _bigbio_entity_row, _nonoverlapping_ner_splits, pubmedqa


def test_new_sources_are_pinned_capped_and_keep_native_splits():
    english = list_tasks().set_index('id')
    for task_id in ['bionlp-cg/entities', 'bioinfer/entities', 'bc4chemd', 'uner/en_ewt', 'harveyner']:
        row = english.loc[task_id]
        task = row.mapping
        assert row.task_type == 'TokenClassification'
        assert task.complete_splits is False
        assert 0 < task.max_examples <= 5000
        assert task.load_dataset_kwargs['revision'] == SOURCE_REVISIONS[row.dataset_name]
        assert len(task.load_dataset_kwargs['revision']) == 40
        assert task_id in NER
    assert not any(i.startswith('mrqa-eval/') for i in english.index)


def test_new_source_license_evidence():
    info = task_licenses(['bionlp-cg/entities', 'bioinfer/entities', 'bc4chemd', 'uner/en_ewt'])
    assert set(info.license_use) == {'commercial'}
    assert dict(zip(info.id, info.license))['bionlp-cg/entities'].startswith('cc-by-sa-3.0')
    assert task_licenses(['harveyner']).iloc[0].license_use == 'unspecified'
    assert task_licenses(['pubmedqa/labeled-fold0']).iloc[0].license_use == 'commercial'


def test_pubmedqa_uses_one_human_fold_and_excludes_answer_leakage():
    assert pubmedqa.load_dataset_kwargs['data_dir'] == 'pubmed_qa_labeled_fold0_bigbio_qa'
    assert pubmedqa.load_dataset_kwargs['revision'] == SOURCE_REVISIONS['bigbio/pubmed_qa']
    assert pubmedqa.max_examples == 2000 and pubmedqa.complete_splits is False
    rows = [{'context': 'Evidence only.', 'question': 'Does it work?', 'answer': [label],
             'document_id': str(i), 'LONG_ANSWER': 'LEAKED CONCLUSION'}
            for i, label in enumerate(['yes', 'no', 'maybe'])]
    result = pubmedqa(DatasetDict(train=Dataset.from_list(rows)))['train']
    assert result['sentence1'] == ['Evidence only.'] * 3
    assert result['sentence2'] == ['Does it work?'] * 3
    assert set(result.features['labels'].names) == {'yes', 'no', 'maybe'}
    assert 'LEAKED' not in str(result.to_dict())


def test_native_ner_splits_remove_only_cross_split_copies():
    def rows(tokens):
        return Dataset.from_dict({'tokens': [[t] for t in tokens], 'ner_tags': [['O']] * len(tokens)})
    ds = DatasetDict(train=rows(['x', 'x', 'y', 'z']), validation=rows(['y', 'z']), test=rows(['z']))
    out = _nonoverlapping_ner_splits(ds)
    assert out['train']['tokens'] == [['x'], ['x']]
    assert out['validation']['tokens'] == [['y']]
    assert out['test']['tokens'] == [['z']]


def test_bigbio_contiguous_runs_keep_exact_offsets_and_drop_overlaps():
    row = {'document_id': 'd', 'passages': [{'offsets': [[2, 17]], 'text': ['muscle or actin']}],
           'entities': [{'type': 'protein', 'offsets': [[2, 8], [12, 17]], 'text': ['muscle', 'actin']}]}
    out = _bigbio_entity_row(row, split_discontinuous=True)
    assert out['tokens'] == ['muscle', 'or', 'actin']
    assert out['ner_tags'] == ['B-protein', 'O', 'B-protein']
    assert out['original_offsets'] == [[2, 8], [9, 11], [12, 17]]
    assert out['dropped_discontinuous'] == 0
    assert _bigbio_entity_row(row)['dropped_discontinuous'] == 1  # existing views unchanged
    row['entities'].append({'type': 'other', 'offsets': [[12, 17]], 'text': ['actin']})
    out = _bigbio_entity_row(row, split_discontinuous=True)
    assert out['dropped_overlapping'] == 2
    assert out['ner_tags'] == ['B-protein', 'O', 'O']
    row['entities'][0]['text'][1] = 'wrong'
    with pytest.raises(ValueError, match='text mismatch'):
        _bigbio_entity_row(row, split_discontinuous=True)


def test_bigbio_rejects_unpaired_annotations():
    row = {'document_id': 'd', 'passages': [{'offsets': [[0, 1]], 'text': ['x']}],
           'entities': [{'type': 'protein', 'offsets': [[0, 1]], 'text': []}]}
    with pytest.raises(ValueError, match='lengths differ'):
        _bigbio_entity_row(row, split_discontinuous=True)


def test_mrqa_eval_uses_only_native_test_and_inclusive_end():
    def row(subset, qid):
        return {'context': 'A Paris.', 'question': 'Where?', 'qid': qid, 'subset': subset,
                'detected_answers': {'text': ['Paris'], 'char_spans': [{'start': [2], 'end': [6]}]}}
    data = DatasetDict(train=Dataset.from_list([row('BioASQ', 'leak')]),
                       test=Dataset.from_list([row('BioASQ', 'gold'), row('DROP', 'other')]))
    result = eval_only.mrqa_bioasq(data)
    assert list(result) == ['train']  # eval-only loader interface, not a training catalog
    assert len(result['train']) == 1
    out = result['train'][0]
    assert out['answers'] == {'text': ['Paris'], 'answer_start': [2]}
    assert json.loads(out['metadata']) == {'id': 'gold', 'mrqa_subset': 'BioASQ', 'source_split': 'test',
                                         'dropped_occurrences': 0}
    for name in ('bioasq', 'drop', 'duorc', 'race', 'relation_extraction', 'textbookqa'):
        assert 'mrqa_' + name in eval_only.REASONS
        assert getattr(eval_only, 'mrqa_' + name).complete_splits is False


def test_mrqa_mismatches_cannot_become_null_or_sliced_answers():
    row = {'context': 'A Paris.', 'detected_answers': {'text': ['Lyon', 'Paris'],
           'char_spans': [{'start': [2], 'end': [6]}, {'start': [2], 'end': [6]}]}}
    out = eval_only._mrqa_eval_answers(row)
    assert out['checked_answers'] == {'text': ['Paris'], 'answer_start': [2]}
    assert out['dropped_occurrences'] == 1
    row['detected_answers'] = {'text': ['Lyon'], 'char_spans': [{'start': [2], 'end': [6]}]}
    assert eval_only._mrqa_eval_answers(row)['checked_answers']['text'] == []
    row['detected_answers']['text'] = ['The PARIS!']
    # Official MRQA normalization verifies this occurrence, but the returned
    # answer must still be verbatim at the original offset.
    assert eval_only._mrqa_eval_answers(row)['checked_answers'] == {'text': ['Paris'], 'answer_start': [2]}

