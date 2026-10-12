import json
from datasets import Dataset, DatasetDict
from tasksource import access
from tasksource.tasks import _nevir_records, _wikiqa_rows


def test_wikiqa_keeps_all_negative_questions_and_duplicate_judgments():
    rows=[dict(question_id='Q1',question='What year?',document_title='Topic',answer='No date here.',label=0)]*2
    source=DatasetDict(train=Dataset.from_list(rows))
    result=_wikiqa_rows(source)['train']
    assert len(result)==2 and list(result['label'])==[0,0]
    assert len({json.loads(r['metadata'])['id'] for r in result})==2
    assert {json.loads(r['metadata'])['split_group_id'] for r in result}=={'Q1'}


def test_nevir_native_query_document_alignment_after_jev_permutation(monkeypatch):
    row=dict(id='1-2',q1='Who left?',q2='Who did not leave?',doc1='Alice left.',doc2='Alice did not leave.')
    native=list(_nevir_records(row))
    assert [r['choices_list'][r['labels']] for r in native]==[row['doc1'],row['doc2']]
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:DatasetDict(train=Dataset.from_list([row])))
    result=access.load_task('nevir',recast='jev')['train']
    for r in result:
        expected=row['doc1'] if r['state']==row['q1'] else row['doc2']
        assert r['criteria'][r['label']]==expected
        assert json.loads(r['metadata'])['split_group_id']=='1'


def test_nevir_splits_hold_out_original_document_across_edits(monkeypatch):
    def row(identity):return dict(id=identity,q1='Who left?',q2='Who stayed?',doc1='Alice left.',doc2='Alice stayed.')
    source=DatasetDict(train=Dataset.from_list([row('1-2'),row('2-2')]),
                       validation=Dataset.from_list([row('1-3')]))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:source)
    result=access.load_task('nevir',recast='jev')
    groups={s:{json.loads(r['metadata'])['split_group_id'] for r in ds} for s,ds in result.items()}
    assert groups==dict(train={'2'},validation={'1'})


def test_antique_retains_four_native_grades_as_ordered_score(monkeypatch):
    rows=[dict(sentence1=f'Question: Q\nCandidate answer: answer {i}',labels=i,
               metadata=json.dumps(dict(id=str(i),split_group_id='q'))) for i in range(1,5)]
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:DatasetDict(train=Dataset.from_list(rows)))
    result=access.load_task('antique',recast='jev')['train']
    assert set(result['kind'])=={'score'}
    assert len(result)==4
    for r in result:
        grade=int(r['state'].rsplit(' ',1)[-1])
        assert r['criteria'][r['label']].startswith(str(grade)+':')

