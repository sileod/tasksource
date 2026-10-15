import json

import pytest
from datasets import Dataset, DatasetDict
from scripts.repackage_dataset.comve import join_answers, prepare_pairs
from tasksource import access, load_task


def test_comve_join_inverts_nonsense_and_checks_ids():
    rows=join_answers('id,sent0,sent1\n7,drink apple,drink milk\n','7,0\n')
    assert rows[0]['labels']==1
    with pytest.raises(ValueError):join_answers('id,sent0,sent1\n7,a,b\n','8,1\n')
    with pytest.raises(ValueError):join_answers('id,sent0,sent1\n7,a,b\n','7,2\n')


def test_comve_reserves_holdouts_and_rejects_ambiguous_pairs():
    def row(id,a,b,gold):return dict(id=id,choices=[a,b],labels=gold)
    output,excluded,_=prepare_pairs(dict(test=[row('t','a','b',0)],validation=[],train=[
        row('1','b','A',1),row('2','same',' same ',0),row('3','c','d',0),row('4','d','c',0),row('5','e','f',1)]))
    assert [r['labels'] for r in output['train']]==[1]
    assert {r['reason'] for r in excluded}=={'duplicate_pair','invalid_choices','conflicting_gold'}


def faithrow(dialog=0,begin=('Hallucination',),original='Unsupported original.',response='SECRET_REWRITE'):
    return dict(dialog_idx=dialog,BEGIN=list(begin),VRM=['Disclosure'],knowledge='Source evidence.',
                history=['A question.'],original_response=original,response=response)


def test_faithdial_original_labels_and_rewrite_separation(monkeypatch):
    raw=DatasetDict(train=Dataset.from_list([faithrow(),faithrow(1,('Generic',)),
        faithrow(2,('Entailment',),None,'Faithful original.')]))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:raw)
    rows=load_task('faithdial/grounding',recast='jev')['train']
    assert len(rows)==2
    for r in rows:
        assert 'SECRET_REWRITE' not in r['state']
        expected='unsupported' if 'Unsupported original.' in r['state'] else 'grounded'
        assert r['criteria'][r['label']]==expected
        assert json.loads(r['metadata'])['annotation_origin']=='human'


def test_faithdial_duplicate_dialogues_stay_in_native_holdout():
    from tasksource.tasks import _faithdial_rows
    raw=DatasetDict(train=Dataset.from_list([faithrow(0)]),test=Dataset.from_list([faithrow(42)]))
    result=_faithdial_rows(raw)
    assert len(result['train'])==0 and len(result['test'])==1


def test_ragtruth_gold_quality_and_wrapper_cleanup(monkeypatch):
    def row(id,spans,quality='good'):
        return dict(id=id,query='Summarize.',context='The source.\n\noutput:',output='Candidate.',
            task_type='Summary',quality=quality,model='source-model',hallucination_labels=json.dumps(spans))
    raw=DatasetDict(train=Dataset.from_list([row('a',[]),row('b',[{'start':0,'end':9,'text':'Candidate'}]),
        row('c',[],'truncated')]))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:raw)
    rows=load_task('ragtruth/grounding',recast='jev')['train']
    assert len(rows)==2
    for r in rows:
        meta=json.loads(r['metadata'])
        assert r['criteria'][r['label']]==('unsupported' if meta['spans'] else 'grounded')
        assert '\n\noutput:' not in r['state']
        assert meta['input_origin']=='LLM'


def test_comve_gold_survives_option_recasting(monkeypatch):
    records,_,_=prepare_pairs(dict(train=[dict(id='1',choices=['Drink an apple.','Drink milk.'],labels=1)],
                                  validation=[],test=[]))
    raw=DatasetDict(train=Dataset.from_list(records['train']))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:raw)
    row=load_task('comve',recast='jev')['train'][0]
    assert row['criteria'][row['label']]=='Drink milk.'
    assert json.loads(row['metadata'])['native_nonsensical_index']==0
