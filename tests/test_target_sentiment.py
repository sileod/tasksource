import json

import pytest
from datasets import Dataset, DatasetDict
from tasksource import access
from tasksource.tasks import _newsmtsc_text, _sentihood_opinions, _mams_aspects
from scripts.repackage_dataset.target_sentiment import uk_targets


def test_newsmtsc_marks_exact_target_and_rejects_bad_offsets():
    row=dict(sentence='Alice praises Bob.',mention='Bob',polarity=1,id='native',**{'from':14,'to':17})
    assert _newsmtsc_text(row)=='Alice praises [TARGET]Bob[/TARGET].'
    with pytest.raises(ValueError,match='span/text mismatch'):_newsmtsc_text(dict(row,mention='Alice'))


def test_uk_keeps_human_added_targets_and_excludes_doesnotapply():
    tweet=dict(content='Alice likes Bob.',entities=[dict(id=1,entity='Alice',offset=0,type='ne'),
                                                  dict(id=2,entity='Bob',offset=12,type='ne')])
    annotation=dict(items={'1':'doesnotapply','2':'positive'},additional_items={'politics':'negative'})
    rows=list(uk_targets(tweet,annotation))
    assert rows[0][-1]=='does_not_apply'
    assert rows[1][:2]==('Bob','positive') and rows[1][-1] is None
    assert rows[2][:2]==('politics','negative') and rows[2][-1] is None
    tweet['entities'][1]['offset']=11
    assert list(uk_targets(tweet,annotation))[1][-1]=='span_mismatch'


def test_sentihood_keeps_entity_aspect_alignment_without_absence_labels():
    row=dict(id=1,text='LOCATION1 is cheap; LOCATION2 is expensive.',opinions=[
        dict(target_entity='LOCATION1',aspect='price',sentiment='Positive'),
        dict(target_entity='LOCATION2',aspect='price',sentiment='Negative')])
    rows=list(_sentihood_opinions(row))
    assert len(rows)==2
    assert rows[0]['labels']=='Positive' and 'LOCATION1' in rows[0]['sentence2']
    assert rows[1]['labels']=='Negative' and 'LOCATION2' in rows[1]['sentence2']
    assert json.loads(rows[0]['metadata'])['split_group_id']==json.loads(rows[1]['metadata'])['split_group_id']
    assert list(_sentihood_opinions(dict(row,opinions=[])))==[]


def test_mams_uses_native_pairs_not_the_generation_prompt():
    row=dict(input=['The food is good but service is bad.'],output="[['food','positive'],['service','negative']]",
             instruction='Ignore this source example and its answer.')
    rows=list(_mams_aspects(row))
    assert [r['labels'] for r in rows]==['positive','negative']
    assert all('Ignore' not in r['sentence1']+r['sentence2'] for r in rows)


@pytest.mark.parametrize('task,rows',[
    ('newsmtsc',[dict(sentence='Alice praises Bob.',mention='Bob',polarity=1,id='native',**{'from':14,'to':17})]),
    ('sentihood/sentiment',[dict(id=1,text='LOCATION1 is cheap.',opinions=[
        dict(target_entity='LOCATION1',aspect='price',sentiment='Positive')])]),
    ('mams/atsa',[dict(input=['Good food.'],output="[['food','positive']]")]),
])
def test_shared_loader_jev_preserves_gold_and_native_splits(task,rows,monkeypatch):
    data=DatasetDict(train=Dataset.from_list(rows))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:data)
    result=access.load_task(task,recast='jev')
    assert set(result)=={'train'}
    example=result['train'][0]
    assert example['criteria'][example['label']]=='positive'
    assert example['instructions'] and 'text_B' not in example['state']


def test_reviewed_ambiguous_sources_stay_out_of_training():
    from tasksource import list_tasks, parked
    assert 'uk-election/sentiment' not in set(list_tasks().id)
    assert 'uk_election__sentiment' in parked.by_kind('unsound')
    opinion=dict(target_entity='LOCATION1',aspect='safety',sentiment='Positive')
    source=DatasetDict(train=Dataset.from_list([dict(id=306,text='LOCATION1 is safe.',opinions=[opinion])]),
        validation=Dataset.from_list([dict(id=306,text='LOCATION1 is safe, but pickpocketing is common.',opinions=[opinion]),
                                      dict(id=307,text='LOCATION1 is secure.',opinions=[opinion])]))
    result=access.load_preprocessing(id='sentihood/sentiment').pre_process(source)
    assert len(result['train'])==len(result['validation'])==1
    assert json.loads(result['train'][0]['metadata'])['native_id']==306
    assert json.loads(result['validation'][0]['metadata'])['native_id']==307
