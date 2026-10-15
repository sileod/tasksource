import json

import pytest
from datasets import Dataset, DatasetDict
from tasksource import access, load_task
from tasksource.tasks import _gap_text, _prosocial_context
from scripts.repackage_dataset.dialogue import esconv_turns


def test_quality_keeps_full_article_and_native_zero_based_gold(monkeypatch):
    article='Start of the article. '+'middle '*3000+'The answer is at the very end.'
    raw=DatasetDict(train=Dataset.from_list([dict(article=article,question='What is the answer?',options=['a','b','c','correct'],answer=3,hard=True)]))
    monkeypatch.setattr(access,'load_dataset',lambda *args,**kwargs:raw)
    ds=load_task('quality',recast='jev')
    row=ds['train'][0]
    assert article in row['state'] and row['criteria'][row['label']]=='correct'
    assert len(row['criteria'])==4 and json.loads(row['metadata'])['hard']
    assert set(ds)=={'train'}


def test_esconv_preserves_consecutive_strategy_changes_and_excludes_future():
    raw=dict(situation='I lost my job.',dialog=[dict(speaker='usr',text='I am worried.'),
        dict(speaker='sys',text='That sounds upsetting.',strategy='Reflection of feelings'),
        dict(speaker='sys',text='Have you considered asking a friend?',strategy='Providing Suggestions'),
        dict(speaker='usr',text='FUTURE_PRIVATE_TEXT')])
    rows=list(esconv_turns(raw,'conversation','train:7'))
    assert [r['labels'] for r in rows]==['Reflection of feelings','Providing Suggestions']
    assert rows[0]['sentence2']=='That sounds upsetting.'
    assert 'That sounds upsetting.' in rows[1]['sentence1']
    assert all('FUTURE_PRIVATE_TEXT' not in r['sentence1']+r['sentence2'] for r in rows)
    assert all('Reflection of feelings' not in r['sentence1'] for r in rows)
    assert len({json.loads(r['metadata'])['split_group_id'] for r in rows})==1


def test_prosocial_history_excludes_current_response_and_handles_split_local_ids():
    train=Dataset.from_list([dict(context='I hurt someone.',response='PRIOR_RESPONSE',dialogue_id=0,response_id=0),
        dict(context='I regret it.',response='HIDDEN_RESPONSE',dialogue_id=0,response_id=1)])
    test=Dataset.from_list([dict(context='Different conversation.',response='TEST_HIDDEN',dialogue_id=0,response_id=0)])
    ds=_prosocial_context(DatasetDict(train=train,test=test))
    assert 'PRIOR_RESPONSE' not in ds['train'][0]['dialogue']
    assert 'PRIOR_RESPONSE' in ds['train'][1]['dialogue']
    assert 'HIDDEN_RESPONSE' not in ds['train'][1]['dialogue']
    assert 'I hurt someone.' not in ds['test'][0]['dialogue']
    assert ds['train'][0]['dialogue_group']==ds['train'][1]['dialogue_group']
    assert ds['test'][0]['dialogue_group']!=ds['train'][0]['dialogue_group']
    with pytest.raises(ValueError,match='out of order'):_prosocial_context(DatasetDict(train=train.select([1])))


def test_clear_keeps_raw_rating_and_preserves_its_mean_in_jev(monkeypatch):
    raw=DatasetDict(train=Dataset.from_list([{'Excerpt':'A reading passage.','BT_easiness':-1.25,'ID':1,'s.e.':0.2,'Title':'Book','Author':'Writer','License':'PD'}]))
    monkeypatch.setattr(access,'load_dataset',lambda *args,**kwargs:raw)
    assert load_task('clear/readability')['train'][0]['labels']==-1.25
    row=load_task('clear/readability',recast='jev')['train'][0]
    assert sum(row['target'])==pytest.approx(1)
    assert sum(i*p for i,p in zip(range(-4,3),row['target']))==pytest.approx(-1.25)


def test_gap_marks_exact_occurrences_and_rejects_wrong_offsets():
    raw={'Text':'Alice told Beth that she won.','Pronoun':'she','Pronoun-offset':21,'A':'Alice','A-offset':0,'B':'Beth','B-offset':11}
    assert _gap_text(raw)=='[A]Alice[/A] told [B]Beth[/B] that [PRONOUN]she[/PRONOUN] won.'
    with pytest.raises(ValueError,match='span/text mismatch'):_gap_text({**raw,'Pronoun-offset':20})


def test_prosocial_shared_openings_keep_entire_conversations_in_holdouts():
    train=Dataset.from_list([dict(context='Shared opening.',response='Reply.',dialogue_id=0,response_id=0),
        dict(context='Later training turn.',response='Reply.',dialogue_id=0,response_id=1),
        dict(context='Unique training opening.',response='Reply.',dialogue_id=1,response_id=0)])
    test=Dataset.from_list([dict(context='Shared opening.',response='Different reply.',dialogue_id=0,response_id=0)])
    ds=_prosocial_context(DatasetDict(train=train,test=test))
    assert len(ds['train'])==len(ds['test'])==1
    assert ds['train'][0]['context']=='Unique training opening.'
    assert not set(ds['train']['dialogue_group']) & set(ds['test']['dialogue_group'])


@pytest.mark.parametrize('task,view',[('trip/cloze','cloze'),('trip/order','order')])
def test_trip_native_stories_keep_gold_and_annotations_out_of_inputs(monkeypatch,task,view):
    raw=DatasetDict(train=Dataset.from_list([dict(example_id='7-C0',label=1,stories=[
        dict(story_id=7,sentences=['The glass broke.','She drank from the intact glass.'],plausible=False),
        dict(story_id=7,sentences=['The glass broke.','She swept up the pieces.'],plausible=True)],
        breakpoint=1,confl_sents=[0])]))
    monkeypatch.setattr(access,'load_dataset',lambda *args,**kwargs:raw)
    row=load_task(task,recast='jev')['train'][0]
    assert len(row['criteria'])==2
    assert row['criteria'][row['label']]=='The glass broke.\nShe swept up the pieces.'
    assert 'physically plausible' in row['instructions']
    assert 'plausible=True' not in row['state'] and 'breakpoint' not in row['state']
    metadata=json.loads(row['metadata'])
    assert metadata['source_view']==view and metadata['split_group_id']=='7'
