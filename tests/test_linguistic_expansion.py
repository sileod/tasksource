import json

from datasets import Dataset, DatasetDict

from tasksource.tasks import (_document_holdouts, _magpie_records, _maven_triggers,
                              _preco_pairs, _redial_preferences)


def test_native_maven_trigger_and_gold_type():
    source=dict(input='Alice left.',output=dict(json_structures={'Departing':[
        dict(trigger=dict(text='left',start=6,end=10))]}))
    row=list(_maven_triggers(source))[0]
    assert row['sentence1']=='Alice [TARGET]left[/TARGET].'
    assert row['labels']=='Departing'
    assert 'Departing' not in row['sentence1']


def test_magpie_marks_native_discontinuous_offsets_and_keeps_full_context():
    row=dict(context=['Before.','Earlier.','He gives me the creeps.','Later.','After.'],
             offsets=[[3,8],[9,11],[16,22]],label='i',confidence=1.,idiom='give someone the creeps',
             id=1,document_id='doc',label_distribution={'i':1.,'l':0.})
    out=list(_magpie_records(row))[0]
    assert 'He [TARGET]gives[/TARGET] [TARGET]me[/TARGET] the [TARGET]creeps[/TARGET].' in out['sentence1']
    assert out['sentence1'].startswith('Before.') and out['sentence1'].endswith('After.')
    assert out['labels']=='i'
    row['confidence']=.5
    assert not list(_magpie_records(row))


def test_shared_source_documents_are_assigned_to_native_holdouts():
    data=DatasetDict({s:Dataset.from_dict(dict(document_id=ids)) for s,ids in
        [('train',['train','shared']),('validation',['dev','shared']),('test',['shared'])]})
    out=_document_holdouts(data,'document_id')
    assert out['train']['document_id']==['train'] and out['validation']['document_id']==['dev']
    assert out['test']['document_id']==['shared']


def test_preco_pairs_use_explicit_clusters_and_complete_document():
    row=dict(id='doc',sentences=[['Alice','saw','Bob','.'],['She','waved','.']],
             mention_clusters=[[[0,0,1],[1,0,1]],[[0,2,3]]])
    out=list(_preco_pairs(row));positive=next(r for r in out if r['labels']=='yes')
    assert positive['sentence1']=='[A]Alice[/A] saw Bob . [B]She[/B] waved .'
    negatives=[r for r in out if r['labels']=='no']
    assert negatives and all('Bob' in r['sentence1'] for r in negatives)
    assert all(json.loads(r['metadata'])['split_group_id']=='doc' for r in out)
    assert len(out)<=8


def test_redial_uses_seeker_consensus_and_native_movie_names():
    row=dict(conversationId=1,initiatorWorkerId=7,movieMentions=[dict(movieId='12',movieName='Arrival')],
        messages=[dict(senderWorkerId=7,text='I loved @12.')],
        initiatorQuestions=[dict(movieId='12',liked=1)],respondentQuestions=[dict(movieId='12',liked=1)])
    out=list(_redial_preferences(row))[0]
    assert out['sentence1']=='Seeker: I loved Arrival.' and out['labels']=='1'
    assert out['sentence2']=='Movie: Arrival'
    row['respondentQuestions'][0]['liked']=0
    assert not list(_redial_preferences(row))


def test_single_text_flattening_does_not_create_empty_second_input():
    from tasksource.tasks import _flatten_text
    source=DatasetDict(train=Dataset.from_list([dict(input='Alice left.',output=dict(json_structures={'Departing':[dict(trigger=dict(text='left',start=6,end=10))]}))]))
    result=_flatten_text(source,_maven_triggers,paired=False)
    assert 'sentence2' not in result['train'].column_names


def test_holdouts_can_detect_duplicate_documents_despite_different_native_ids():
    data=DatasetDict(train=Dataset.from_list([dict(id='train1',text='same'),dict(id='train2',text='unique')]),
                     validation=Dataset.from_list([dict(id='dev1',text='same')]))
    result=_document_holdouts(data,lambda r:r['text'])
    assert result['train']['id']==['train2'] and result['validation']['id']==['dev1']
