import json

from datasets import Dataset, DatasetDict

from tasksource.tasks import (_coqa_yesno, _dialogue_records, _hatexplain_majority,
                              _hotpot_facts, _qasper_yesno, _quac_records, _top_intent)


def test_quac_prior_answers_only_and_unanswerable_sentinel_removed():
    row=dict(context='Alice won. CANNOTANSWER',wikipedia_page_title='Article',section_title='Section',
        dialogue_id='d',turn_ids=['d0','d1'],questions=['Who won?','Did Bob win?'],
        orig_answers=dict(texts=['Alice','CANNOTANSWER']),yesnos=[2,1],followups=[0,1])
    results=list(_quac_records(row))
    assert [r['labels'] for r in results]==['yes','no']
    assert 'CANNOTANSWER' not in results[1]['sentence1']
    assert 'Question: Who won?' in results[1]['sentence1']
    assert 'Did Bob win?' not in results[0]['sentence1']
    assert 'Current answer:' not in results[1]['sentence2']


def test_hatexplain_majority_ties_are_not_arbitrarily_resolved():
    assert _hatexplain_majority(dict(annotators=dict(label=[0,0,2])))==0
    assert _hatexplain_majority(dict(annotators=dict(label=[0,1,2]))) is None


def test_hotpot_uses_native_title_and_sentence_identity_not_text_matching():
    row=dict(id='q',question='Who won?',context=dict(title=['A','B'],sentences=[['Alice won.'],['Alice won.']]),
        supporting_facts=dict(title=['A'],sent_id=[0]))
    results=list(_hotpot_facts(row))
    assert [r['labels'] for r in results]==['yes','no']
    assert len({json.loads(r['metadata'])['id'] for r in results})==2


def test_dialogue_state_targets_and_future_turns_never_enter_input():
    row=dict(dialogue_id='d',turns=dict(speaker=[0,1,0],utterance=['Find a cheap hotel.','FUTURE RESPONSE','Thanks.'],
        frames=[dict(service=['hotel'],state=[dict(active_intent='SECRET_INTENT',
                    slots_values=dict(slots_values_name=['hotel-pricerange'],slots_values_list=[['cheap']]))]),
                dict(service=[],state=[]),dict(service=[],state=[])]))
    intent=list(_dialogue_records(row,'intent'))[0]
    assert intent['labels']=='SECRET_INTENT'
    assert 'SECRET_INTENT' not in intent['sentence1'] and 'FUTURE' not in intent['sentence1']
    slot=list(_dialogue_records(row,'slot'))[0]
    assert slot['labels']=='cheap' and 'hotel-pricerange' in slot['sentence2']


def test_coqa_only_native_exact_binary_answers_with_previous_nonbinary_context():
    row=dict(story='Alice won.',source='wikipedia',questions=['Who won?','Was it Alice?','Was it Bob?'],
        answers=dict(input_text=['Alice','yes','no']))
    results=list(_coqa_yesno(row))
    assert [r['labels'] for r in results]==['yes','no']
    assert 'Answer: Alice' in results[0]['sentence1']
    assert 'Was it Bob?' not in results[0]['sentence1']


def test_qasper_requires_consistent_binary_annotations_and_text_evidence():
    row=dict(id='paper',title='Title',abstract='Abstract',full_text=dict(paragraphs=[['Our model won.']]),
        qas=dict(question=['Did it win?'],question_id=['q'],answers=[dict(annotation_id=['a','b'],answer=[
            dict(yes_no=True,unanswerable=False,evidence=['Our model won.']),
            dict(yes_no=True,unanswerable=False,evidence=['Our model won.'])])]))
    assert list(_qasper_yesno(row))[0]['labels']=='yes'
    row['qas']['answers'][0]['answer'][1]['yes_no']=False
    assert list(_qasper_yesno(row))==[]
    row['qas']['answers'][0]['answer'][1]['yes_no']=True
    row['qas']['answers'][0]['answer'][1]['evidence'].append('FLOAT SELECTED: figure only')
    assert list(_qasper_yesno(row))==[]
    for a in row['qas']['answers'][0]['answer']:a['evidence']=['FLOAT SELECTED: figure only']
    assert list(_qasper_yesno(row))==[]


def test_topv2_uses_only_root_intent():
    assert _top_intent(dict(semantic_parse='[IN:PLAY_MUSIC Play [SL:ARTIST Queen ] ]'))=='play music'


def test_data_only_loader_override_keeps_source_and_subset(monkeypatch):
    from tasksource import access
    from tasksource.preprocess import Classification
    mapping=Classification('text',labels='label',dataset_name='org/source',config_name='native-subset',
        complete_splits=False,load_dataset_kwargs={'revision':'pinned','name':'default','data_dir':'{config_name}'})
    calls=[]
    def loader(repo,config,**kwargs):
        calls.append((repo,config,kwargs))
        return DatasetDict(train=Dataset.from_dict(dict(text=['First','Second'],label=['yes','no'])))
    monkeypatch.setattr(access,'load_preprocessing',lambda *a,**k:mapping)
    monkeypatch.setattr(access,'load_dataset',loader)
    result=access.load_task('example')
    assert calls==[('org/source','default',dict(revision='pinned',data_dir='native-subset'))]
    assert mapping.config_name=='native-subset' and len(result['train'])==2


def test_slurp_scenario_action_disambiguates_raw_aliases():
    from scripts.repackage_dataset.dialogue import _slurp_intent
    assert _slurp_intent(dict(scenario='weather',action='query',intent='query'))=='weather_query'
    assert _slurp_intent(dict(scenario='calendar',action='query',intent='query'))=='calendar_query'
    assert _slurp_intent(dict(scenario='play',action='music',intent='play_music'))=='play_music'
