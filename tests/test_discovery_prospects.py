import json
from types import SimpleNamespace
import sys

from tasksource.task_discovery import discover_task_prospects as discovery
from tasksource.task_discovery import jev_review


def test_training_check_records_native_split_and_labeled_preview(monkeypatch):
    calls=[]
    def cached(url,*args):
        calls.append(url)
        data={'splits':[{'config':'main','split':'train'}]} if '/splits?' in url else {'features':[{'name':'label'}],'rows':[{'row':{'label':1}}]}
        return dict(data=data,sha256='source-hash')
    monkeypatch.setattr(discovery,'cached_json',cached)
    result=discovery.training_evidence({'name':'Source','huggingface':['org/data']},'unused')
    assert result['train_status']=='present' and len(result['evidence'])==2
    assert result['split_policy']=='train_only' and result['source_splits']==['train']
    assert 'length=1' in calls[1] and result['evidence'][1]['metadata']['rows'][0]['row']['label']==1


def test_incomplete_viewer_metadata_does_not_prove_no_training_split(monkeypatch):
    monkeypatch.setattr(discovery,'cached_json',lambda *a:dict(sha256='hash',data={
        'splits':[{'config':'main','split':'test'}],'pending':[{'config':'other'}]}))
    result=discovery.training_evidence({'name':'Source','huggingface':['org/data']},'unused')
    assert result['train_status']=='unknown'


def reviewed(*train_votes,evidence_count=1,train_status='present'):
    reviews=[]
    for vote in train_votes:
        answers={q:dict(probabilities={'yes':0.95,'no':0.025,'unknown':0.025}) for q in jev_review.CHECKS}
        answers['train']['probabilities']={v:0.95 if v==vote else 0.025 for v in ('yes','no','unknown')}
        reviews.append(dict(reviewer='jev_review',data={'answers':answers}))
    return dict(reviews=reviews,review_status='reviewed',review_evidence_count=evidence_count,train_status=train_status)


def test_prospect_selection_preserves_unknown_disagreement_and_native_split_evidence():
    assert discovery.prospect_status(reviewed('yes','yes'))=='promising'
    assert discovery.prospect_status(reviewed('no','no'))=='not_suitable'
    assert discovery.prospect_status(reviewed('yes','no'))=='needs_evidence'
    assert discovery.prospect_status(reviewed('yes','yes',evidence_count=0))=='needs_evidence'
    assert discovery.prospect_status(reviewed('yes','yes',train_status='not_listed'))=='needs_evidence'


def test_review_preserves_history_model_answers_and_evidence_without_promoting(tmp_path,monkeypatch):
    monkeypatch.setenv('JEV_OPENROUTER_API_KEY','private-test-key')
    monkeypatch.setattr(jev_review,'evidence',lambda *a:dict(evidence=[{'url':'source','text':'Native training labels.'}]))
    result=SimpleNamespace(failed=False,model_used='openrouter/test',data={'model':'dated-model','answers':{}})
    calls=[]
    def complete(states,**kwargs):calls.append(kwargs);kwargs['on_result'](0,result);return [result]
    monkeypatch.setitem(sys.modules,'litlm',SimpleNamespace(complete=complete,ensemble=None))
    monkeypatch.setitem(sys.modules,'litlm_cli',SimpleNamespace(_record=lambda r:dict(model=r.model_used,data=r.data,failed=r.failed)))
    source=dict(name='Candidate',reviews=[{'reviewer':'earlier-reviewer'}])
    manifest=jev_review.run([source],tmp_path/'review',tmp_path/'cache',['openrouter/test'])
    output=json.loads((tmp_path/'review/reviews.jsonl').read_text())
    assert output['reviews'][0]==source['reviews'][0]
    assert output['reviews'][1]['data']['model']=='dated-model'
    assert output['requires_processing'] and output['review_evidence_count']==1
    assert manifest['records']==1 and calls[0]['decisions']==jev_review.QUESTIONS
    assert 'private-test-key' not in ''.join(p.read_text() for p in (tmp_path/'review').glob('*'))


def test_discovery_uses_current_evidence_and_question_reviews_not_stale_history():
    row=reviewed('yes','yes')
    row.update(review_state_sha256='current',review_questions_sha256='prompt-v2')
    for r in row['reviews']:r.update(state_sha256='current',questions_sha256='prompt-v2')
    stale=reviewed('no')['reviews'][0];stale.update(state_sha256='old',questions_sha256='prompt-v1')
    row['reviews'].append(stale)
    assert discovery.prospect_status(row)=='promising'


def test_language_filter_prefers_english_and_keeps_unknown():
    assert discovery.language_priority({'languages':['English']},['en'])==0
    assert discovery.language_priority({'languages':['de','en']},['en'])==0
    assert discovery.language_priority({'languages':'en'},['en'])==0
    assert discovery.language_priority({'languages':[]},['en'])==1
    assert discovery.language_priority({'languages':['tr']},['en'])==2
    assert discovery.language_priority({'languages':['tr']},[])==0


def test_multiple_splits_do_not_claim_disjointness(monkeypatch):
    monkeypatch.setattr(discovery,'cached_json',lambda *a:dict(sha256='hash',data={
        'splits':[{'config':'main','split':'train'},{'config':'main','split':'test'}]}))
    result=discovery.training_evidence({'name':'Source','huggingface':['org/data']},'unused')
    assert result['source_splits']==['test','train'] and result['split_policy']=='multiple_splits'
    assert 'disjoint' not in result
