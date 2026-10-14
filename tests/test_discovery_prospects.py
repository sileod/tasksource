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
    assert calls[0]['adaptive_concurrency'] and calls[0]['max_concurrency']==32
    assert 'per_key_rpm' not in calls[0]
    assert 'private-test-key' not in ''.join(p.read_text() for p in (tmp_path/'review').glob('*'))
    jev_review.run([source],tmp_path/'english',tmp_path/'cache',['openrouter/test'],languages=['en'])
    assert 'dataset_language' in calls[-1]['decisions']
    english=json.loads((tmp_path/'english/reviews.jsonl').read_text())
    assert english['review_languages']==['en']
    assert 'dataset_language' not in jev_review.QUESTIONS


def test_language_scope_checks_dataset_not_paper_language():
    row=reviewed('yes')
    row.update(paper_language='en',review_languages=['en'])
    assert discovery.prospect_status(row)=='needs_evidence'
    answer=row['reviews'][0]['data']['answers']
    answer['dataset_language']=dict(probabilities={'yes':0,'no':1,'unknown':0})
    assert discovery.prospect_status(row)=='not_suitable'
    answer['dataset_language']=dict(probabilities={'yes':1,'no':0,'unknown':0})
    assert discovery.prospect_status(row)=='promising'


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


def test_annotation_provenance_is_preserved_without_becoming_an_eligibility_vote():
    row=reviewed('yes','yes')
    for review in row['reviews']:
        review['data']['answers']['annotation_origin']={'type':'choice','probabilities':{'model':0.95,'unknown':0.05}}
    assert discovery.prospect_status(row)=='promising'
    assert set(jev_review.PROVENANCE).isdisjoint(jev_review.CHECKS)
    assert all('unknown' in jev_review.QUESTIONS[q]['criteria'] for q in jev_review.PROVENANCE)


def test_prior_screening_excludes_names_and_hub_copies_but_not_shared_papers():
    previous=[dict(name='Old source',huggingface=['https://huggingface.co/datasets/Org/Data'],papers=[{'url':'arxiv:1234.5678'}])]
    candidates=[dict(name='OLD-source'),dict(name='Copy',huggingface=['org/data']),
                dict(name='New subset',papers=[{'url':'arxiv:1234.5678'}])]
    assert discovery.unseen_candidates(candidates,previous)==[candidates[2]]


def test_origin_filter_keeps_procedural_and_requires_current_consensus(tmp_path):
    row=reviewed('yes','yes')
    for review in row['reviews']:
        review['data']['answers']['annotation_origin']={'probabilities':{'human':0.1,'procedural':0.85,'model':0.05}}
    assert discovery.origin_matches(row,['human','procedural'])
    row['reviews'][1]['data']['answers']['annotation_origin']={'probabilities':{'model':0.9,'unknown':0.1}}
    assert not discovery.origin_matches(row,['human','procedural'])
    assert discovery.origin_matches(row,[])
    manifest=discovery.save_prospects([row],tmp_path,['human','procedural'])
    assert manifest['annotation_filtered']==1 and manifest['handoff_count']==0
    assert json.loads((tmp_path/'prospects.jsonl').read_text())['prospect_status']=='promising'
    assert (tmp_path/'handoff.jsonl').read_text()==''
    for review in row['reviews']:review['data']['answers'].pop('annotation_origin')
    assert not discovery.origin_matches(row,['human','procedural'])


def test_saved_reviews_filter_never_calls_catalog_or_models(tmp_path,monkeypatch):
    from tasksource.task_discovery import discover_task_prospects as cli
    source=tmp_path/'reviews.jsonl';source.write_text(json.dumps(reviewed('yes'))+'\n')
    monkeypatch.setattr(cli,'catalog_index',lambda:(_ for _ in ()).throw(AssertionError('catalog called')))
    monkeypatch.setattr(cli,'run',lambda *a:(_ for _ in ()).throw(AssertionError('model called')))
    monkeypatch.setattr(sys,'argv',['discovery',str(source),'--from-reviews','--annotation-origin','human','--output',str(tmp_path/'filtered')])
    cli.main()
    manifest=json.loads((tmp_path/'filtered/manifest.json').read_text())
    assert manifest['from_reviews'] and manifest['annotation_filtered']==1


def test_training_round_splits_and_config_are_respected(monkeypatch):
    data={'splits':[{'config':'wrong','split':'train'}, {'config':'right','split':'train_r1'}, {'config':'right','split':'test_r1'}]}
    calls=[]
    def cached(url,*a):
        calls.append(url)
        return dict(sha256='hash',data=data if '/splits?' in url else {'rows':[{'row':{'label':1}}]})
    monkeypatch.setattr(discovery,'cached_json',cached)
    result=discovery.training_evidence({'huggingface':['org/data'],'config_name':'right'},'unused')
    assert result['train_status']=='present' and result['source_splits']==['test_r1','train_r1']
    assert 'config=right' in calls[1] and 'split=train_r1' in calls[1]
    assert discovery.observed_resource(result)
    assert not discovery.is_train_split('untrained') and not discovery.is_train_split('trainval')


def test_downloaded_row_establishes_resource_without_changing_training_or_fit_gates():
    row=reviewed('yes','yes')
    for r in row['reviews']:r['data']['answers']['resources']['probabilities']={'no':0.99,'yes':0.01}
    assert discovery.prospect_status(row)=='not_suitable'
    row['evidence']=[{'url':'https://datasets-server.huggingface.co/rows?dataset=org/data','sha256':'hash','metadata':{'rows':[{'row':{'label':1}}]}}]
    assert discovery.prospect_status(row)=='promising'
    for r in row['reviews']:r['data']['answers']['decision_fit']['probabilities']={'no':0.99,'yes':0.01}
    assert discovery.prospect_status(row)=='not_suitable'


def test_verified_synthetic_filter_requires_independent_checks_and_full_coverage():
    row=reviewed('yes','yes')
    for review in row['reviews']:
        answers=review['data']['answers']
        answers.update(annotation_origin={'probabilities':{'model':0.95}},
                       annotation_checks={'probabilities':{'executable':0.95}},
                       verification_scope={'probabilities':{'all':0.95}})
    assert not discovery.origin_matches(row,['human','procedural'])
    assert discovery.origin_matches(row,['human','procedural'],allow_verified_synthetic=True)
    answers['verification_scope']={'probabilities':{'sample':0.95}}
    assert not discovery.origin_matches(row,['human','procedural'],allow_verified_synthetic=True)
    answers['verification_scope']={'probabilities':{'all':0.95}}
    answers['annotation_checks']={'probabilities':{'model_review':0.95}}
    assert not discovery.origin_matches(row,['human','procedural'],allow_verified_synthetic=True)
    answers.pop('annotation_checks')
    assert not discovery.origin_matches(row,['human','procedural'],allow_verified_synthetic=True)


def test_allow_verified_synthetic_alone_filters_unverified_model_labels(tmp_path,monkeypatch):
    verified=reviewed('yes');unverified=reviewed('yes')
    for row in (verified,unverified):
        row['reviews'][0]['data']['answers'].update(annotation_origin={'probabilities':{'model':0.95}},
            annotation_checks={'probabilities':{'executable':0.95}},verification_scope={'probabilities':{'all':0.95}})
    unverified['reviews'][0]['data']['answers']['annotation_checks']={'probabilities':{'model_review':0.95}}
    source=tmp_path/'reviews.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in (verified,unverified)))
    monkeypatch.setattr(sys,'argv',['discovery',str(source),'--from-reviews','--allow-verified-synthetic','--output',str(tmp_path/'filtered')])
    discovery.main()
    manifest=json.loads((tmp_path/'filtered/manifest.json').read_text())
    assert manifest['annotation_origins']==['human','procedural'] and manifest['handoff_count']==1
    assert manifest['annotation_filtered']==1


def test_prior_paper_identity_excludes_paper_leads_without_merging_dataset_views():
    prior=[dict(name='Original dataset',huggingface=[],papers=[{'url':'https://arxiv.org/abs/2401.12345v2'}])]
    paper=dict(name='Different paper title',candidate_kind='paper',papers=[{'url':'https://arxiv.org/abs/2401.12345'}])
    dataset=dict(paper,name='Distinct dataset from the same paper',candidate_kind='dataset')
    assert discovery.unseen_candidates([paper,dataset],prior)==[dataset]


def test_large_review_documents_are_bounded_without_mutating_source_evidence():
    docs=[dict(url='source',sha256='original-hash',metadata={'labels':['x'*10000]})]
    bounded=jev_review.bounded_documents(docs,limit=100)
    assert bounded[0]['truncated'] and bounded[0]['sha256']=='original-hash'
    assert len(bounded[0]['source_document_json_excerpt'])==100
    assert len(docs[0]['metadata']['labels'][0])==10000
    assert jev_review.bounded_documents([{'text':'small'}])==[{'text':'small'}]


def test_investigation_keeps_evidence_gaps_without_promoting_or_admitting_bad_fit(tmp_path):
    row=reviewed('yes');row['review_languages']=['en']
    answers=row['reviews'][0]['data']['answers']
    answers['dataset_language']={'probabilities':{'yes':0.65,'unknown':0.35}}
    answers['annotation_origin']={'probabilities':{'human':0.65,'unknown':0.35}}
    discovery.save_prospects([row],tmp_path,['human','procedural'],True)
    assert (tmp_path/'handoff.jsonl').read_text()==''
    lead=json.loads((tmp_path/'investigate.jsonl').read_text())
    assert lead['prospect_status']=='needs_evidence'
    assert 'dataset_language below strict confidence' in lead['investigation_reasons']
    answers['decision_fit']={'probabilities':{'yes':0.5,'unknown':0.5}}
    discovery.save_prospects([row],tmp_path,['human','procedural'],True)
    assert (tmp_path/'investigate.jsonl').read_text()==''


def test_investigation_excludes_known_unverified_synthetic_and_reviewer_disagreement(tmp_path):
    row=reviewed('yes')
    row['reviews'][0]['data']['answers']['annotation_origin']={'probabilities':{'model':0.95}}
    result=discovery.save_prospects([row],tmp_path,['human','procedural'],True)
    assert result['handoff_count']==result['investigate_count']==0
    row=reviewed('yes','no');row['prospect_status']='needs_evidence'
    assert discovery.investigation_reasons(row)==[]


def test_null_visual_preview_is_not_a_strict_handoff(tmp_path):
    row=reviewed('yes')
    row['evidence']=[{'url':'https://datasets-server.huggingface.co/rows?dataset=org/data',
                     'sha256':'hash','metadata':{'features':[{'name':'image','type':{'_type':'Image'}}],
                                               'rows':[{'row':{'image':None,'label':1}}]}}]
    assert discovery.prospect_status(row)=='needs_evidence'
    result=discovery.save_prospects([row],tmp_path)
    assert result['handoff_count']==0 and result['investigate_count']==1
    row['evidence'][0]['metadata']['rows'][0]['row']['image']={'src':'actual-image'}
    assert discovery.prospect_status(row)=='promising'


def test_documented_quality_concerns_hold_strong_handoff_without_erasing_lead(tmp_path):
    row=reviewed('yes')
    row['reviews'][0]['data']['answers']['supervision_quality']={'probabilities':{'concerns':0.95}}
    result=discovery.save_prospects([row],tmp_path)
    assert result['handoff_count']==0 and result['investigate_count']==1
    assert 'documented supervision quality concerns' in json.loads((tmp_path/'investigate.jsonl').read_text())['investigation_reasons']
