import json

import pytest
from datasets import Dataset, DatasetDict
from tasksource import eval_only, list_tasks


def garage_row():
    return dict(sample_id='sample',question='What happened?',question_date='1 January 2024',
        question_complexity='Multi-Hop',question_valid='YES',
        grounding=[dict(cite_1='Direct answer.',cite_2=None,date='2023-12-01',age='1 month',provider='web'),
                   dict(cite_1=None,cite_2='Old answer.',date='2020-01-01',age='4 years',provider='ent'),
                   dict(cite_3='Unannotated passage.',date='not defined')],
        evidence_correct=['ANSWER-THE-QUESTION','OUTDATED',''],
        evidence_relevant=['YES','YES','NO'],evidence_cited=['YES','NO','NO'],
        answer_generate='SECRET GOLD ANSWER')


def test_garage_flattens_aligned_citations_and_skips_unannotated():
    row=garage_row();records=list(eval_only._garage_records(row))
    assert [r['labels'] for r in records]==['ANSWER-THE-QUESTION','OUTDATED']
    meta=[json.loads(r['metadata']) for r in records]
    assert [m['id'] for m in meta]==['sample:0','sample:1']
    assert [m['citation'] for m in meta]==['cite_1','cite_2']
    assert meta[1]['passage_date']=='2020-01-01' and meta[1]['question_complexity']=='Multi-Hop'
    assert meta[0]['evidence_cited']=='YES' and meta[1]['provider']=='ent'
    assert all('SECRET' not in r['sentence1'] and 'cite_' not in r['sentence1'] for r in records)
    assert 'Question date: 1 January 2024' in records[0]['sentence1']


def test_garage_rejects_misaligned_or_unrecognized_annotations():
    row=garage_row();row['evidence_cited']=[]
    with pytest.raises(ValueError,match='length mismatch'):list(eval_only._garage_records(row))
    row=garage_row();row['evidence_correct'][0]='MODEL GUESS'
    with pytest.raises(ValueError,match='Unknown GaRAGe'):list(eval_only._garage_records(row))
    row=garage_row();row['grounding'][0]['cite_2']='Ambiguous second passage'
    with pytest.raises(ValueError,match='exactly one'):list(eval_only._garage_records(row))


def test_garage_templates_have_explicit_labels_and_no_invented_splits():
    source=DatasetDict(train=Dataset.from_list([garage_row()]))
    for task,expected in [(eval_only.garage__answer_support,['yes','other evidence']),
                          (eval_only.garage__evidence_type,['ANSWER-THE-QUESTION','OUTDATED'])]:
        result=task(source)
        assert set(result)=={'train'} and len(result['train'])==2
        names=result['train'].features['labels'].names
        assert [names[v] for v in result['train']['labels']]==expected
        assert task.complete_splits is False
    assert not any(x.startswith(('garage/','bright/','followir/','trec-rag/')) for x in list_tasks().id)


def test_bright_loads_original_queries_and_only_matching_corpus(monkeypatch):
    calls=[]
    def load(repo,config,**kwargs):
        calls.append((repo,config,kwargs));return Dataset.from_list([dict(id='native')])
    monkeypatch.setattr(eval_only,'load_dataset',load)
    result=eval_only.load_bright('biology')
    assert set(result)=={'queries','corpus'}
    assert [c[1] for c in calls]==['examples','documents']
    assert all(c[2]['split']=='biology' and len(c[2]['revision'])==40 for c in calls)
    assert all('reason' not in c[1] for c in calls)


def test_followir_preserves_original_and_changed_human_judgments(monkeypatch):
    calls=[]
    def load(repo,config,**kwargs):
        calls.append((repo,config,kwargs));return Dataset.from_list([dict(score=16)])
    monkeypatch.setattr(eval_only,'load_dataset',load)
    result=eval_only.load_followir('news21')
    assert set(result)=={'queries','corpus','qrels_og','qrels_changed','top_ranked'}
    assert result['qrels_changed'][0]['score']==16  # Never binarize native News21 grades.
    assert all(c[0]=='jhu-clsp/news21-instructions' for c in calls)
    assert all('train' not in c[2]['split'] for c in calls)
    with pytest.raises(ValueError):eval_only.load_followir('synthetic-train')


def test_trec_rag_keeps_candidates_separate_from_manual_qrels(monkeypatch):
    calls=[]
    def load(loader,**kwargs):
        calls.append((loader,kwargs));return Dataset.from_list([dict(native=True)])
    monkeypatch.setattr(eval_only,'load_dataset',load)
    result=eval_only.load_trec_rag_2025()
    assert set(result)=={'requests','qrels'}
    assert calls[1][1]['column_names']==['query_id','iteration','document_id','grade']
    assert calls[1][1]['data_files']['test'].endswith('/qrels/qrels.rag25.test.txt')
    assert 'umbrela' not in calls[1][1]['data_files']['test']
    assert all(c[1]['split']=='test' for c in calls)


def test_garage_jev_retains_instructions_gold_and_source_identity():
    from tasksource.jev.recast import recast_jev
    task=eval_only.garage__evidence_type
    canonical=task(DatasetDict(train=Dataset.from_list([garage_row()])))
    result=recast_jev(canonical,task=task.task_id,question=task.question)['train']
    assert [r['criteria'][r['label']] for r in result]==['ANSWER-THE-QUESTION','OUTDATED']
    assert all(r['instructions']==task.question for r in result)
    assert {json.loads(r['metadata'])['split_group_id'] for r in result}=={'sample'}
