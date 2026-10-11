import csv
import io
import json
import tarfile

from datasets import Dataset, DatasetDict
import pytest

from tasksource import load_task
from tasksource.tasks import (_absa_records, _hoc_records, _maud_question, _multidogo_context,
                              _record_candidates, _wikisql_conditions, _wikisql_inputs)
from scripts.repackage_dataset import specialized_text as mirrors


def test_record_retains_multiple_gold_candidates_and_no_hidden_test_labels():
    source=dict(passage='Alice and Alicia are the same person. Bob is another person.',
        query='@placeholder won.',entities=['Alice','Alicia','Bob','Alice'],answers=['Alice','Alicia'],
        idx=dict(passage=3,query=2))
    rows=list(_record_candidates(source))
    assert [r['labels'] for r in rows]==['yes','yes','no']
    assert len({json.loads(r['metadata'])['id'] for r in rows})==3
    assert list(_record_candidates({**source,'answers':[]}))==[]


def test_hoc_binary_projection_preserves_multilabel_supervision():
    labels=['inducing angiogenesis','resisting cell death']
    rows=list(_hoc_records(dict(document_id='123_4',text='A sentence.',labels=labels)))
    assert len(rows)==10 and sum(r['labels']=='yes' for r in rows)==2
    assert all(json.loads(r['metadata'])['split_group_id']=='123' for r in rows)
    assert len({json.loads(r['metadata'])['id'] for r in rows})==10


def test_absa_checks_spans_and_keeps_conflict():
    row=dict(sentenceId='1',text='Great food, uneven service.',aspectTerms=[
        dict(term='service',polarity='conflict',**{'from':'19','to':'26'})])
    result=list(_absa_records(row))
    assert result[0]['labels']=='conflict'
    row['aspectTerms'][0]['to']='25'
    with pytest.raises(ValueError,match='span'):
        list(_absa_records(row))


def test_dialogue_context_has_only_earlier_text_and_no_annotations():
    rows=[dict(conversationId='c',turnNumber=i,sentenceNumber=0,utteranceId=str(i),
               utterance=text,intent=label) for i,text,label in
          [(2,'Tomorrow.','date'),(0,'Book a flight.','bookflight'),(1,'What date?','requestdate')]]
    out=_multidogo_context(DatasetDict(train=Dataset.from_list(rows)))['train']
    by_id={r['utteranceId']:r for r in out}
    assert by_id['0']['sentence1']==''
    assert 'Book a flight.' in by_id['1']['sentence1']
    assert 'Tomorrow.' not in by_id['1']['sentence1']
    assert 'bookflight' not in by_id['2']['sentence1']


def test_wikisql_inputs_omit_gold_sql_and_conditions_keep_native_operators():
    row=dict(question='How many players scored more than five goals?',
        table=dict(id='table',header=['Player','Goals'],types=['text','real']),
        sql=dict(agg=3,sel=0,human_readable='SECRET GOLD SQL',
                 conds=dict(column_index=[1],operator_index=[1],condition=['5'])))
    assert 'SECRET' not in _wikisql_inputs(row)
    result=list(_wikisql_conditions(row))
    assert result[0]['labels']=='1' and 'Goals' in result[0]['sentence2']
    assert 'greater than' not in result[0]['sentence2']


def test_maud_keeps_one_native_question_without_subquestion_label_confusion():
    rows=Dataset.from_list([dict(question=q,subquestion=s,text=q+s,answer='native') for q,s in
                           [('consideration','<NONE>'),('other','<NONE>'),('consideration','a subquestion')]])
    assert len(_maud_question(DatasetDict(train=rows),'consideration')['train'])==1


@pytest.mark.parametrize('task', ['squad-v2/answerability','cuad/answerability'])
def test_answerability_shared_loader_preserves_negative_and_positive(task,monkeypatch):
    data=DatasetDict({s:Dataset.from_list([dict(id=str(i),title=s,context='The answer is here.',question='What is here?',
        answers=dict(text=['The answer'] if i else [],answer_start=[0] if i else [])) for i in range(2)])
        for s in ('train','test' if task.startswith('cuad') else 'validation')})
    monkeypatch.setattr('tasksource.access.load_dataset',lambda *a,**kw:data)
    out=load_task(task,recast='jev')
    assert set(out)==set(data)
    assert [r['answer'] for r in out['train']]==['no','yes']
    assert all('answer_spans' in json.loads(r['metadata']) for r in out['train'])


def test_clinical_join_uses_prompt_not_first_evidence_from_article(tmp_path,monkeypatch):
    text='Treatment increased pain. Treatment decreased nausea.'
    annotations=[]
    for pid,code,start,end in [('1','1',0,25),('2','-1',26,len(text))]:
        annotations.append(dict(PromptID=pid,PMCID='10',**{'Valid Label':'True','Valid Reasoning':'True',
            'Label Code':code,'Evidence Start':start,'Evidence End':end}))
    prompts=[dict(PromptID=str(i),PMCID='10',Intervention='treatment',Comparator='placebo',Outcome=o)
             for i,o in [(1,'pain'),(2,'nausea')]]
    def as_csv(rows):
        out=io.StringIO();w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return out.getvalue()
    files={'annotations_merged.csv':as_csv(annotations),'prompts_merged.csv':as_csv(prompts),
        'txt_files/PMC10.txt':text,'splits/train_article_ids.txt':'10',
        'splits/validation_article_ids.txt':'','splits/test_article_ids.txt':''}
    archive=io.BytesIO()
    with tarfile.open(fileobj=archive,mode='w:gz') as tar:
        for name,value in files.items():
            raw=value.encode();info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    raw=archive.getvalue()
    monkeypatch.setattr(mirrors,'EVIDENCE_SHA256',mirrors.hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(mirrors.urllib.request,'urlopen',lambda url:io.BytesIO(raw))
    script=tmp_path/'upstream.py';script.write_text('INCORRECT_PROMPT_IDS = set([999])')
    monkeypatch.setattr(mirrors,'hf_hub_download',lambda *a,**kw:str(script))
    monkeypatch.setattr(mirrors,'write_mirror',lambda name,splits,*a:splits)
    rows=mirrors.evidence_inference()['train']
    assert [r['labels'] for r in rows]==['1','-1']
    evidence=[json.loads(r['metadata'])['evidence_spans'][0]['text'] for r in rows]
    assert 'pain' in evidence[0] and 'nausea' in evidence[1]
    assert all(r['sentence1']==text for r in rows)


def test_identical_text_holdouts_take_priority_and_conflicting_labels_are_excluded():
    from tasksource.tasks import _unique_text_splits
    data=DatasetDict({s:Dataset.from_list(rows) for s,rows in {
        'train':[{'text':'shared','label':0},{'text':'conflicting','label':0},{'text':'unique','label':1}],
        'validation':[{'text':'shared','label':0}],
        'test':[{'text':'shared','label':0},{'text':'conflicting','label':1}]}.items()})
    result=_unique_text_splits(data,lambda r:r['text'],'label')
    assert result['train']['text']==['unique']
    assert result['validation']['text']==[]
    assert result['test']['text']==['shared']


def test_dialogue_multilabel_rows_are_context_but_not_fused_intent_classes():
    rows=Dataset.from_list([dict(conversationId='c',turnNumber=i,sentenceNumber=0,utteranceId=str(i),
        utterance=t,intent=label) for i,t,label in [(0,'Pizza and soda.','pizza<div>drink'),(1,'Thanks.','thankyou')]])
    result=_multidogo_context(DatasetDict(train=rows))['train']
    assert len(result)==1 and result[0]['intent']=='thankyou'
    assert 'Pizza and soda.' in result[0]['sentence1']
