import json

from datasets import Dataset, DatasetDict
from tasksource import access
from tasksource.tasks import _fabsa_records, _tab_mentions


def review(text='The service was excellent.', original=1, counterfactual=False):
    return dict(text=text,original_id=original,counterfactual=counterfactual,rating=4,
                food=None,ambiance=None,service='Positive',noise='unknown')


def test_cebab_direct_views_preserve_pairs_and_median_rating(monkeypatch):
    source=DatasetDict(train=Dataset.from_list([review(),review('The service was awful.',counterfactual=True)]))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:source)
    result=access.load_task('cebab/rating',recast='jev')['train']
    assert len(result)==2 and set(result['kind'])=={'score'}
    assert all(r['criteria'][r['label']]=='4 stars' for r in result)
    metadata=[json.loads(r['metadata']) for r in result]
    assert {r['split_group_id'] for r in metadata}=={'1'}
    assert {r['counterfactual'] for r in metadata}=={True,False}
    assert all('counterfactual' not in r['state'] for r in result)


def test_cebab_unknown_is_not_neutral_and_null_is_not_a_label(monkeypatch):
    source=DatasetDict(train=Dataset.from_list([review(),dict(review('No service comment.'),service=None)]))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:source)
    result=access.load_task('cebab/service',recast='jev')['train']
    assert len(result)==1 and result[0]['criteria'][result[0]['label']]=='positive'
    assert result[0]['instructions']=='What sentiment does this restaurant review express about the service?'
    noise=access.load_task('cebab/noise',recast='jev')['train']
    assert all(r['criteria'][r['label']]=='no clear positive or negative sentiment' for r in noise)


def fabsa(text='The app is usable.', identity=1):
    return dict(id=identity,text=text,labels=[['Online experience: App website','neutral']])


def test_fabsa_retains_neutral_but_not_absent_or_conflicting_aspects():
    r=fabsa();r['labels'] += [['Staff support: Email','positive'],['Staff support: Email','negative']]
    rows=list(_fabsa_records(r))
    assert len(rows)==1 and rows[0]['labels']=='neutral'
    assert 'app website' in rows[0]['sentence2']
    assert len(json.loads(rows[0]['metadata'])['annotations'])==3


def test_fabsa_jev_holds_out_duplicate_reviews(monkeypatch):
    source=DatasetDict(train=Dataset.from_list([fabsa(),fabsa('Another app works.',2)]),
                       test=Dataset.from_list([fabsa(identity=3)]))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:source)
    result=access.load_task('fabsa/sentiment',recast='jev')
    assert len(result['train'])==1 and len(result['test'])==1
    assert result['train'][0]['criteria'][result['train'][0]['label']]=='neutral'
    assert result['train'][0]['instructions']=='What sentiment is expressed about online experience: app website?'
    assert 'text_B' not in result['train'][0]['state']


def document(identity='doc', annotator='a1', label='DIRECT'):
    return dict(doc_id=identity,text='Alice sued Bob.',task='Anonymise Alice.',annotator_id=annotator,
                entity_mentions=[dict(start_offset=0,end_offset=5,span_text='Alice',identifier_type=label,
                    entity_id=annotator+'-e1',entity_mention_id=annotator+'-m1',related_mentions=[])])


def test_tab_preserves_each_annotators_native_privacy_judgment():
    rows=[next(_tab_mentions(document())),next(_tab_mentions(document(annotator='a2',label='QUASI')))]
    assert [r['labels'] for r in rows]==['DIRECT','QUASI']
    assert rows[0]['sentence1']==rows[1]['sentence1']=='Anonymise Alice.\n\n[TARGET]Alice[/TARGET] sued Bob.'
    metadata=[json.loads(r['metadata']) for r in rows]
    assert {r['split_group_id'] for r in metadata}=={'doc'}
    assert {r['annotator'] for r in metadata}=={'a1','a2'}
    assert metadata[0]['annotation']['entity_id']=='a1-e1'


def test_tab_rejects_misaligned_spans():
    import pytest
    row=document();row['entity_mentions'][0]['span_text']='Bob'
    with pytest.raises(ValueError,match='span/text mismatch'):list(_tab_mentions(row))


def test_tab_jev_keeps_document_holdouts_and_target_person(monkeypatch):
    source=DatasetDict(train=Dataset.from_list([document(),document('other')]),
                       test=Dataset.from_list([document(annotator='a3')]))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:source)
    result=access.load_task('tab/identifier',recast='jev')
    assert {json.loads(r['metadata'])['split_group_id'] for r in result['train']}=={'other'}
    assert {json.loads(r['metadata'])['split_group_id'] for r in result['test']}=={'doc'}
    row=result['train'][0]
    assert row['criteria'][row['label']]=='direct identifier: mask'
    assert 'Anonymise Alice.' in row['state'] and '[TARGET]Alice[/TARGET]' in row['state']
    assert 'a1-e1' not in row['state']


def test_cebab_keeps_holdout_text_and_original_source_rows(monkeypatch):
    source=DatasetDict(train=Dataset.from_list([review(),review('Another review.',original=2)]),
                       test=Dataset.from_list([review(original=3)]))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:source)
    result=access.load_task('cebab/rating',recast='jev')
    assert len(result['train'])==1 and len(result['test'])==1
    assert result['train'][0]['source_index']=='train:1'
    assert result['test'][0]['source_index']=='test:0'
