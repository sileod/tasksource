import json

from datasets import Dataset, DatasetDict
import pytest

from tasksource import load_task
from scripts.repackage_dataset.relevance import parse_sentfin, search_record, text_key


def test_sentfin_repairs_only_quoting_and_validates_labels():
    assert parse_sentfin("{'Gold': 'positive', 'Silver': 'negative'}") == ({'Gold':'positive','Silver':'negative'},False)
    assert parse_sentfin("{'Osian's Art Fund': 'neutral'}") == ({"Osian's Art Fund":'neutral'},True)
    with pytest.raises(ValueError):
        parse_sentfin("{'Company': 'invented-label'}")
    with pytest.raises(ValueError):
        parse_sentfin("{'Gold': 'positive'} trailing junk")
    assert text_key(' Market   news ') == text_key('market news')


def test_search_keeps_score_and_omits_label_and_popularity_from_input():
    record=search_record({'doc_id':'query','query':'cell biology'},
                        dict(doc_id='paper',title='Cell structure',abstract='A native abstract.',score=3,n_citations=999))
    assert record['labels'] == 3.0
    assert record['sentence2'] == 'Cell structure\nA native abstract.'
    metadata=json.loads(record['metadata'])
    assert metadata['source_score'] == 3
    assert metadata['score_anchors'] == list(range(0,15,2))
    with pytest.raises(ValueError,match='score'):
        search_record({'doc_id':'query','query':'query'},dict(doc_id='p',title='title',score=15))


@pytest.mark.parametrize('task,labels', [('sentfin',['negative','positive']),('wands',[0,2]),('scirepeval/search',[1.,3.,14.])])
def test_shared_loading_and_jev_supervision(monkeypatch,task,labels):
    data=DatasetDict({s:Dataset.from_list([dict(sentence1=f'Query {i}',sentence2=f'Content {i}',labels=v,
                metadata=json.dumps(dict(id=f'{s}:{i}',split_group_id=f'{s}:{i}'))) for i,v in enumerate(labels)])
                for s in ('train','validation')})
    monkeypatch.setattr('tasksource.access.load_dataset',lambda *a,**kw:data)
    raw=load_task(task)
    assert set(raw)=={'train','validation'}
    if task=='scirepeval/search':
        assert list(raw['train']['labels']) == labels
    recast=load_task(task,recast='jev')
    for row,value in zip(recast['train'],labels):
        assert json.loads(row['metadata'])['split_group_id'].startswith('train:')
        if task=='sentfin':
            assert row['answer']==value
            assert 'entity text_B' in row['instructions']
        elif task=='wands':
            assert row['kind']=='score'
            assert row['criteria']==['irrelevant','partial match','exact match']
            assert row['label']==value
        else:
            assert row['kind']=='score' and len(row['target'])==8
            assert sum(float(option)*weight for option,weight in zip(row['criteria'],row['target']))==pytest.approx(value)
            assert 'click-derived' in row['instructions']
