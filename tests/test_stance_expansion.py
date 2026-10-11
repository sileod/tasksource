import io
import json

import pytest
from datasets import Features, Value

from scripts.repackage_dataset import stance
from scripts.repackage_dataset.specialized_text import pubmed_rows
from tasksource.metadata.discourse import DISCO_RELATIONS


def capture(monkeypatch, files):
    monkeypatch.setattr(stance,'_source',lambda repo:(lambda name:files[name],{}))
    monkeypatch.setattr(stance,'write_mirror',lambda name,rows,label,report,*args:(rows,report))


def test_vast_keeps_original_text_and_excludes_explicit_synthetic_neutrals(monkeypatch):
    header='post,new_topic,label,new_id,arc_id,type_idx,seen?\n'
    files={f'data/VAST/vast_{s}.csv':header+f'Original {s} comment,topic,1,{s},article,1,0\n'+
           f'Synthetic {s},other,2,{s}-synthetic,article,4,0\n' for s in ('train','dev','test')}
    capture(monkeypatch,files)
    rows,report=stance.vast()
    assert rows['train'][0]['sentence1']=='Original train comment'
    assert rows['train'][0]['sentence2']=='topic' and rows['train'][0]['labels']=='1'
    assert all(len(r)==1 for r in rows.values())
    assert report['excluded_synthetic_neutral']==dict(train=1,validation=1,test=1)


def test_fnc_joins_complete_bodies_and_keeps_native_holdout_over_duplicate_text(monkeypatch):
    files={prefix+'_bodies.csv':'Body ID,articleBody\n1,"The complete article.\nA second paragraph."\n' for prefix in ('train','competition_test')}
    files.update({prefix+'_stances.csv':'Headline,Body ID,Stance\nA claim,1,agree\n' for prefix in ('train','competition_test')})
    capture(monkeypatch,files)
    rows,report=stance.fnc1()
    assert rows['train']==[] and 'validation' not in rows
    assert rows['test'][0]['sentence2']=='The complete article.\nA second paragraph.'
    assert report['excluded_group_overlap']['train']==1


def test_perspectrum_joins_native_primary_perspective_and_records_cluster(monkeypatch):
    base='data/dataset/'
    files={base+'perspective_pool_v1.0.json':json.dumps([dict(pId=2,text='A reason in favor.')]),
           base+'dataset_split_v1.0.json':json.dumps({'1':'train'}),
           base+'perspectrum_with_answers_v1.0.json':json.dumps([dict(cId=1,text='A claim',perspectives=[
               dict(pids=[2,3],stance_label_3='SUPPORT',voter_counts=[3,0,0,0,0])])])}
    capture(monkeypatch,files)
    rows,_=stance.perspectrum()
    assert rows['train'][0]['sentence2']=='A reason in favor.' and rows['train'][0]['labels']=='SUPPORT'
    assert json.loads(rows['train'][0]['metadata'])['native_cluster']==[2,3]


def test_convincingness_groups_both_stance_sides_and_preserves_native_gold(monkeypatch):
    files={f'data/UKPConvArg1Strict-CSV/topic-{i}_{side}.csv':'#id\tlabel\ta1\ta2\npair\ta2\tFirst argument\tSecond argument\n'
           for i in range(5) for side in ('yes','no')}
    capture(monkeypatch,files)
    tree=json.dumps(dict(tree=[dict(path=p) for p in files])).encode()
    monkeypatch.setattr(stance,'urlopen',lambda *a,**k:io.BytesIO(tree))
    rows,_=stance.ukpconvarg()
    assert {s:len(r) for s,r in rows.items()}==dict(train=6,validation=2,test=2)
    groups={s:{json.loads(r['metadata'])['split_group_id'] for r in rs} for s,rs in rows.items()}
    assert all(not groups[a]&groups[b] for a in groups for b in groups if a!=b)
    assert all(r['choices_list'][r['labels']]=='Second argument' for rs in rows.values() for r in rs)


def test_discogem_recovers_integer_counts_from_rounded_native_shares():
    counts={code:0 for code in DISCO_RELATIONS};counts.update(precedence=4,reason=3,result=4)
    native='; '.join(f'{k}:{v/11:.3f}' for k,v in counts.items())
    assert stance.discourse_votes(native,11)==list(counts.values())
    with pytest.raises(ValueError):stance.discourse_votes(native,10)


def test_empty_native_evaluation_keeps_declared_mirror_features(tmp_path,monkeypatch):
    from scripts.repackage_dataset.relevance import write_mirror
    monkeypatch.chdir(tmp_path)
    row=stance._row('Example','Topic','pro','id','group')
    data=write_mirror('tiny',dict(train=[row],validation=[]),'string',{})
    assert data['train'].features==data['validation'].features and len(data['validation'])==0


def test_pubmed_preserves_pmids_indices_and_numbers_and_rejects_malformed_roles(tmp_path):
    p=tmp_path/'native.txt';p.write_text('###123\nMETHODS\tWe enrolled 80 patients.\n  \n###456\nRESULTS\tTen recovered.\n')
    rows=list(pubmed_rows(p))
    assert rows[0]['sentence1']=='We enrolled 80 patients.'
    assert 'sentence2' not in rows[0]
    assert [json.loads(r['metadata'])['id'] for r in rows]==['123:0','456:0']
    p.write_text('###123\nUNKNOWN\tText\n')
    with pytest.raises(ValueError,match='Invalid PubMed'):list(pubmed_rows(p))

