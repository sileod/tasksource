import json
from types import SimpleNamespace

from scripts.append_jev_webinstruct import convert
from scripts.audit_jev_progressive import call, round_robin, render, restore_raw
from scripts.reconfirm_jev_filtering import MODEL


def test_mc_conversion_preserves_semantic_gold_and_all_options():
    raw = dict(id='42',prompt='Pick the largest.',options=['1','2','3','4','5'],gold=4)
    row = convert(raw,'mc','train')
    assert row['state'] == raw['prompt']
    assert set(row['options']) == set(raw['options'])
    assert row['options'][row['target'].index(1)] == '5'
    assert row == convert(raw,'mc','train')
    assert convert(raw,'mc','test')['group_id'] != row['group_id']


def test_binary_conversion_preserves_label_polarity():
    for gold in [0,1]:
        row = convert(dict(id='42',prompt='The sky is blue.',label=gold),'binary','train')
        assert row['options'] == ['no / false','yes / true']
        assert row['target'][gold] == 1
        assert row['kind'] == 'choice'


def test_round_robin_completes_each_task_round_before_next():
    rows = [dict(source=source,index=index) for source,count in [('a',3),('b',1),('c',2)]
            for index in range(count)]
    ordered = list(round_robin(rows))
    assert {row['source'] for row in ordered[:3]} == {'a','b','c'}
    assert {row['source'] for row in ordered[3:5]} == {'a','c'}
    assert ordered[-1] == dict(source='a',index=2)


def test_raw_checkpoint_replays_verdict_callback_without_api(tmp_path,monkeypatch):
    import hashlib
    import sys
    row = dict(example_id='one',source='a',state='hello',question='Pick',kind='choice',
               options=['a','b'],target=[1,0])
    prompts = [render([row],{})]
    fingerprint = hashlib.sha256(json.dumps(dict(prompts=prompts,model=MODEL,reasoning=True),sort_keys=True).encode()).hexdigest()
    item = dict(example_id='one',decision='keep',answer=0,confidence='high',reason='Evidence',
                gold_assessment='Gold defensible',uncertainty='')
    (tmp_path/'test-raw.jsonl').write_text(json.dumps(dict(index=0,fingerprint=fingerprint,data={'items':[item]}))+'\n')
    def unexpected(*args,**kwargs):
        raise AssertionError('Completed raw checkpoint must not make another API call')
    monkeypatch.setitem(sys.modules,'litlm',SimpleNamespace(complete=unexpected))
    monkeypatch.setitem(sys.modules,'litlm_cli',SimpleNamespace(_record=unexpected))
    replayed=[]
    assert call([[row]],{},SimpleNamespace(),tmp_path,'test',on_items=lambda items,rows: replayed.extend(items)) == {'one':item}
    assert replayed == [item]


def test_full_recovery_uses_stable_ids_after_pending_chunks_shift(tmp_path):
    row=dict(example_id='one',source='a',kind='choice',options=['a','b'],target=[1,0])
    item=dict(example_id='one',decision='keep',answer=0,confidence='high',reason='Evidence',
              gold_assessment='Gold defensible',uncertainty='')
    raw=dict(index=900,model=MODEL,schema_valid=True,data={'items':[item]})
    (tmp_path/'chunk-0000000-raw.jsonl').write_text(json.dumps(raw)+'\n')
    done=set()
    assert restore_raw(tmp_path,[row],done) == [dict(source='a',**item)]
    assert restore_raw(tmp_path,[row],done) == []
    assert len((tmp_path/'verdicts.jsonl').read_text().splitlines()) == 1
