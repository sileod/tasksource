import json
from datasets import Dataset, DatasetDict
from tasksource import access
from tasksource.tasks import _causal_transitivity_records


def test_transitivity_excludes_ambiguous_self_queries_and_cycles():
    base=dict(premise='A causes B. B causes C.',hypothesis='Does A cause C?',label='Yes')
    assert len(list(_causal_transitivity_records(base)))==1
    assert not list(_causal_transitivity_records(dict(base,hypothesis='Does A cause A?')))
    assert not list(_causal_transitivity_records(dict(base,premise='A causes B. B causes A.')))


def test_causal_graph_identity_ignores_edge_order_and_duplicate_edges():
    base=dict(premise='A causes B. B causes C.',hypothesis='Does A cause C?',label='Yes')
    other=dict(base,premise='B causes C. A causes B. A causes B.')
    a=next(_causal_transitivity_records(base));b=next(_causal_transitivity_records(other))
    assert json.loads(a['metadata'])==json.loads(b['metadata'])


def test_transitivity_jev_preserves_gold_and_graph_holdouts(monkeypatch):
    def row(premise,hypothesis,label):return dict(premise=premise,hypothesis=hypothesis,label=label)
    source=DatasetDict(train=Dataset.from_list([
        row('A causes B. B causes C.','Does A cause C?','Yes'),
        row('D causes E. E causes F.','Does D cause F?','Yes'),
        row('G causes H.','Does H cause G?','No')]),
        test=Dataset.from_list([row('B causes C. A causes B.','Does C cause A?','No')]))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:source)
    result=access.load_task('causal-reasoning/transitivity',recast='jev')
    assert len(result['train'])==2 and len(result['test'])==1
    groups={s:{json.loads(r['metadata'])['split_group_id'] for r in ds} for s,ds in result.items()}
    assert not groups['train']&groups['test']
    for split,ds in result.items():
        for r in ds:
            expected='yes' if 'Does D cause F?' in r['state'] else 'no'
            assert r['criteria'][r['label']]==expected
            assert r['instructions'] and r['state']
