import gzip
import hashlib
import json

import pytest

from tasksource.task_discovery import paperswithcode as pwc
from tasksource.task_discovery.match import catalog_hash, deduplicate, match


def test_retrieve_preserves_paper_and_explicit_links_without_claiming_train():
    source = [dict(name='MAVEN', url='https://paperswithcode.com/dataset/maven',
        homepage='https://github.com/THU-KEG/MAVEN-dataset',
        description='See https://huggingface.co/datasets/org/maven.',
        paper=dict(title='MAVEN: Event Detection', url='https://example.org/paper'),
        tasks=[dict(task='Event Detection')]),
        dict(name='Generator', url='https://example.org/generator', tasks=[dict(task='Text Generation')])]
    row, = list(pwc.candidates(source))
    assert row['papers'][0]['title'] == 'MAVEN: Event Detection'
    assert row['github'] == ['https://github.com/THU-KEG/MAVEN-dataset']
    assert row['huggingface'] == ['https://huggingface.co/datasets/org/maven']
    assert row['train_status'] == 'unchecked'
    assert len(list(pwc.candidates(source, decision_only=False))) == 2


def test_snapshot_replay_checks_bytes(tmp_path, monkeypatch):
    snapshot = tmp_path/'snapshot.gz'
    snapshot.write_bytes(gzip.compress(json.dumps([dict(name='QA',url='u',tasks=[dict(task='Question Answering')])]).encode()))
    monkeypatch.setattr(pwc, 'SHA256', hashlib.sha256(snapshot.read_bytes()).hexdigest())
    assert pwc.retrieve(snapshot=snapshot)[1]['candidates'] == 1
    snapshot.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        pwc.retrieve(snapshot=snapshot)


def test_candidate_dedup_keeps_versions_and_all_source_evidence():
    rows = deduplicate([dict(name='SST-5',sources=[dict(source='pwc',url='a')]),
        dict(name='sst5',sources=[dict(source='other',url='b')]),dict(name='SST-2')])
    assert len(rows) == 2
    assert len(next(r for r in rows if r['name']=='SST-5')['sources']) == 2


def test_match_reports_evidence_without_fuzzy_or_shared_paper_dedup():
    names = {'sst2': {'glue/sst2'}}; repos = {'original/data': {'task/view'}}
    rows = list(match([dict(name='SST-5'), dict(name='SST 2'),
        dict(name='Mirror',huggingface=['https://huggingface.co/datasets/Original/Data'])], names, repos))
    indexed = {r['name']:r for r in rows}
    assert indexed['SST-5']['matches'] == []
    assert indexed['SST 2']['matches'][0]['tasks'] == ['glue/sst2']
    assert indexed['Mirror']['matches'][0]['by'] == 'hub_repository'
    assert catalog_hash(names,repos) == catalog_hash(dict(reversed(list(names.items()))),repos)


def test_previous_decisions_match_source_ids_and_do_not_block_other_views():
    from tasksource.task_discovery.match import decision_matches
    record=dict(name='Known source',scope='task',tasks=['source/intent'],
        github=['org/data'],papers=['arxiv:1808.07036'],huggingface=[],reviews=[
            dict(reviewer=dict(kind='model',model='model-a'),verdict='defer'),
            dict(reviewer=dict(kind='model',model='model-b'),verdict='retain')])
    assert list(decision_matches(dict(name='Renamed mirror',github=['https://github.com/Org/Data.git']),[record]))==[record]
    row=dict(name='New spelling',papers=[dict(title='Title',url='https://arxiv.org/pdf/1808.07036v2.pdf')])
    assert list(decision_matches(row,[record]))==[record]
    result=list(match([row],{},{},[record]))[0]
    assert result['status']=='previously_reviewed' and result['previous_decisions'][0]['scope']=='task'
    assert result['previous_decisions'][0]['reviews']==record['reviews']
    record['scope']='dataset'
    assert list(match([row],{},{},[record]))[0]['status']=='previously_reviewed'


def test_structured_history_preserves_existing_reasons():
    from tasksource import parked,eval_only
    from tasksource.task_discovery.match import previous_decisions
    history=previous_decisions()
    indexed={r['annotation']:r for r in history if 'annotation' in r}
    for module,reasons in ((parked,parked.REASONS),(eval_only,eval_only.REASONS)):
        for name,reason in reasons.items():
            record=indexed[module.__name__+'.'+name]
            assert record['reason']==reason
            assert all(key in record for key in ('huggingface','github','papers','scope','status','reason_code'))


def test_linking_requires_exact_title_and_keeps_hub_suggestions_unverified(monkeypatch):
    from importlib import import_module
    linker=import_module('tasksource.task_discovery.link')
    def cached(url,*args):
        data=([dict(id='org/MAVEN'),dict(id='org/MAVEN-Arg')] if 'huggingface.co' in url else
              dict(data=[dict(title='MAVEN: Event Detection',paperId='s2id',url='paper',externalIds={'ArXiv':'1234.56789'})]))
        return dict(url=url,sha256='hash',data=data)
    monkeypatch.setattr(linker,'cached_json',cached)
    row=linker.link(dict(name='MAVEN',huggingface=[],papers=[dict(title='MAVEN: Event Detection')]),'unused')
    assert row['papers'][0]['semantic_scholar_id']=='s2id'
    assert row['link_candidates'][0]['verified'] is False and row['huggingface']==[]
    assert len(row['link_candidates'])==1
    row=linker.link(dict(name='MAVEN',papers=[dict(title='MAVEN: Different Paper')]),'unused')
    assert 'semantic_scholar_id' not in row['papers'][0]


def test_link_cache_replays_without_network_or_storing_headers(tmp_path,monkeypatch):
    from importlib import import_module
    import io
    linker=import_module('tasksource.task_discovery.link')
    calls=[]
    def fetch(request,**kwargs):
        calls.append(request.full_url)
        return io.BytesIO(b'{"paperId":"id"}')
    monkeypatch.setattr(linker,'urlopen',fetch)
    first=linker.cached_json('https://example.org/paper',tmp_path,headers={'x-api-key':'secret'})
    assert first==linker.cached_json('https://example.org/paper',tmp_path,offline=True)
    assert len(calls)==1 and 'secret' not in next(tmp_path.iterdir()).read_text()
    with pytest.raises(FileNotFoundError):
        linker.cached_json('https://example.org/other',tmp_path,offline=True)
