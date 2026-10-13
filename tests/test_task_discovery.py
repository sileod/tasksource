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


def test_human_queue_matches_names_without_merging_versions_or_overriding_catalog(tmp_path):
    from tasksource.task_discovery.match import queued_sources
    path=tmp_path/'queue.md'
    text='| Source | Requested views | Status |\n|---|---|---|\n| SST-5 | Sentiment | Pending review |\n'
    path.write_text(text)
    queue=queued_sources(path)
    rows={r['name']:r for r in match([dict(name='sst5'),dict(name='SST-2'),
        dict(name='Mirror',aliases=['SST-5'])],{'mirror':{'existing/view'}},{},queue=queue)}
    assert rows['sst5']['status']=='queued'
    assert rows['sst5']['queue_matches'][0]['status']=='Pending review'
    assert rows['SST-2']['status']=='candidate' and not rows['SST-2']['queue_matches']
    assert rows['Mirror']['status']=='catalog_match' and rows['Mirror']['queue_matches']
    assert path.read_text()==text
    actual=queued_sources()
    assert {'maven','mavenarg','sst5'} <= actual.keys()


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


def test_hub_retrieval_snapshots_metadata_and_replays_without_api(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tasksource.task_discovery import hf_hub as hub
    calls=[]
    info=SimpleNamespace(id='org/Example',sha='source-revision',tags=['task_categories:text-classification','arxiv:1234.56789'],
        card_data=SimpleNamespace(to_dict=lambda:dict(license='cc-by-4.0',language=['en'])),
        downloads=42,likes=3,last_modified=None)
    def search(**kwargs):
        calls.append(kwargs)
        return iter([info])
    monkeypatch.setattr(hub,'HfApi',lambda:SimpleNamespace(list_datasets=search))
    rows,manifest=hub.retrieve(tmp_path,queries=['Example','alternate'],limit=2)
    assert len(rows)==1 and len(calls)==2
    row=rows[0]
    assert row['huggingface']==['org/Example'] and row['train_status']=='unchecked'
    assert row['hub_metadata']['revision']=='source-revision' and row['hub_metadata']['license']=='cc-by-4.0'
    assert row['papers'][0]['url']=='https://arxiv.org/abs/1234.56789'
    assert all(c['limit']==2 and c['full'] and c['task_categories'] is None for c in calls)
    _,defaults=hub.retrieve(tmp_path,limit=2)
    assert 'multiple-choice' in {c['task_categories'] for c in calls}
    assert defaults['default_tasks']==list(hub.TASKS)
    assert defaults['snapshot']!=manifest['snapshot']
    info.card_data=None;info.tags.append('language:en')
    english,filtered=hub.retrieve(tmp_path,queries=['Example','alternate'],limit=2,languages=['en'])
    assert english[0]['languages']==['en']
    assert calls[-1]['language']==['en'] and filtered['snapshot']!=manifest['snapshot']
    monkeypatch.setattr(hub,'HfApi',lambda:pytest.fail('Offline replay called Hub'))
    assert hub.retrieve(tmp_path,queries=['Example','alternate'],limit=2,offline=True)==(rows,manifest)
    with pytest.raises(FileNotFoundError):hub.retrieve(tmp_path,queries=['uncached'],offline=True)


def test_openalex_bounded_pages_preserve_paper_identity_without_inventing_dataset(tmp_path,monkeypatch):
    from urllib.parse import parse_qs,urlsplit
    from tasksource.task_discovery import openalex
    calls=[]
    def query(url,*args):
        params=parse_qs(urlsplit(url).query);page=int(params['page'][0]);size=int(params['per_page'][0])
        calls.append((page,size,params.get('filter')))
        return dict(url=url,sha256='hash',data=dict(results=[dict(id=f'https://openalex.org/W{page*100+i}',
            title=f'Classification dataset paper {page*100+i}',doi=f'https://doi.org/10.1234/{page*100+i}',
            publication_year=2024,locations=[dict(landing_page_url='https://github.com/org/data',pdf_url=None)],
            language='en',abstract_inverted_index={'https://huggingface.co/datasets/org/data':[2],
                'training':[1], 'Human':[0]}) for i in range(size)]))
    monkeypatch.setattr(openalex,'cached_json',query)
    rows,manifest=openalex.retrieve(tmp_path,queries=['classification dataset'],limit=101,languages=['en'])
    assert calls==[(1,100,['language:en']),(2,1,['language:en'])] and len(rows)==101
    assert rows[0]['description']=='Human training https://huggingface.co/datasets/org/data'
    assert rows[0]['paper_language']=='en' and rows[0]['languages']==[]
    assert manifest['paper_languages']==['en']
    assert rows[0]['candidate_kind']=='paper' and rows[0]['train_status']=='unchecked'
    assert rows[0]['github']==['https://github.com/org/data']
    assert rows[0]['huggingface']==['https://huggingface.co/datasets/org/data']
    assert rows[0]['papers'][0]['openalex_id']=='https://openalex.org/W100'
    assert len(manifest['requests'])==2
    from tasksource.task_discovery.identifiers import paper_ids
    assert 'openalex.org/w100' in paper_ids(rows[0]['papers'])


def test_retrieve_combines_sources_through_same_cli_and_preserves_evidence(tmp_path,monkeypatch):
    import sys
    from tasksource.task_discovery import __main__ as cli,hf_hub,openalex
    def retrieve(source):
        return lambda *a,**k:([dict(name='Shared',sources=[dict(source=source,url=source)],
                                    train_status='unchecked')],dict(source=source))
    monkeypatch.setattr(hf_hub,'retrieve',retrieve('hf_hub'))
    monkeypatch.setattr(openalex,'retrieve',retrieve('openalex'))
    output=tmp_path/'candidates.jsonl'
    monkeypatch.setattr(sys,'argv',['discovery','retrieve','--source','hf_hub','--source','openalex',
                                  '--query','classification dataset','--output',str(output)])
    cli.main()
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows)==1 and [s['source'] for s in rows[0]['sources']]==['hf_hub','openalex']
    assert rows[0]['train_status']=='unchecked'
    manifest=json.loads(output.with_suffix('.manifest.json').read_text())
    assert [s['source'] for s in manifest['sources']]==['hf_hub','openalex']


def test_card_links_extract_repositories_and_papers_without_infrastructure():
    from tasksource.task_discovery.cards import card_links
    from tasksource.task_discovery.identifiers import paper_ids
    links=card_links('''https://github.com/Org/Data/blob/main/LICENSE
        https://github.com/huggingface/datasets https://github.com/sileod/tasksource
        Paper [https://arxiv.org/pdf/1808.07036v2.pdf](https://arxiv.org/pdf/1808.07036v2.pdf)
        doi https://doi.org/10.1234/example). eprint={2401.12345}''')
    assert links['github']==['https://github.com/Org/Data']
    assert paper_ids(links['papers'])=={'arxiv:1808.07036','doi:10.1234/example','arxiv:2401.12345'}


def test_mine_cards_records_snapshot_and_failures_without_loading_data(tmp_path,monkeypatch):
    from tasksource.task_discovery import cards
    path=tmp_path/'snapshots'/'source-commit'/'README.md';path.parent.mkdir(parents=True)
    path.write_text('https://github.com/Org/Data')
    def read(repo,filename,**kwargs):
        assert filename=='README.md' and kwargs==dict(repo_type='dataset',local_files_only=True)
        if repo=='org/missing':raise FileNotFoundError('Not cached')
        return str(path)
    monkeypatch.setattr(cards,'hf_hub_download',read)
    rows=cards.mine_cards({'org/known':{'task/view'},'org/missing':{'other/view'}},offline=True)
    assert rows[0]['revision']=='source-commit' and rows[0]['tasks']==['task/view']
    assert rows[0]['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert '/source-commit/README.md' in rows[0]['sources'][0]['url']
    assert rows[1]['error']=='Not cached'


def test_catalog_reference_matches_flag_shared_sources_without_discarding_new_datasets(tmp_path):
    from tasksource.task_discovery.match import catalog_references
    path=tmp_path/'links.jsonl'
    path.write_text(json.dumps(dict(name='org/known',tasks=['known/view'],github=['https://github.com/Org/Data'],
        papers=[dict(url='https://arxiv.org/abs/1808.07036')]))+'\n')
    refs=catalog_references(path)
    rows=list(match([dict(name='New task',github=['https://github.com/org/data.git'],
        papers=[dict(url='https://arxiv.org/pdf/1808.07036v2.pdf')])],{},{},references=refs))
    assert rows[0]['status']=='candidate' and rows[0]['matches']==[]
    assert {m['by'] for m in rows[0]['related_matches']}=={'github_repository','paper'}
    assert all(m['tasks']==['known/view'] for m in rows[0]['related_matches'])


def test_semantic_scholar_rate_limit_retries_then_caches(tmp_path,monkeypatch):
    from importlib import import_module
    from urllib.error import HTTPError
    import io
    linker=import_module('tasksource.task_discovery.link');calls=[]
    monkeypatch.setattr(linker.time,'sleep',lambda delay:None)
    def fetch(request,**kwargs):
        calls.append(request.full_url)
        if len(calls)<3:raise HTTPError(request.full_url,429,'Rate limited',{},None)
        return io.BytesIO(b'{"paperId":"canonical-id"}')
    monkeypatch.setattr(linker,'urlopen',fetch)
    url=linker.S2+'/paper/arxiv:1808.07036'
    result=linker.cached_json(url,tmp_path)
    assert len(calls)==3 and result['data']['paperId']=='canonical-id'
    assert linker.cached_json(url,tmp_path,offline=True)==result and len(calls)==3


def test_semantic_scholar_private_key_fallback_and_environment_precedence(tmp_path,monkeypatch):
    from importlib import import_module
    linker=import_module('tasksource.task_discovery.link')
    monkeypatch.setattr(linker.Path,'home',lambda:tmp_path)
    path=tmp_path/'.config/tasksource/semantic_scholar_api_key';path.parent.mkdir(parents=True)
    path.write_text('private-test-key');monkeypatch.delenv('SEMANTIC_SCHOLAR_API_KEY',raising=False)
    headers=[]
    def cached(url,cache,offline,header):
        headers.append(header)
        return dict(url=url,sha256='hash',data=dict(paperId='id',url='paper',externalIds={'DOI':'10.1234/test'}))
    monkeypatch.setattr(linker,'cached_json',cached)
    candidate=dict(name='Dataset',papers=[dict(url='arxiv:1808.07036')])
    row=linker.link(candidate,tmp_path,search_hub=False)
    monkeypatch.setenv('SEMANTIC_SCHOLAR_API_KEY','environment-test-key')
    linker.link(candidate,tmp_path,search_hub=False)
    assert headers==[{'x-api-key':'private-test-key'},{'x-api-key':'environment-test-key'}]
    assert 'private-test-key' not in json.dumps(row)
    assert row['papers'][0]['external_ids']['DOI']=='10.1234/test'


def test_semantic_scholar_accepts_acl_anthology_identifiers(monkeypatch):
    from importlib import import_module
    linker=import_module('tasksource.task_discovery.link');urls=[]
    def cached(url,*args):
        urls.append(url)
        return dict(url=url,sha256='hash',data=dict(paperId='canonical-id',url='paper',externalIds={}))
    monkeypatch.setattr(linker,'cached_json',cached)
    row=linker.link(dict(name='Dataset',papers=[dict(url='https://aclanthology.org/P19-1474.pdf')]),
        'unused',search_hub=False)
    assert '/paper/ACL:P19-1474?' in urls[0]
    assert row['papers'][0]['semantic_scholar_id']=='canonical-id'


def test_canonical_paper_titles_link_pwc_candidates_without_discarding_them(tmp_path):
    from tasksource.task_discovery.match import catalog_references
    path=tmp_path/'links.jsonl';path.write_text(json.dumps(dict(tasks=['known/view'],papers=[
        dict(url='arxiv:1808.07036',semantic_scholar_id='canonical-id',canonical_title='Dataset: A New Benchmark')]))+'\n')
    result=list(match([dict(name='New dataset view',papers=[dict(title='Dataset - A New Benchmark')])],
                      {},{},references=catalog_references(path)))[0]
    assert result['status']=='candidate'
    assert result['related_matches']==[dict(by='paper_title',value='datasetanewbenchmark',tasks=['known/view'])]


def test_hf_papers_keeps_native_evidence_explicit_links_and_replays(tmp_path,monkeypatch):
    from tasksource.task_discovery import hf_papers
    from urllib.parse import parse_qs,urlsplit
    calls=[]
    def cached(url,cache,offline):
        calls.append((url,offline))
        if '/search?' in url:
            data=[{'paper':{'id':'2401.12345','title':'A labeled dataset','summary':'Native abstract'}}]
        elif '/api/papers/' in url:
            data=dict(title='A labeled dataset',summary='Native abstract',ai_summary='Invented training claims',
                      githubRepo='https://github.com/org/data',projectPage='https://example.org')
        else:
            assert parse_qs(urlsplit(url).query)['filter']==['arxiv:2401.12345']
            data=[{'id':'Org/Data'}]
        return dict(url=url,sha256='hash',data=data)
    monkeypatch.setattr(hf_papers,'cached_json',cached)
    rows,manifest=hf_papers.retrieve(tmp_path,queries=['first','second'],limit=3)
    assert len(rows)==1 and rows[0]['candidate_kind']=='paper'
    assert rows[0]['description']=='Native abstract'
    assert rows[0]['huggingface']==['Org/Data'] and rows[0]['github']==['https://github.com/org/data']
    assert rows[0]['languages']==[] and rows[0]['train_status']=='unchecked'
    assert len(manifest['requests'])==4
    assert hf_papers.retrieve(tmp_path,queries=['first','second'],limit=3,offline=True)==(rows,manifest)
    assert calls[-1][1]
    with pytest.raises(ValueError):hf_papers.retrieve(tmp_path,limit=121)
