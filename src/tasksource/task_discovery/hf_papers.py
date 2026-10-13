"""Search Hugging Face paper pages and their explicitly linked data resources."""
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

from .identifiers import repository_links, source_id
from .link import cached_json

QUERIES = ('English classification annotated dataset', 'English multiple choice training dataset',
           'English relevance ranking judgments dataset', 'biomedical annotated classification corpus',
           'legal annotated classification dataset')


def retrieve(cache, queries=(), limit=30, offline=False):
    if not 1 <= limit <= 120:
        raise ValueError('HF Papers search accepts 1–120 results per query')
    papers, requests = {}, []
    for query in queries or QUERIES:
        url = 'https://huggingface.co/api/papers/search?'+urlencode(dict(q=query, limit=limit))
        response = cached_json(url, cache, offline)
        requests.append({k:v for k,v in response.items() if k!='data'})
        for item in response['data']:
            paper = item.get('paper', item)
            papers[paper['id']] = paper

    def enrich(paper):
        identifier = paper['id']
        documents, errors = [], []
        def fetch(url, fallback):
            try:
                response = cached_json(url, cache, offline)
                documents.append({k:v for k,v in response.items() if k!='data'})
                return response['data']
            except (OSError, ValueError) as error:
                errors.append(dict(url=url, error=str(error)))
                return fallback
        detail = fetch('https://huggingface.co/api/papers/'+identifier, paper)
        datasets = fetch('https://huggingface.co/api/datasets?'+urlencode(
            dict(filter='arxiv:'+identifier)), [])
        # The native abstract is evidence; AI-generated summaries are not substituted.
        abstract = detail.get('summary') or paper.get('summary') or ''
        links = repository_links(' '.join([abstract, detail.get('githubRepo') or '',
                                          detail.get('projectPage') or '']))
        repos = sorted({source_id(v,'huggingface'): v.split('/datasets/',1)[-1].strip('/')
                        for v in links['huggingface']+[d['id'] for d in datasets]}.values())
        page = 'https://huggingface.co/papers/'+identifier
        row = dict(name=detail.get('title') or paper['title'], aliases=[], candidate_kind='paper',
            sources=[dict(source='hf_papers', url=page)],
            papers=[dict(title=detail.get('title') or paper['title'], url='https://arxiv.org/abs/'+identifier)],
            huggingface=repos, github=links['github'], dataset_urls=[
                v if v.startswith('https://') else 'https://huggingface.co/datasets/'+v for v in repos],
            description=abstract, tasks=[], decision_tasks=[], languages=[], modalities=[],
            train_status='unchecked', published_at=detail.get('publishedAt'),
            project_page=detail.get('projectPage'), retrieval_errors=errors)
        return row, documents

    with ThreadPoolExecutor(max_workers=4) as pool:
        enriched = list(pool.map(enrich, papers.values()))
    for _, documents in enriched:
        requests.extend(documents)
    return [row for row,_ in enriched], dict(source='hf_papers', queries=list(queries or QUERIES),
        limit=limit, candidates=len(enriched), requests=requests)
