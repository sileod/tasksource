"""Find dataset-paper leads; paper titles are not asserted to be dataset names."""
from urllib.parse import urlencode

from .link import cached_json
from .identifiers import repository_links

QUERIES = ('classification dataset', 'multiple choice dataset', 'decision benchmark')


def retrieve(cache, queries=(), limit=100, offline=False):
    rows, requests = {}, []
    for query in queries or QUERIES:
        page, fetched = 1, 0
        while fetched < limit:
            size = min(100, limit-fetched)
            url = 'https://api.openalex.org/works?'+urlencode(dict(search=query, per_page=size, page=page))
            response = cached_json(url, cache, offline)
            requests.append({k:v for k,v in response.items() if k!='data'})
            results = response['data']['results']
            for work in results:
                if not work.get('title'):
                    continue
                locations = [loc for loc in work.get('locations') or [] if loc]
                urls = sorted({loc[k] for loc in locations for k in ('landing_page_url','pdf_url') if loc.get(k)})
                paper = dict(title=work['title'], url=work.get('doi') or work['id'], openalex_id=work['id'])
                links = repository_links(' '.join(urls+list(work.get('abstract_inverted_index') or {})))
                rows[work['id']] = dict(name=work['title'], aliases=[], candidate_kind='paper',
                    sources=[dict(source='openalex', url=work['id'])], papers=[paper],
                    dataset_urls=links['huggingface'], **links, tasks=[], decision_tasks=[], languages=[], modalities=[],
                    train_status='unchecked', publication_year=work.get('publication_year'), paper_urls=urls)
            fetched += len(results)
            if len(results) < size:
                break
            page += 1
    return list(rows.values()), dict(source='openalex', queries=list(queries or QUERIES),
        limit=limit, requests=requests, candidates=len(rows))
