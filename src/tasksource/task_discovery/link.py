"""Cached metadata-only linking. Name matches suggest mirrors; they do not verify them."""
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .identifiers import normalized, paper_ids

S2 = 'https://api.semanticscholar.org/graph/v1'


def cached_json(url, directory, offline=False, headers=None):
    path = Path(directory)/ (hashlib.sha256(url.encode()).hexdigest()+'.json')
    if not path.exists():
        if offline:
            raise FileNotFoundError(f'No cached response for {url}')
        if url.startswith(S2):
            time.sleep(1)
        with urlopen(Request(url, headers=headers or {}), timeout=20) as response:
            raw = response.read()
        payload = dict(url=url, retrieved_at=datetime.now(timezone.utc).isoformat(),
                       sha256=hashlib.sha256(raw).hexdigest(), data=json.loads(raw))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True)+'\n')
    return json.loads(path.read_text())


def link(candidate, cache, offline=False, search_hub=True):
    row = dict(candidate)
    requests, errors = [], []
    def query(url, headers=None):
        try:
            response = cached_json(url, cache, offline, headers)
            requests.append({k:v for k,v in response.items() if k!='data'})
            return response['data']
        except (HTTPError, URLError, FileNotFoundError) as error:
            errors.append(dict(url=url, error=str(error)))
            return None
    suggestions = []
    if search_hub:
        url = 'https://huggingface.co/api/datasets?'+urlencode(dict(search=row['name'], limit=10))
        names = {normalized(n) for n in [row['name'], *row.get('aliases', [])]}
        for item in query(url) or []:
            if normalized(item['id'].rsplit('/', 1)[-1]) in names:
                suggestions.append(dict(kind='huggingface', id=item['id'],
                    url='https://huggingface.co/datasets/'+item['id'], method='exact_name', verified=False))
    papers = []
    key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
    headers = {'x-api-key':key} if key else {}
    fields = 'paperId,title,url,externalIds'
    for original in row.get('papers', []):
        paper = dict(original) if isinstance(original, dict) else dict(url=original)
        identities = sorted(v for v in paper_ids([paper]) if v.startswith(('arxiv:', 'doi:', 's2:')))
        identity = identities[0] if identities else None
        if identity:
            identity = identity.removeprefix('s2:')
            url = S2+'/paper/'+quote(identity, safe=':')+'?'+urlencode(dict(fields=fields))
            found = query(url, headers)
        elif paper.get('title'):
            url = S2+'/paper/search/match?'+urlencode(dict(query=paper['title'], fields=fields))
            response = query(url, headers) or {}
            matches = [p for p in response.get('data', []) if normalized(p['title'])==normalized(paper['title'])]
            found = matches[0] if len(matches)==1 else None
        else:
            found = None
        if found and found.get('paperId'):
            paper.update(semantic_scholar_id=found['paperId'], external_ids=found.get('externalIds') or {},
                         semantic_scholar_url=found['url'], link_method='identifier' if identity else 'exact_title')
        papers.append(paper)
    row.update(papers=papers, link_candidates=suggestions, link_requests=requests, link_errors=errors)
    return row
