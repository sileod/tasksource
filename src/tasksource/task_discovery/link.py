"""Cached metadata-only linking. Name matches suggest mirrors; they do not verify them."""
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from threading import Lock
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .identifiers import normalized, paper_ids

S2 = 'https://api.semanticscholar.org/graph/v1'
_VIEWER_LOCK=Lock()
_VIEWER_LAST=0.0


def cached_json(url, directory, offline=False, headers=None):
    path = Path(directory)/ (hashlib.sha256(url.encode()).hexdigest()+'.json')
    if not path.exists():
        if offline:
            raise FileNotFoundError(f'No cached response for {url}')
        for attempt in range(3):
            if url.startswith('https://datasets-server.huggingface.co/'):
                global _VIEWER_LAST
                with _VIEWER_LOCK:
                    time.sleep(max(0,1-(time.monotonic()-_VIEWER_LAST)))
                    _VIEWER_LAST=time.monotonic()
            if url.startswith(S2):time.sleep(1.1 * 2**attempt)
            try:
                with urlopen(Request(url, headers=headers or {}), timeout=20) as response:
                    raw = response.read()
                break
            except HTTPError as error:
                if error.code!=429 or attempt==2:raise
                if url.startswith('https://datasets-server.huggingface.co/'):
                    delay=error.headers.get('Retry-After','') if error.headers else ''
                    time.sleep(min(30,float(delay) if delay.isdigit() else 2**attempt))
                elif not url.startswith(S2):raise
        payload = dict(url=url, retrieved_at=datetime.now(timezone.utc).isoformat(),
                       sha256=hashlib.sha256(raw).hexdigest(), data=json.loads(raw))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
        temporary.write_text(json.dumps(payload, sort_keys=True)+'\n');temporary.replace(path)
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
    key_path = Path.home()/'.config/tasksource/semantic_scholar_api_key'
    key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY') or (key_path.read_text().strip() if key_path.is_file() else None)
    headers = {'x-api-key':key} if key else {}
    fields = 'paperId,title,url,externalIds'
    for original in row.get('papers', []):
        paper = dict(original) if isinstance(original, dict) else dict(url=original)
        identities = sorted(v for v in paper_ids([paper]) if v.startswith(('arxiv:', 'doi:', 's2:', 'aclanthology.org/')))
        identity = identities[0] if identities else None
        if identity:
            identity = identity.removeprefix('s2:')
            if identity.startswith('aclanthology.org/'):
                identifier = identity.split('/',1)[1]
                identity = 'ACL:'+(identifier.upper() if identifier[:1].isalpha() else identifier)
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
            paper.update(semantic_scholar_id=found['paperId'], canonical_title=found.get('title',''), external_ids=found.get('externalIds') or {},
                         semantic_scholar_url=found['url'], link_method='identifier' if identity else 'exact_title')
        papers.append(paper)
    row.update(papers=papers, link_candidates=suggestions, link_requests=requests, link_errors=errors)
    return row
