"""Deduplicate candidate names and report conservative catalog matches."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .identifiers import normalized, source_id, paper_ids



def deduplicate(candidates):
    merged = {}
    fields = ('aliases', 'sources', 'dataset_urls', 'github', 'huggingface', 'papers',
              'tasks', 'decision_tasks', 'languages', 'modalities')
    for candidate in candidates:
        key = normalized(candidate['name'])
        if not key:
            raise ValueError('Candidate must have a nonempty dataset name')
        if key not in merged:
            merged[key] = dict(candidate)
        else:
            row = merged[key]
            for field in fields:
                values = row.get(field, []) + candidate.get(field, [])
                row[field] = list({json.dumps(v, sort_keys=True): v for v in values}.values())
    return [merged[key] for key in sorted(merged)]


def catalog_index(normalize_repos=True):
    from tasksource.access import _task_repos, list_tasks
    names, repos = defaultdict(set), defaultdict(set)
    for options in ({}, {'soft': True}, {'multilingual': True}, {'vision': True}):
        for row in list_tasks(**options).itertuples():
            for name in (row.id, row.id.split('/')[0]):
                names[normalized(name)].add(row.id)
            for repo in _task_repos(row.mapping, row.dataset_name, row.id):
                repos[repo.casefold() if normalize_repos else repo].add(row.id)
                names[normalized(repo.rsplit('/', 1)[-1])].add(row.id)
    return names, repos


def catalog_references(path=None):
    path=Path(path) if path else Path(__file__).with_name('catalog_links.jsonl')
    references=defaultdict(set)
    for line in path.read_text().splitlines() if path.exists() else []:
        row=json.loads(line)
        for identity in (source_id(v,'github') for v in row.get('github',[])):
            references['github:'+identity].update(row['tasks'])
        for identity in paper_ids(row.get('papers',[])):
            references['paper:'+identity].update(row['tasks'])
        for paper in row.get('papers',[]):
            if isinstance(paper,dict) and paper.get('canonical_title'):
                references['paper_title:'+normalized(paper['canonical_title'])].update(row['tasks'])
    return references


def previous_decisions():
    return json.loads(Path(__file__).with_name('decisions.json').read_text())


def queued_sources(path=None):
    """Read source names from the human queue's tables without maintaining a second list."""
    path = Path(path) if path else Path(__file__).with_name('todo_batch.md')
    sources = {}
    for line in path.read_text().splitlines():
        if not line.startswith('| '):
            continue
        name, views, status = (cell.strip() for cell in line.strip('|').split('|'))
        if name == 'Source':
            continue
        sources[normalized(name)] = dict(name=name, views=views, status=status)
    return sources



def decision_matches(row, records):
    names = {normalized(n) for n in [row['name'], *row.get('aliases', [])]}
    identities = {kind: {source_id(v, kind) for v in row.get(kind, [])} for kind in ('huggingface', 'github')}
    identities['papers'] = paper_ids(row.get('papers', []))
    for record in records:
        aliases = [record['name'], *record.get('aliases', [])]
        aliases += [repo.rsplit('/', 1)[-1] for repo in record.get('huggingface', [])]
        name_match = names.intersection(map(normalized, aliases))
        id_match = any(identities[kind].intersection(source_id(v, kind) for v in record.get(kind, [])) for kind in identities)
        if name_match or id_match:
            yield record


def match(candidates, names, repos, decisions=(), queue=None, references=None):
    references=catalog_references() if references is None else references
    for row in deduplicate(candidates):
        evidence = []
        for name in [row['name'], *row.get('aliases', [])]:
            if normalized(name) in names:
                evidence.append(dict(by='name', value=name, tasks=sorted(names[normalized(name)])))
        for url in row.get('huggingface', []):
            repo = url.split('/datasets/', 1)[-1].strip('/').casefold()
            if repo in repos:
                evidence.append(dict(by='hub_repository', value=repo, tasks=sorted(repos[repo])))
        related=[]
        identities=[('github_repository','github:'+source_id(v,'github')) for v in row.get('github',[])]
        identities += [('paper','paper:'+v) for v in sorted(paper_ids(row.get('papers',[])))]
        identities += [('paper_title','paper_title:'+normalized(p.get('canonical_title') or p.get('title')))
                       for p in row.get('papers',[]) if isinstance(p,dict) and (p.get('canonical_title') or p.get('title'))]
        for kind,identity in identities:
            if identity in references:
                related.append(dict(by=kind,value=identity.split(':',1)[1],tasks=sorted(references[identity])))
        history = list(decision_matches(row, decisions))
        queued = {normalized(n) for n in [row['name'], *row.get('aliases', [])]} & (queue or {}).keys()
        yield dict(row, matches=evidence, related_matches=related, previous_decisions=history,
            queue_matches=[queue[n] for n in sorted(queued)],
            status='catalog_match' if evidence else 'previously_reviewed' if history else 'queued' if queued else 'candidate')


def catalog_hash(names, repos):
    value = {kind: {k: sorted(v) for k, v in sorted(index.items())}
             for kind, index in [('names', names), ('repositories', repos)]}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
