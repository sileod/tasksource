"""Bounded Hub metadata searches, saved for offline replay; no dataset downloads."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

from .paperswithcode import DECISION_TASKS
from .identifiers import repository_links

TASKS = ('multiple-choice', 'text-classification', 'token-classification', 'image-classification', 'question-answering')


def retrieve(cache, queries=(), tasks=(), limit=100, offline=False, languages=(), modalities=(), sort='downloads'):
    settings = dict(queries=list(queries), tasks=list(tasks), limit=limit,
                    default_tasks=list(TASKS) if not queries and not tasks else [], languages=list(languages))
    if modalities:
        settings['modalities'] = list(modalities)
    if sort != 'downloads':
        settings['sort'] = sort
    path = Path(cache)/('hf_hub-'+hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()+'.json')
    if not path.exists():
        if offline:
            raise FileNotFoundError(f'No Hub snapshot: {path}')
        api, records = HfApi(), {}
        # Explicit name searches also find untagged datasets; defaults use decision tags.
        categories = tasks or (() if queries else TASKS)
        for query in queries or (None,):
            for category in categories or (None,):
                for info in api.list_datasets(search=query, task_categories=category,
                                             sort=sort, limit=limit, full=True, language=list(languages) or None,
                                             filter=[f'modality:{m}' for m in modalities] or None):
                    card = info.card_data.to_dict() if info.card_data else {}
                    records[info.id] = dict(id=info.id, sha=info.sha, tags=info.tags or [],
                        card=card, downloads=info.downloads, likes=info.likes,
                        last_modified=info.last_modified.isoformat() if info.last_modified else None)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(settings=settings, retrieved_at=datetime.now(timezone.utc).isoformat(),
                                        records=list(records.values())), sort_keys=True)+'\n')
    snapshot = json.loads(path.read_text())
    rows = []
    for item in snapshot['records']:
        card, repo = item['card'], item['id']
        tags = item['tags']
        tasks = [t.split(':', 1)[1] for t in tags if t.startswith(('task_categories:', 'task_ids:'))]
        url = 'https://huggingface.co/datasets/'+repo
        papers = [dict(url='https://arxiv.org/abs/'+t[6:]) for t in tags if t.startswith('arxiv:')]
        links = repository_links(json.dumps(card))
        rows.append(dict(name=repo, aliases=[repo.rsplit('/', 1)[-1]],
            sources=[dict(source='hf_hub', url=url)], dataset_urls=[url], huggingface=[repo], github=links['github'],
            papers=papers, tasks=tasks, decision_tasks=[t for t in tasks if any(k in t.replace('-', ' ') for k in DECISION_TASKS)],
            languages=([card['language']] if isinstance(card.get('language'), str) else card.get('language'))
                      or [t.split(':', 1)[1] for t in tags if t.startswith('language:')],
            modalities=card.get('modalities') or [t.split(':', 1)[1] for t in tags if t.startswith('modality:')], train_status='unchecked',
            hub_metadata=dict(revision=item['sha'], license=card.get('license'),
                              downloads=item['downloads'], likes=item['likes'], last_modified=item['last_modified'])))
    return rows, dict(source='hf_hub', **settings,
        snapshot=str(path), snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        retrieved_at=snapshot['retrieved_at'], candidates=len(rows))
