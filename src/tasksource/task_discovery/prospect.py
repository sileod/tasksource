"""Ranked prospect search: cheap signals first, model review only for the head of the ranking.

    PYTHONPATH=src python -m tasksource.task_discovery.prospect --output build/discovery/prospect-YYYY-MM-DD

Stages, each cached under ``--output`` and resumable:
1. inventory: Hub listings for decision task tags (English), from the snapshot cache.
2. identity: drop catalog names/repos, linked papers/GitHub, decisions.json and the queue.
3. prior: metadata score (downloads, likes, paper/GitHub links, age) minus slop signals.
4. structure: viewer ``/info`` and ``/first-rows`` for the top ``--probe``: a train split with
   enough rows, a label-like column, and no text segments shared with the catalog
   (``fingerprint.py``) or with a higher-ranked candidate.
5. review: Jev-style decision models on the top ``--review`` with sampled rows as evidence.
6. ranking: ``ranked.jsonl`` and ``report.md``; nothing enters the catalog automatically.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlencode

from .discover_task_prospects import current_answers, language_priority, write_rows
from .fingerprint import DatasetIndex, Index, row_segments, signature
from .hf_hub import retrieve
from .identifiers import normalized
from .link import cached_json
from .match import catalog_index, catalog_references, match, previous_decisions, queued_sources

TAGS = ('text-classification', 'token-classification', 'multiple-choice', 'zero-shot-classification',
        'sentence-similarity', 'text-ranking', 'table-question-answering', 'question-answering')
VIEWER = 'https://datasets-server.huggingface.co/'
LABEL_COLUMN = re.compile(r'(^|_)(labels?|answer(_?key|_?idx|_?index)?|gold(_label)?|target|class|category|choices|options|'
                          r'endings|tags|score|rating|stars|verdict|stance|sentiment|emotion|polarity)$|^is_\w+$', re.I)
TAG_PREFIXES = ('task_', 'size_', 'language', 'format:', 'library:', 'modality:', 'region:', 'arxiv:', 'doi:')
# Tags and names that mark generated, personal or placeholder uploads.
SLOP = re.compile(r'synthetic|generated|gpt-?[345o]|llama|mistral|claude|gemini|distil|test|demo|sample|dummy|toy|tmp|'
                  r'mini|small|subset|debug|my[-_]|assignment|homework|pii|benchmark-v\d|v0\.\d', re.I)


def prior(row, authors):
    """Metadata score; weights fit catalog sources against the tag inventory (CV AUC 0.96)."""
    meta = row.get('hub_metadata', {})
    age = min(3, max(0, datetime.now(timezone.utc).year - int((meta.get('last_modified') or '9999')[:4])))
    score = 2 * math.log10((meta.get('downloads') or 0) + 1) + math.log10((meta.get('likes') or 0) + 1)
    score += 0.5 * bool(row.get('papers')) + 0.5 * age  # age capped: recent sources are the ones to find
    flags = []
    if SLOP.search(row['name']):
        flags.append('slop_name')
    if any(SLOP.search(t) for t in row.get('hub_tags', []) if not t.startswith(TAG_PREFIXES)):
        flags.append('slop_tag')
    uploads = authors.get(row['name'].split('/')[0], [])
    if len(uploads) >= 25 and statistics.median(uploads) < 50:
        flags.append('bulk_author')  # many uploads nobody uses
    score -= 1.5 * ('slop_name' in flags) + 1.5 * ('slop_tag' in flags) + 3 * ('bulk_author' in flags)
    return round(score, 3), flags


def class_names(feature):
    """ClassLabel names inside a viewer feature, through sequences and lists."""
    if isinstance(feature, list):
        return class_names(feature[0]) if feature else None
    if not isinstance(feature, dict):
        return None
    if feature.get('_type') == 'ClassLabel':
        return feature.get('names')
    return class_names(feature.get('feature')) if 'feature' in feature else None


def label_columns(features):
    return [name for name, feature in (features or {}).items() if class_names(feature) or LABEL_COLUMN.search(name)]


def discrete(name, features, preview):
    """A label column is discrete: a ClassLabel, a choice list, or few short values in the preview."""
    if class_names((features or {}).get(name)):
        return True
    values = [row.get(name) for row in preview if row.get(name) is not None]
    if not values:
        return False
    if all(isinstance(v, list) for v in values):
        return all(1 < len(v) <= 26 and all(isinstance(x, str) and len(x) < 300 for x in v) for v in values)
    if any(isinstance(v, float) and not v.is_integer() for v in values):
        return len(set(values)) <= 11  # coarse ratings; continuous scores are not decisions
    distinct = {json.dumps(v, sort_keys=True) for v in values}
    return len(distinct) <= max(12, len(values) // 4) and all(len(v) <= 40 for v in distinct)


def structure(repo, cache, target=None):
    """Train rows, label columns and preview rows of the largest configuration with a train split."""
    try:
        info = cached_json(VIEWER + 'info?' + urlencode(dict(dataset=repo)), cache)['data']
    except (OSError, ValueError) as error:
        return dict(viewer='unavailable', viewer_error=str(error)[:200])
    best = None
    for config, data in (info.get('dataset_info') or {}).items():
        splits = data.get('splits') or {}
        train = next((s for s in splits if re.match(r'^(train|training)($|[_/-])', s.casefold())), None)
        rows = (splits.get(train) or {}).get('num_examples') or 0 if train else 0
        if train and (best is None or rows > best['train_rows']):
            best = dict(config=config, split=train, train_rows=rows, splits={k: v.get('num_examples') for k, v in splits.items()},
                        labels=label_columns(data.get('features')), columns=list(data.get('features') or {}),
                        signature=signature(info),
                        configs=len(info['dataset_info']))
    if best is None:
        return dict(viewer='no_train', splits=sorted({s for d in (info.get('dataset_info') or {}).values() for s in d.get('splits') or {}}))
    try:
        preview = cached_json(VIEWER + 'first-rows?' + urlencode(dict(dataset=repo, config=best['config'], split=best['split'])), cache)['data']
        rows = [r['row'] for r in preview.get('rows', [])]
    except (OSError, ValueError, KeyError):
        rows = []
    best['labels'] = [name for name in best['labels'] if discrete(name, info['dataset_info'][best['config']].get('features'), rows)]
    if target:  # requested target kinds (boxes, points, actions) need no discreteness check
        best['labels'] += [name for name in best['columns'] if name not in best['labels'] and re.search(target, name, re.I)]
    return dict(best, viewer='ok', preview=rows)


def short(value, limit=300):
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, list):
        return [short(v, limit) for v in value[:12]]
    if isinstance(value, dict):
        return {k: short(v, limit) for k, v in list(value.items())[:20]}
    return value


def review_score(row, extra=()):
    """Least favorable reviewer on fit, train, sampled-row quality, supervision and label origin."""
    answers = current_answers(row)
    if not answers:
        return None
    least = lambda q, k: min(a.get(q, {}).get('probabilities', {}).get(k, 0) for a in answers)
    most = lambda q, k: max(a.get(q, {}).get('probabilities', {}).get(k, 0) for a in answers)
    origin = min(sum(a.get('annotation_origin', {}).get('probabilities', {}).get(k, 0) for k in ('human', 'procedural'))
                 for a in answers)
    score = least('decision_fit', 'yes') + least('train', 'yes') + 2 * least('sample_quality', 'yes')
    score += least('supervision_quality', 'usable') + origin - 2 * most('annotation_origin', 'model')
    score += sum(2 * least(question, 'yes') for question in extra)
    if any('dataset_language' in a for a in answers):
        score += 2 * least('dataset_language', 'yes') - 2
    return round(score, 3)


def origin(row):
    votes = [max(p, key=p.get) for a in current_answers(row) if (p := a.get('annotation_origin', {}).get('probabilities'))]
    return '/'.join(dict.fromkeys(votes))


QUALITY = dict(type='choice', criteria={
    'yes': 'The sampled rows support this statement.', 'no': 'The sampled rows contradict this statement.',
    'unknown': 'The rows are missing, unreadable or insufficient.'},
    instructions='Use the sampled training rows, not the dataset name. Are these rows suitable for a discrete-decision '
    'training task: inputs readable and self-contained, gold labels plausibly correct, the task non-trivial (not '
    'answerable from a keyword or surface pattern), and rows not repetitive template or model-generated filler?')


def question(text):
    return dict(QUALITY, instructions='Use the supplied evidence and sampled rows, not the dataset name. ' + text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('build/discovery/cache'))
    parser.add_argument('--index', type=Path, default=Path('build/discovery/index/jev-segments.npz'))
    parser.add_argument('--signatures', type=Path, default=Path('build/discovery/index/catalog-signatures.json'))
    parser.add_argument('--tag', action='append', default=[], help='Hub task tag; repeat (default: decision tags unless --query)')
    parser.add_argument('--query', action='append', default=[], help='Hub name search, e.g. screenshot; repeat for OR')
    parser.add_argument('--language', action='append', help='Dataset language (default en); judged on the evidence')
    parser.add_argument('--untagged', action='store_true',
                        help='Also keep datasets without language tags (most GUI data); other declared languages stay out')
    parser.add_argument('--candidates', type=Path, action='append', default=[],
                        help='Extra candidate JSONL, e.g. from web_agent.py; always probed, ahead of the prior')
    parser.add_argument('--target', help='Regex of extra target columns accepted without a discreteness check, e.g. bbox|point|action')
    parser.add_argument('--question', action='append', default=[], metavar='NAME=TEXT',
                        help='Extra yes/no judge question, added to the review score; repeat')
    parser.add_argument('--probe', type=int, default=1500, help='Structure checks for the top of the prior')
    parser.add_argument('--review', type=int, default=300, help='Model reviews for the top structural survivors')
    parser.add_argument('--min-train', type=int, default=500)
    parser.add_argument('--model', action='append')
    parser.add_argument('--media', action='store_true', help='Keep image/video/audio sources (default: text only)')
    parser.add_argument('--include-known', action='store_true', help='Calibration: keep catalog sources, marked')
    parser.add_argument('--refresh', action='store_true', help='Fetch new Hub listings instead of cached snapshots')
    args = parser.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    languages = args.language or ['en']
    extra = dict(item.split('=', 1) for item in args.question)
    (out / 'command.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')

    # 1-2. inventory and identity
    rows = {}
    searches = [dict(queries=[q]) for q in args.query] + [dict(tasks=[t]) for t in args.tag or ([] if args.query or args.candidates else TAGS)]
    for search in searches:
        search.update(limit=None, languages=[] if args.untagged else languages)
        found, _ = retrieve(args.cache, offline=not args.refresh and _cached(args.cache, search), **search)
        for r in found:
            if language_priority(r, languages) < 2:  # untagged kept, explicit other languages dropped
                rows.setdefault(r['name'], r)
    for path in args.candidates:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if language_priority(row, languages) < 2:
                rows[row['name']] = dict(row, sources=rows.get(row['name'], {}).get('sources', []) + row['sources'])
    names, repos = catalog_index()
    matched = list(match(rows.values(), names, repos, previous_decisions(), queued_sources(), catalog_references()))
    known = lambda r: r['status'] != 'candidate' or r.get('related_matches')
    pool = matched if args.include_known else [r for r in matched if not known(r)]
    print(f'inventory {len(rows)}; unknown to catalog {sum(not known(r) for r in matched)}', flush=True)

    # 3. prior
    authors = defaultdict(list)
    for r in matched:
        authors[r['name'].split('/')[0]].append(r.get('hub_metadata', {}).get('downloads') or 0)
    for r in pool:
        r['prior'], r['flags'] = prior(r, authors)
        r['known'] = bool(known(r))
    if not args.media:
        pool = [r for r in pool if not {'image', 'video', 'audio'} & set(r.get('modalities') or [])]
    agent = lambda r: any(s.get('source') == 'web_agent' for s in r.get('sources', []))
    pool.sort(key=lambda r: (not agent(r), -r['prior']))  # agent finds were chosen by meaning, not popularity
    head = pool[:args.probe + sum(map(agent, pool))]

    # 4. structure and content identity
    with ThreadPoolExecutor(max_workers=4) as executor:
        probed = list(executor.map(lambda r: structure(r['name'], args.cache, args.target), head))
    index = Index(args.index) if args.index.exists() else None
    signatures = DatasetIndex(args.signatures) if args.signatures.exists() else None
    claimed = {}
    survivors = []
    for r, s in zip(head, probed):
        preview = s.pop('preview', [])
        r['structure'] = s
        reasons = []
        if s['viewer'] != 'ok':
            reasons.append(s['viewer'])
        else:
            if s['train_rows'] < args.min_train:
                reasons.append('small_train')
            if not s['labels']:
                reasons.append('no_label_column')
        if signatures is not None and s.get('signature'):
            r['signature_matches'] = signatures.matches(s['signature'])
            if r['signature_matches']:
                reasons.append('signature_in_catalog')
        segments = set().union(*map(row_segments, preview)) if preview else set()
        if index is not None and segments:
            # a few shared segments are boilerplate; mirrors and derivatives share many
            enough = max(3, 0.02 * len(segments))
            r['content_matches'] = {k: v for k, v in index.owners_of(segments).most_common(5) if v >= enough}
            if r['content_matches']:
                reasons.append('content_in_catalog')
        dup = Counter(claimed[h] for h in segments if h in claimed)
        if dup and dup.most_common(1)[0][1] >= 3:
            r['duplicate_of'] = dup.most_common(1)[0][0]; reasons.append('duplicate_candidate')
        for h in segments:
            claimed.setdefault(h, r['name'])
        r['rejections'] = reasons
        r['preview'] = [short(p) for p in preview[:5]]
        if not reasons:
            survivors.append(r)
    write_rows(out / 'probed.jsonl', head)
    print(f'probed {len(head)}; structural survivors {len(survivors)}', flush=True)

    # 5. review
    reviewed = []
    if args.review and survivors:
        from . import jev_review
        jev_review.QUESTIONS['sample_quality'] = QUALITY
        jev_review.QUESTIONS.update({name: question(text) for name, text in extra.items()})
        batch = []
        for r in survivors[:args.review]:
            s = r['structure']
            evidence = [dict(kind='viewer_structure', url=VIEWER + 'info?' + urlencode(dict(dataset=r['name'])),
                             metadata={k: s[k] for k in ('config', 'split', 'splits', 'columns', 'labels', 'train_rows')}),
                        dict(kind='sampled_training_rows', rows=r['preview'])]
            batch.append(dict(r, evidence=evidence, train_status='present', source_splits=sorted(s['splits']),
                              split_policy='multiple_splits' if len(s['splits']) > 1 else 'train_only'))
        jev_review.run(batch, out / 'review', args.cache, args.model or [jev_review.MODEL, 'openrouter/~typesafe/jev-latest'],
                       languages=languages)
        reviewed = [json.loads(l) for l in (out / 'review/reviews.jsonl').read_text().splitlines()]

    # 6. ranking
    for r in reviewed:
        r['review_score'] = review_score(r, extra)
    ranked = sorted(reviewed, key=lambda r: -((r['review_score'] or 0) + 0.1 * r['prior']))
    write_rows(out / 'ranked.jsonl', ranked)
    lines = [f'# Prospects {datetime.now(timezone.utc).date()}', '',
             f'Inventory {len(rows)}, unknown to the catalog {sum(not known(r) for r in matched)}, probed {len(head)}, '
             f'structural survivors {len(survivors)}, reviewed {len(reviewed)}.', '',
             '| # | Dataset | Score | Prior | Train rows | Labels | Label origin |', '|---:|---|---:|---:|---:|---|---|']
    for i, r in enumerate(ranked[:100], 1):
        lines.append(f"| {i} | [{r['name']}](https://huggingface.co/datasets/{r['name']}) | {r['review_score']} | {r['prior']} | "
                     f"{r['structure']['train_rows']} | {', '.join(r['structure']['labels'][:3])} | {origin(r)} |")
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:30]))


def _cached(cache, search):
    try:
        retrieve(cache, offline=True, **search)
        return True
    except FileNotFoundError:
        return False


if __name__ == '__main__':
    main()
