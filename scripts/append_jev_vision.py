"""Grow the published vision training split without replacing existing rows.

Reuse the canonical builder, resume per task, and append only unseen requests.
Evaluation shards remain byte-identical. Dataset families share the requested cap.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

from datasets import concatenate_datasets, disable_progress_bars, load_from_disk
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, HfFileSystem, hf_hub_download
import pyarrow.parquet as pq
import yaml

from scripts.build_jev_dataset import build_vision, source_family, slug, validate_decisions

REPO = 'tasksource/tasksource-jev-typed-decisions'
COLUMNS = ['id', 'source', 'state', 'question', 'kind', 'options', 'target', 'metadata']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def identities(row):
    meta = json.loads(row['metadata'])
    images = list(meta['image_ids'])
    original = meta.get('augmentation', {}).get('original_image_sha256')
    if original:
        images.append(original)
    return set(images)


def request_key(row):
    # A permutation of the same options is still the same training example.
    meta = json.loads(row['metadata'])
    payload = [row['source'], meta['image_ids'], row['state'], row['question'],
               row['kind'], sorted(row['options'])]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def select_additions(candidates, baseline, evaluation, cap):
    counts = Counter(source_family(row['source']) for row in baseline)
    seen = {request_key(row) for row in baseline + evaluation}
    ids = {row['id'] for row in baseline + evaluation}
    forbidden = set().union(*(identities(row) for row in evaluation))
    buckets = defaultdict(deque)
    for index, row in enumerate(candidates):
        buckets[row['source']].append((index, row))
    selected, dropped = [], Counter()
    # Share the remaining family budget across task views, preserving old allocations.
    while buckets:
        for source in list(buckets):
            index, row = buckets[source].popleft()
            family, key = source_family(source), request_key(row)
            if counts[family] >= cap:
                dropped['family_cap'] += 1
            elif key in seen or row['id'] in ids:
                dropped['existing_or_duplicate_request'] += 1
            elif identities(row) & forbidden:
                dropped['published_evaluation_image'] += 1
            elif json.loads(row['metadata']).get('source_redistribution') == 'restricted':
                dropped['restricted_redistribution'] += 1
            else:
                selected.append(index); seen.add(key); ids.add(row['id']); counts[family] += 1
            if not buckets[source]:
                del buckets[source]
    return sorted(selected), dict(dropped)


def run(args):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    api, fs = HfApi(), HfFileSystem()
    baseline_path = output / 'baseline.json'
    if baseline_path.exists():
        base = json.loads(baseline_path.read_text())
    else:
        info = api.dataset_info(REPO, files_metadata=True)
        base = {'revision': info.sha, 'cap': args.cap, 'seed': args.seed,
                'parquets': {s.rfilename: s.blob_id for s in info.siblings
                             if s.rfilename.startswith('vision/') and s.rfilename.endswith('.parquet')},
                'train': [], 'evaluation': []}
        for filename in base['parquets']:
            print('Index baseline', filename, flush=True)
            # Read only text/provenance column chunks, not eleven GB of image payloads.
            with fs.open(f'datasets/{REPO}@{info.sha}/{filename}', block_size=1024 * 1024) as stream:
                rows = pq.ParquetFile(stream).read(columns=COLUMNS).to_pylist()
            base['train' if filename.startswith('vision/train-') else 'evaluation'].extend(rows)
        dump(baseline_path, base)
    if (base['cap'], base['seed']) != (args.cap, args.seed):
        raise ValueError('Resume requires the original cap and seed')
    sources = (['vision/' + task.removeprefix('vision/') for task in args.tasks]
               if getattr(args, 'tasks', None) else sorted({row['source'] for row in base['train']}))
    records, frames = {}, []
    for source in sources:
        directory = output / 'candidates' / slug(source)
        complete = directory / 'dataset' / 'dataset_dict.json'
        # Existing FigureQA license review prohibits further redistribution.
        if source == 'vision/figureqa':
            records[source] = {'status': 'skipped', 'reason': 'original terms prohibit redistribution'}
            continue
        print('Build', source, flush=True)
        try:
            if complete.exists():
                data = load_from_disk(str(directory / 'dataset'))
            else:
                data = build_vision(SimpleNamespace(output=directory, tasks=[source.removeprefix('vision/')],
                    excluded_sources=[], license_use=None, limit=None, max_rows=args.cap,
                    max_rows_eval=1, seed=args.seed, repo_id=REPO, upload=False,
                    grounding_probabilities=None))
            frames.append(data['train'])
            records[source] = {'status': 'ok', 'candidate_rows': len(data['train']),
                'provenance': yaml.safe_load((directory / 'sources.yaml').read_text())['sources'][source],
                'build_manifest': json.loads((directory / 'build-manifest.json').read_text())}
        except Exception as error:
            records[source] = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
            print(source, records[source], flush=True)
        dump(output / 'progress.json', records)
    failed = [source for source, record in records.items() if record['status'] == 'failed']
    if args.push and args.tasks and failed:
        raise ValueError(f'Requested source builds failed; publication stopped: {failed}')
    if not frames:
        raise ValueError('No source builds succeeded')
    candidates = concatenate_datasets(frames)
    selected, dropped = select_additions(list(candidates.select_columns(COLUMNS)),
                                         base['train'], base['evaluation'], args.cap)
    addition = candidates.select(selected).shuffle(seed=args.seed)
    validate_decisions(addition, 'train')
    if not len(addition):
        raise ValueError('No unseen eligible training rows')
    stamp = hashlib.sha256((base['revision'] + str(args.cap) + str(args.seed) + '\n'.join(sources)).encode()).hexdigest()[:12]
    files = []; count = max(1, math.ceil(addition.data.nbytes / 256_000_000))
    for index in range(count):
        path = output / f'train-append-{stamp}-{index:05d}-of-{count:05d}.parquet'
        addition.shard(count, index, contiguous=True).to_parquet(path)
        files.append(path)
    counts = Counter(addition['source'])
    manifest = dict(baseline_revision=base['revision'], cap=args.cap, seed=args.seed,
        added_rows=len(addition), sources=dict(counts), exclusions=dropped, builds=records,
        evaluation='unchanged', preserved_parquets=base['parquets'],
        sampling='union of preserved rows and unseen seeded reservoir candidates; shared family cap',
        ordering='new rows shuffled with the recorded seed; published shards unchanged',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(output / 'manifest.json', manifest)
    print('Staged', json.dumps(manifest), flush=True)
    if not args.push:
        return
    # Rebase metadata over unrelated text updates, but never over a changed vision release.
    head = api.dataset_info(REPO, files_metadata=True)
    if args.release in {ref.name for ref in api.list_repo_refs(REPO, repo_type='dataset').tags}:
        raise ValueError(f'Release tag already exists: {args.release}')
    current = {s.rfilename: s.blob_id for s in head.siblings
               if s.rfilename.startswith('vision/') and s.rfilename.endswith('.parquet')}
    if current != base['parquets']:
        raise ValueError('Vision shards changed during build; publication needs reconciliation')
    def download(name):
        return Path(hf_hub_download(REPO, name, repo_type='dataset', revision=head.sha))
    # Publish only this entry: the local changelog may include a pending text rebuild.
    changelog = Path(__file__).resolve().parents[1] / 'DATA_CHANGELOG.md'
    entry = re.search(r'^## tasksource-jev-typed-decisions ' + re.escape(args.release) + r'\n.*?(?=^## |\Z)',
                      changelog.read_text(), re.M | re.S)
    if entry is None:
        raise ValueError(f'No changelog entry for {args.release}')
    published_notes = download('CHANGELOG.md').read_text()
    offset = published_notes.index('\n## ')
    (output / 'CHANGELOG.md').write_text(published_notes[:offset] + '\n' + entry[0].rstrip() + '\n' + published_notes[offset:])
    card = DatasetCard.load(str(download('README.md')))
    audit = json.loads(download('vision/release-audit.json').read_text())
    sources_doc = yaml.safe_load(download('vision/sources.yaml').read_text())
    train = audit['splits']['train']; train['rows'] += len(addition)
    for key, values in [('sources', counts), ('families', Counter(source_family(s) for s in addition['source'])),
                        ('license_use', Counter(addition['license_use']))]:
        train[key] = dict(Counter(train[key]) + values)
    for source, n in counts.items():
        if source not in sources_doc['sources']:
            sources_doc['sources'][source] = {**records[source]['provenance'], 'rows': {'train': 0}}
        sources_doc['sources'][source]['rows']['train'] += n
    sources_doc['datasets'] = sorted({repo for info in sources_doc['sources'].values()
        for repo in [info.get('dataset'), *info.get('data_files_from', []), *info.get('originals', [])] if repo})
    audit.setdefault('incremental_additions', []).append({'manifest': f'vision/additions/{stamp}/manifest.json',
                                                          'train_rows': len(addition)})
    info = next(x for x in card.data.dataset_info if x['config_name'] == 'vision')
    train_info = next(x for x in info['splits'] if x['name'] == 'train')
    train_info['num_examples'] += len(addition); train_info['num_bytes'] += addition.data.nbytes
    info['dataset_size'] += addition.data.nbytes
    info['download_size'] += sum(p.stat().st_size for p in files)
    card.save(output / 'README.md'); dump(output / 'release-audit.json', audit)
    (output / 'sources.yaml').write_text(yaml.safe_dump(sources_doc, sort_keys=False))
    operations = [CommitOperationAdd(path_in_repo='vision/' + p.name, path_or_fileobj=p) for p in files]
    operations += [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=output / local)
                   for remote, local in [('README.md', 'README.md'), ('vision/release-audit.json', 'release-audit.json'),
                                         ('CHANGELOG.md', 'CHANGELOG.md'),
                                         ('vision/sources.yaml', 'sources.yaml'),
                                         (f'vision/additions/{stamp}/manifest.json', 'manifest.json')]]
    result = api.create_commit(REPO, repo_type='dataset', parent_commit=head.sha, operations=operations,
                               commit_message=f'Append {len(addition):,} vision training rows; preserve previous data and evaluation')
    after = api.dataset_info(REPO, revision=result.oid, files_metadata=True)
    blobs = {s.rfilename: s.blob_id for s in after.siblings}
    assert all(blobs.get(name) == digest for name, digest in base['parquets'].items())
    api.create_tag(REPO, repo_type='dataset', tag=args.release, revision=result.oid)
    dump(output / 'published.json', dict(revision=result.oid, train_rows=train['rows'], **manifest))
    print('Published', result.oid, 'train rows', train['rows'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/vision-append-20k'))
    parser.add_argument('--cap', type=int, default=20_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tasks', nargs='+', help='Append only these vision tasks, including newly registered sources')
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--release', help='Version with a "## tasksource-jev-typed-decisions VERSION" entry in DATA_CHANGELOG.md; required for --push')
    args = parser.parse_args()
    if args.cap <= 0:
        parser.error('--cap must be positive')
    if args.push and not args.release:
        parser.error('--push requires --release')
    disable_progress_bars()
    run(args)
