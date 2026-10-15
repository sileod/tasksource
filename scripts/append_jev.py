"""Add sources to tasksource-jev-typed-decisions without republishing it.

    python -m scripts.append_jev --config full --tasks TASK ... --push --release vYYYY.MM.DD.n
    python -m scripts.append_jev --config vision --tasks TASK ... --cap 30000 --push --release vYYYY.MM.DD.n

``full`` builds the new text sources with the canonical builder (``build_jev_dataset.py``) and
applies the release row pipeline: bad examples, label balance, prompt styling, license columns.
Rows overlapping the published train or evaluation splits are dropped. Only training rows are
appended; published shards stay byte-identical. ``default`` (the steered mix) changes only at the
next full rebuild, which includes these sources from the catalog. ``vision`` runs
``append_jev_vision``. A release needs its ``DATA_CHANGELOG.md`` entry and gets a Hub tag.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from datasets import Dataset, Features, concatenate_datasets, disable_progress_bars, load_dataset
import pyarrow as pa
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, HfFileSystem, hf_hub_download
import pyarrow.parquet as pq
import yaml

from scripts import append_jev_vision
from scripts.build_jev_dataset import (
    TRAINING_FEATURES, add_example_ids, add_license_columns, add_question_groups, add_source_indices,
    bad_examples, balance_labels, check_gold_positions, content_keys, diversify_published_prompts,
    drop_train_overlap, exclude_publish_sources, latest_records, row_format, source_family,
    source_indices, source_licenses, source_provenance, validate_decisions,
)

ROOT = Path(__file__).resolve().parents[1]
REPO = append_jev_vision.REPO
CONFIG = 'full'
KEY_COLUMNS = ['id', 'example_id', 'source', 'state', 'options', 'variant']
TEXT_COLUMNS = ['state', 'kind', 'id', 'options', 'target', 'question', 'source', 'variant', 'split']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def changelog_entry(release):
    entry = re.search(r'^## tasksource-jev-typed-decisions ' + re.escape(release) + r'\n.*?(?=^## |\Z)',
                      (ROOT / 'DATA_CHANGELOG.md').read_text(), re.M | re.S)
    if entry is None:
        raise ValueError(f'No changelog entry for {release}')
    return entry[0].rstrip()


def index_baseline(output, api, fs):
    """Published key columns of the config, cached as Parquet with the revision and shard blobs."""
    path = output / 'baseline.json'
    if not path.exists():
        info = api.dataset_info(REPO, files_metadata=True)
        base = {'revision': info.sha, 'parquets': {s.rfilename: s.blob_id for s in info.siblings
                if s.rfilename.startswith(CONFIG + '/') and s.rfilename.endswith('.parquet')}}
        tables = {}
        for name in sorted(base['parquets']):
            print('Index baseline', name, flush=True)
            with fs.open(f'datasets/{REPO}@{info.sha}/{name}', block_size=1024 * 1024) as stream:
                split = 'train' if name.startswith(CONFIG + '/train-') else 'evaluation'
                tables.setdefault(split, []).append(pq.ParquetFile(stream).read(columns=KEY_COLUMNS))
        for split, parts in tables.items():
            pq.write_table(pa.concat_tables(parts), output / f'baseline-{split}.parquet')
        dump(path, base)
    base = json.loads(path.read_text())
    base['rows'] = {split: Dataset.from_parquet(str(output / f'baseline-{split}.parquet'))
                    for split in ('train', 'evaluation')}
    return base


def build_rows(output, tasks, args):
    """Canonical per-task shards, then the release row pipeline on their training rows."""
    work = output / 'build'
    command = [sys.executable, str(ROOT / 'scripts' / 'build_jev_dataset.py'), '--output', str(work),
               '--tasks', *tasks, '--max-rows', str(args.max_rows), '--max-rows-eval', str(args.max_rows_eval),
               '--skip-migrate']
    subprocess.run(command, check=True, env={**os.environ, 'PYTHONPATH': f'{ROOT}:{ROOT / "src"}'})
    records = latest_records(work / 'build-report.jsonl')
    failed = [task for task in tasks if records.get(task, {}).get('status') != 'ok']
    if failed:
        raise ValueError(f'Source builds failed: {failed}')
    files = [str(path) for task in tasks for path in sorted((work / 'data').glob('train-*.parquet'))
             if path.name == f'train-{append_jev_vision.slug(task)}.parquet']
    rows = load_dataset('parquet', data_files=files, split='train', columns=TEXT_COLUMNS,
                        features=Features({name: TRAINING_FEATURES[name] for name in TEXT_COLUMNS}))
    rows = exclude_publish_sources(rows, allowed_sources=set(tasks))
    rows = add_source_indices(rows, source_indices(work))
    rows = rows.filter(lambda states: [state is not None for state in states], input_columns='state', batched=True)
    bad = bad_examples()
    rows = add_example_ids(rows).filter(lambda key: key not in bad, input_columns='example_id')
    return balance_labels(rows), records, work


def select_new(rows, base):
    """Drop rows whose example or content is already published (train or evaluation)."""
    published = {e for split in base['rows'].values() for e in split.unique('example_id')}
    rows = rows.filter(lambda ids: [i not in published for i in ids], input_columns='example_id', batched=True)
    dropped = {}
    for split, baseline in base['rows'].items():
        rows, dropped[split] = drop_train_overlap(rows, content_keys(baseline))
    return rows, dropped


def run(args):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    api, fs = HfApi(), HfFileSystem()
    base = index_baseline(output, api, fs)
    existing = sorted(set(args.tasks) & set(base['rows']['train'].unique('source')))
    if existing:
        raise ValueError(f'Already in {CONFIG}; changing a published source needs a full rebuild: {existing}')
    rows, records, work = build_rows(output, args.tasks, args)
    before = len(rows)
    rows, overlap = select_new(rows, base)
    rows = diversify_published_prompts(add_question_groups(rows)).shuffle(seed=args.seed).flatten_indices()
    licenses = source_licenses(sorted(set(rows.unique('source'))))
    rows = add_license_columns(rows, licenses)
    validate_decisions(rows, 'train')
    choice = {task for task, record in records.items() if record.get('task_type') == 'MultipleChoice'}
    check_gold_positions(rows, 'train', choice)
    if not len(rows):
        raise ValueError('No unseen training rows')
    with fs.open(f'datasets/{REPO}@{base["revision"]}/{min(n for n in base["parquets"] if "/train-" in n)}') as stream:
        schema = pq.ParquetFile(stream).schema_arrow
    rows = rows.select_columns(schema.names).cast(Features.from_arrow_schema(schema))
    stamp = hashlib.sha256((base['revision'] + '\n'.join(sorted(args.tasks))).encode()).hexdigest()[:12]
    count = max(1, math.ceil(rows.data.nbytes / 256_000_000))
    files = [output / f'train-append-{stamp}-{i:05d}-of-{count:05d}.parquet' for i in range(count)]
    for index, path in enumerate(files):
        rows.shard(count, index, contiguous=True).to_parquet(path)
    counts = Counter(rows['source'])
    manifest = dict(config=CONFIG, baseline_revision=base['revision'], seed=args.seed, max_rows=args.max_rows,
        added_rows=len(rows), sources=dict(counts), dropped_overlap={'built': before, **overlap},
        revisions={task: records[task].get('revisions') for task in args.tasks},
        evaluation='unchanged', default='unchanged until the next full rebuild', preserved_parquets=base['parquets'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(output / 'manifest.json', manifest)
    print('Staged', json.dumps({k: manifest[k] for k in ('added_rows', 'sources', 'dropped_overlap')}), flush=True)
    if args.push:
        publish(args, api, base, rows, files, counts, licenses, records, manifest, stamp, output)


def publish(args, api, base, rows, files, counts, licenses, records, manifest, stamp, output):
    if args.release in {ref.name for ref in api.list_repo_refs(REPO, repo_type='dataset').tags}:
        raise ValueError(f'Release tag already exists: {args.release}')
    head = api.dataset_info(REPO, files_metadata=True)
    current = {s.rfilename: s.blob_id for s in head.siblings
               if s.rfilename.startswith(CONFIG + '/') and s.rfilename.endswith('.parquet')}
    if current != base['parquets']:
        raise ValueError(f'{CONFIG} shards changed since staging; restage')

    def download(name):
        return Path(hf_hub_download(REPO, name, repo_type='dataset', revision=head.sha))
    notes = download('CHANGELOG.md').read_text()
    offset = notes.index('\n## ')
    (output / 'CHANGELOG.md').write_text(notes[:offset] + '\n' + changelog_entry(args.release) + '\n' + notes[offset:])
    audit = json.loads(download('release-audit.json').read_text())
    train = audit['splits']['train']; train['rows'] += len(rows)
    formats = {task: record.get('task_type') for task, record in records.items()}
    columns = rows.select_columns(['source', 'kind', 'variant', 'license_use'])[:]
    for key, values in [('sources', counts), ('families', Counter(map(source_family, columns['source']))),
                        ('formats', Counter(row_format(s, formats) for s in columns['source'])),
                        ('kinds', Counter(columns['kind'])), ('variants', Counter(columns['variant'])),
                        ('license_use', Counter(columns['license_use']))]:
        train[key] = dict(Counter(train[key]) + values)
    audit.setdefault('incremental_additions', []).append(
        {'manifest': f'{CONFIG}/additions/{stamp}/manifest.json', 'train_rows': len(rows)})
    sources_doc = yaml.safe_load(download('sources.yaml').read_text())
    for source, n in counts.items():
        info = {'rows': {'train': n}, **source_provenance(source)}
        if records[source].get('revisions'):
            info['revisions'] = {repo: sha for repo, sha in records[source]['revisions'].items() if sha}
        sources_doc['sources'][source] = {**info, **licenses[source]}
    sources_doc['datasets'] = sorted({repo for info in sources_doc['sources'].values()
        for repo in [info.get('dataset'), *info.get('data_files_from', []), *info.get('originals', [])] if repo})
    card = DatasetCard.load(str(download('README.md')))
    info = next(x for x in card.data.dataset_info if x['config_name'] == CONFIG)
    split_info = next(x for x in info['splits'] if x['name'] == 'train')
    split_info['num_examples'] += len(rows); split_info['num_bytes'] += rows.data.nbytes
    info['dataset_size'] += rows.data.nbytes
    info['download_size'] += sum(p.stat().st_size for p in files)
    card.save(output / 'README.md'); dump(output / 'release-audit.json', audit)
    (output / 'sources.yaml').write_text(yaml.safe_dump(sources_doc, sort_keys=False, allow_unicode=True, width=120))
    operations = [CommitOperationAdd(path_in_repo=f'{CONFIG}/{p.name}', path_or_fileobj=p) for p in files]
    operations += [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=output / local) for remote, local in
                   [('README.md', 'README.md'), ('CHANGELOG.md', 'CHANGELOG.md'), ('release-audit.json', 'release-audit.json'),
                    ('sources.yaml', 'sources.yaml'), (f'{CONFIG}/additions/{stamp}/manifest.json', 'manifest.json')]]
    result = api.create_commit(REPO, repo_type='dataset', parent_commit=head.sha, operations=operations,
                               commit_message=f'Append {len(rows):,} {CONFIG} training rows; preserve published data')
    blobs = {s.rfilename: s.blob_id for s in api.dataset_info(REPO, revision=result.oid, files_metadata=True).siblings}
    assert all(blobs.get(name) == digest for name, digest in base['parquets'].items())
    api.create_tag(REPO, repo_type='dataset', tag=args.release, revision=result.oid)
    dump(output / 'published.json', dict(revision=result.oid, train_rows=train['rows'], **manifest))
    print('Published', result.oid, 'train rows', train['rows'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', choices=['full', 'vision'], required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cap', type=int, default=30_000, help='vision: rows per dataset family')
    parser.add_argument('--max-rows', type=int, default=30_000, help='full: training rows built per source')
    parser.add_argument('--max-rows-eval', type=int, default=3_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--release', help='Version with a "## tasksource-jev-typed-decisions VERSION" entry in DATA_CHANGELOG.md')
    args = parser.parse_args()
    if args.push and not args.release:
        parser.error('--push requires --release')
    if args.release:
        changelog_entry(args.release)
    args.output = args.output or ROOT / 'build' / f'append-{args.config}'
    disable_progress_bars()
    (append_jev_vision.run if args.config == 'vision' else run)(args)


if __name__ == '__main__':
    main()
