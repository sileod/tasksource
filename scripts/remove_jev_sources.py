"""Remove sources from the published Jev ``vision`` config without rebuilding it.

    PYTHONPATH=.:src python -m scripts.remove_jev_sources --sources vision/rico-widget/grid7 \
        --push --release vYYYY.MM.DD.n

Only shards holding a removed source are rewritten, keeping their other rows in order and their
names; every other shard stays byte-identical. The card counts, ``vision/release-audit.json``,
``vision/sources.yaml`` and ``CHANGELOG.md`` are updated, ``vision/removals/<release>.json`` records
the rows removed and the rewritten shards, and the release is tagged. A release needs its
``DATA_CHANGELOG.md`` entry. Text configs change through ``build_jev_dataset.py`` rebuilds.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, HfFileSystem, hf_hub_download

from scripts.build_jev_dataset import source_family

REPO = 'tasksource/tasksource-jev-typed-decisions'
CHANGELOG = Path(__file__).resolve().parents[1] / 'DATA_CHANGELOG.md'


def run(args):
    api, fs = HfApi(), HfFileSystem()
    head = api.dataset_info(REPO, files_metadata=True)
    if args.release in {ref.name for ref in api.list_repo_refs(REPO, repo_type='dataset').tags}:
        raise ValueError(f'Release tag already exists: {args.release}')
    entry = re.search(r'^## tasksource-jev-typed-decisions ' + re.escape(args.release) + r'\n.*?(?=^## |\Z)',
                      CHANGELOG.read_text(), re.M | re.S)
    if entry is None:
        raise ValueError(f'No changelog entry for {args.release}')
    removed_sources = set(args.sources)
    shards = sorted(s.rfilename for s in head.siblings
                    if s.rfilename.startswith('vision/train') and s.rfilename.endswith('.parquet'))
    sizes = {s.rfilename: s.size for s in head.siblings}
    args.output.mkdir(parents=True, exist_ok=True)
    removed, rewritten, files = Counter(), {}, []
    licenses, removed_bytes = Counter(), 0
    for name in shards:
        sources = pq.read_table(fs.open(f'datasets/{REPO}@{head.sha}/{name}'), columns=['source']).column('source')
        hit = pc.is_in(sources, value_set=pa.array(sorted(removed_sources)))
        if not pc.any(hit).as_py():
            continue
        print('Rewrite', name, flush=True)
        table = pq.read_table(hf_hub_download(REPO, name, repo_type='dataset', revision=head.sha))
        mask = pc.is_in(table.column('source'), value_set=pa.array(sorted(removed_sources)))
        dropped = table.filter(mask)
        removed.update(dropped.column('source').to_pylist())
        licenses.update(dropped.column('license_use').to_pylist())
        removed_bytes += dropped.nbytes
        path = args.output / Path(name).name
        pq.write_table(table.filter(pc.invert(mask)), path)
        rewritten[name] = {'rows_removed': dropped.num_rows, 'rows_kept': table.num_rows - dropped.num_rows,
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        files.append((name, path))
    missing = removed_sources - set(removed)
    if missing:
        raise ValueError(f'Sources not in the vision config: {sorted(missing)}')

    def download(name):
        return Path(hf_hub_download(REPO, name, repo_type='dataset', revision=head.sha))
    notes = download('CHANGELOG.md').read_text()
    offset = notes.index('\n## ')
    (args.output / 'CHANGELOG.md').write_text(notes[:offset] + '\n' + entry[0].rstrip() + '\n' + notes[offset:])
    audit = json.loads(download('vision/release-audit.json').read_text())
    train = audit['splits']['train']
    train['rows'] -= sum(removed.values())
    families = Counter(source_family(source) for source in removed.elements())
    for key, values in [('sources', removed), ('families', families), ('license_use', licenses)]:
        train[key] = {k: v for k, v in (Counter(train[key]) - values).items() if v}
    audit.setdefault('removals', []).append({'manifest': f'vision/removals/{args.release}.json',
                                             'train_rows': sum(removed.values())})
    sources_doc = yaml.safe_load(download('vision/sources.yaml').read_text())
    for source in removed:
        sources_doc['sources'].pop(source, None)
    sources_doc['datasets'] = sorted({repo for info in sources_doc['sources'].values()
        for repo in [info.get('dataset'), *info.get('data_files_from', []), *info.get('originals', [])] if repo})
    card = DatasetCard.load(str(download('README.md')))
    info = next(x for x in card.data.dataset_info if x['config_name'] == 'vision')
    split = next(x for x in info['splits'] if x['name'] == 'train')
    split['num_examples'] -= sum(removed.values())
    split['num_bytes'] -= removed_bytes
    info['dataset_size'] -= removed_bytes
    info['download_size'] += sum(path.stat().st_size - sizes[name] for name, path in files)
    manifest = {'release': args.release, 'baseline_revision': head.sha, 'removed_sources': dict(removed),
                'reason': args.reason, 'rewritten_shards': rewritten,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    card.save(args.output / 'README.md')
    (args.output / 'release-audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    (args.output / 'sources.yaml').write_text(yaml.safe_dump(sources_doc, sort_keys=False))
    print('Staged', json.dumps({k: manifest[k] for k in ('removed_sources',)}), len(files), 'shards', flush=True)
    if not args.push:
        return
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=path) for name, path in files]
    operations += [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=args.output / local)
                   for remote, local in [('README.md', 'README.md'), ('CHANGELOG.md', 'CHANGELOG.md'),
                                         ('vision/release-audit.json', 'release-audit.json'),
                                         ('vision/sources.yaml', 'sources.yaml'),
                                         (f'vision/removals/{args.release}.json', 'manifest.json')]]
    result = api.create_commit(REPO, repo_type='dataset', parent_commit=head.sha, operations=operations,
                               commit_message=f'Remove {sum(removed.values()):,} vision rows: {", ".join(sorted(removed))}')
    api.create_tag(REPO, repo_type='dataset', tag=args.release, revision=result.oid)
    print('Published', result.oid, 'train rows', train['rows'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sources', nargs='+', required=True, help='Published source names, e.g. vision/rico-widget/grid7')
    parser.add_argument('--reason', default='')
    parser.add_argument('--output', type=Path, default=Path('build/vision-removal'))
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--release')
    args = parser.parse_args()
    if args.push and not args.release:
        parser.error('--push requires --release')
    run(args)
