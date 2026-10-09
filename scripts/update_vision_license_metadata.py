"""Apply reviewed component-level licenses to an existing, pinned vision export.

Images and decisions remain byte-for-byte unchanged. Default is staging only;
--publish commits the staged files atomically against the reviewed parent.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, hf_hub_download

from tasksource.licenses import provenance_repos, source_license
from tasksource.metadata.source_license_evidence import SOURCE_LICENSE_REVIEWS


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='tasksource/tasksource-jev-typed-decisions')
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.dataset_info(args.repo, revision=args.parent, files_metadata=True)

    def remote(name):
        return Path(hf_hub_download(args.repo, name, repo_type='dataset', revision=args.parent))

    sources = yaml.safe_load(remote('vision/sources.yaml').read_text())
    release = json.loads(remote('vision/release-audit.json').read_text())
    card = DatasetCard.load(remote('README.md'))
    splits = {s: Counter() for s in release['splits']}
    counts, files, byte_changes = Counter(), [], Counter()
    for file in info.siblings:
        name = file.rfilename
        if not (name.startswith('vision/') and name.endswith('.parquet')):
            continue
        # Use already verified local shards only if their SHA matches this parent.
        local = next((p for d in ('build/vision-prompt-license-repair', 'build/vision-append-release')
                      if (p := Path(d) / Path(name).name).exists()
                      and file.lfs and hashlib.sha256(p.read_bytes()).hexdigest() == file.lfs.sha256), None)
        table = pq.read_table(local or remote(name))
        split = Path(name).name.split('-')[0]
        updates = {field: [] for field in ('metadata', 'license', 'license_use')}
        changed = False
        for row in table.select(['source', *updates]).to_pylist():
            family = row['source'].removeprefix('vision/').split('/')[0]
            if family in SOURCE_LICENSE_REVIEWS:
                metadata = json.loads(row['metadata'])
                old = metadata['licenses']
                new = source_license(row['source'], provenance_repos(metadata['provenance']), old.get('card_licenses', {}))
                if old['license_use'] == 'non-commercial':
                    new['license_use'] = 'non-commercial'
                metadata['licenses'] = new
                metadata['license_review_parent'] = args.parent
                if family == 'vsr':
                    metadata.update(source_license='cc-by-4.0', source_code_license='apache-2.0',
                                    source_license_url=new['license_review']['annotations']['url'])
                row.update(metadata=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                           license=new['license'], license_use=new['license_use'])
                sources['sources'][row['source']].update(new)
                counts[row['source']] += 1
                changed = True
            splits[split][row['license_use']] += 1
            for key in updates:
                updates[key].append(row[key])
        if not changed:
            continue
        patched = table
        for key, values in updates.items():
            field = table.schema.field(key)
            patched = patched.set_column(table.schema.get_field_index(key), field, pa.array(values, type=field.type))
        path = root / Path(name).name
        pq.write_table(patched, path, compression='zstd')
        restored = pq.read_table(path)
        for key in table.column_names:
            assert restored[key].equals(patched[key] if key in updates else table[key]), (name, key)
        byte_changes[split] += patched.nbytes - table.nbytes
        files.append({'path': name, 'local': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'size_change': path.stat().st_size - file.size})
        print(f'Staged {name}', flush=True)
    for split, uses in splits.items():
        assert sum(uses.values()) == release['splits'][split]['rows']
        release['splits'][split]['license_use'] = dict(uses)
    report = {'parent_revision': args.parent, 'checked': '2026-10-09', 'reviewed_rows': dict(counts),
              'splits': {s: dict(c) for s, c in splits.items()}, 'sources': SOURCE_LICENSE_REVIEWS,
              'checks': ['all non-license columns, including images, gold targets, IDs and splits unchanged',
                         'all retained rows counted; no source excluded'],
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    release['license_review'] = report
    dump(root / 'license-review.json', report)
    dump(root / 'release-audit.json', release)
    (root / 'sources.yaml').write_text(yaml.safe_dump(sources, sort_keys=False))
    manifest = {'operation': 'review original annotation and image terms', 'parent_revision': args.parent,
                'files': files, 'license_review': report,
                'parent_manifest': json.loads(remote('vision/build-manifest.json').read_text())}
    dump(root / 'build-manifest.json', manifest)
    quality = json.loads(remote('vision/quality-audit.json').read_text())
    quality['license_review'] = report
    dump(root / 'quality-audit.json', quality)
    vision = next(entry for entry in card.data.dataset_info if entry['config_name'] == 'vision')
    for split in vision['splits']:
        split['num_bytes'] += byte_changes[split['name']]
    vision['dataset_size'] = sum(s['num_bytes'] for s in vision['splits'])
    vision['download_size'] += sum(f['size_change'] for f in files)
    uses = splits['train']
    card.text += f'''
### Component-level license review (2026-10-09)

This review supersedes earlier license-use counts. All sources remain included.
Training: **{uses['commercial']:,} commercial, {uses['non-commercial']:,} non-commercial,
{uses['unspecified']:,} unresolved complete-row coverage**. Counts describe scoped
license evidence, not legal clearance of each image.

**FigureQA's official archive contains Microsoft Research Open Data License terms:
non-commercial research/testing only; dataset redistribution and standalone hosting
are prohibited.** Its generator's MIT license covers code. The retained FigureQA
rows are explicitly flagged `redistribution: prohibited`; their inclusion here does
not grant redistribution rights. BAPPS training patches inherit Adobe / Adobe-MIT
research image terms; its software BSD license does not clear the images.

NLVR2 annotations and VSR annotation cards specify CC BY 4.0. TallyQA and A-OKVQA
repositories specify Apache 2.0; InterGPS's data/code repository specifies MIT.
These terms do not clear third-party images. COCO per-image grants, NLVR2 photo
rights, imported TallyQA QAs and SPair PASCAL/Flickr terms still need row-level
resolution. No explicit Visual7W annotation grant was found in checked original
project/toolkit documentation. These are documented coverage gaps, not a claim
that every upstream component has no license.

Every reviewed row has `metadata.licenses.license_review` with separate annotation
and image terms, evidence URLs, review status and unresolved reasons. See
[license-review.json](vision/license-review.json), [sources.yaml](vision/sources.yaml)
and [the reproducible update script](vision/update_vision_license_metadata.py).
Images, questions, choices, targets, IDs and source splits are unchanged.
'''
    card.save(root / 'README.md')
    dump(root / 'staged.json', {'files': files, 'report': report})
    print(json.dumps(report['splits'], indent=2), flush=True)
    if args.publish:
        operations = [CommitOperationAdd(path_in_repo=f['path'], path_or_fileobj=f['local']) for f in files]
        for name in ('sources.yaml', 'release-audit.json', 'license-review.json', 'build-manifest.json', 'quality-audit.json'):
            operations.append(CommitOperationAdd(path_in_repo='vision/' + name, path_or_fileobj=str(root / name)))
        operations += [CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=str(root / 'README.md')),
                       CommitOperationAdd(path_in_repo='vision/update_vision_license_metadata.py', path_or_fileobj=__file__)]
        result = api.create_commit(args.repo, repo_type='dataset', parent_commit=args.parent,
                                   operations=operations, commit_message='Resolve scoped vision licenses and flag research/redistribution restrictions')
        dump(root / 'published.json', {'revision': result.oid, 'url': result.commit_url})
        print(result.commit_url, flush=True)


if __name__ == '__main__':
    main()
