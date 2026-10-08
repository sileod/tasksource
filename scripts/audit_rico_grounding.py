"""Coarse pinned RICO candidate and Jev/SoM audit: one shard per native split.

This is a mapping smoke test, not a representative source-quality estimate.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from huggingface_hub import HfApi, HfFileSystem

from tasksource import load_task, list_tasks
from tasksource.grounding import grounding_row
from tasksource.vision_tasks import rico_candidates


def audit(output, build_dir, limit=64):
    task = list_tasks(vision=True).set_index('id').loc['rico-widget/element', 'mapping']
    repo, revision = task.dataset_name, task.load_dataset_kwargs['revision']
    paths = HfApi().list_repo_files(repo, repo_type='dataset', revision=revision)
    report = {'source': repo, 'revision': revision, 'splits': {},
              'scope': f'First {limit} rows of one pinned parquet shard per native split; not full-source rates'}
    fixtures = {}
    features = Features({'screenId': Value('int64'), 'image': Image(decode=False),
        'bbox': Sequence(Value('float64')), 'captions': Sequence(Value('string')),
        'semantic_annotations': Value('string')})
    for split in ('train', 'val', 'test'):
        path = next(p for p in paths if p.startswith(f'data/{split}-') and p.endswith('.parquet'))
        with HfFileSystem().open(f'datasets/{repo}@{revision}/{path}', 'rb', block_size=65536) as handle:
            rows = next(pq.ParquetFile(handle).iter_batches(batch_size=limit, columns=list(features))).to_pylist()
        counts, candidate_counts, usable = Counter(), Counter(), []
        for index, row in enumerate(rows):
            try:
                candidates = rico_candidates(row['semantic_annotations'])
                candidate_counts[len(candidates)] += 1
                grounding_row([row['image']], row['captions'][0], row['bbox'],
                              candidates=candidates, metadata={'source_row': str(index)})
            except (ValueError, TypeError, KeyError) as error:
                counts[str(error)] += 1
            else:
                counts['retained'] += 1
                usable.append(row)
        fixtures[split] = Dataset.from_list(usable[:3], features=features)
        report['splits'][split] = {'examined': len(rows), 'counts': dict(counts),
                                  'candidate_count_distribution': dict(candidate_counts)}
        print(split, report['splits'][split], flush=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    source = DatasetDict(fixtures)
    source.save_to_disk(str(build_dir / 'fixture'))
    # Run the public API against the pinned native fixture, without rescanning the full source.
    with patch('tasksource.access.load_dataset', return_value=source):
        data = load_task('rico-widget/element', vision=True, recast='jev',
                        grounding={'probabilities': {'som-only': 1}}, seed=42)
    data.save_to_disk(str(build_dir / 'dataset'))
    report['recast_splits'] = {split: len(rows) for split, rows in data.items()}
    report['samples'] = []
    for index, row in enumerate(data['train']):
        row['images'][0].save(build_dir / f'sample-{index}.png')
        report['samples'].append({key: row[key] for key in ('state', 'criteria', 'label', 'answer', 'metadata')})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=64)
    parser.add_argument('--output', type=Path, default=Path('docs/jev/rico-grounding-audit.json'))
    parser.add_argument('--build-dir', type=Path, default=Path('build/rico-grounding-smoke'))
    args = parser.parse_args()
    audit(args.output, args.build_dir, args.rows)
