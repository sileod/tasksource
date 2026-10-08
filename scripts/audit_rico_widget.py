"""Audit pinned RICO screen disjointness without downloading screenshot columns."""
import argparse
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from tasksource import list_tasks
from tasksource.vision_tasks import grid_labels


def audit(output):
    task = list_tasks(vision=True).set_index('id').loc['rico-widget/grid7', 'mapping']
    repo, revision = task.dataset_name, task.load_dataset_kwargs['revision']
    screens, counts = {}, {}
    eligibility = {}
    paths = [path for path in HfApi().list_repo_files(repo, repo_type='dataset', revision=revision)
             if path.startswith('data/') and path.endswith('.parquet')]
    def shard(path):
        split = Path(path).name.split('-')[0]
        fs = HfFileSystem()
        ids, rows, captions, dropped, labels = set(), 0, 0, Counter(), Counter()
        with fs.open(f'datasets/{repo}@{revision}/{path}', 'rb', block_size=65536) as handle:
            for batch in pq.ParquetFile(handle).iter_batches(columns=['screenId', 'bbox', 'captions']):
                for row in batch.to_pylist():
                    ids.add(row['screenId'])
                    rows += 1
                    captions += len(row['captions'])
                    box = row['bbox']
                    if len(box) != 4 or not all(math.isfinite(v) for v in box) or not (
                            0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                        dropped['invalid_box'] += len(row['captions'])
                        continue
                    r, c = grid_labels(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))[0]
                    for caption in row['captions']:
                        if caption.strip():
                            labels[f'r{r}c{c}'] += 1
                        else:
                            dropped['empty_caption'] += 1
        print(path, flush=True)
        return split, ids, rows, captions, dropped, labels
    with ThreadPoolExecutor(max_workers=6) as pool:
        for split, ids, rows, captions, dropped, labels in pool.map(shard, paths):
            screens.setdefault(split, set()).update(ids)
            counts[split] = counts.get(split, 0) + rows
            stats = eligibility.setdefault(split, {'source_captions': 0, 'retained_captions': 0,
                'dropped_by_reason': Counter(), 'label_distribution': Counter()})
            stats['source_captions'] += captions
            stats['retained_captions'] += sum(labels.values())
            stats['dropped_by_reason'].update(dropped)
            stats['label_distribution'].update(labels)
    overlap = {f'{a}/{b}': len(screens[a] & screens[b])
               for a in screens for b in screens if a < b}
    report = {'dataset': repo, 'revision': revision,
        'scope': 'Full native screenId, box, and caption audit; screenshot integrity is checked separately on smoke fixtures.',
        'rows': counts, 'screens': {key: len(value) for key, value in screens.items()},
        'screen_overlap': overlap, 'caption_eligibility': eligibility}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    if any(overlap.values()):
        raise ValueError(f'RICO screens cross native split boundaries: {overlap}')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output)
