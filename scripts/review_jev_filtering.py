"""Prepare per-task screening statistics and bounded review packs without model calls.

    PYTHONPATH=.:src python scripts/review_jev_filtering.py --tasks discosense cloth ReSQ

A flag is a review candidate, never an exclusion. Samples are stratified by verdict,
not prevalence estimates. Missing source text can be restored from local build shards.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def prepare(verdicts, output, tasks, shards, per_verdict=1):
    counts = defaultdict(Counter)
    samples = defaultdict(list)
    selected = set(tasks)
    for line in verdicts.open():
        record = json.loads(line)
        source, verdict = record['source'], record['verdict']
        counts[source][verdict] += 1
        if source not in selected:
            continue
        score = hashlib.sha256(f"jev-task-review/{record['example_id']}".encode()).hexdigest()
        group = samples[source, verdict]
        group.append((score, record))
        group.sort(key=lambda pair: pair[0])
        del group[per_verdict:]
    records = [record for group in samples.values() for _, record in group]
    missing = {r['id']: r for r in records if not r.get('row')}
    stems = {key.split(':', 1)[0] for key in missing}
    paths = {path for stem in stems for path in shards.glob(f'*-{stem}.parquet')}
    # Some share-valued tasks derive IDs from the original source, while shards
    # are named after the recast task. Recover only exact IDs within that source.
    for record in missing.values():
        slug = re.sub(r'[^A-Za-z0-9]+', '-', record['source']).strip('-')
        paths.update(shards.glob(f'*-{slug}-*.parquet'))
    for path in sorted(paths):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                record = missing.get(row['id'])
                if row['variant'] == 'direct' and record and row['source'] == record['source']:
                    missing.pop(row['id'])
                    record['row'] = {**row, 'example_id': record['example_id']}
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'per-task.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['source', 'screened', 'ok', 'wrong', 'malformed',
                                                   'ambiguous', 'flagged', 'flag_rate', 'review_status'])
        writer.writeheader()
        for source, count in sorted(counts.items()):
            total = sum(count.values())
            flagged = total - count['ok']
            writer.writerow(dict(source=source, screened=total, **{v: count[v] for v in
                                 ['ok', 'wrong', 'malformed', 'ambiguous']}, flagged=flagged,
                                 flag_rate=round(flagged / total, 6), review_status='pending'))
    records.sort(key=lambda r: (r['source'], r['verdict'], r['example_id']))
    (output / 'examples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    (output / 'sampling.json').write_text(json.dumps(dict(tasks=tasks, per_verdict=per_verdict,
        screened=sum(sum(c.values()) for c in counts.values()), task_count=len(counts),
        sampled=len(records), missing_source_text=sorted(missing),
        warning='Stratified review sample; model flags are not confirmed errors.'), indent=2) + '\n')
    print(f'{len(counts)} task summaries; {len(records)} review examples; {len(missing)} missing source texts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verdicts', type=Path, default=Path('build/jev-error-pass/verdicts.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('build/jev-task-filter-review'))
    parser.add_argument('--shards', type=Path, default=Path('build/tasksource-jev-typed-decisions-v11/data'))
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--per-verdict', type=int, default=1)
    args = parser.parse_args()
    if args.per_verdict < 1:
        parser.error('--per-verdict must be positive')
    prepare(args.verdicts, args.output, args.tasks, args.shards, args.per_verdict)
