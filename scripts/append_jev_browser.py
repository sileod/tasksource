"""Append selected text tasks to default/full, preserving all existing data.

Stages a reviewable, parent-revision-guarded patch; publication requires --push.
Only direct decisions are added. Source identity/provenance remain in a sidecar
because the existing text release schema has no per-row metadata column.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from datasets import Dataset
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tasksource import load_task, task_provenance, list_tasks
from tasksource.jev.length import render_request
from tasksource.licenses import provenance_repos
from scripts.build_jev_dataset import (PUBLISH_EXCLUDED_PREFIXES, example_key, slug,
                                      to_training_row, validate_decisions, source_licenses)

REPO = 'tasksource/tasksource-jev-typed-decisions'


def build(output, max_rows=1000, max_rows_eval=100, tasks=None, description=None):
    tasks = tasks or ['mind2web/action', 'mind2web/dom-element', 'weblinx/action', 'weblinx/dom-element', 'websrc/yesno', 'websrc/element']
    formats = dict(zip(list_tasks().id, list_tasks().task_type))
    licenses = source_licenses(tasks)
    api = HfApi()
    before = api.dataset_info(REPO, files_metadata=True)
    def download(name):
        return Path(hf_hub_download(REPO, name, repo_type='dataset', revision=before.sha))
    card = DatasetCard.load(str(download('README.md')))
    audit = json.loads(download('release-audit.json').read_text())
    sources = yaml.safe_load(download('sources.yaml').read_text())
    if any(task in sources['sources'] or task.startswith(PUBLISH_EXCLUDED_PREFIXES) for task in tasks):
        raise ValueError('Tasks must be new, non-excluded sources')
    schema = pq.read_schema(download('full/test-00000-of-00001.parquet'))
    output.mkdir(parents=True, exist_ok=True)
    stamp = hashlib.sha256(('\n'.join(tasks) + before.sha).encode()).hexdigest()[:12]
    stats, provenance, evidence, additions, exclusions = {}, {}, {}, [], []
    rows = {split: [] for split in ('train', 'validation', 'test')}
    trajectory_sets = {split: set() for split in rows}
    page_sets = {split: set() for split in rows}
    split_groups = {split: set() for split in rows}
    for task in tasks:
        print('Stage', task, flush=True)
        provenance[task] = task_provenance(task)
        data = load_task(task, recast='jev', max_rows=max_rows, max_rows_eval=max_rows_eval)
        counts = {}
        for native_split, dataset in data.items():
            split = 'test' if native_split.startswith('test') else native_split
            if split not in rows:
                raise ValueError(f'Unrecognized native split: {task}/{native_split}')
            for index, example in enumerate(dataset):
                meta = json.loads(example['metadata'])
                meta['native_split'] = native_split
                row = to_training_row(example, index, task, split)
                # Source identity is stable across text/visual views and permutations.
                group = f"{task.split('/')[0]}:{split}:{meta.get('source_row') or meta['id']}"
                row.update(id=f'{group}:{slug(task)}', group_id=group, question_id=slug(task),
                           license=licenses[task]['license'], license_use=licenses[task]['license_use'])
                if len(render_request(row['state'], [row]).encode()) > 131_072:
                    exclusions.append(dict(id=row['id'], source=task, split=split, reason='request_exceeds_131072_bytes'))
                    continue
                row['example_id'] = example_key(task, row['state'], row['options'], row['target'])
                evidence[row['id']] = meta
                rows[split].append({key: row[key] for key in schema.names})
                if meta.get('split_group_id'):
                    split_groups[split].add((task.split('/')[0],meta['split_group_id']))
                if meta.get('trajectory_id') or meta.get('demo'):
                    trajectory_sets[split].add((task.split('/')[0], meta.get('trajectory_id') or meta['demo']))
                if meta.get('page_group_id'):
                    page_sets[split].add((task.split('/')[0], meta['page_group_id']))
            counts[split] = sum(row['source'] == task for row in rows[split])
        sources['sources'][task] = dict(**provenance[task], rows=counts,
            revisions={provenance[task]['dataset']:provenance[task]['revision']}, **licenses[task])
    for a in rows:
        for b in rows:
            if a != b:
                assert not trajectory_sets[a] & trajectory_sets[b], 'Trajectory leakage'
                assert not page_sets[a] & page_sets[b], 'Page leakage'
                assert not split_groups[a] & split_groups[b], 'Source group leakage'
    for split, examples in rows.items():
        table = pa.Table.from_pylist(examples, schema=schema)
        validate_decisions(Dataset(table), split)
        assert len({row['id'] for row in examples}) == len(examples)
        filename = f'{split}-tasksource-addition-{stamp}.parquet'
        path = output / filename
        pq.write_table(table, path)
        assert pq.read_table(path).equals(table)
        counts = dict(Counter(row['source'] for row in examples))
        stats[split] = dict(rows=len(examples), sources=counts, num_bytes=table.nbytes,
            file_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        additions.extend((f'{folder}/{filename}', path) for folder in ('data', 'full'))
        published = audit['splits'][split]
        published['rows'] += len(examples)
        published['sources'].update(counts)
        for field, increments in dict(
            formats={kind:sum(count for task,count in counts.items() if formats[task] == kind)
                     for kind in ('Classification','MultipleChoice')},
            families={family:sum(count for task,count in counts.items() if task.split('/')[0] == family)
                      for family in {task.split('/')[0] for task in counts}}, kinds=dict(Counter(row['kind'] for row in examples)),
            variants={'direct':len(examples)}, license_use=dict(Counter(row['license_use'] for row in examples))).items():
            for key, count in increments.items():
                published[field][key] = published[field].get(key, 0) + count
        audit['mix'][split]['rows'] += len(examples)
        audit['mix'][split]['sources'] += len(counts)
    for info in card.data.dataset_info:
        if info['config_name'] not in ('default', 'full'):
            continue
        for split in info['splits']:
            addition = stats[split['name']]
            split['num_examples'] += addition['rows']
            split['num_bytes'] += addition['num_bytes']
        info['dataset_size'] += sum(v['num_bytes'] for v in stats.values())
        info['download_size'] += sum(v['file_bytes'] for v in stats.values())
    sources['datasets'] = sorted(set(sources['datasets']) | {repo for p in provenance.values() for repo in provenance_repos(p)})
    manifest = dict(parent_revision=before.sha, tasks=tasks, configs=['default','full'],
        max_rows=max_rows, max_rows_eval=max_rows_eval, splits=stats, provenance=provenance,
        excluded=exclusions, max_request_bytes=131072, licenses=licenses,
        conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        row_metadata=f'additions/{stamp}/rows.json',
        preserved_parquets={s.rfilename:s.blob_id for s in before.siblings if s.rfilename.endswith('.parquet')})
    audit.setdefault('incremental_additions', []).append(dict(tasks=tasks,
        manifest=f'additions/{stamp}/manifest.json', rows={s:v['rows'] for s,v in stats.items()}))
    card.text += (description + f'\n\nSource-row metadata: additions/{stamp}/rows.json.\n') if description else ('\n\n## HTML/browser text additions\n\n'
        'WebLINX action/DOM-element and WebSRC yes/no/element tasks retain their source partitions. '
        'WebLINX named test subsets are combined into the hosted test split, with native split names in row provenance. '
        '`mind2web/action` and `mind2web/dom-element` use only the original public training trajectories. '
        'Tasksource-derived train/validation/test holdouts keep whole trajectories and identical DOM snapshots together '
        '(80/10/10 deterministic group hashing). These are not the official benchmark test partitions. '
        'The same partition is used for both annotations. Original annotation/action IDs and page hashes '
        f'are recorded in [row provenance](additions/{stamp}/rows.json). '
        'Only past actions enter context; four-way choices use native DOM candidates. '
        'Direct decisions are appended to default/full; existing shards, vision and filtered-full are unchanged. '
        'Alignment with the Decision Index private evaluation rows has not been independently verified.\n')
    card.save(output/'README.md')
    (output/'sources.yaml').write_text(yaml.safe_dump(sources,sort_keys=False,allow_unicode=True))
    for filename, obj in [('release-audit.json',audit), ('manifest.json',manifest), ('rows.json',evidence)]:
        (output/filename).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    additions.extend((name,output/name) for name in ('README.md','sources.yaml','release-audit.json'))
    additions.extend((f'additions/{stamp}/{name}',output/name) for name in ('manifest.json','rows.json'))
    return manifest, additions


def push(manifest, additions):
    api = HfApi()
    commit = api.create_commit(REPO,repo_type='dataset',parent_commit=manifest['parent_revision'],
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=path) for name,path in additions],
        commit_message='Append validated Tasksource decisions: ' + ', '.join(manifest['tasks']))
    after = api.dataset_info(REPO,revision=commit.oid,files_metadata=True)
    blobs = {s.rfilename:s.blob_id for s in after.siblings}
    assert all(blobs.get(name) == blob for name,blob in manifest['preserved_parquets'].items())
    print('Published',commit.oid,'; existing Parquet blobs preserved',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/jev-browser-addition'))
    parser.add_argument('--max-rows', type=int, default=1000)
    parser.add_argument('--max-rows-eval', type=int, default=100)
    parser.add_argument('--tasks', nargs='+', help='New task IDs (defaults to browser tasks)')
    parser.add_argument('--description', help='Dataset-card note for this addition')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    manifest, additions = build(args.output,args.max_rows,args.max_rows_eval,args.tasks,args.description)
    print(json.dumps(manifest['splits'],indent=2),flush=True)
    if args.push:
        push(manifest,additions)
