"""Recover reusable, attributed selection evidence by comparing frozen configs.

No LLM requests; no inferred exclusion is promoted to a verified labeling error.
The report identifies two reproducible policy rules. Remaining missing rows are
published as unadjudicated candidates because per-row GLM verdicts are unavailable.
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
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, snapshot_download

from tasksource.quality import decision_key

REPO = 'tasksource/tasksource-jev-typed-decisions'
RELEASE = '84008c8a8f4640a2b8a3b222cb62f23ae5fa6e73'
META = ['id','source','group_id','question_id','variant','split','license_use']


def policy_category(row, rule):
    root = re.sub('[^a-z0-9]','',row['source'].split('/')[0].casefold())
    blocked = {re.sub('[^a-z0-9]','',name.casefold()) for name in rule['blocked_roots']}
    if root in blocked or any(row['source'].casefold().startswith(prefix)
                              for prefix in rule['casefolded_source_prefixes']):
        return 'benchmark_policy'
    if row['license_use'] != 'commercial':
        return 'license_policy'
    return 'unattributed_exclusion'


def extract(source_files, selected_files, report, output, validate_hashes=True):
    output.mkdir(parents=True,exist_ok=True)
    retained = set()
    for path in selected_files:
        ids = pq.read_table(path,columns=['id'])['id'].to_pylist()
        if len(set(ids)) != len(ids) or retained.intersection(ids):
            raise ValueError('Duplicate retained IDs')
        retained.update(ids)
    if len(retained) != report['retained_rows']:
        raise ValueError('Retained count disagrees with report')
    values = pa.array(sorted(retained))
    found, counts, source_counts = set(), Counter(), Counter()
    policies = {}
    candidates, candidate_writer = 0, None
    schema = pa.schema([(name,pa.string()) for name in META+['category','decision_key']])
    with pq.ParquetWriter(output/'excluded.parquet',schema) as writer:
        for path in source_files:
            if validate_hashes:
                expected = report['source_file_sha256'][f'full/{path.name}']
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Source checksum mismatch: {path}')
            table = pq.read_table(path)
            selected = pc.is_in(table['id'],value_set=values)
            found.update(table.filter(selected)['id'].to_pylist())
            absent = table.filter(pc.invert(selected))
            metadata = absent.select(META).to_pylist()
            for row in metadata:
                identity = (row['source'],row['license_use'])
                if identity not in policies:
                    policies[identity] = policy_category(row,report['benchmark_source_holdout_rule'])
                row['category'] = policies[identity]
                row['decision_key'] = None
                counts[row['category']] += 1
                source_counts[row['source']] += 1
            mask = pa.array([row['category']=='unattributed_exclusion' for row in metadata])
            candidate_rows = absent.filter(mask).to_pylist()
            keys = {row['id']:decision_key(row) for row in candidate_rows}
            for row in metadata:
                row['decision_key'] = keys.get(row['id'])
            writer.write_table(pa.Table.from_pylist(metadata,schema=schema))
            for row in candidate_rows:
                row['category'] = 'unattributed_exclusion'
                row['decision_key'] = keys[row['id']]
            if candidate_rows:
                candidate_schema = pa.schema([*table.schema,pa.field('category',pa.string()),
                                              pa.field('decision_key',pa.string())])
                candidate_table = pa.Table.from_pylist(candidate_rows,schema=candidate_schema)
                if candidate_writer is None:
                    candidate_writer = pq.ParquetWriter(output/'candidates.parquet',candidate_table.schema)
                candidate_writer.write_table(candidate_table)
                candidates += len(candidate_rows)
            print(path.name,'excluded',len(metadata),'candidates',len(candidate_rows),flush=True)
    if candidate_writer: candidate_writer.close()
    if found != retained:
        raise ValueError('Retained IDs absent from frozen source')
    if sum(counts.values()) != report['source_rows']-report['retained_rows']:
        raise ValueError('Source/excluded count mismatch')
    for category,reason in [('license_policy','license_not_commercial'),
                            ('benchmark_policy','benchmark_source_holdout')]:
        if counts[category] != report['prefilter_exclusions'][reason]:
            raise ValueError(f'Reconstructed {category} count differs from published policy')
    manifest = dict(schema_version=1,repository=REPO,release_revision=RELEASE,
        source_revision=report['source_revision'],source_config='full',source_split='train',
        selected_config='filtered-full',source_rows=report['source_rows'],retained_rows=len(retained),
        excluded_rows=sum(counts.values()),categories=dict(counts),candidate_rows=candidates,
        llm_rejected_rows_reported=report['llm_rejected_rows'],
        unresolved_api_errors_reported=report['unresolved_api_error_rows_excluded'],
        decision_key='tasksource-decision-v1; SHA256; exact task/input/question/gold; neutral choice order canonicalized',
        identified_error_ids_available=False,approved_removals=[],
        attribution='Set difference proves exclusion membership. License/benchmark policy predicates are reproduced from the report. Remaining exclusions cannot be assigned individual GLM/overlap/budget/conflict reasons.',
        matching='Decision-level content lookup only; no group-wide, row-index-only or fuzzy transfer. Carry the canonical key before changing recast rendering. Corrected gold/context no longer matches.',
        benchmark_rule=report['benchmark_source_holdout_rule'],
        policy_precedence=['benchmark_policy','license_policy'],
        per_source_excluded=dict(sorted(source_counts.items())))
    manifest['files']={path.name:dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                       for path in output.glob('*.parquet')}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    return manifest


def build(output):
    report = json.loads(Path(hf_hub_download(REPO,'filtered-full/filtering-report.json',repo_type='dataset',revision=RELEASE)).read_text())
    source_root = Path(snapshot_download(REPO,repo_type='dataset',revision=report['source_revision'],
        allow_patterns=list(report['source_file_sha256']),max_workers=4))
    selected_root = Path(snapshot_download(REPO,repo_type='dataset',revision=RELEASE,
        allow_patterns=[shard['path'] for shard in report['shards']],max_workers=4))
    source_files = [source_root/name for name in sorted(report['source_file_sha256'])]
    selected_files = [selected_root/shard['path'] for shard in report['shards']]
    manifest = extract(source_files,selected_files,report,output)
    print(json.dumps({k:manifest[k] for k in ['excluded_rows','categories','candidate_rows']},indent=2),flush=True)


def publish(output):
    api = HfApi()
    before = api.dataset_info(REPO,files_metadata=True)
    prefix = 'quality/filtered-full-v1/'
    if any(file.rfilename.startswith(prefix) for file in before.siblings):
        raise ValueError('Resource already published; use a new version for changed evidence')
    manifest = json.loads((output/'manifest.json').read_text())
    for name, info in manifest['files'].items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Output checksum mismatch: {name}')
    root = Path(__file__).resolve().parent.parent
    readme = (root/'dataset_cards/jev-exclusion-resource.md').read_text()
    (output/'README.md').write_text(readme)
    card = Path(hf_hub_download(REPO,'README.md',repo_type='dataset',revision=before.sha)).read_text()
    (output/'dataset-card.md').write_text(card+'\n\n## Reusable selection evidence\n\n'
        'The [exclusion evidence resource](quality/filtered-full-v1/README.md) records '
        'historical filtered-full selection membership separately from licensing and '
        'benchmark policies. Remaining candidates are unattributed exclusions, not '
        'verified errors. Canonical decision keys support evidence reuse across recasts.\n')
    operations = [CommitOperationAdd(path_in_repo=prefix+name,path_or_fileobj=output/name)
                  for name in ['excluded.parquet','candidates.parquet','manifest.json','README.md']]
    operations.append(CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=output/'dataset-card.md'))
    commit = api.create_commit(REPO,repo_type='dataset',parent_commit=before.sha,
        operations=operations,commit_message='Publish reusable historical exclusion evidence')
    after = api.dataset_info(REPO,revision=commit.oid,files_metadata=True)
    blobs = {file.rfilename:file.blob_id for file in after.siblings}
    for file in before.siblings:
        if file.rfilename != 'README.md' and blobs.get(file.rfilename) != file.blob_id:
            raise RuntimeError(f'Existing file changed: {file.rfilename}')
    print('Published sidecar resource:',commit.oid,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('build/jev-exclusion-resource'))
    parser.add_argument('--publish',action='store_true',help='publish sidecars after successful extraction')
    args = parser.parse_args()
    build(args.output)
    if args.publish: publish(args.output)
