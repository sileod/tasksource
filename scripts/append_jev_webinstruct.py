"""Append pinned, filtered WebInstruct shards without rebuilding existing Jev data.

Build a reviewable patch by default; --push commits only that patch, guarded by
the destination revision. Existing Parquet blobs must remain byte-identical.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from datasets import Dataset
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, hf_hub_download

from scripts.build_jev_dataset import example_key, slug, validate_decisions
from tasksource.jev.options import permute_choices
from tasksource.jev.recast import JEV_MULTIPLE_CHOICE_INSTRUCTIONS

REPO = 'tasksource/tasksource-jev-typed-decisions'
WEB = 'tasksource/webinstruct'
WEB_REVISION = '05e8b528d68c628c1af682d47bdb53f85535876a'


def convert(row, config, split):
    source = f'webinstruct/{config}'
    identifier = f'{slug(source)}:{split}:{row["id"]}'
    if config == 'mc':
        options, gold = permute_choices(row['options'], row['gold'], identifier)
        question = JEV_MULTIPLE_CHOICE_INSTRUCTIONS
    else:
        options, gold = ['no / false', 'yes / true'], row['label']
        question = 'Is the answer to the question or statement yes/true or no/false?'
    if type(gold) is not int or not 0 <= gold < len(options):
        raise ValueError(f'Invalid gold: {identifier}')
    target = [float(index == gold) for index in range(len(options))]
    result = dict(state=row['prompt'], kind='choice', id=identifier, options=options,
                  target=target, question=question, source=source, variant='direct',
                  split=split, group_id=identifier, question_id='decision',
                  license='apache-2.0', license_use='commercial')
    result['example_id'] = example_key(source, result['state'], options, target)
    return result


def build(output):
    api = HfApi()
    base = api.dataset_info(REPO, files_metadata=True)
    revision = base.sha
    def download(name, repo=REPO, rev=revision):
        return Path(hf_hub_download(repo, name, repo_type='dataset', revision=rev))
    card = DatasetCard.load(str(download('README.md')))
    audit = json.loads(download('release-audit.json').read_text())
    sources = yaml.safe_load(download('sources.yaml').read_text())
    if any(f'webinstruct/{config}' in sources['sources'] for config in ['mc', 'binary']):
        raise ValueError('WebInstruct already present; refusing duplicate append')
    schema = pq.read_schema(download('full/test-00000-of-00001.parquet'))
    output.mkdir(parents=True, exist_ok=True)
    additions, stats, ids, excluded = [], {}, set(), []
    source_counts = {f'webinstruct/{config}': {} for config in ['mc', 'binary']}
    for split in ['train', 'test']:
        rows = []
        for config in ['mc', 'binary']:
            table = pq.read_table(download(f'{config}/{split}-00000-of-00001.parquet', WEB, WEB_REVISION))
            converted = [convert(row, config, split) for row in table.to_pylist()]
            valid = []
            for row in converted:
                texts = [option.strip() for option in row['options']]
                if len(texts) < 2 or not all(texts) or len(set(texts)) != len(texts):
                    excluded.append(dict(id=row['id'],source=row['source'],split=split,
                                         reason='Empty or duplicate options violate Jev choice contract'))
                else:
                    valid.append(row)
            converted = valid
            for row in converted:
                if row['id'] in ids:
                    raise ValueError(f'Duplicate decision: {row["id"]}')
                ids.add(row['id'])
            source_counts[f'webinstruct/{config}'][split] = len(converted)
            rows.extend(converted)
        table = pa.Table.from_pylist(rows, schema=schema)
        validate_decisions(Dataset(table), split)
        assert table.schema.equals(schema, check_metadata=True)
        name = f'{split}-webinstruct-{WEB_REVISION[:12]}.parquet'
        path = output / name
        pq.write_table(table, path)
        assert pq.read_schema(path).equals(schema, check_metadata=True)
        stats[split] = dict(rows=len(rows),num_bytes=table.nbytes,file_bytes=path.stat().st_size,
                            sources=dict(Counter(row['source'] for row in rows)),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for folder in ['data', 'full']:
            additions.append((f'{folder}/{name}', path))
        published = audit['splits'][split]
        published['rows'] += len(rows)
        published['sources'].update(stats[split]['sources'])
        for name, increments in [('formats', {'MultipleChoice':source_counts['webinstruct/mc'][split],
                                              'Classification':source_counts['webinstruct/binary'][split]}),
                                 ('families', {'webinstruct':len(rows)}),
                                 ('kinds', {'choice':len(rows)}), ('variants', {'direct':len(rows)}),
                                 ('license_use', {'commercial':len(rows)})]:
            for key, count in increments.items():
                published[name][key] = published[name].get(key, 0) + count
        audit['mix'][split]['rows'] += len(rows)
        audit['mix'][split]['sources'] += 2
    for info in card.data.dataset_info:
        for split in info['splits']:
            if split['name'] in stats:
                addition = stats[split['name']]
                split['num_examples'] += addition['rows']
                split['num_bytes'] += addition['num_bytes']
        info['dataset_size'] += sum(value['num_bytes'] for value in stats.values())
        info['download_size'] += sum(value['file_bytes'] for value in stats.values())
    for source, counts in source_counts.items():
        sources['sources'][source] = dict(rows=counts, dataset=WEB,
            revisions={WEB:WEB_REVISION}, license='apache-2.0',license_use='commercial',
            card_licenses={WEB:['apache-2.0']}, config=source.split('/')[1])
    sources['datasets'] = sorted(set(sources['datasets']) | {WEB})
    manifest = dict(destination=REPO,parent_revision=revision,source=WEB,source_revision=WEB_REVISION,
        converter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        configs=['default','full'],splits=stats,source_counts=source_counts,excluded=excluded,
        transformations=['Filtered mc/binary only; no repaired or unfiltered examples.',
            'Prompt becomes state; source train/test splits preserved; no new validation split.',
            'MC retains every option; standard deterministic Jev permutation/remapped one-hot gold.',
            'Position-dependent options keep order; trailing all/none-of-above normalized by canonical Jev helper.',
            'Empty or duplicate option text is excluded under the existing Jev choice contract; IDs recorded here.',
            'Binary uses explicit no/false and yes/true options and one-hot target.',
            'Direct rows only; stable source-row group IDs and canonical content example IDs.',
            'Identical additions to default/full; existing mixture not resampled.'],
        preserved_parquets={s.rfilename:s.blob_id for s in base.siblings if s.rfilename.endswith('.parquet')})
    audit.setdefault('incremental_additions',[]).append(dict(source=WEB,revision=WEB_REVISION,
        manifest='webinstruct-addition.json',rows={split:stat['rows'] for split,stat in stats.items()}))
    card.text += ('\n\n## Incremental filtered WebInstruct addition\n\n'
        'Filtered `tasksource/webinstruct` configs `mc` and `binary`, pinned at '
        f'`{WEB_REVISION}`, are appended as standalone shards to both `default` and `full`. '
        'The original train/test split assignment is preserved; validation is unchanged. '
        'All choices of included rows are retained, with canonical Jev option permutation and one-hot targets; '
        f'{len(excluded)} rows with invalid option text are omitted under the existing choice contract. '
        'binary questions use explicit no/false and yes/true choices. No repairs or derived '
        'variants are included. Existing shards and mixture sampling are unchanged. '
        'See `webinstruct-addition.json` for exact counts, transformations and provenance.\n')
    card.save(output/'README.md')
    for name, obj in [('release-audit.json',audit), ('webinstruct-addition.json',manifest)]:
        (output/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    (output/'sources.yaml').write_text(yaml.safe_dump(sources,sort_keys=False,allow_unicode=True))
    additions.extend((name,output/name) for name in ['README.md','release-audit.json','sources.yaml','webinstruct-addition.json'])
    return manifest, additions


def push(manifest, additions):
    api = HfApi()
    commit = api.create_commit(REPO,repo_type='dataset',parent_commit=manifest['parent_revision'],
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=path) for name,path in additions],
        commit_message='Append filtered WebInstruct as canonical Jev decisions')
    after = api.dataset_info(REPO,revision=commit.oid,files_metadata=True)
    blobs = {s.rfilename:s.blob_id for s in after.siblings}
    for name, blob in manifest['preserved_parquets'].items():
        if blobs.get(name) != blob:
            raise RuntimeError(f'Existing shard changed: {name}')
    print('Published',commit.oid,'; all existing Parquet blobs preserved',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('build/jev-webinstruct-addition'))
    parser.add_argument('--push',action='store_true')
    args = parser.parse_args()
    manifest, additions = build(args.output)
    print(json.dumps(dict(parent_revision=manifest['parent_revision'],splits=manifest['splits']),indent=2),flush=True)
    if args.push:
        push(manifest, additions)
