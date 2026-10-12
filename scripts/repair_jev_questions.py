"""Repair published question surfaces without resampling or changing supervision.

PYTHONPATH=src:. python -m scripts.repair_jev_questions --revision COMMIT --push
Only the current training configs are patched; historical quality snapshots stay pinned.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from datasets import Dataset
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, hf_hub_download

from tasksource.access import load_preprocessing
from tasksource.preprocess import classification_prompt
from scripts.build_jev_dataset import example_key, question_context_sources, validate_decisions

REPO = 'tasksource/tasksource-jev-typed-decisions'
CONFIGS = {'data': 'default', 'full': 'full', 'filtered-full': 'filtered-full'}
REMOVED = {'wouldyourather', 'puzzte', 'linguisticprobing/sentence_length'}
EMPTY = {'edos', 'slurp/intent', 'pubmed-rct/role'}
DIALOGUE = {f'multidogo/{c}/intent' for c in ('airline', 'finance', 'fastfood', 'insurance', 'media', 'software')}
PAIR_FORMATS = [('text_A: ', '\ntext_B: '), ('First text:\n', '\n\nSecond text:\n'),
                ('Passage A:\n', '\n\nPassage B:\n'), ('A: ', '\nB: ')]
REPAIR = 'repairs/question-surfaces'


def pair(state):
    for prefix, delimiter in PAIR_FORMATS:
        if state.startswith(prefix) and delimiter in state:
            first, second = state[len(prefix):].split(delimiter, 1)
            return {'sentence1': first, 'sentence2': second}
    return None


def rewrite(row, questions):
    """Keep answers and source identity; move context only when explicitly declared."""
    if row['source'] in REMOVED:
        return None
    if row['source'] not in questions and row['source'] not in EMPTY:
        return row
    if row['variant'] == 'packed_derived':
        # Each item has its own context/conclusion; a common per-row template cannot replace it.
        if not row['source'].startswith('probability_words_nli/') or '\ntext_B: ' not in row['state']:
            raise ValueError(f"Unreviewed packed surface: {row['id']}")
        return row
    if row['variant'] not in {'direct', 'criteria_permutation', 'paired_text_format', 'label_verification'}:
        raise ValueError(f"Unreviewed variant: {row['id']}")
    fields = pair(row['state'])
    question = questions.get(row['source'], row['question'])
    if fields is None:
        if '{sentence2}' in question:
            raise ValueError(f"Missing context delimiter: {row['id']}")
        fields = {'sentence1': row['state']}
    state, question = classification_prompt(fields, question)
    if row['variant'] == 'paired_text_format':
        if not row['source'].startswith('probability_words_nli/'):
            raise ValueError(f"Unreviewed alternate surface: {row['id']}")
        state = f"Context:\n{fields['sentence1']}\n\nConclusion:\n{fields['sentence2']}"
        question = 'Given the context, is the conclusion valid?'
    if row['variant'] == 'label_verification':
        marker = ' Is "'
        position = row['question'].rfind(marker)
        if position < 0 or not row['question'].endswith('" the correct answer?'):
            raise ValueError(f"Missing verification proposal: {row['id']}")
        question += row['question'][position:]
    return {**row, 'state': state, 'question': question}


def scan(base, output, affected):
    groups = defaultdict(list)
    for file in base.siblings:
        if file.rfilename.endswith('.parquet') and file.rfilename.split('/')[0] in CONFIGS:
            groups[file.blob_id].append(file)
    def inspect(files):
        path = hf_hub_download(REPO, files[0].rfilename, repo_type='dataset', revision=base.sha)
        table = pq.read_table(path, columns=['source', 'variant'])
        count = pc.sum(pc.cast(pc.is_in(table['source'], value_set=pa.array(sorted(affected))), pa.int64())).as_py()
        return dict(paths=[f.rfilename for f in files], local=path, rows=len(table), affected=count)
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(inspect, groups.values()))


def build(output, revision):
    output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    base = api.dataset_info(REPO, revision=revision, files_metadata=True)
    questions = {source: load_preprocessing(id=source).question for source in question_context_sources() | DIALOGUE}
    affected = set(questions) | EMPTY | REMOVED
    files = scan(base, output, affected)
    aliases = {}
    for file in files:
        if not file['affected'] or not any(p.startswith('full/') for p in file['paths']):
            continue
        table = pq.read_table(file['local'])
        mask = pc.and_(pc.is_in(table['source'], value_set=pa.array(sorted(affected - REMOVED))),
                       pc.equal(table['variant'], 'direct'))
        for row in table.filter(mask).to_pylist():
            updated = rewrite(row, questions)
            key = example_key(row['source'], updated['state'], row['options'], row['target'], updated['question'])
            identity = (row['source'], row['example_id'])
            if identity in aliases and aliases[identity] != key:
                raise ValueError(f'Conflicting content identity: {identity}')
            aliases[identity] = key
    stats, operations, examples = [], [], defaultdict(list)
    removed_counts = defaultdict(Counter)
    for file in files:
        if not file['affected']:
            continue
        before = pq.read_table(file['local'])
        mask = pc.is_in(before['source'], value_set=pa.array(sorted(affected)))
        selected = before.filter(mask).to_pylist()
        replacements, removed, changes = [], [], Counter()
        for row in selected:
            updated = rewrite(row, questions)
            if updated is None:
                removed.append(row)
                continue
            if row['variant'] != 'packed_derived' and 'example_id' in row:
                updated['example_id'] = aliases[(row['source'], row['example_id'])]
            replacements.append(updated)
            if updated != row:
                changes[row['source']] += 1
                if len(examples[row['source']]) < 3:
                    examples[row['source']].append(dict(id=row['id'], before_state=row['state'],
                        before_question=row['question'], state=updated['state'], question=updated['question'],
                        options=row['options'], target=row['target'], variant=row['variant']))
        after = before.filter(pc.invert(pc.is_in(before['source'], value_set=pa.array(sorted(REMOVED)))))
        replacement_mask = pc.is_in(after['source'], value_set=pa.array(sorted(affected - REMOVED)))
        for column in [c for c in ('state', 'question', 'example_id') if c in after.column_names]:
            values = pa.array([r[column] for r in replacements], type=after.schema.field(column).type)
            after = after.set_column(after.schema.get_field_index(column), after.schema.field(column),
                                     pc.replace_with_mask(after[column], replacement_mask.combine_chunks(), values))
        untouched = pc.invert(pc.is_in(after['source'], value_set=pa.array(sorted(affected))))
        assert after.filter(untouched).equals(before.filter(pc.invert(mask)))
        kept_before = before.filter(pc.invert(pc.is_in(before['source'], value_set=pa.array(sorted(REMOVED)))))
        invariant = [c for c in before.column_names if c not in {'state', 'question', 'example_id'}]
        assert kept_before.select(invariant).equals(after.select(invariant))
        validate_decisions(Dataset(after.filter(replacement_mask)), file['paths'][0])
        if not changes and not removed:
            continue
        path = output / 'shards' / file['paths'][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(after, path)
        written = pq.read_table(path)
        assert written.equals(after) and written.schema.equals(before.schema, check_metadata=True)
        for name in file['paths']:
            operations.append((name, path))
            config, split = CONFIGS[name.split('/')[0]], name.split('/')[1].split('-')[0]
            removed_counts[config, split].update(row['source'] for row in removed)
        stats.append(dict(paths=file['paths'], changed=dict(changes), removed=dict(Counter(r['source'] for r in removed)),
            before_rows=len(before), after_rows=len(after), num_bytes_delta=after.nbytes-before.nbytes,
            file_bytes_delta=path.stat().st_size-Path(file['local']).stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(file['paths'][0], dict(changes), 'removed', len(removed), flush=True)
    def download(name):
        return Path(hf_hub_download(REPO, name, repo_type='dataset', revision=base.sha))
    card = DatasetCard.load(str(download('README.md')))
    for info in card.data.dataset_info:
        for split in info['splits']:
            relevant = [s for s in stats if any(CONFIGS[p.split('/')[0]] == info['config_name']
                        and p.split('/')[1].startswith(split['name']+'-') for p in s['paths'])]
            split['num_examples'] += sum(s['after_rows']-s['before_rows'] for s in relevant)
            split['num_bytes'] += sum(s['num_bytes_delta'] for s in relevant)
        relevant = [s for s in stats if any(CONFIGS[p.split('/')[0]] == info['config_name'] for p in s['paths'])]
        info['dataset_size'] += sum(s['num_bytes_delta'] for s in relevant)
        info['download_size'] += sum(s['file_bytes_delta'] for s in relevant)
    sources = yaml.safe_load(download('sources.yaml').read_text())
    from tasksource import parked
    for source, annotation in [('wouldyourather','wouldyourather'), ('puzzte','puzzte'),
                               ('linguisticprobing/sentence_length','linguisticprobing___sentence_length')]:
        sources['sources'].pop(source, None)
        sources.setdefault('excluded_views', {})[source] = parked.REASONS[annotation]
    audit = json.loads(download('release-audit.json').read_text())
    for split, summary in audit['splits'].items():
        removed = removed_counts['full', split]
        summary['rows'] -= sum(removed.values())
        for source, count in removed.items():
            summary['sources'].pop(source)
            summary['families'].pop(source)
            for field, key in [('kinds','choice'), ('variants','direct'), ('license_use','commercial'), ('formats','graded')]:
                summary[field][key] -= count
    for split, summary in audit['mix'].items():
        removed = removed_counts['default', split]
        summary['rows'] -= sum(removed.values())
        summary['sources'] -= len(removed)
    manifest = dict(parent_revision=base.sha, code_revision=__import__('subprocess').check_output(
        ['git','rev-parse','HEAD'], text=True).strip(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        configs=list(CONFIGS.values()), shards=stats,
        removed={config:{split:dict(counts) for (c,split),counts in removed_counts.items() if c==config}
                 for config in CONFIGS.values()},
        rules=['Only state, question and content example_id change on retained rows.',
               'IDs, groups, splits, options, targets, licenses and other metadata remain exact.',
               'Packed probability examples keep their valid per-item paired surface.',
               'Probability formatting variants retain context/conclusion fields; verification proposals are unchanged.',
               'No new tasks, rows, distractors, splits or samples; three parked task IDs excluded.',
               'Root dataset card body and all non-training Parquet blobs remain exact.'],
        checks=['Arrow equality of all unrelated rows and retained supervision/identity columns',
                'Typed decision validation for affected retained rows', 'Exact Parquet round-trip and schema equality'],
        preserved_parquets={s.rfilename:s.blob_id for s in base.siblings
            if s.rfilename.endswith('.parquet') and s.rfilename not in {name for name,_ in operations}})
    audit['question_surface_repair'] = dict(manifest=f'{REPAIR}/manifest.json', parent_revision=base.sha)
    card.save(output/'README.md')
    assert DatasetCard.load(str(output/'README.md')).text == DatasetCard.load(str(download('README.md'))).text
    for name, obj in [('release-audit.json',audit), (f'{REPAIR}/manifest.json',manifest),
                      (f'{REPAIR}/examples.json',dict(examples))]:
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');operations.append((name,path))
    (output/'sources.yaml').write_text(yaml.safe_dump(sources,sort_keys=False,allow_unicode=True))
    alias_path=output/'example-ids.parquet'
    pq.write_table(pa.Table.from_pylist([dict(source=s, previous_example_id=old, example_id=new)
        for (s,old),new in sorted(aliases.items()) if old!=new]),alias_path)
    operations.extend([('README.md',output/'README.md'), ('sources.yaml',output/'sources.yaml'),
        (f'{REPAIR}/example-ids.parquet',alias_path), (f'{REPAIR}/repair.py',Path(__file__))])
    (output/'upload-paths.json').write_text(json.dumps([(name,str(path)) for name,path in operations],indent=2)+'\n')
    return manifest, operations


def publish(manifest, operations):
    api=HfApi()
    result=api.create_commit(REPO, repo_type='dataset', parent_commit=manifest['parent_revision'],
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=path) for name,path in operations],
        commit_message='Clarify English question surfaces and remove parked preference task')
    blobs={s.rfilename:s.blob_id for s in api.dataset_info(REPO,revision=result.oid,files_metadata=True).siblings}
    assert all(blobs.get(name)==blob for name,blob in manifest['preserved_parquets'].items())
    print('Published',result.oid,flush=True)
    return result.oid


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--output',type=Path,default=Path('build/jev-question-repair/staged'))
    parser.add_argument('--push',action='store_true')
    args=parser.parse_args()
    manifest,operations=build(args.output,args.revision)
    if args.push:
        (args.output/'published.txt').write_text(publish(manifest,operations)+'\n')
