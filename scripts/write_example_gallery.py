"""Render real published Jev examples, indexed by the complete English catalog.

PYTHONPATH=src python scripts/write_example_gallery.py [--revision COMMIT] [--load-missing]
Writes docs/examples.md. --load-missing previews tasks absent from the release by loading their source.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi, hf_hub_download
from tasksource import list_tasks

REPO = 'tasksource/tasksource-jev-typed-decisions'
ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['source', 'id', 'state', 'question', 'kind', 'options', 'target', 'split']


def collect(revision, count):
    info = HfApi().dataset_info(REPO, revision=revision)
    examples = {}
    files = sorted(s.rfilename for s in info.siblings
                   if s.rfilename.startswith('full/train') and s.rfilename.endswith('.parquet'))
    for name in files:
        path = hf_hub_download(REPO, name, repo_type='dataset', revision=info.sha)
        for batch in pq.ParquetFile(path).iter_batches(columns=FIELDS):
            # Avoid materializing large text columns once a batch's tasks are filled.
            sources = batch.column('source').to_pylist()
            for index, source in enumerate(sources):
                selected = examples.setdefault(source, [])
                if len(selected) < count:
                    row = batch.slice(index, 1).to_pylist()[0]
                    if all(row['id'] != previous['id'] for previous in selected):
                        selected.append(row)
        print(name, len(examples), 'sources', flush=True)
    return info.sha, examples


def source_examples(task, count, current_recast=False):
    """Canonical fallback for tasks absent from the exported snapshot."""
    from datasets import ClassLabel
    from tasksource import load_task, task_provenance
    data = load_task(task, max_rows=count, max_rows_eval=count)
    if 'train' not in data or not len(data['train']):
        raise ValueError('No labeled training rows')
    split = data['train']
    def decode(value, feature):
        if isinstance(feature, ClassLabel):
            return feature.int2str(value) if value >= 0 else value
        if hasattr(feature, 'feature') and isinstance(value, list):
            return [decode(v, feature.feature) for v in value]
        return value
    question = getattr(data, 'question', None)
    label_feature = split.features.get('labels')
    if hasattr(label_feature, 'feature'):
        label_feature = label_feature.feature
    criteria = list(label_feature.names) if isinstance(label_feature, ClassLabel) else []
    if not question and 'tokens' not in split.features:
        from tasksource.jev.questions import default_question
        question = default_question(criteria, 'sentence2' in split.features)
    if current_recast and (isinstance(split.features.get('labels'), ClassLabel) or 'inputs' in split.features):
        from tasksource.recast import recast_jev
        from scripts.build_jev_dataset import to_training_row
        recast = recast_jev(data, task=task, question=question)['train']
        return [dict(to_training_row(row, i, task, 'train'),
                     provenance=task_provenance(task), origin='current local Jev recast (not yet published)')
                for i, row in enumerate(recast.select(range(min(count, len(recast)))))]
    return [dict(source=task, id=f'{task}:canonical:sample:{i}', split='train',
                 canonical={k: decode(v, split.features[k]) for k, v in row.items() if k != 'metadata'},
                 question=question or '', criteria=criteria, provenance=task_provenance(task), row_metadata=row.get('metadata'), origin='canonical preprocessing (not a published Jev row)')
            for i, row in enumerate(split.select(range(min(count, len(split)))))]


def load_missing(tasks, examples, count, timeout, current_recast=False):
    """Bound each source load in its own process; retain errors rather than guessing."""
    def load(task):
        try:
            result = subprocess.run([sys.executable, '-m', 'scripts.write_example_gallery', '--source-task', task,
                                     '--examples', str(count)] + (['--current-recast'] if current_recast else []), capture_output=True, text=True, timeout=timeout, cwd=ROOT)
            if result.returncode:
                raise ValueError(result.stderr.strip().splitlines()[-1])
            return task, json.loads(result.stdout.split('GALLERY_RESULT=')[-1]), None
        except (subprocess.TimeoutExpired, ValueError) as error:
            return task, [], str(error) if not isinstance(error, subprocess.TimeoutExpired) else f'Source load exceeded {timeout}s'
    errors = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for task, rows, error in pool.map(load, [t for t in tasks if not examples.get(t)]):
            examples[task] = rows
            if error:
                errors[task] = error
            print(task, error or f'{len(rows)} examples', flush=True)
    return errors


def format_example(row):
    """Keep criteria order and the complete gold distribution, including soft labels."""
    if 'canonical' in row:
        from tasksource.preprocess import classification_prompt
        values = row['canonical']
        state, question = classification_prompt(values, row['question']) if 'sentence1' in values else (None, row['question'])
        fields = ([('State', state)] if state is not None else []) + [('Question', question)]
        if row.get('criteria'):
            fields.append(('Criteria (label vocabulary)', '\n'.join(row['criteria'])))
        if 'tokens' in values and isinstance(values.get('labels'), list) and len(values['tokens']) == len(values['labels']):
            return fields + [('Tokens / gold tags', '\n'.join(f'{token} → {label}' for token, label in zip(values['tokens'], values['labels'])))]
        return fields + [('Gold labels' if key == 'labels' else key,
                          value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))
                         for key, value in values.items() if key not in ('sentence1', 'sentence2')]
    if row['kind'] == 'noul' and not row['options'] and len(row['target']) == 1:
        return [('State', row['state']), ('Question', row['question']),
                ('Decision', 'Boolean (noul)'), ('Gold target', f"Probability of true: {row['target'][0]:g}")]
    if len(row['options']) != len(row['target']):
        raise ValueError(f"Misaligned target: {row['id']}")
    options = '\n'.join(f'{i + 1}. {option}' for i, option in enumerate(row['options']))
    gold = '\n'.join(f'{i + 1}. {option} — {weight:g}'
                     for i, (option, weight) in enumerate(zip(row['options'], row['target'])) if weight)
    return [('State', row['state']), ('Question', row['question']),
            (f"Criteria ({row['kind']})", options), ('Gold target (weight)', gold)]


CROP = 1200  # longer texts keep their beginning and end


def crop(value):
    if not isinstance(value, str) or len(value) <= CROP:
        return value
    return f"{value[:CROP * 2 // 3]}\n[... {len(value) - CROP:,} characters omitted ...]\n{value[-CROP // 3:]}"


def entry(task, row):
    """One example as plain data: every field visible, long texts middle-cropped."""
    if 'canonical' in row:
        fields = {name.lower(): crop(value) for name, value in format_example(row)}
        return dict(task=task['id'], note='source-loaded preview; not in the published Jev config', **fields)
    out = dict(task=task['id'], state=crop(row['state']), question=row['question'])
    if row['kind'] == 'noul' and not row['options']:
        out['probability_true'] = row['target'][0]
        return out
    out['options'] = list(row['options'])
    gold = {option: weight for option, weight in zip(row['options'], row['target']) if weight}
    out['answer'] = next(iter(gold)) if list(gold.values()) == [1.0] else gold
    if row['kind'] != 'choice':
        out['kind'] = row['kind']
    return out


class Literal(yaml.SafeDumper):
    pass


Literal.add_representer(str, lambda dumper, value: dumper.represent_scalar(
    'tag:yaml.org,2002:str', value, style='|' if '\n' in value else None))


def write(output, revision, examples, errors=None):
    """A single scrollable file: one example per task, each a YAML block (human-readable, copyable, parseable)."""
    lines = ['# Example catalog', '',
             f'One real training example per task from [{REPO}](https://huggingface.co/datasets/{REPO}) '
             f'(revision `{revision}`), in catalog order. Each block is YAML (`yaml.safe_load`); texts over '
             f'{CROP:,} characters are cropped in the middle. Regenerate with `python scripts/write_example_gallery.py`.', '']
    missing = []
    for r in list_tasks().itertuples():
        rows = [row for row in examples.get(r.id, []) if 'canonical' in row or row.get('kind') in ('choice', 'score', 'noul')]
        if not rows:
            missing.append(r.id)
            continue
        task = dict(id=r.id)
        block = yaml.dump(entry(task, rows[0]), Dumper=Literal, allow_unicode=True, sort_keys=False, width=10_000)
        lines += [f'## {r.id}', '', '````yaml', block.rstrip(), '````', '']  # four backticks: texts may contain fences
    if missing:
        lines += ['## Without examples', '', ', '.join(f'`{task}`' for task in missing), '']
    output.write_text('\n'.join(lines))
    print(f'{output}: {len(list_tasks()) - len(missing)} tasks, {len(missing)} without examples')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-task', help=argparse.SUPPRESS)
    parser.add_argument('--current-recast', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--load-missing', action='store_true', help='Try canonical source loads for missing exports')
    parser.add_argument('--timeout', type=int, default=90, help='Seconds per fallback source load')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/examples.md')
    args = parser.parse_args()
    if args.source_task:
        print('GALLERY_RESULT=' + json.dumps(source_examples(args.source_task, 1, args.current_recast)))
    else:
        revision, examples = collect(args.revision, 1)
        errors = load_missing(list_tasks().id, examples, 1, args.timeout) if args.load_missing else {}
        write(args.output, revision, examples, errors)
