"""Load the English structural sources, check every output span and record counts/examples.

PYTHONPATH=.:src python -m scripts.audit_span_structure --output build/span-structure/audit.json
--loaded points to DatasetDicts previously obtained by load_task (for offline audit).
--pending includes locally converted QA-SRL/BioScope/Sherlock; no network writes occur.
"""
import argparse
import json
from pathlib import Path

from datasets import DatasetDict, Dataset, load_from_disk

from tasksource import load_task, task_licenses, task_provenance
from tasksource.jev.extraction import ner_rows, qa_rows, check
from scripts.build_jev_extraction import NER, row_id

TASKS = ['ud/ewt', 'ud/gum', 'preco/mentions', 'redocred/entities', 'conll2000']


def summarize(dataset, task, token_task=False):
    stats, examples = {}, []
    for split, inputs in dataset.items():
        rows = ner_rows(inputs, task, split, inputs.features['labels'].feature.names,
                        instruction=NER[task]) if token_task else qa_rows(
            inputs, task, split, kind='list')
        count, fields, nulls = 0, 0, 0
        candidates = []
        for row in rows:
            check(row)
            row['id'] = row_id(row)
            count += 1
            fields += len(row['fields'])
            nulls += sum(a['null'] for a in row['answers'])
            if (split == 'train' and len(row['state']) >= 20
                    and sum(not a['null'] for a in row['answers']) >= 2
                    and all(r['state'] != row['state'] for r in candidates)):
                candidates.append(row)
                candidates.sort(key=lambda r: (len(r['state']), r['state']))
                del candidates[2:]
        stats[split] = dict(input_rows=len(inputs), extraction_rows=count, fields=fields, null_fields=nulls,
                            null_share=round(nulls / fields, 6) if fields else 0)
        if split == 'train':
            examples = candidates
    return dict(splits=stats, examples=examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loaded', type=Path)
    parser.add_argument('--pending', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for task in TASKS:
        token_task = task == 'conll2000'
        rows = load_from_disk(str(args.loaded / task.replace('/', '--'))) if args.loaded else load_task(
            task, extractive=not token_task)
        result[task] = summarize(rows, task, token_task)
        result[task]['provenance'] = task_provenance(task, extractive=not token_task)
        result[task]['license'] = task_licenses([task], extractive=not token_task).iloc[0][['license', 'license_use']].to_dict()
        print(task, result[task]['splits'], flush=True)
    if args.pending:
        for task, path in [('qasrl/orig', 'pending-qasrl'), ('bioscope/abstracts', 'pending-bioscope/abstracts'),
                           ('bioscope/papers', 'pending-bioscope/papers'), ('sherlock/scopes', 'pending-sherlock')]:
            directory = args.pending / path
            rows = DatasetDict({p.stem: Dataset.from_parquet(str(p)) for p in sorted(directory.glob('*.parquet'))})
            result[task] = dict(status='local conversion only; no active task', **summarize(rows, task))
            print(task, result[task]['splits'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + '\n')


if __name__ == '__main__':
    main()
