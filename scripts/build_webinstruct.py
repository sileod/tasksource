"""Prepare WebInstruct's single-answer MC and explicit Boolean rows.

    PYTHONPATH=src python scripts/build_webinstruct.py
    PYTHONPATH=.:src python scripts/upload_repackaged.py webinstruct

See dataset_cards/webinstruct.md for parsing rules and exclusions.
"""

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from datasets import Dataset
from huggingface_hub import hf_hub_download

SOURCE = 'TIGER-Lab/WebInstruct-verified'
REVISION = '3e8a350b3a935d68fe70bdf379692500abc9ff51'
OPTION_MARKERS = r'(?<!\S)(?:\(([A-Za-z])\)|([A-Za-z])[.):](?:\.)?)\s+'


def _webinstruct_choices(x):
    matches = list(re.finditer(OPTION_MARKERS, x['question']))
    letters = [(m[1] or m[2]).lower() for m in matches]
    choices = [x['question'][m.end():matches[i+1].start() if i+1 < len(matches) else None].strip()
               for i, m in enumerate(matches)]
    answer = x['answer'].strip().lower().strip('(). ')
    valid = (len(letters) >= 2 and len(set(letters)) == len(letters)
             and set(letters) == set('abcdefghijklmnopqrstuvwxyz'[:len(letters)])
             and answer in letters and all(choices) and x['question'][:matches[0].start()].strip())
    return {'prompt': x['question'][:matches[0].start()].strip() if matches else '',
            'options': choices, 'gold': letters.index(answer) if valid else None}


def prepare(rows):
    mc, binary = [], []
    for row in rows:
        if row['answer_type'] == 'Multiple Choice':
            parsed = _webinstruct_choices(row)
            if parsed['gold'] is not None:
                mc.append({**row, **parsed, 'method': 'regex'})
        elif row['answer_type'] == 'Boolean' and row['answer'].strip().lower() in {'yes', 'no', 'true', 'false'}:
            binary.append({**row, 'prompt': row['question'],
                           'label': int(row['answer'].strip().lower() in {'yes', 'true'}), 'method': 'exact'})
    return {'mc': mc, 'binary': binary}


def build(output, bad_examples=None, repairs=None):
    rejected = [json.loads(line) for line in bad_examples.read_text().splitlines()] if bad_examples else []
    bad_keys = {row['key'] for row in rejected}
    if any(row.get('verdict') not in {'wrong', 'malformed', 'ambiguous'} for row in rejected):
        raise ValueError('The removal manifest must contain only confirmed bad examples')
    edits = [json.loads(line) for line in repairs.read_text().splitlines()] if repairs else []
    edited = {row['key']: row for row in edits}
    if len(edited) != len(edits) or bad_keys & edited.keys():
        raise ValueError('Repair keys must be unique and disjoint from removals')
    counts = {}
    removed, repaired, found = {}, {}, set()
    filtered = bad_examples is not None or repairs is not None
    for split in ['train', 'test']:
        path = hf_hub_download(SOURCE, f'data/{split}-00000-of-00001.parquet', repo_type='dataset', revision=REVISION)
        source = Dataset.from_parquet(path)
        for config, rows in prepare(source.to_list()).items():
            target = output / f'{config}-unfiltered'
            target.mkdir(parents=True, exist_ok=True)
            Dataset.from_list(rows).to_parquet(str(target / f'{split}.parquet'))
            counts[f'{config}-unfiltered/{split}'] = len(rows)
            if not filtered:
                continue
            removed[f'{config}/{split}'] = sum(f"{config}/{split}/{row['id']}" in bad_keys for row in rows)
            found.update(f"{config}/{split}/{row['id']}" for row in rows
                         if f"{config}/{split}/{row['id']}" in bad_keys | edited.keys())
            rows = [row for row in rows if f"{config}/{split}/{row['id']}" not in bad_keys]
            repaired[f'{config}/{split}'] = 0
            for row in rows:
                edit = edited.get(f"{config}/{split}/{row['id']}")
                if not edit:
                    continue
                if not isinstance(edit.get('prompt'), str) or not edit['prompt'].strip():
                    raise ValueError('A repair must include a nonempty prompt')
                if config == 'mc':
                    options = edit.get('options')
                    if (not isinstance(options, list) or len(options) != len(row['options'])
                            or any(not isinstance(new, str) or not new.strip()
                                   or ' '.join(new.split()) not in ' '.join(old.split())
                                   for new, old in zip(options, row['options']))):
                        raise ValueError('Repaired options must preserve source order and content')
                    row['options'] = options
                row.update(prompt=edit['prompt'], method=row['method'] + '+presentation-repair')
                repaired[f'{config}/{split}'] += 1
            target = output / config
            target.mkdir(parents=True, exist_ok=True)
            Dataset.from_list(rows).to_parquet(str(target / f'{split}.parquet'))
            counts[f'{config}/{split}'] = len(rows)
    unknown = (bad_keys | edited.keys()) - found
    if unknown:
        raise ValueError(f'Audit manifests contain {len(unknown)} unknown example keys')
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'dataset_cards/webinstruct.md', output / 'README.md')
    provenance = {'source': SOURCE, 'revision': REVISION, 'counts': counts}
    if bad_examples:
        shutil.copyfile(bad_examples, output / 'bad-examples.jsonl')
        provenance.update(removed=removed, audit_models=sorted({row['model'] for row in rejected}),
                          removal_manifest_sha256=hashlib.sha256(bad_examples.read_bytes()).hexdigest())
        with (output / 'README.md').open('a') as card:
            card.write(f"\n## Filtered examples\n\nRemoved {sum(removed.values())} confirmed problem examples; "
                       "[bad-examples.jsonl](bad-examples.jsonl) preserves their IDs and audit evidence. "
                       "Counts by config and split are recorded in [provenance.json](provenance.json).\n")
    if repairs:
        shutil.copyfile(repairs, output / 'repairs.jsonl')
        provenance.update(repaired=repaired, repairs_sha256=hashlib.sha256(repairs.read_bytes()).hexdigest())
        with (output / 'README.md').open('a') as card:
            card.write(f"\nApplied {sum(repaired.values())} source-grounded, rechecked presentation repairs. "
                       "[repairs.jsonl](repairs.jsonl) records the changed prompts/options. "
                       "Original questions, answers, labels and option order are preserved.\n")
    if bad_examples and (bad_examples.parent / 'screen-verdicts.jsonl').exists():
        audit_dir = bad_examples.parent
        screened = [json.loads(line) for line in (audit_dir / 'screen-verdicts.jsonl').read_text().splitlines()]
        confirmed = [json.loads(line) for line in (audit_dir / 'confirm-verdicts.jsonl').read_text().splitlines()]
        baseline_count = sum(count for key, count in counts.items() if '-unfiltered/' in key)
        if len({row['key'] for row in screened}) != baseline_count:
            raise ValueError('Presentation screening coverage is incomplete')
        provenance['audit'] = {'screened': len(screened), 'confirmed': len(confirmed),
                               'screen_counts': dict(Counter(row['verdict'] for row in screened)),
                               'confirm_counts': dict(Counter(row['verdict'] for row in confirmed)),
                               'input_sha256': {name: hashlib.sha256((audit_dir / name).read_bytes()).hexdigest()
                                                for name in ['screen-inputs.jsonl', 'confirm-inputs.jsonl',
                                                             'repair-inputs.jsonl', 'verify-repairs-inputs.jsonl']
                                                if (audit_dir / name).exists()}}
        for name in ['audit-prompt.txt', 'audit-settings.json', 'screen-verdicts.jsonl', 'confirm-verdicts.jsonl']:
            if (audit_dir / name).exists():
                shutil.copyfile(audit_dir / name, output / name)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(counts, indent=2))
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/webinstruct'))
    parser.add_argument('--bad-examples', type=Path, help='explicit confirmed-removal manifest from the audit')
    parser.add_argument('--repairs', type=Path, help='source-grounded, verified presentation edits')
    args = parser.parse_args()
    build(args.output, args.bad_examples, args.repairs)
