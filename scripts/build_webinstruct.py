"""Prepare WebInstruct's single-answer MC and explicit Boolean rows.

    PYTHONPATH=src python scripts/build_webinstruct.py
    PYTHONPATH=.:src python scripts/upload_repackaged.py webinstruct

See dataset_cards/webinstruct.md for parsing rules and exclusions.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from datasets import Dataset
from huggingface_hub import hf_hub_download

SOURCE = 'TIGER-Lab/WebInstruct-verified'
REVISION = '3e8a350b3a935d68fe70bdf379692500abc9ff51'


def _webinstruct_choices(x):
    matches = list(re.finditer(r'(?<!\S)(?:\(([A-Za-z])\)|([A-Za-z])[.):](?:\.)?)\s+', x['question']))
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
            binary.append({**row, 'label': int(row['answer'].strip().lower() in {'yes', 'true'}), 'method': 'exact'})
    return {'mc': mc, 'binary': binary}


def build(output):
    counts = {}
    for split in ['train', 'test']:
        path = hf_hub_download(SOURCE, f'data/{split}-00000-of-00001.parquet', repo_type='dataset', revision=REVISION)
        source = Dataset.from_parquet(path)
        for config, rows in prepare(source.to_list()).items():
            target = output / config
            target.mkdir(parents=True, exist_ok=True)
            Dataset.from_list(rows).to_parquet(str(target / f'{split}.parquet'))
            counts[f'{config}/{split}'] = len(rows)
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'dataset_cards/webinstruct.md', output / 'README.md')
    (output / 'provenance.json').write_text(json.dumps({'source': SOURCE, 'revision': REVISION, 'counts': counts}, indent=2) + '\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/webinstruct'))
    build(parser.parse_args().output)
