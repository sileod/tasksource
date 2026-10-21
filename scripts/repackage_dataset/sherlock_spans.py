"""Locally convert the *SEM 2012 ConanDoyle-neg gold triplet columns; never upload.

Download the two pinned files in FILES into --input. Reconstruct the official 3644/787/1089
sentence split by story identity (the source mirror concatenates training and development).
This keeps separate negation instances, discontinuous scopes and sub-token affix cues.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from datasets import Dataset

from scripts.repackage_dataset.span_structure import FEATURES
from tasksource.citation import card_section
from tasksource.span_structure import record, token_offsets

REVISION = '03cb052efdb205efdf403223a71764237a353f6e'
BASE = f'https://raw.githubusercontent.com/mosharafhossain/negation-cue/{REVISION}/data/cd-sco/'
FILES = {
    'training_dev.txt': ('corpus/training_dev/training_dev.txt', 'ea966b062893cdc18d8d74b1c635f50db623f3f6e9d605fa4e3e262272901e64'),
    'single_gold.txt': ('corpus/test-gold/single_gold.txt', 'cb71fad09c44b8d4d7147cd7275fd025583d1c8d6f1458f039565c685c85b996'),
}


def column_spans(text, tokens, starts, column):
    spans = []
    for token, start, value in zip(tokens, starts, column):
        if value == '_':
            continue
        matches = [m.start() for m in re.finditer(f'(?={re.escape(value)})', token)]
        if len(matches) != 1:
            return None
        a = start + matches[0]
        b = a + len(value)
        if spans and not text[spans[-1][1]:a].strip():
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans


def sherlock_records(lines, counts):
    columns = [line.split('\t') for line in lines if line.strip()]
    tokens = [r[3] for r in columns]
    text, starts = token_offsets(tokens)
    group = ':'.join(columns[0][:2])
    if len(columns[0]) == 8 and columns[0][7] == '***':
        yield record(text, 'Every negation cue in the text, as written.', [], 'negation_cues', group, id=group)
        return
    n = len(columns[0])
    if (n - 7) % 3 or any(len(r) != n for r in columns):
        raise ValueError('Invalid native negation columns')
    all_cues, cue_failure = [], False
    for i, col in enumerate(range(7, n, 3)):
        cues = column_spans(text, tokens, starts, [r[col] for r in columns])
        scopes = column_spans(text, tokens, starts, [r[col + 1] for r in columns])
        if cues is None or not cues:
            counts['unlocated_cue_fields'] += 1
            cue_failure = True
            continue
        all_cues.extend(cues)
        if scopes is None:
            counts['unlocated_scope_fields'] += 1
            continue
        cue = ', '.join(f'{text[a:b]!r} at [{a}, {b})' for a, b in cues)
        yield record(text, f'The scope of the negation cue {cue} (zero-based characters, end-exclusive). '
                     'Return every scope segment: the negated event with all its participants, including its '
                     'subject, excluding the cue itself.', scopes,
                     f'negation_scope_{i}', group, id=f'{group}:{i}', cue_spans=cues)
    if not cue_failure:
        yield record(text, 'Every negation cue in the text, as written.', all_cues, 'negation_cues', group, id=group)


def convert(input_path, output):
    split_lines = {'train': [], 'validation': [], 'test': []}
    for filename, (_, sha) in FILES.items():
        content = (input_path / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != sha:
            raise ValueError(f'Hash mismatch: {filename}')
        for block in re.split(r'\n\s*\n', content.decode().strip()):
            chapter = block.split('\t', 1)[0]
            if chapter.startswith('baskervilles'):
                split = 'train'
            elif chapter in ('wisteria01', 'wisteria02'):
                split = 'validation'
            elif chapter in ('cardboard', 'circle01', 'circle02'):
                split = 'test'
            else:
                raise ValueError(f'Unexpected story: {chapter}')
            split_lines[split].append(block.splitlines())
    expected = {'train': 3644, 'validation': 787, 'test': 1089}
    assert {s: len(v) for s, v in split_lines.items()} == expected
    output.mkdir(parents=True, exist_ok=True)
    sizes, reports = {}, {}
    for split, sentences in split_lines.items():
        counts = Counter()
        records = [r for sentence in sentences for r in sherlock_records(sentence, counts)]
        rows = Dataset.from_list(records, features=FEATURES)
        rows.to_parquet(str(output / f'{split}.parquet'))
        sizes[split], reports[split] = len(rows), dict(counts)
    mirror = 'tasksource/sherlock-negation-spans'
    provenance = dict(mirror=mirror, revision=REVISION, files={name: {'url': BASE + path, 'sha256': sha}
        for name, (path, sha) in FILES.items()}, native_sentences=expected, rows=sizes, dropped=reports,
        license='unspecified (freely available annotations; public-domain source stories)',
        license_evidence='https://www.aclweb.org/portal/content/sem-shared-task-2012-resolving-scope-and-focus-negation',
        settings={'text': 'tokens joined by one space', 'scopes': 'merge adjacent literal segments; preserve discontinuity',
                  'split': 'official story partitions, no resampling', 'uploads': False})
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'README.md').write_text(f'''---
license: unknown
language:
- en
tags:
- tasksource
- extraction
- negation
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: validation
    path: validation.parquet
  - split: test
    path: test.parquet
---
# ConanDoyle-neg negation spans

Negation cues and their scopes in Conan Doyle stories, as annotated for the *SEM 2012 shared task: every sentence
asks for its negation cues (including affixes such as "in" of "infrequent") and the scope of each cue, which can be
discontinuous.

Data-only mirror of the [original *SEM 2012 shared task distribution](https://www.clips.uantwerpen.be/sem2012-st-neg/),
using a [pinned copy of the native gold columns](https://github.com/mosharafhossain/negation-cue/tree/03cb052efdb205efdf403223a71764237a353f6e/data/cd-sco),
prepared for `{mirror}`.
Reproduce with [scripts/repackage_dataset/sherlock_spans.py](https://github.com/sileod/tasksource/blob/main/scripts/repackage_dataset/sherlock_spans.py).
[provenance.json](provenance.json) records file hashes, revisions, splits and settings.

Native train/validation/test story partitions (Baskervilles / Wisteria / Cardboard+Circle).
Schema: `context`, `question`, list `answers` (`text`, `answer_start`), JSON-string `metadata`
(`field`, `group`, cue offsets). Offsets preserve sub-token cues, cue identity and discontinuous
scopes. Human gold only. Stories are public domain; no standard annotation license established
(unspecified). Declared in tasksource's extraction catalog as `sherlock/scopes`.

Every sentence is kept; sentences without negation answer the cue field with null.

## Citation

Please cite the original dataset:

```bibtex
@inproceedings{{morante-daelemans-2012-conandoyle,
    title = "{{C}}onan{{D}}oyle-neg: Annotation of negation cues and their scope in {{C}}onan {{D}}oyle stories",
    author = "Morante, Roser and Daelemans, Walter",
    booktitle = "Proceedings of the Eighth International Conference on Language Resources and Evaluation ({{LREC}}'12)",
    year = "2012",
    url = "http://www.lrec-conf.org/proceedings/lrec2012/pdf/221_Paper.pdf",
}}
```

{card_section(title='Citing tasksource')}''')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    convert(args.input, args.output)
