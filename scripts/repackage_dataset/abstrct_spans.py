"""Locally convert AbstRCT (Mayer, Cabrio and Villata, ECAI 2020) into major-claim/claim/premise
argument-component spans; never upload.

AbstRCT's original brat distribution (gitlab.com/tomaye/abstrct) is one .txt/.ann pair per MEDLINE
RCT abstract: .ann ``T`` lines are ``Tn\\t<Label> <start> <end>\\t<text>`` character spans into the
paired .txt, native to the whole abstract (not per sentence), so a claim/premise/major-claim answer
needs the full abstract as context -- unlike a per-sentence BIO view, which strips the surrounding
abstract a claim or premise is judged against.

PYTHONPATH=.:src python -m scripts.repackage_dataset.abstrct_spans --input /tmp/abstrct.zip --output build/abstrct-spans

Download the pinned GitLab archive into --input first:
    curl -L -o /tmp/abstrct.zip \\
      "https://gitlab.com/tomaye/abstrct/-/archive/<REVISION>/abstrct-<REVISION>.zip"
"""
import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from datasets import Dataset

from scripts.repackage_dataset.span_structure import FEATURES
from tasksource.citation import card_section
from tasksource.span_structure import record

REVISION = 'f856f1ca7514caa4094194b5623e84c159d9bf1d'
ARCHIVE_SHA256 = '32150598d2e9d3d8d5df82f4c059aee140cf769fde8b4842c6df83e7a76f5d4b'
ARCHIVE_URL = f'https://gitlab.com/tomaye/abstrct/-/archive/{REVISION}/abstrct-{REVISION}.zip'
LICENSE_URL = f'https://gitlab.com/tomaye/abstrct/-/blob/{REVISION}/LICENSE'
# (config, native directory, split): neoplasm has a genuine train/dev/test split; glaucoma and mixed
# are evaluation-only transfer sets (Mayer et al. train on neoplasm and test cross-domain on them).
DIRECTORIES = [
    ('neoplasm', 'train/neoplasm_train', 'train'),
    ('neoplasm', 'dev/neoplasm_dev', 'validation'),
    ('neoplasm', 'test/neoplasm_test', 'test'),
    ('glaucoma', 'test/glaucoma_test', 'test'),
    ('mixed', 'test/mixed_test', 'test'),
]
FIELDS = [('MajorClaim', 'major_claim', 'Every major claim in the abstract, as written.'),
          ('Claim', 'claim', 'Every claim in the abstract, as written.'),
          ('Premise', 'premise', 'Every premise in the abstract, as written.')]


def parse_ann(ann_text, text, counts):
    """``{label: [(start, end)]}`` from brat ``T`` lines; every span is checked verbatim against
    ``text`` (never trusted blindly), discontinuous (``;``-joined) spans are rejected rather than
    silently merged or dropped."""
    spans = {}
    for line in ann_text.splitlines():
        if not line.startswith('T') or '\t' not in line:
            continue
        _, label_span, span_text = line.split('\t', 2)
        label, offsets = label_span.split(' ', 1)
        if ';' in offsets:
            counts['discontinuous_spans_rejected'] += 1
            raise ValueError(f'Discontinuous brat span not supported: {line!r}')
        start, end = (int(x) for x in offsets.split(' '))
        if not 0 <= start < end <= len(text) or text[start:end] != span_text:
            raise ValueError(f'Span does not match document text: {line!r}')
        spans.setdefault(label, []).append((start, end))
    return spans


def abstract_records(doc_id, text, spans):
    for label, field, question in FIELDS:
        yield record(text, question, spans.get(label, []), field, doc_id, id=f'{doc_id}:{field}')


def convert(input_path, output, revision=REVISION, sha256=ARCHIVE_SHA256):
    """``revision``/``sha256`` are overridable so tests can run this against a small fixture archive
    without downloading or faking the real multi-hundred-abstract release."""
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f'Archive hash mismatch: {digest}')
    output.mkdir(parents=True, exist_ok=True)
    rows, counts = {}, {}
    root = f'abstrct-{revision}/AbstRCT_corpus/data'
    with zipfile.ZipFile(input_path) as archive:
        names = set(archive.namelist())
        license_text = archive.read(f'abstrct-{revision}/LICENSE').decode()
        if 'Attribution-NonCommercial-ShareAlike 4.0' not in license_text:
            raise ValueError('Unexpected AbstRCT license text; re-check the pinned revision')
        by_config_split = {}
        for config, directory, split in DIRECTORIES:
            counter = Counter()
            prefix = f'{root}/{directory}/'
            doc_ids = sorted({name[len(prefix):-4] for name in names
                               if name.startswith(prefix) and name.endswith('.txt')})
            records = []
            for doc_id in doc_ids:
                text = archive.read(f'{prefix}{doc_id}.txt').decode()
                ann_text = archive.read(f'{prefix}{doc_id}.ann').decode()
                counter['abstracts'] += 1
                spans = parse_ann(ann_text, text, counter)
                records += list(abstract_records(f'{config}:{doc_id}', text, spans))
            by_config_split.setdefault(config, {})[split] = records
            counts[f'{config}/{split}'] = dict(counter)
    native_abstracts = {}
    for config, splits in by_config_split.items():
        path = output if config == 'neoplasm' else output / config
        path.mkdir(exist_ok=True)
        for split, records in splits.items():
            data = Dataset.from_list(records, features=FEATURES)
            data.to_parquet(str(path / f'{split}.parquet'))
            rows[f'{config}/{split}'] = len(data)
            native_abstracts[f'{config}/{split}'] = counts[f'{config}/{split}']['abstracts']
    archive_url = f'https://gitlab.com/tomaye/abstrct/-/archive/{revision}/abstrct-{revision}.zip'
    license_url = f'https://gitlab.com/tomaye/abstrct/-/blob/{revision}/LICENSE'
    provenance = dict(mirror='tasksource/abstrct-spans', revision=revision,
        archive={'url': archive_url, 'sha256': sha256},
        license='cc-by-nc-sa-4.0', license_evidence=license_url,
        native_abstracts=native_abstracts,
        rows=rows, dropped=counts,
        settings={'context': 'the full abstract (not per-sentence): claim/premise status is judged '
                              'against the rest of the abstract, not standalone', 'fields': [f for _, f, _ in FIELDS],
                  'discontinuous_spans': 'none in this corpus; conversion raises if one appears',
                  'configs': 'neoplasm: native train/dev/test; glaucoma, mixed: test-only cross-domain sets',
                  'uploads': False})
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2, default=str) + '\n')
    configs_yaml = ''.join(
        f"- config_name: {config}\n  data_files:\n" + ''.join(
            f"  - split: {split}\n    path: {'' if config == 'neoplasm' else config + '/'}{split}.parquet\n"
            for split in splits)
        for config, splits in by_config_split.items())
    (output / 'README.md').write_text(f'''---
license: cc-by-nc-sa-4.0
language:
- en
tags:
- tasksource
- extraction
configs:
{configs_yaml}---
# AbstRCT argument-component spans

Major claim, claim and premise argument-component spans in MEDLINE randomized-controlled-trial (RCT) abstracts
(Mayer, Cabrio and Villata, ECAI 2020), expert-annotated with the brat tool (two annotators, adjudicated). Every
context is the full abstract, not an isolated sentence: whether a sentence is a claim or a premise is judged
against the rest of the abstract, which a per-sentence view would strip away.

Data-only mirror of the [original brat distribution](https://gitlab.com/tomaye/abstrct) (revision
`{revision}`, archive SHA256 `{sha256[:12]}`). Reproduce with
[scripts/repackage_dataset/abstrct_spans.py](https://github.com/sileod/tasksource/blob/main/scripts/repackage_dataset/abstrct_spans.py);
[provenance.json](provenance.json) records the archive hash, revision, native abstract counts and settings.
License: CC BY-NC-SA 4.0 ([evidence]({license_url})).

Schema: `context` (the abstract), `question`, `answers` (`text`, `answer_start`: every span, not alternatives) and
JSON-string `metadata` (`field` is one of `major_claim`, `claim`, `premise`; `group` is `{{config}}:{{PMID}}`).
Every field is always asked; an abstract with none of a kind answers with null. Used by the `extraction` config of
[tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions)
(`neoplasm` config, as `abstrct/neoplasm`).

`neoplasm` has the native train/dev/test split used for training. `glaucoma` and `mixed` are test-only
cross-domain transfer sets (no training abstracts) and are evaluation-only.

## Citation

Please cite the original dataset:

```bibtex
@inproceedings{{mayer2020ecai,
  author    = {{Tobias Mayer and Elena Cabrio and Serena Villata}},
  title     = {{Transformer-Based Argument Mining for Healthcare Applications}},
  booktitle = {{{{ECAI}} 2020 - 24th European Conference on Artificial Intelligence}},
  series    = {{Frontiers in Artificial Intelligence and Applications}},
  volume    = {{325}},
  pages     = {{2108--2115}},
  publisher = {{{{IOS}} Press}},
  year      = {{2020}},
}}
```

{card_section(title='Citing tasksource')}''')
    print(json.dumps(provenance, indent=2, default=str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    convert(args.input, args.output)
