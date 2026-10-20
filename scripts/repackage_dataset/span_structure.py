"""Prepare unpublished English structural-span mirrors locally; never upload.

PYTHONPATH=.:src python -m scripts.repackage_dataset.span_structure qasrl --input /tmp/qasrl-v2.tar --output build/pending-qasrl
PYTHONPATH=.:src python -m scripts.repackage_dataset.span_structure bioscope --input /tmp/bioscope.zip --output build/pending-bioscope

QA-SRL uses orig only, native train/dev/test, independent human validation, and one complete
validator reading per question. BioScope uses public abstracts/full papers only (the clinical
file masks words and requires restricted originals); the publisher's BigBio loader assigns
this corpus to train, with no fabricated evaluation split. Archive hashes below pin bytes.
"""
import argparse
import gzip
import hashlib
import json
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

from datasets import Dataset, Features, List, Value

from tasksource.span_structure import record, token_offsets, token_span

SOURCES = {
    'qasrl': {
        'url': 'https://qasrl.org/data/qasrl-v2.tar',
        'sha256': 'caba7a2d18584d6210b9a176ef6b4808d96f3d48194083dcb20db03697793e07',
        'revision': '29034398c26e419d530a2ec1e81994ca4a4d4e42',
        'documentation': 'https://github.com/uwnlp/qasrl-bank/blob/29034398c26e419d530a2ec1e81994ca4a4d4e42/FORMAT.md',
        'license': 'unspecified',
        'license_evidence': 'https://github.com/uwnlp/qasrl-bank/blob/29034398c26e419d530a2ec1e81994ca4a4d4e42/LICENSE',
        'license_note': 'Repository MIT license is phrased for software, not an explicit grant over underlying TQA/Wikipedia/Wikinews texts. Data scope remains unspecified.',
    },
    'bioscope': {
        'url': 'https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/bioscope.zip',
        'sha256': 'b20b525506f783ab9c120548ac4022ab587bf7cc419beaf759f7253f0cf0b1d1',
        'revision': '43ba162359f01d6436590ba2a014421b6ad09621',
        'documentation': 'https://huggingface.co/datasets/bigbio/bioscope/blob/43ba162359f01d6436590ba2a014421b6ad09621/bioscope.py',
        'license': 'cc-by-2.0',
        'license_evidence': 'https://huggingface.co/datasets/bigbio/bioscope/blob/43ba162359f01d6436590ba2a014421b6ad09621/README.md',
        'license_note': 'Public biological abstracts and papers only; article texts retain their original terms. Clinical masked text excluded.',
    },
}
FEATURES = Features({'context': Value('string'), 'question': Value('string'),
    'answers': {'text': List(Value('string')), 'answer_start': List(Value('int32'))},
    'metadata': Value('string')})


def qasrl_records(row, counts):
    tokens = row['sentenceTokens']
    text, starts = token_offsets(tokens)
    for entry in row['verbEntries'].values():
        verb = entry['verbIndex']
        va, vb = token_span(tokens, starts, verb, verb + 1)
        for index, (question, label) in enumerate(sorted(entry['questionLabels'].items())):
            # Only human judgments, once per worker. Conflicting duplicate worker judgments fail closed.
            by_worker = defaultdict(set)
            for judgment in label['answerJudgments']:
                if judgment['sourceId'].startswith('turk-'):
                    reading = tuple(sorted(set(tuple(s) for s in judgment.get('spans', []))))
                    worker = re.sub(r'-(expansion|eval)$', '', judgment['sourceId'])
                    by_worker[worker].add((judgment['isValid'], reading))
            votes = [next(iter(v)) for v in by_worker.values() if len(v) == 1]
            valid = [spans for yes, spans in votes if yes and spans]
            if len(valid) < 2 or 2 * len(valid) <= len(votes):
                counts['unvalidated_questions'] += 1
                continue
            # Select a complete real reading; disagreement never creates a union of incompatible sets.
            reading = min(Counter(valid), key=lambda spans: (-valid.count(spans), spans))
            spans = [token_span(tokens, starts, a, b) for a, b in reading]
            if any(b > c for (_, b), (c, _) in zip(spans, spans[1:])):
                counts['overlapping_readings'] += 1
                continue
            yield record(text, f'{question} For predicate {text[va:vb]!r} at characters [{va}, {vb}) '
                         '(zero-based, end-exclusive). Return every answer span.', spans,
                         f'predicate_{verb}_argument_{index}', row['sentenceId'],
                         id=f'{row["sentenceId"]}:{verb}:{index}', predicate_index=verb,
                         valid_judgments=len(valid), judgments=len(votes), selection='modal complete human reading')


def bioscope_records(sentence, group, counts):
    chunks, annotations, pos = [], [], 0
    def append(text):
        nonlocal pos
        chunks.append(text or '')
        pos += len(text or '')
    def visit(element):
        begin = pos
        append(element.text)
        for child in element:
            visit(child)
            append(child.tail)
        if element.tag in ('cue', 'xcope'):
            annotations.append((element.tag, dict(element.attrib), begin, pos))
    visit(sentence)
    text = ''.join(chunks)
    cues, scopes = defaultdict(list), defaultdict(list)
    for tag, attrs, a, b in annotations:
        if tag == 'xcope':
            scopes[attrs['id']].append((a, b))
        else:
            cues[(attrs['type'], attrs['ref'])].append((a, b))
    for kind in ('negation', 'speculation'):
        spans = [s for (t, _), values in cues.items() if t == kind for s in values]
        yield record(text, f'Every {kind} cue in the text, as written.', spans, f'{kind}_cues', group,
                     id=f'{group}:{kind}')
    for index, ((kind, ref), spans) in enumerate(sorted(cues.items())):
        if ref not in scopes:
            counts['missing_scope_fields'] += 1
            continue  # Never convert an unlocated positive scope into null.
        description = ', '.join(f'{text[a:b]!r} at [{a}, {b})' for a, b in spans)
        yield record(text, f'The full scope of the {kind} cue {description} '
                     '(zero-based characters, end-exclusive). Return every scope segment.', scopes[ref],
                     f'{kind}_scope_{index}', group, id=f'{group}:{ref}', cue_spans=spans)


def convert(source, input_path, output):
    info = SOURCES[source]
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if digest != info['sha256']:
        raise ValueError(f'Archive hash mismatch: {digest}')
    output.mkdir(parents=True, exist_ok=True)
    counts, sizes = {}, {}
    if source == 'qasrl':
        with tarfile.open(input_path) as archive:
            for split, native in [('train', 'train'), ('validation', 'dev'), ('test', 'test')]:
                counter = Counter()
                def generate():
                    with gzip.GzipFile(fileobj=archive.extractfile(f'qasrl-v2/orig/{native}.jsonl.gz')) as stream:
                        for line in stream:
                            counter['input_sentences'] += 1
                            yield from qasrl_records(json.loads(line), counter)
                rows = Dataset.from_generator(generate, features=FEATURES)
                rows.to_parquet(str(output / f'{split}.parquet'))
                counts[split], sizes[split] = dict(counter), len(rows)
    else:
        with zipfile.ZipFile(input_path) as archive:
            for config, member in [('abstracts', 'abstracts.xml'), ('papers', 'full_papers.xml')]:
                counter = Counter()
                def generate():
                    root = ET.fromstring(archive.read(member))
                    for index, sentence in enumerate(root.iter('sentence')):
                        counter['input_sentences'] += 1
                        yield from bioscope_records(sentence, f'{config}:{index}:{sentence.attrib["id"]}', counter)
                rows = Dataset.from_generator(generate, features=FEATURES)
                path = output / config
                path.mkdir(exist_ok=True)
                rows.to_parquet(str(path / 'train.parquet'))
                counts[config], sizes[config] = dict(counter), len(rows)
                (path / 'README.md').write_text(f'# {config}\n\nPublic BioScope {config}; train only, following BigBio. '
                    'Native XML text, nested cue/scope offsets; clinical masked words excluded.\n')
    provenance = dict(source=source, upstream=info, settings={'qasrl_partition': 'orig',
        'qasrl_validation': '>=2 valid human judgments and strict majority; modal complete reading',
        'bioscope_members': ['abstracts.xml', 'full_papers.xml'], 'uploads': False},
        rows=sizes, counts=counts,
        script='scripts/repackage_dataset/span_structure.py')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (output / 'README.md').write_text(f'''---
license: {info['license']}
language: en
---
# {source} span annotations

Local, unpublished conversion of the [original distribution]({info['url']}).
Reproduce with [scripts/repackage_dataset/span_structure.py](https://github.com/sileod/tasksource/blob/main/scripts/repackage_dataset/span_structure.py).
See [provenance.json](provenance.json) for pinned source documentation, archive SHA256,
conversion settings, validation policy and counts. {info['license_note']}

Schema: `context`, `question`, `answers` (`text`, `answer_start`) and JSON-string `metadata`.
Answer spans are verbatim lists, not alternatives. Metadata contains readable `field` names and
sentence `group` ids for the shared QA extraction path. Source native splits are preserved;
BioScope has train-only configs `abstracts` and `papers`. QA-SRL has train/validation/test.
No catalog declaration is active until a maintainer publishes and pins a data-only mirror.
''')
    print(json.dumps(provenance, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=sorted(SOURCES))
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    convert(args.source, args.input, args.output)


if __name__ == '__main__':
    main()
