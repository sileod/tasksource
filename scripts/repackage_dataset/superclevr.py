"""Join pinned Super-CLEVR programs, native split tables and original images."""
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import inspect
from pathlib import Path
import zipfile

from datasets import Features, Image, IterableDataset, IterableDatasetDict, Sequence, Value
from huggingface_hub import snapshot_download
import pyarrow.parquet as pq

from .vision import _materialize

SOURCE = 'RyanWW/Super-CLEVR'
REVISION = '2ecf1d00d1fa06b4a4ed9330269ac57d2f411090'
SHAPES = ('airliner', 'articulated bus', 'biplane', 'chopper', 'cruiser', 'dirtbike',
          'double bus', 'fighter', 'jet', 'minivan', 'mountain bike', 'regular bus',
          'road bike', 'school bus', 'scooter', 'sedan', 'suv', 'tandem bike', 'truck',
          'utility bike', 'wagon')
VOCABULARIES = {
    'yesno': ('no', 'yes'), 'count': tuple(map(str, range(11))), 'shape': SHAPES,
    'color': ('gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'),
    'size': ('small', 'large'), 'material': ('rubber', 'metal'),
}
BOOLEAN = {'exist', 'equal_size', 'equal_color', 'equal_material', 'equal_shape',
           'less_than', 'greater_than', 'equal_integer'}


def canonical_qa(row):
    """Partition by native program semantics, not guessed question wording."""
    terminal = row['program'][-1]['type']
    view = 'yesno' if terminal in BOOLEAN else terminal.removeprefix('query_')
    answer = row['answer']
    if view == 'yesno':
        if not isinstance(answer, bool):
            raise ValueError('Expected native boolean answer')
        answer = 'yes' if answer else 'no'
    else:
        answer = str(answer)
    if answer not in VOCABULARIES[view] or not row['question'].strip():
        raise ValueError('Unknown answer ontology or empty question')
    # Shape outputs use internal tokens ("utility" -> "utility bike").
    raw = row['program'][-1]['_output']
    if view == 'shape':
        raw = next((v for v in SHAPES if v.split()[0] == raw), raw)
    if raw != row['answer']:
        raise ValueError('Native program output disagrees with answer')
    metadata = {'id': f"superclevr:{row['question_index']}", 'source_dataset': SOURCE,
                'source_revision': REVISION, 'question_family_index': row['question_family_index'],
                'terminal_type': terminal, 'source_answer': row['answer']}
    return {'inputs': row['question'], 'labels': answer, 'view': view,
            'metadata': json.dumps(metadata, sort_keys=True)}


def superclevr(max_rows=5000, max_rows_eval=100, seed=42):
    """Super-CLEVR: original PNG bytes with grouped, program-typed questions.

    Native train/validation/test image partitions are retained. Bounded uniform
    question samples share one image per group. Yes/no, count, color, vehicle
    subtype, size and material views use explicit answer vocabularies. Programs
    and answer traces never enter model inputs. Source dataset license: MIT.
    """
    root = Path(snapshot_download(SOURCE, revision=REVISION, repo_type='dataset',
        local_dir='build/superclevr-original', allow_patterns=[
            'images.zip', 'viewer/*', 'superCLEVR_questions_30k.json']))
    questions = json.loads((root / 'superCLEVR_questions_30k.json').read_text())['questions']
    by_id = {row['question_index']: row for row in questions}
    assert len(by_id) == len(questions)
    memberships, seen_questions, seen_images = {}, set(), set()
    audit = {}
    for split in ('train', 'validation', 'test'):
        native = pq.read_table(root / 'viewer' / (split + '.parquet')).to_pylist()
        ids = [row['question_index'] for row in native]
        images = {row['image_index'] for row in native}
        assert len(set(ids)) == len(ids) and not seen_questions.intersection(ids)
        assert not seen_images.intersection(images), 'Native image split overlap'
        seen_questions.update(ids)
        seen_images.update(images)
        labels = Counter()
        for row in native:
            original = by_id[row['question_index']]
            assert all(original[key] == row[key] for key in ('question', 'image_index', 'image_filename'))
            assert str(original['answer']) == row['answer']
            qa = canonical_qa(original)
            labels[(qa['view'], qa['labels'])] += 1
        limit = max_rows if split == 'train' else max_rows_eval
        memberships[split] = heapq.nsmallest(limit, ids, key=lambda i:
            hashlib.sha256(f'{seed}:{i}'.encode()).digest())
        audit[split] = {'eligible': len(ids), 'selected': len(memberships[split]),
                        'source_labels': {f'{v}/{a}': n for (v, a), n in sorted(labels.items())}}
    assert seen_questions == set(by_id), 'Unassigned native question'
    selected = {split: [by_id[i] for i in ids] for split, ids in memberships.items()}
    del questions, by_id
    features = Features({'images': Sequence(Image(decode=False)), 'image_group_id': Value('string'),
        'qa': [{'inputs': Value('string'), 'labels': Value('string'), 'view': Value('string'),
                'metadata': Value('string')}]})
    def rows(split):
        grouped = defaultdict(list)
        for row in selected[split]:
            grouped[row['image_filename']].append(row)
        with zipfile.ZipFile(root / 'images.zip') as archive:
            for filename, records in grouped.items():
                yield {'images': [{'bytes': archive.read('images/' + filename), 'path': None}],
                       'image_group_id': f"superclevr:{records[0]['image_index']}",
                       'qa': [canonical_qa(row) for row in records]}
    dataset = IterableDatasetDict({split: IterableDataset.from_generator(rows,
        gen_kwargs={'split': split}, features=features) for split in memberships})
    result = _materialize(dataset, 'superclevr', SOURCE, REVISION, [], superclevr,
                          'mit', f'https://huggingface.co/datasets/{SOURCE}/blob/{REVISION}/README.md')
    report_path = Path('build/superclevr-release/provenance.json')
    report = json.loads(report_path.read_text())
    report.update(selection={'unit': 'question', 'seed': seed, 'max_rows': max_rows,
                             'max_rows_eval': max_rows_eval}, audit=audit,
                  vocabularies=VOCABULARIES,
                  qa_conversion_sha256=hashlib.sha256(inspect.getsource(canonical_qa).encode()).hexdigest())
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return result
