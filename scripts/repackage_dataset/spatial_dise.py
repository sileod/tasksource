"""Join Spatial-DISE CSVs to native image archives; keep reliable panel-based tasks."""
from collections import Counter
import csv
import json
from pathlib import Path
import re
import tarfile

from huggingface_hub import snapshot_download
from .procedural.common import file_sha256, write_static

SOURCE = 'TACPS-liv/Spatial-DISE'
REVISION = 'c1067b763ab4199d0dd481efefeae384b738f9fd'
FILES = {'train':'dataset/train.csv', 'validation':'dataset/val.csv',
         'test':'dataset/test.csv', 'benchmark':'DISE-bench/DISE-benchmark.csv'}
CATEGORIES = {'3D Rotation', '3D Folding', '3D Shape Finding', '3D Combination'}


def source_group(row):
    # Difficulty variants can reuse the generator's question ID / seed.
    if row['source'] == 'synthetic':
        return row['category'] + ':' + re.findall(r'\d+', row['image'])[-1]
    return row['image']


def select_rows(splits):
    """Keep native partitions, reserving benchmark and held-out generator families."""
    seen = {source_group(r) for r in splits['benchmark']}
    selected, excluded = {}, []
    for split in ('test', 'validation', 'train'):
        selected[split] = []
        for row in splits[split]:
            options = [v.strip() for v in row['options'].split(',')]
            reason = ('unaudited_category' if row['category'] not in CATEGORIES else
                      'heldout_source_group' if source_group(row) in seen else
                      'invalid_choices' if options != list('ABCDEF'[:len(options)]) or
                          not 2 <= len(options) <= 6 or row['answer'] not in options else
                      'missing_question' if not row['question'].strip() else None)
            if reason:
                excluded.append(dict(split=split, image=row['image'], reason=reason));continue
            selected[split].append(row)
        seen.update(source_group(r) for r in splits[split])
    return selected, excluded


def spatial_dise():
    """Spatial-DISE native spatial MC questions, with unchanged composite image bytes.

    Includes rotation, folding, combination and hidden-face tasks. Projection is
    excluded after a source-image audit found missing arrow and option labels.
    Original train/validation/test partitions are retained, removing generator-ID
    families shared with a held-out split or the separate 559-question benchmark.
    Answer choices name the image's panels; Jev may reorder these references without
    changing panel identity. No generated distractors. Native procedural or curated
    wild labels are identified in metadata. CC BY-NC-SA 4.0; embedded Icons8 assets
    and collected aptitude-test images retain their original third-party terms.
    """
    root = Path(snapshot_download(SOURCE, repo_type='dataset', revision=REVISION,
        allow_patterns=['image/*.tar', *FILES.values()], local_dir='build/spatial-dise-source', max_workers=8))
    splits = {}
    for split, name in FILES.items():
        with (root/name).open(newline='', encoding='utf-8-sig') as stream:
            splits[split] = [{k.strip():v.strip() for k,v in r.items()}
                             for r in csv.DictReader(stream, skipinitialspace=True)]
    selected, excluded = select_rows(splits)
    wanted = {r['image'].removeprefix('images/') for rows in selected.values() for r in rows}
    images = {}
    for archive in sorted((root/'image').glob('*.tar')):
        with tarfile.open(archive) as source:
            for member in source:
                if member.isfile() and member.name in wanted:
                    payload = source.extractfile(member).read()
                    if member.name in images and images[member.name] != payload:
                        raise ValueError(f'Conflicting image bytes: {member.name}')
                    images[member.name] = payload
    if wanted != images.keys():raise ValueError(f'Missing {len(wanted-images.keys())} source images')
    output = {}
    for split, rows in selected.items():
        output[split] = [dict(id=r['image'], image=images[r['image'].removeprefix('images/')],
            inputs=r['question']+'\nChoose the image panel identified by the answer text.',
            choices_list=['Panel '+v.strip() for v in r['options'].split(',')],
            labels=[v.strip() for v in r['options'].split(',')].index(r['answer']),
            metadata=dict(source=SOURCE, revision=REVISION, native_split=split,
                source_group_id=source_group(r), category=r['category'], difficulty=r['difficulty'],
                annotation_origin='procedural' if r['source']=='synthetic' else 'curated wild',
                native_answer=r['answer'])) for r in rows]
    report = dict(source=SOURCE, revision=REVISION, license='cc-by-nc-sa-4.0',
        license_url='https://huggingface.co/datasets/'+SOURCE+'#license',
        source_sha256={str(p.relative_to(root)):file_sha256(p)
                       for p in [*(root/n for n in FILES.values()), *sorted((root/'image').glob('*.tar'))]},
        native_counts={s:len(rs) for s,rs in splits.items()}, excluded_rows=len(excluded),
        excluded_counts=dict(Counter(r['reason'] for r in excluded)),
        configuration=dict(categories=sorted(CATEGORIES), pixels='original encoded bytes',
            split_policy='benchmark > test > validation > train; category + generator ID across difficulty'))
    result = write_static('spatial-dise', output, report, spatial_dise)
    directory = Path('build/spatial-dise-release')
    with (directory/'excluded-questions.jsonl').open('a') as stream:
        for row in excluded:stream.write(json.dumps(row)+'\n')
    return result
