"""Pinned native grounding joins; canonical MC QAs share unchanged encoded images."""
import hashlib
import inspect
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Features, Image, Sequence, Value
from huggingface_hub import hf_hub_download

from tasksource.grounding import coordinate_choices, normalized_box
from .procedural.common import file_sha256
from .vision import _materialize, _parquet_source

REFCOCO = {
    'refcocog': ('jxu124/refcocog', '55319436ad54b9480cefda6b9d64397de92456dd'),
    'refcoco': ('jxu124/refcoco', '9fba8200c5326e996f789191f095bd464ef1d09e'),
    'refcocoplus': ('jxu124/refcocoplus', '12e20b0b6039fbf656e89e2f26597e84c1037847'),
}
COCO = ('Untitled111/refcoco-train2014-assets', '5daa32cfe87ea355fab400b5ec5a8a9bb476ffd2')
COCO_ANNOTATIONS_URL = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip'
COCO_ANNOTATIONS_SHA256 = '031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009'
FEATURES = Features({'images': Sequence(Image(decode=False)), 'image_group_id': Value('string'),
    'qa': [{'view': Value('string'), 'inputs': Value('string'), 'choices_list': Sequence(Value('string')),
            'labels': Value('int64'), 'metadata': Value('string')}]})


def hub_file(source, filename):
    return Path(hf_hub_download(source[0], filename, repo_type='dataset', revision=source[1],
                               local_dir='build/grounding-sources/' + source[0].replace('/', '--')))


def ordered(values, seed):
    """Stable whole-source ordering, independent of file prefixes and target labels."""
    return sorted(values, key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).digest())


def xywh_box(box, width=1, height=1):
    x, y, w, h = box
    return normalized_box([x/width, y/height, (x+w)/width, (y+h)/height])


def grounding_qa(instruction, target, candidates, metadata, view, excluded):
    """Use only native candidates, requiring one exact target; keep SoM geometry aligned."""
    identity = metadata['source_row']
    def reject(reason):
        excluded.append({'source_row': identity, 'reason': reason})
    try:
        target = normalized_box(target)
        pool = list(dict.fromkeys(tuple(normalized_box(box)) for box in candidates))
    except (ValueError, TypeError):
        reject('invalid_box');return
    if not instruction.strip():
        reject('empty_instruction');return
    if tuple(target) not in pool:
        reject('target_not_in_native_candidates');return
    picked = coordinate_choices(target, pool, identity, return_geometry=True)
    if picked is None:
        reject('no_distinct_native_distractor');return
    metadata = {**metadata, 'target_bbox': target, 'candidate_boxes': picked[2],
                'coordinate_units': 'normalized_xyxy', 'choice_units': '0–1000',
                'image_group_id': metadata['image_group_id']}
    return {'view': view, 'inputs': instruction.strip(), 'choices_list': picked[0], 'labels': picked[1],
            'metadata': json.dumps(metadata, ensure_ascii=False, sort_keys=True)}


def materialize(generators, name, sources, excluded, conversion, license_url, max_rows, seed, provenance=None):
    script_hash = file_sha256(Path(inspect.getsourcefile(conversion)))
    fingerprint = hashlib.sha256(json.dumps([name, sources, max_rows, seed, script_hash, file_sha256(Path(__file__))]).encode()).hexdigest()
    result = _materialize(generators, name, sources[0][0], sources[0][1], excluded, conversion, 'other', license_url,
                          features=FEATURES, fingerprint=fingerprint)
    directory = Path('build') / (name + '-release')
    path = directory / 'provenance.json'
    report = json.loads(path.read_text())
    report.update({'mirror': 'tasksource/' + name, 'sources': dict(sources), 'max_train_questions': max_rows,
        'seed': seed, 'selection': 'stable hash order of source image groups; cap counts QAs, not images',
        'excluded_by_reason': dict(Counter(row['reason'] for row in excluded)),
        'conversion_file_sha256': script_hash, 'grounding_helpers_sha256': file_sha256(Path(__file__)),
        'parquet_sha256': {p.name: file_sha256(p) for p in directory.glob('*.parquet')},
        'representation': 'canonical grouped MC; original images, native candidate boxes, no synthetic negatives',
        'choice_config': {'max_options': 8, 'min_center_distance': 0.03, 'units': '0–1000'}})
    if name == 'refcoco-grounding':
        report['annotation_archive'] = {'url': COCO_ANNOTATIONS_URL, 'sha256': COCO_ANNOTATIONS_SHA256}
        report['split_policy'] = 'native test subsets merged; held-out images reserved by both COCO ID and encoded-image SHA256 across variants'
    report.update(provenance or {})
    path.write_text(json.dumps(report, indent=2) + '\n')
    return result


def refcoco_grounding(max_rows=100000, seed=0):
    """Human referring expressions from RefCOCO, RefCOCO+ and RefCOCOg, joined to COCO2014.

    One shared image cache and mirror; views retain their native question identities.
    The union of all held-out image IDs is reserved from training across variants;
    overlapping validation/test images stay in test only. Original split names and
    per-image Flickr licenses are recorded without excluding noncommercial or
    unresolved sources. Original images are redistributed unchanged; optional pixel
    augmentations remain subject to each image's original terms.
    Only native COCO object boxes supply options; no invented distractors.
    """
    from urllib.request import urlretrieve
    annotation_path = Path('build/grounding-assets/annotations_trainval2014.zip')
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    if not annotation_path.exists():
        temporary = annotation_path.with_suffix('.part')
        urlretrieve(COCO_ANNOTATIONS_URL, temporary)
        temporary.replace(annotation_path)
    if file_sha256(annotation_path) != COCO_ANNOTATIONS_SHA256:
        raise ValueError('COCO2014 annotation archive checksum mismatch')
    with zipfile.ZipFile(annotation_path) as archive:
        coco = json.loads(archive.read('annotations/instances_train2014.json'))
    images = {image['id']: image for image in coco['images']}
    licenses = {item['id']: item for item in coco['licenses']}
    objects = defaultdict(dict)
    for item in coco['annotations']:
        if not item.get('iscrowd'):
            objects[item['image_id']][item['id']] = item
    grouped, heldout, tests = defaultdict(list), set(), set()
    for view, source in REFCOCO.items():
        native = _parquet_source(*source, 'build/grounding-sources/' + view, 'data/*.parquet')
        for split, rows in native.items():
            for row in rows:
                canonical = 'train' if split == 'train' else 'validation' if split in ('validation', 'val') else 'test'
                grouped[row['image_id']].append((canonical, split, view, row))
                if canonical != 'train':heldout.add(row['image_id'])
                if canonical == 'test':tests.add(row['image_id'])
    archive_path = Path('build/grounding-assets/train2014.zip')
    if not archive_path.exists():archive_path = hub_file(COCO, 'train2014.zip')
    # COCO contains duplicate photographs under different image IDs. Reserve bytes too.
    with zipfile.ZipFile(archive_path) as archive:
        image_hashes = {image_id: hashlib.sha256(archive.read('train2014/' + images[image_id]['file_name'])).hexdigest()
                        for image_id in grouped}
    heldout_hashes = {image_hashes[key] for key in heldout}
    test_hashes = {image_hashes[key] for key in tests}
    excluded, generators = [], {}
    for split in ('train', 'validation', 'test'):
        def examples(split=split):
            count = 0
            with zipfile.ZipFile(archive_path) as archive:
                for image_id in ordered(grouped, seed):
                    rows = [item for item in grouped[image_id] if item[0] == split]
                    if not rows:continue
                    if (split == 'train' and image_hashes[image_id] in heldout_hashes
                            or split == 'validation' and image_hashes[image_id] in test_hashes):
                        excluded.append({'source_row': f'{split}:coco:{image_id}', 'reason': 'cross_variant_heldout_image'});continue
                    info = images[image_id]
                    license = licenses[info['license']]
                    width, height = info['width'], info['height']
                    pool = {}
                    for ann_id, item in objects[image_id].items():
                        try:pool[ann_id] = xywh_box(item['bbox'], width, height)
                        except ValueError:continue
                    qas, phrases = [], defaultdict(set)
                    for _, _, view, row in rows:
                        for sentence in row['sentences']:
                            phrases[sentence['raw'].strip().casefold()].add(row['ann_id'])
                    for _, native_split, view, row in rows:
                        for sentence in row['sentences']:
                            identity = f"{view}:{sentence['sent_id']}"
                            if len(phrases[sentence['raw'].strip().casefold()]) > 1:
                                excluded.append({'source_row': identity, 'reason': 'ambiguous_duplicate_instruction'});continue
                            if row['ann_id'] not in pool:
                                excluded.append({'source_row': identity, 'reason': 'invalid_native_target'});continue
                            qa = grounding_qa(sentence['raw'], pool[row['ann_id']], pool.values(), {
                                'source_row': identity, 'source_dataset': REFCOCO[view][0], 'source_revision': REFCOCO[view][1],
                                'source_split': native_split, 'image_group_id': f'coco:{image_id}', 'coco_image_id': image_id,
                                'image_size': [width, height], 'ann_id': row['ann_id'], 'ref_id': row['ref_id'],
                                'image_license': license, 'image_url': info.get('flickr_url'), 'annotation_origin': 'human'}, view, excluded)
                            if qa is not None:qas.append(qa)
                    if split == 'train':qas = qas[:max_rows-count]
                    if qas:
                        encoded = archive.read('train2014/' + info['file_name'])
                        yield {'images': [{'bytes': encoded, 'path': None}], 'image_group_id': f'coco:{image_id}', 'qa': qas}
                        count += len(qas)
                    if split == 'train' and count >= max_rows:break
        generators[split] = examples
    return materialize(generators, 'refcoco-grounding', [*REFCOCO.values(), COCO], excluded,
        refcoco_grounding, 'https://cocodataset.org/#termsofuse', max_rows, seed)
