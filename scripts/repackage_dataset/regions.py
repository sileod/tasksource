"""Reproducible region, preference and correspondence mirrors; no runtime transforms."""
from collections import Counter, defaultdict
from functools import cache
import hashlib
import heapq
import io
import json
import math
from pathlib import Path
import tarfile
import zipfile

import numpy as np
from PIL import Image as PILImage, ImageChops, ImageDraw, ImageFilter
from datasets import ClassLabel, Features, Image, IterableDataset, IterableDatasetDict, Sequence, Value, load_dataset
from huggingface_hub import snapshot_download

from tasksource.grounding import grid_labels
from .vision import _materialize

ROOT = Path('build/region-expansion')
COCO = ('srishti-kaushik/COCO-2017', '3e3628d52b78d7e18860087b297efeb282b5a6a1')
DOCLAYNET = ('docling-project/DocLayNet-v1.1', '5e89392376049f4d589ea339ed64468310ed5c3f')
SPAIR = ('0jl/SPair-71k', '1982fc62b2c1db92db05f67599f2d79943735441')
ARCHIVES = {
    'lvis_v1_train.json.zip': ('https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip',
        '334a4caa374030a7817cf050364525e910f7960f9b6968cef47cffbf3893f8ba'),
    'lvis_v1_val.json.zip': ('https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip',
        '5cae9a3c79aadb667550c2b5dcf7f4d86e059a41ec91ef690225b667e28e9ba5'),
    'SPair-71k.tar.gz': ('https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz',
        'd145eebe4d7d02ff0b9e2746dd3d72cfcb872df0dcfd6f3227576eb594408809'),
    'twoafc_val.tar.gz': ('https://perceptual-similarity.s3.us-west-2.amazonaws.com/dataset/twoafc_val.tar.gz',
        '41486671b06de923e7ba277c4797d4d9d86d609863ba35ecdb576133f8b9d859'),
    'twoafc_train.tar.gz': ('https://perceptual-similarity.s3.us-west-2.amazonaws.com/dataset/twoafc_train.tar.gz',
        'b4f5c772c7cd88ac44bcb4ff0741745d434c05f0094f2bfb262ddcd0055bf393'),
}
LAYOUT_CLASSES = ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header',
                  'Picture', 'Section-header', 'Table', 'Text', 'Title']
RENDER_CONFIG = {'version': 1, 'color': [255, 0, 0], 'mask_opacity': .15,
                 'line_width_fraction': .004, 'encoding': 'PNG', 'resize': None}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive_file(filename):
    """Cache versioned upstream bytes, checking the pinned content hash."""
    from urllib.request import urlopen
    url, expected = ARCHIVES[filename]
    path = ROOT / 'archives' / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.partial')
        # A failed download never becomes a usable source archive.
        with urlopen(url, timeout=90) as response, temporary.open('wb') as output:
            while chunk := response.read(2**20):
                output.write(chunk)
        temporary.replace(path)
    with path.open('rb') as handle:
        check = hashlib.sha256()
        for chunk in iter(lambda: handle.read(2**20), b''):
            check.update(chunk)
        if check.hexdigest() != expected:
            raise ValueError(f'Upstream content changed: {filename}')
    return path


def image_bytes(image):
    return image['bytes'] if image.get('bytes') is not None else Path(image['path']).read_bytes()


def native_dataset(source, patterns):
    """The same pinned COCO directory and Arrow cache serve both annotation families."""
    directory = Path(snapshot_download(source[0], revision=source[1], repo_type='dataset',
        local_dir=ROOT / 'native' / source[0].replace('/', '--'), allow_patterns=patterns, max_workers=8))
    return directory


@cache
def coco_images():
    directory = native_dataset(COCO, ['data/images/train2017/*', 'data/images/val2017/*'])
    data = load_dataset('parquet', data_files={split: [str(p) for p in sorted(
        (directory / 'data/images' / native).glob('*.parquet'))]
        for split, native in [('train', 'train2017'), ('validation', 'val2017')]})
    return data.cast_column('image', Image(decode=False))


def pixel_box(box, size):
    """COCO xywh to visible xyxy; reject malformed/out-of-frame source boxes."""
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError('invalid_box')
    x, y, width, height = map(float, box)
    if not (0 <= x < x + width <= size[0] and 0 <= y < y + height <= size[1]):
        raise ValueError('out_of_frame_box')
    return [x, y, x + width, y + height]


def render_region(encoded, box, *, polygons=None, mask=None):
    """Uniform red marks carry location only; never category-dependent styling."""
    with PILImage.open(io.BytesIO(encoded)) as source:
        image = source.convert('RGB')
    visible = pixel_box(box, image.size)
    width = max(2, round(min(image.size) * RENDER_CONFIG['line_width_fraction']))
    if polygons is not None:
        mask = PILImage.new('L', image.size)
        draw = ImageDraw.Draw(mask)
        for polygon in polygons:
            if len(polygon) < 6 or len(polygon) % 2 or not all(math.isfinite(v) for v in polygon):
                raise ValueError('invalid_polygon')
            draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=255)
    if mask is None:
        ImageDraw.Draw(image).rectangle(visible, outline='red', width=width)
    else:
        if mask.size != image.size or not mask.getbbox():
            raise ValueError('invalid_mask')
        # Check that the selected native segment and annotation box agree.
        if max(abs(a-b) for a,b in zip(mask.getbbox(), visible)) > 2:
            raise ValueError('mask_box_mismatch')
        tint = mask.point(lambda value: round(value * RENDER_CONFIG['mask_opacity']))
        image = PILImage.composite(PILImage.new('RGB', image.size, 'red'), image, tint)
        boundary = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(2*width+1)), mask)
        image.paste((255, 0, 0), mask=boundary)
    output = io.BytesIO(); image.save(output, format='PNG')
    return output.getvalue(), {'type': 'region_highlight', **RENDER_CONFIG,
        'original_image_sha256': digest(encoded), 'image_size': list(image.size), 'box_xyxy': visible,
        'mask_sha256': digest(mask.tobytes()) if mask is not None else None,
        'pillow_version': PILImage.__version__}


def region_example(encoded, box, label, metadata, *, polygons=None, mask=None):
    marked, rendering = render_region(encoded, box, polygons=polygons, mask=mask)
    return {'images': [{'bytes': marked, 'path': None}],
            'inputs': 'Identify the region highlighted in red.', 'labels': label,
            'metadata': json.dumps({**metadata, 'augmentation': rendering}, sort_keys=True)}


def selected(rows, limit, seed):
    """Uniform stable hash selection over eligible source annotations, not a prefix."""
    key = lambda row: digest(f'{seed}:{row["source_row"]}'.encode())
    return heapq.nsmallest(limit, rows, key=key) if limit is not None else sorted(rows, key=key)


def materialize(generators, name, labels, source, excluded, conversion, license, license_url, config):
    features = Features({'images': Sequence(Image(decode=False)), 'inputs': Value('string'),
                         'labels': ClassLabel(names=labels), 'metadata': Value('string')})
    if config.get('multiple_choice'):
        features['choices_list'] = Sequence(Value('string'))
        features['labels'] = Value('int64')
    dataset = IterableDatasetDict({split: IterableDataset.from_generator(fn, features=features)
                                  for split, fn in generators.items()})
    result = _materialize(dataset, name, source[0], source[1], excluded, conversion, license, license_url)
    path = Path('build') / (name + '-release') / 'provenance.json'
    report = json.loads(path.read_text())
    report.update({'configuration': config,
        'partial_source': config['max_rows'] is not None or config['max_rows_eval'] is not None,
        'selection': 'stable hash of eligible source-row identity, independent of target class',
        'excluded_by_reason': dict(Counter(item['reason'] for item in excluded)),
        'conversion_file_sha256': digest(Path(__file__).read_bytes()),
        'label_names': labels, 'ontology_sha256': digest(json.dumps(labels).encode())})
    path.write_text(json.dumps(report, indent=2) + '\n')
    return result


def coco_regions(max_rows=5000, max_rows_eval=100, seed=0):
    """LVIS v1 (1,203 classes) and COCO Panoptic (133), sharing one COCO image cache.

    Image licenses remain per-image; NoDerivatives images are excluded from
    rendered mirrors. Only COCO train2017 images can enter training; evaluation
    uses the intersection of native annotation validation and COCO val2017.
    This deliberately excludes conflicting LVIS/COCO image partitions, rather
    than silently relabeling them. No unlabeled test images are included.
    """
    images = coco_images()
    indices = {split: {int(image_id): i for i, image_id in enumerate(rows['image_id'])}
               for split, rows in images.items()}
    image_info = {split: {row['image_id']: row for row in rows.select_columns(
        ['image_id', 'width', 'height', 'license'])} for split, rows in images.items()}
    directory = native_dataset(COCO, ['data/panoptic/*', 'data/panoptic_masks/*',
        'data/categories/panoptic*', 'data/licenses.parquet'])
    import pyarrow.parquet as pq
    licenses = {row['id']: row for row in pq.read_table(directory / 'data/licenses.parquet').to_pylist()}
    results, reports, all_excluded = {}, {}, []
    for family in ('lvis', 'panoptic'):
        generators, excluded = {}, []
        names = None
        for split, native in [('train', 'train2017'), ('validation', 'val2017')]:
            if family == 'lvis':
                filename = 'lvis_v1_' + ('train' if split == 'train' else 'val') + '.json.zip'
                with zipfile.ZipFile(archive_file(filename)) as z:
                    annotations = json.load(z.open(filename.removesuffix('.zip')))
                categories = {int(row['id']): row['name'] for row in annotations['categories']}
                objects = annotations['annotations']
            else:
                categories = {int(row['id']): row['name'] for row in pq.read_table(
                    directory / f'data/categories/panoptic_{native}.parquet').to_pylist()}
                table = pq.read_table(directory / f'data/panoptic/{native}/{native}.parquet').to_pylist()
                objects = [{**obj, 'image_id': row['image_id']} for row in table for obj in row['objects']]
            current_names = [categories[key] for key in sorted(categories)]
            assert len(current_names) == (1203 if family == 'lvis' else 133)
            assert names is None or names == current_names
            names = current_names
            label_indices = {key: index for index, key in enumerate(sorted(categories))}
            def eligible(objects=objects, split=split):
                for obj in objects:
                    identity = f"{family}:{obj['image_id']}:{obj['id']}"
                    if obj['image_id'] not in indices[split]:
                        excluded.append({'source_row': identity, 'reason': 'outside_matching_coco_split'})
                        continue
                    if obj.get('iscrowd'):
                        excluded.append({'source_row': identity, 'reason': 'crowd_region'})
                        continue
                    info = image_info[split][obj['image_id']]
                    if info['license'] in (3, 6):
                        excluded.append({'source_row': identity, 'reason': 'no_derivatives_image_license'})
                        continue
                    try:
                        pixel_box(obj['bbox'], (info['width'], info['height']))
                    except ValueError as error:
                        excluded.append({'source_row': identity, 'reason': str(error)})
                        continue
                    yield {**obj, 'source_row': identity}
            selection = selected(eligible(), max_rows if split == 'train' else max_rows_eval, seed)
            mask_data, mask_indices = None, None
            if family == 'panoptic':
                mask_data = load_dataset('parquet', data_files=[str(p) for p in sorted(
                    (directory / f'data/panoptic_masks/{native}').glob('*.parquet'))], split='train').cast_column('mask', Image(decode=False))
                mask_indices = {int(image_id): i for i, image_id in enumerate(mask_data['image_id'])}
            def examples(selection=selection, split=split, masks=mask_data, mask_indices=mask_indices,
                         family=family, label_indices=label_indices):
                for obj in selection:
                    row = images[split][indices[split][obj['image_id']]]
                    license = licenses[row['license']]
                    mask = None
                    if masks is not None:
                        with PILImage.open(io.BytesIO(image_bytes(masks[mask_indices[obj['image_id']]]['mask']))) as image:
                            rgb = np.asarray(image.convert('RGB'), dtype=np.uint32)
                            ids = rgb[:,:,0] + 256*rgb[:,:,1] + 65536*rgb[:,:,2]
                            mask = PILImage.fromarray(np.uint8(ids == obj['id']) * 255)
                    metadata = {'source_row': obj['source_row'], 'source_dataset': COCO[0],
                        'source_revision': COCO[1], 'image_group_id': f"coco2017:{obj['image_id']}",
                        'coco_image_id': obj['image_id'], 'annotation_id': obj['id'],
                        'source_category_id': obj['category_id'], 'image_license': license,
                        'image_attribution': {key: row[key] for key in ('flickr_url', 'coco_url', 'file_name')},
                        'annotation_source': family, 'bbox_xywh': obj['bbox']}
                    if family == 'lvis':
                        metadata['annotation_archive'] = dict(zip(('url', 'sha256'), ARCHIVES[
                            'lvis_v1_' + ('train' if split == 'train' else 'val') + '.json.zip']))
                    try:
                        yield region_example(image_bytes(row['image']), obj['bbox'], label_indices[obj['category_id']],
                            metadata, polygons=obj.get('segmentation') if family == 'lvis' else None, mask=mask)
                    except ValueError as error:
                        excluded.append({'source_row': obj['source_row'], 'reason': str(error)})
            generators[split] = examples
            # Release the full polygon table before rendering this family's sample.
        results[family] = materialize(generators, 'coco-regions-' + family, names, COCO, excluded,
            coco_regions, 'other', 'https://cocodataset.org/#termsofuse',
            {'max_rows': max_rows, 'max_rows_eval': max_rows_eval, 'seed': seed,
             'rendering': RENDER_CONFIG, 'coco_revision': COCO[1], 'family': family,
             'partition': 'native_annotation_split_intersect_same_coco_image_split'})
        report_dir = Path('build/coco-regions-' + family + '-release')
        reports[family] = json.loads((report_dir / 'provenance.json').read_text())
        all_excluded.extend(excluded)
    output = Path('build/coco-regions-release'); output.mkdir(exist_ok=True)
    (output / 'provenance.json').write_text(json.dumps(reports, indent=2) + '\n')
    (output / 'excluded-questions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in all_excluded))
    return results


def doclaynet_region(max_rows=5000, max_rows_eval=100, seed=0):
    """Native DocLayNet v1.1 page regions, with 11 source-defined layout classes.

    CDLA-Permissive-1.0. Native train/val/test partitions are preserved (val is
    named validation). PDF text cells never enter the inputs. A uniform red box
    identifies the selected region. The original page/document identity and
    geometry remain in metadata; out-of-frame boxes are excluded, not clipped.
    """
    directory = native_dataset(DOCLAYNET, ['data/*.parquet'])
    files = {split: [str(p) for p in sorted(
        (directory/'data').glob(native+'-*.parquet'))]
        for split,native in [('train','train'),('validation','val'),('test','test')]}
    # Inspect annotations without copying every full-resolution page to Arrow.
    data = load_dataset('parquet', data_files=files,
        columns=['metadata', 'bboxes', 'category_id'])
    generators, excluded, owners = {}, [], {}
    for split, rows in data.items():
        def eligible(rows=rows, split=split):
            for index, row in enumerate(rows.select_columns(['metadata', 'bboxes', 'category_id'])):
                info = row['metadata']; group = info['page_hash']
                document = (info['collection'], info['original_filename'])
                for key in (('page', group), ('document', document)):
                    if key in owners and owners[key] != split:
                        raise ValueError('DocLayNet native page/document crosses source splits')
                    owners[key] = split
                if len(row['bboxes']) != len(row['category_id']):
                    raise ValueError('DocLayNet box/label lengths differ')
                labels_by_box = defaultdict(set)
                for box, category in zip(row['bboxes'], row['category_id']):
                    labels_by_box[tuple(box)].add(category)
                for number,(box,category) in enumerate(zip(row['bboxes'],row['category_id'])):
                    identity = f'{group}:{number}'
                    if len(labels_by_box[tuple(box)]) != 1:
                        excluded.append({'source_row': identity, 'reason': 'conflicting_region_labels'})
                        continue
                    if not 1 <= category <= 11:
                        raise ValueError('Unknown DocLayNet source category')
                    try:
                        pixel_box(box, (info['coco_width'], info['coco_height']))
                    except ValueError as error:
                        excluded.append({'source_row': identity, 'reason': str(error)})
                        continue
                    yield {'index': index, 'annotation': number, 'bbox': box,
                           'label': category-1, 'source_row': identity}
        selection = selected(eligible(), max_rows if split=='train' else max_rows_eval, seed)
        def examples(selection=selection, rows=rows, paths=files[split]):
            import pyarrow.parquet as pq
            wanted = defaultdict(list)
            for item in selection:
                wanted[item['index']].append(item)
            def page_regions():
                offset = 0
                for path in paths:
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=['image']):
                        for index in sorted(wanted.keys() & set(range(offset, offset + len(batch)))):
                            encoded = image_bytes(batch.column('image')[index-offset].as_py())
                            for item in wanted[index]:
                                yield item, encoded
                        offset += len(batch)
            for item, encoded in page_regions():
                row=rows[item['index']]; info=row['metadata']
                metadata={**info, 'source_row':item['source_row'], 'image_group_id':'doclaynet:'+info['page_hash'],
                    'source_dataset':DOCLAYNET[0], 'source_revision':DOCLAYNET[1],
                    'source_category_id':item['label']+1, 'bbox_xywh':item['bbox'],
                    'source_license':'CDLA-Permissive-1.0', 'source_license_url':'https://cdla.io/permissive-1-0/'}
                try:
                    yield region_example(encoded, item['bbox'], item['label'], metadata)
                except ValueError as error:
                    excluded.append({'source_row':item['source_row'], 'reason':str(error)})
        generators[split]=examples
    return materialize(generators, 'doclaynet-region', LAYOUT_CLASSES, DOCLAYNET, excluded,
        doclaynet_region, 'cdla-permissive-1.0', 'https://cdla.io/permissive-1-0/',
        {'max_rows':max_rows,'max_rows_eval':max_rows_eval,'seed':seed,'rendering':RENDER_CONFIG})


def bapps_label(judgment):
    """Native judge is the fraction preferring p1. Two-vote ties have no hard gold."""
    value=float(judgment)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('invalid_human_judgment')
    return None if value == .5 else int(value > .5)


def bapps(max_rows=5000, max_rows_eval=100, seed=0):
    """BAPPS native 2AFC human preferences; ordered reference, p0, p1 images.

    Two training votes and five validation votes per triplet. Ties are excluded
    from the hard-choice view; fractions and vote counts remain in metadata.
    Only 2AFC is included, not JND. No perceptual model generates labels. Original
    code is BSD-2-Clause; it does not establish a license for the image dataset.
    """
    generators, excluded, reference_owners = {}, [], {}
    for split,filename in [('train','twoafc_train.tar.gz'),('validation','twoafc_val.tar.gz')]:
        path=archive_file(filename)
        def examples(path=path,split=split):
            with path.open('rb', buffering=8*2**20) as handle, tarfile.open(fileobj=handle,mode='r:*') as archive:
                members={member.name:member for member in archive if member.isfile()}
                def eligible():
                    for name,member in members.items():
                        if '/judge/' not in name or not name.endswith('.npy'):
                            continue
                        with archive.extractfile(member) as f:
                            fraction=float(np.load(io.BytesIO(f.read()),allow_pickle=False).item())
                        label=bapps_label(fraction)
                        if label is None:
                            excluded.append({'source_row':name,'reason':'tied_human_votes'})
                            continue
                        yield {'source_row':name,'fraction':fraction,'label':label}
                for item in selected(eligible(),max_rows if split=='train' else max_rows_eval,seed):
                    name=item['source_row']; stem=name.removesuffix('.npy')
                    paths=[stem.replace('/judge/',f'/{part}/')+'.png' for part in ('ref','p0','p1')]
                    images=[archive.extractfile(members[p]).read() for p in paths]
                    reference = digest(images[0])
                    if reference in reference_owners and reference_owners[reference] != split:
                        excluded.append({'source_row': name, 'reason': 'reference_image_crosses_splits'})
                        continue
                    reference_owners[reference] = split
                    metadata={'source_row':name,'source_dataset':'richzhang/PerceptualSimilarity',
                        'image_group_id':'bapps:'+reference, 'source_archive':dict(zip(('url','sha256'),ARCHIVES[filename])),
                        'image_roles':['reference','p0','p1'],'human_preference_p1':item['fraction'],
                        'annotators':2 if split=='train' else 5,'aggregation':'votes',
                        'source_license':'unspecified_dataset_license', 'code_license':'BSD-2-Clause'}
                    yield {'images':[{'bytes':data,'path':None} for data in images],
                        'inputs':'Image 1 is the reference. Which alternative looks closer to it?',
                        'choices_list':['Image 2','Image 3'],'labels':item['label'],
                        'metadata':json.dumps(metadata,sort_keys=True)}
        generators[split]=examples
    return materialize(generators,'bapps',[],('richzhang/PerceptualSimilarity',ARCHIVES['twoafc_train.tar.gz'][1]),
        excluded,bapps,'other','https://github.com/richzhang/PerceptualSimilarity',
        {'max_rows':max_rows,'max_rows_eval':max_rows_eval,'seed':seed,'multiple_choice':True})


def correspondence_example(source, target, source_point, target_point, metadata):
    with PILImage.open(io.BytesIO(source)) as decoded:
        marked=decoded.convert('RGB')
    with PILImage.open(io.BytesIO(target)) as decoded:
        target_size=decoded.size
    for point,size in [(source_point,marked.size),(target_point,target_size)]:
        if len(point)!=2 or not all(math.isfinite(v) and 0<=v<size[i] for i,v in enumerate(point)):
            raise ValueError('out_of_frame_keypoint')
    radius=max(4,round(min(marked.size)*.012)); x,y=source_point
    draw=ImageDraw.Draw(marked)
    draw.ellipse((x-radius-2,y-radius-2,x+radius+2,y+radius+2),fill='white')
    draw.line((x-radius,y,x+radius,y),fill='red',width=3)
    draw.line((x,y-radius,x,y+radius),fill='red',width=3)
    output=io.BytesIO(); marked.save(output,format='PNG')
    r,c=grid_labels((target_point[0]/target_size[0],target_point[1]/target_size[1]))[0]
    metadata={**metadata,'source_point':source_point,'target_point':target_point,
        'source_image_size':list(marked.size),'target_image_size':list(target_size),
        'augmentation':{'type':'source_keypoint_mark','version':1,'original_image_sha256':digest(source),
            'radius':radius,'encoding':'PNG','resize':None,'pillow_version':PILImage.__version__}}
    return {'images':[{'bytes':output.getvalue(),'path':None},{'bytes':target,'path':None}],
        'inputs':'The first image marks a keypoint with a red cross. Locate the corresponding point in the second image.',
        'labels':r*7+c,'metadata':json.dumps(metadata,sort_keys=True)}


def spair71k_grid(max_rows=5000, max_rows_eval=100, seed=0):
    """SPair-71k native correspondences: mark the source, classify the target 7x7 cell.

    Native train/validation/test image sets must be disjoint. Every supplied
    keypoint correspondence is eligible; a bounded mirror samples keypoints
    uniformly rather than selecting one fixed point per pair. Target images
    retain their original encoded bytes. PASCAL/Flickr source terms apply.
    """
    images, pairs= {}, defaultdict(list)
    with tarfile.open(archive_file('SPair-71k.tar.gz'),mode='r|*') as archive:
        for member in archive:
            if not member.isfile():continue
            if '/JPEGImages/' in member.name and member.name.endswith('.jpg'):
                images[member.name.split('/JPEGImages/')[1]]=archive.extractfile(member).read()
            elif '/PairAnnotation/' in member.name and member.name.endswith('.json'):
                split,name=member.name.split('/PairAnnotation/')[1].split('/',1)
                pairs[split].append((name,json.load(archive.extractfile(member))))
    excluded,generators,owners=[],{},{}
    for split,native in [('train','trn'),('validation','val'),('test','test')]:
        def eligible(native=native,split=split):
            for pair,row in pairs[native]:
                src=row['category']+'/'+row['src_imname']; trg=row['category']+'/'+row['trg_imname']
                for name in (src,trg):
                    if name in owners and owners[name]!=split:
                        raise ValueError('SPair native image crosses source splits')
                    owners[name]=split
                if not len(row['src_kps'])==len(row['trg_kps'])==len(row['kps_ids']):
                    raise ValueError('SPair correspondence arrays differ')
                for i,keypoint in enumerate(row['kps_ids']):
                    yield {'source_row':pair+':'+str(keypoint),'pair':pair,'keypoint':keypoint,
                        'source':src,'target':trg,'source_point':row['src_kps'][i],'target_point':row['trg_kps'][i]}
        selection=selected(eligible(),max_rows if split=='train' else max_rows_eval,seed)
        def examples(selection=selection):
            for row in selection:
                metadata={key:row[key] for key in ('source_row','pair','keypoint')}
                metadata.update({'image_group_id':'spair:'+row['pair'],'source_dataset':SPAIR[0],
                    'source_revision':SPAIR[1],'source_archive':dict(zip(('url','sha256'),ARCHIVES['SPair-71k.tar.gz'])),
                    'source_image':row['source'],'target_image':row['target'],'bins':[7,7],
                    'source_license':'PASCAL-VOC/Flickr terms','source_license_url':'https://cvlab.postech.ac.kr/research/SPair-71k/'})
                try:
                    yield correspondence_example(images[row['source']],images[row['target']],
                        row['source_point'],row['target_point'],metadata)
                except ValueError as error:
                    excluded.append({'source_row':row['source_row'],'reason':str(error)})
        generators[split]=examples
    return materialize(generators,'spair71k-grid',[f'r{i//7}c{i%7}' for i in range(49)],SPAIR,excluded,
        spair71k_grid,'other','https://cvlab.postech.ac.kr/research/SPair-71k/',
        {'max_rows':max_rows,'max_rows_eval':max_rows_eval,'seed':seed,'bins':[7,7],'native_image_split_disjoint':True})
