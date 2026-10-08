"""Pinned visual source conversions; expensive preparation runs once before upload."""
import hashlib
import inspect
import io
import json
import math
import random
import re
from pathlib import Path
from PIL import Image as PILImage
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value, IterableDataset, load_dataset
from tasksource.preprocess import disable_image_decoding
from tasksource.vision_tasks import grid_labels

CAULDRON_REVISION = '847a98a779b1652d65111daf20c972dfcd333605'
_MIND2WEB_REVISION = '1b4c6a8cf9f77b7a5e0d641959935c80c4a05889'

def prepare_cauldron(dataset, config, excluded=None):
    """Parse prompt-encoded MCQs once, retaining one ordered image set per group."""
    def convert(row, index):
        images = row['images']
        group = hashlib.sha256(b''.join(hashlib.sha256(image['bytes']).digest() for image in images)).hexdigest()
        qas = []
        for qa_index, qa in enumerate(row['texts']):
            identity = f'{config}:{index}:{qa_index}'
            if '\nChoices:\n' not in qa['user']:
                if config != 'iconqa':
                    raise ValueError('Unexpected pinned AI2D question format')
                if excluded is not None:
                    excluded.append({'id': identity, 'reason': 'no explicit text choices'})
                continue
            question, options = qa['user'].split('\nChoices:\n')
            matches = re.findall(r'^([A-Z])\. (.*)$', options.removesuffix('\nAnswer with the letter.'), re.MULTILINE)
            answer = qa['assistant'].removeprefix('Answer: ').strip()
            keys = [letter for letter, _ in matches]
            if not matches or keys != list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(matches)]) or answer not in keys:
                raise ValueError('Unexpected pinned Cauldron option format')
            source_options = dict(matches)
            gold = source_options[answer]
            if list(source_options.values()).count(gold) != 1:
                if excluded is not None:
                    excluded.append({'id': identity, 'reason': 'ambiguous duplicate gold option'})
                continue
            kept = {}
            for key, value in matches:
                if value not in kept.values():
                    kept[key] = value
            metadata = {'id': identity, 'source_dataset': config, 'source_revision': CAULDRON_REVISION,
                        'source_options': source_options, 'source_answer': answer, 'option_keys': list(kept)}
            qas.append({'inputs': question.removeprefix('Question: '), 'choices_list': list(kept.values()),
                        'labels': list(kept).index(answer), 'metadata': json.dumps(metadata, sort_keys=True)})
        return {'images': images, 'image_group_id': group, 'qa': qas}
    features = Features({'images': Sequence(Image(decode=False)), 'image_group_id': Value('string'),
        'qa': [{'inputs': Value('string'), 'choices_list': Sequence(Value('string')),
                'labels': Value('int64'), 'metadata': Value('string')}]})
    return type(dataset)({split: rows.map(convert, with_indices=True, remove_columns=list(rows.features),
        features=features).filter(lambda row: bool(row['qa'])) for split, rows in dataset.items()})


def prepare_mind2web(dataset, excluded=None):
    """One eligible action pool for all views; boxes supervise element grounding."""
    def decode(value):
        return json.loads(value) if isinstance(value, str) else value

    def candidate(value, width, height):
        item = decode(value)
        attrs = decode(item['attributes'])
        if not isinstance(attrs, dict):
            return None
        box = [float(v) for v in attrs.get('bounding_box_rect', '').split(',')]
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            return None
        x, y, w, h = box
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            return None
        node = str(item['backend_node_id'])
        # Every candidate uses the same visible/DOM attributes, never positive flags.
        description = json.dumps({'tag': item['tag'], 'node': node, 'bbox': box,
            **{k: attrs[k] for k in ('aria_label', 'placeholder', 'title', 'role', 'type', 'id', 'text') if attrs.get(k)}},
            ensure_ascii=False, sort_keys=True)
        return item, node, box, description

    def reject(source, reason):
        if excluded is not None:
            excluded.append({'source_row': f"{source['annotation_id']}:{source['action_uid']}", 'reason': reason})

    def rows(batch):
        output = {k: [] for k in ('images', 'inputs', 'action', 'x10', 'y10', 'grid5', 'grid7',
                                  'choices_list', 'labels', 'metadata')}
        for source in (dict(zip(batch, values)) for values in zip(*batch.values())):
            operation = decode(source['operation'])
            if operation['op'] not in ('CLICK', 'TYPE', 'SELECT'):
                reject(source, 'unsupported operation')
                continue
            image = source['screenshot']
            if image is None or not (image.get('bytes') or image.get('path')):
                reject(source, 'missing screenshot')
                continue
            # Read the encoded header for dimensions; never decode the pixel array.
            try:
                with PILImage.open(io.BytesIO(image['bytes']) if image.get('bytes') is not None else image['path']) as header:
                    width, height = header.size
            except (OSError, ValueError):
                reject(source, 'invalid screenshot header')
                continue
            positives = [decode(v) for v in source['pos_candidates']]
            originals = [v for v in positives if v.get('is_original_target') is True]
            if len(originals) != 1:
                reject(source, 'no unambiguous original target')
                continue
            try:
                gold = candidate(originals[0], width, height)
            except (ValueError, KeyError, TypeError):
                reject(source, 'malformed target candidate')
                continue
            if gold is None:
                reject(source, 'target box outside screenshot')
                continue
            positive_ids = {str(v['backend_node_id']) for v in positives}
            negatives = {}
            for value in source['neg_candidates']:
                try:
                    option = candidate(value, width, height)
                except (ValueError, KeyError, TypeError):
                    continue
                if option and option[1] not in positive_ids and not option[0].get('is_original_target'):
                    negatives[option[1]] = option
            if not negatives:
                reject(source, 'no valid negative candidates')
                continue
            identity = f"{source['annotation_id']}:{source['action_uid']}"
            rng = random.Random(int(hashlib.sha256(identity.encode()).hexdigest(), 16))
            options = [gold] + rng.sample(sorted(negatives.values(), key=lambda v: v[1]), min(23, len(negatives)))
            rng.shuffle(options)
            target = next(i for i, option in enumerate(options) if option[1] == gold[1])
            index = int(source['target_action_index'])
            history = source['action_reprs']
            if not 0 <= index < len(history):
                reject(source, 'invalid action history index')
                continue
            context = source['confirmed_task']
            if index:
                context += '\nPast actions:\n' + '\n'.join(history[:index])
            x, y, w, h = gold[2]
            point = ((x + w / 2) / width, (y + h / 2) / height)
            r10, c10 = grid_labels(point, (10, 10))[0]
            r5, c5 = grid_labels(point, (5, 5))[0]
            r7, c7 = grid_labels(point, (7, 7))[0]
            metadata = {'group': 'osunlp/Multimodal-Mind2Web@' + _MIND2WEB_REVISION, 'source_row': identity,
                'action_uid': source['action_uid'], 'trajectory_id': source['annotation_id'],
                'website': source['website'], 'domain': source['domain'], 'bbox': gold[2],
                'image_size': [width, height], 'point': point, 'coordinate_source': 'original_target_bbox_center',
                'backend_node_id': gold[1], 'original_op': operation['original_op']}
            values = [[image], context, operation['op'], c10, r10, r5 * 5 + c5, r7 * 7 + c7,
                      [option[3] for option in options], target, json.dumps(metadata, sort_keys=True)]
            for key, value in zip(output, values):
                output[key].append(value)
        return output

    features = Features({'images': Sequence(Image(decode=False)), 'inputs': Value('string'),
        'action': Value('string'), **{k: Value('int64') for k in ('x10', 'y10', 'grid5', 'grid7', 'labels')},
        'choices_list': Sequence(Value('string')), 'metadata': Value('string')})
    return type(dataset)({split: rows_.map(rows, batched=True, batch_size=16,
        remove_columns=list(rows_.features), features=features)
        for split, rows_ in dataset.items()})


def view2space():
    """VIEW2SPACE's native training MCQs, grouped by ordered source images.

    Source: Pokerme/view2space-train at 1af36aa39a18143db6599a049da6de657e18215c.
    Only explicit multiple-choice questions are included; counting and detection are
    excluded. Original PNG bytes, image order, question IDs, input boxes and reasoning
    are preserved. Reasoning is metadata, never question input. Related QAs share one
    image group to avoid repeating image bytes for every question. The source has only
    a training split; no evaluation split is fabricated. The source README states CC BY 4.0.
    Duplicate distractors are deduplicated with gold indices remapped; ambiguous duplicate
    gold options are excluded and recorded in excluded-questions.jsonl.
    """
    import hashlib
    import zipfile
    import shutil
    from huggingface_hub import hf_hub_download

    repo, revision = 'Pokerme/view2space-train', '1af36aa39a18143db6599a049da6de657e18215c'
    root = Path('build/view2space-original')
    root.mkdir(parents=True, exist_ok=True)
    paths = {name: hf_hub_download(repo, name, repo_type='dataset', revision=revision, local_dir=root)
             for name in ('overall.jsonl', 'images.zip')}
    groups, excluded, deduplicated = {}, [], 0
    with open(paths['overall.jsonl']) as handle:
        for line in handle:
            row = json.loads(line)
            if row['q_type'] != 'mcq':
                continue
            options = row['options']
            if not 2 <= len(options) <= 26 or row['answer'] not in options:
                raise ValueError(f"Invalid source options: {row['q_idx']}")
            gold = options[row['answer']]
            if list(options.values()).count(gold) != 1:
                excluded.append({'id': row['q_idx'], 'reason': 'ambiguous duplicate gold option'})
                continue
            kept = {}
            for key, value in options.items():
                if value not in kept.values():
                    kept[key] = value
            deduplicated += len(kept) != len(options)
            supporting = row.get('supporting') or {}
            groups.setdefault(tuple(row['image_paths']), []).append({
                'inputs': '\n'.join(filter(None, [row['question'], row['question_prompt'],
                    'Image files in order: ' + json.dumps(row['image_paths']) if supporting.get('draw_boxes') else '',
                    'Input boxes: ' + json.dumps(supporting['draw_boxes'], sort_keys=True) if supporting.get('draw_boxes') else ''])),
                'choices_list': list(kept.values()), 'labels': list(kept).index(row['answer']),
                'metadata': json.dumps({'id': row['q_idx'], 'image_paths': row['image_paths'],
                    'source_answer': row['answer'], 'option_keys': list(kept), 'source_options': options,
                    'draw_boxes': supporting.get('draw_boxes'),
                    'reasoning': supporting.get('chain_of_thought'), 'source_revision': revision},
                    ensure_ascii=False, sort_keys=True)})
    output = Path('build/view2space-release-imagefolder')
    output.mkdir(parents=True, exist_ok=True)
    conversion = hashlib.sha256(inspect.getsource(view2space).encode()).hexdigest()
    # Keep each original PNG once. Native ImageFolder supports ZIP + metadata.jsonl.
    packaged = output / 'data.zip'
    if not packaged.exists() or packaged.stat().st_size != Path(paths['images.zip']).stat().st_size:
        shutil.copyfile(paths['images.zip'], packaged)
    with zipfile.ZipFile(packaged, 'a') as archive:
        members = {name[name.index('images/'):]: name for name in archive.namelist()
                   if 'images/' in name and not name.endswith('/')}
        needed = {name for images in groups for name in images}
        # Append one canonical manifest while retaining the original compressed images.
        manifest = zipfile.ZipInfo('metadata.jsonl')  # fixed timestamp for reproducible bytes
        manifest.compress_type = zipfile.ZIP_DEFLATED
        with archive.open(manifest, 'w', force_zip64=True) as handle:
            for images, qas in groups.items():
                handle.write((json.dumps({'file_names': [members[name] for name in images],
                    'image_group_id': hashlib.sha256('\x1f'.join(images).encode()).hexdigest(),
                    'qa': qas}, ensure_ascii=False) + '\n').encode())
    report = {'source': repo, 'revision': revision, 'rows': len(groups), 'images': len(needed),
              'questions': sum(len(qas) for qas in groups.values()), 'license': 'cc-by-4.0',
              'format': 'imagefolder ZIP: ordered images + grouped canonical MC QAs', 'format_version': 1,
              'conversion_sha256': conversion, 'excluded_questions': len(excluded),
              'deduplicated_distractor_rows': deduplicated}
    output.joinpath('provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    output.joinpath('excluded-questions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in excluded))
    # Native imagefolder maps file_names to Sequence(Image); no custom loader.
    output.joinpath('README.md').write_text('---\nlicense: cc-by-4.0\nconfigs:\n'
        '- config_name: default\n  data_files:\n  - split: train\n'
        '    path: data.zip\n---\n')
    return output



def _materialize(dataset, name, source, revision, excluded, conversion, license, license_url):
    """Write bounded-memory Parquet conversions and their reproducibility evidence."""
    output = Path('build') / (name + '-release')
    output.mkdir(parents=True, exist_ok=True)
    import tempfile
    (output / 'cache').mkdir(exist_ok=True)
    cache = tempfile.mkdtemp(prefix='run-', dir=output / 'cache')
    result, counts = {}, {}
    for split, rows in dataset.items():
        def examples(rows=rows):
            yield from rows
        result[split] = Dataset.from_generator(examples, features=rows.features,
            cache_dir=cache)
        result[split].to_parquet(str(output / (split + '.parquet')))
        counts[split] = {'rows': len(result[split]), 'questions': sum(len(qas) for qas in result[split]['qa'])
                         if 'qa' in result[split].features else len(result[split])}
        print(name, split, counts[split], flush=True)
    assert counts.get('train', {}).get('questions'), 'Mirror requires labeled training examples'
    code = inspect.getsource(conversion) + inspect.getsource(_materialize) + inspect.getsource(grid_labels)
    report = {'source': source, 'revision': revision, 'format_version': 1, 'splits': counts,
        'conversion_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'excluded_rows': len(excluded), 'license': license, 'license_url': license_url}
    (output / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'excluded-questions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in excluded))
    return DatasetDict(result)



def _parquet_source(repo, revision, directory, pattern, columns=None):
    """Cache pinned source shards once, avoiding repeated remote byte-range requests."""
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(repo, repo_type='dataset', revision=revision,
        local_dir=directory, allow_patterns=pattern, max_workers=8))
    files = {}
    for path in sorted(root.glob(pattern)):
        files.setdefault(path.name.split('-')[0], []).append(str(path))
    return disable_image_decoding(load_dataset('parquet', data_files=files, columns=columns, streaming=True))

def ai2d():
    """AI2D's Cauldron training release, with prompt-encoded choices parsed once.

    Source: HuggingFaceM4/the_cauldron, ai2d config, pinned at
    847a98a779b1652d65111daf20c972dfcd333605. Original encoded images and question
    order are retained. Canonical QAs share one ordered image set. No splits are
    fabricated. Ambiguous duplicate gold options are recorded and excluded;
    repeated distractors are deduplicated with gold indices remapped.
    Original AI2D license: CC BY-SA 4.0, as linked by the AWS dataset registry:
    https://registry.opendata.aws/allenai-diagrams/
    """
    source = _parquet_source('HuggingFaceM4/the_cauldron', CAULDRON_REVISION,
        'build/cauldron-original', 'ai2d/*.parquet')
    excluded = []
    return _materialize(prepare_cauldron(source, 'ai2d', excluded), 'ai2d',
        'HuggingFaceM4/the_cauldron', CAULDRON_REVISION, excluded, prepare_cauldron,
        'cc-by-sa-4.0', 'https://registry.opendata.aws/allenai-diagrams/')


def iconqa_text():
    """IconQA text-choice QAs from the pinned Cauldron training release.

    Source: HuggingFaceM4/the_cauldron, iconqa config, pinned at
    847a98a779b1652d65111daf20c972dfcd333605. Open-answer questions are excluded;
    image-choice questions are not included. Original image bytes and text-choice
    option order are preserved. Canonical QAs share one image set. No splits are
    fabricated. Duplicate gold options are excluded, repeated distractors deduplicated.
    Original IconQA license: CC BY-NC-SA 4.0:
    https://github.com/lupantech/IconQA#license
    """
    source = _parquet_source('HuggingFaceM4/the_cauldron', CAULDRON_REVISION,
        'build/cauldron-original', 'iconqa/*.parquet')
    excluded = []
    return _materialize(prepare_cauldron(source, 'iconqa', excluded), 'iconqa-text',
        'HuggingFaceM4/the_cauldron', CAULDRON_REVISION, excluded, prepare_cauldron,
        'cc-by-nc-sa-4.0', 'https://github.com/lupantech/IconQA#license')


def mind2web():
    """Multimodal Mind2Web's eligible GUI actions, prepared once for six annotations.

    Source: osunlp/Multimodal-Mind2Web at 1b4c6a8cf9f77b7a5e0d641959935c80c4a05889.
    Retains native train, test_task, test_website and test_domain splits, encoded
    screenshots, stable action IDs and geometry metadata. Inputs contain only the
    task and past actions. Candidate descriptions omit positive flags. A uniquely
    marked original target with a valid on-screen box and valid negatives is required.
    At most 23 negatives are sampled and option order shuffled using the action ID.
    Canonical labels supervise element selection; action/x10/y10/grid5/grid7 fields
    provide the derived annotations. No evaluation splits are fabricated.
    Original Hub card license: OpenRAIL.
    """
    columns = ['screenshot', 'operation', 'pos_candidates', 'neg_candidates', 'action_uid',
               'annotation_id', 'confirmed_task', 'action_reprs', 'target_action_index', 'website', 'domain']
    source = _parquet_source('osunlp/Multimodal-Mind2Web', _MIND2WEB_REVISION,
        'build/mind2web-original', 'data/*.parquet', columns)
    excluded = []
    return _materialize(prepare_mind2web(source, excluded), 'multimodal-mind2web',
        'osunlp/Multimodal-Mind2Web', _MIND2WEB_REVISION, excluded, prepare_mind2web,
        'openrail', 'https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web')
