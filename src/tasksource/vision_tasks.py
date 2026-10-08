"""Visual tasks, using the shared loader and pinned, data-only Hub sources."""
import hashlib
import io
import json
import math
import random
import re
from PIL import Image as PILImage
from datasets import Features, Image, Sequence, Value, IterableDataset
from .preprocess import VisualClassification, VisualMultipleChoice


def figureqa_rows(dataset):
    """Expand native grouped QAs within each source split, without decoding images."""
    def flatten(batch):
        rows = [(image, qa) for image, qas in zip(batch['image'], batch['qa']) for qa in qas]
        return {'images': [[image] for image, qa in rows],
                'inputs': [qa['question'] for image, qa in rows],
                'labels': [qa['answer'].rstrip('.') for image, qa in rows]}
    features = Features({'images': Sequence(Image(decode=False)), 'inputs': Value('string'), 'labels': Value('string')})
    return type(dataset)({split: rows.map(flatten, batched=True, batch_size=32,
                                        remove_columns=list(rows.features),
                                        **({'features': features} if not isinstance(rows, IterableDataset) else {}))
                          for split, rows in dataset.items()})


def ai2d_rows(dataset):
    """Expand the Cauldron AI2D training subset's fixed Question/Choices format."""
    def flatten(batch):
        result = {'images': [], 'inputs': [], 'choices_list': [], 'labels': []}
        for images, texts in zip(batch['images'], batch['texts']):
            for qa in texts:
                question, options = qa['user'].split('\nChoices:\n')
                options = options.removesuffix('\nAnswer with the letter.')
                matches = re.findall(r'^([A-Z])\. (.*)$', options, re.MULTILINE)
                answer = qa['assistant'].removeprefix('Answer: ').strip()
                if not matches or [letter for letter, _ in matches] != list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(matches)]) or answer not in [letter for letter, _ in matches]:
                    raise ValueError('Unexpected pinned AI2D option format')
                result['images'].append(images)
                result['inputs'].append(question.removeprefix('Question: '))
                result['choices_list'].append([text for _, text in matches])
                result['labels'].append(ord(answer) - ord('A'))
        return result
    features = Features({'images': Sequence(Image(decode=False)), 'inputs': Value('string'),
                         'choices_list': Sequence(Value('string')), 'labels': Value('int64')})
    return type(dataset)({split: rows.map(flatten, batched=True, batch_size=32,
                                        remove_columns=list(rows.features),
                                        **({'features': features} if not isinstance(rows, IterableDataset) else {}))
                          for split, rows in dataset.items()})


nlvr2 = VisualClassification(
    images=lambda x: [x['image0'], x['image1']], inputs='sentence', labels='label',
    dataset_name='pingzhili/nlvr2', task_id='nlvr2', metadata=lambda x: {'image_group': x['identifier'].rsplit('-', 1)[0]}, label_values={'False': 'False', 'True': 'True'},
    question='Is the statement true for these two images?',
    load_dataset_kwargs={'revision': '6ad9994db49bf1162feaecacc5554e5ca4a36487'},
)

snli_ve = VisualClassification(
    images=lambda x: [x['image']], inputs='sentence', labels='gold_label',
    dataset_name='pingzhili/snli-ve', task_id='snli-ve', metadata=lambda x: {'image_id': x['Flickr30K_ID']},
    question='Does the image entail, contradict, or leave the statement neutral?',
    label_values={'entailment': 'entailment', 'neutral': 'neutral', 'contradiction': 'contradiction'},
    load_dataset_kwargs={'revision': '176e5ba43a2219043ebdaa057d9aaf42d2f34dc8'},
)

aokvqa = VisualMultipleChoice(
    images=lambda x: [x['image']], inputs='question', choices_list='choices', labels='correct_choice_idx',
    dataset_name='HuggingFaceM4/A-OKVQA', task_id='aokvqa', metadata=lambda x: {'question_id': x['question_id']}, splits=('train', 'validation', None),
    load_dataset_kwargs={'revision': 'd1b0efa3a436e9101dfbde3752db7607da696c35'},
)

scienceqa_img = VisualMultipleChoice(
    images=lambda x: [x['image']], inputs=lambda x: '\n'.join(filter(None, [x['hint'], x['question']])),
    choices_list='choices', labels='answer', dataset_name='derek-thomas/ScienceQA', task_id='scienceqa-img',
    pre_process=lambda ds: ds.filter(lambda x: x['image'] is not None),
    load_dataset_kwargs={'revision': 'f18b0a70359ebfb41f658fd564208d0355b013f4'},
)

ai2d = VisualMultipleChoice(
    choices_list='choices_list', dataset_name='HuggingFaceM4/the_cauldron', config_name='ai2d', task_id='ai2d',
    pre_process=ai2d_rows,
    load_dataset_kwargs={'revision': '847a98a779b1652d65111daf20c972dfcd333605'},
)

figureqa = VisualClassification(
    dataset_name='vikhyatk/figureqa', task_id='figureqa', pre_process=figureqa_rows,
    label_values={'No': 'No', 'Yes': 'Yes'}, question='Answer the question about the chart.',
    load_dataset_kwargs={'revision': '1afe55949decfebe4ccaaa0ce175ff4bce28d24c'},
)


# Mind2Web stores operations, candidates and candidate attributes as nested JSON.
_MIND2WEB_REVISION = '1b4c6a8cf9f77b7a5e0d641959935c80c4a05889'


def grid_labels(point, bins=(7, 7), depth=1):
    """Row/column labels from a normalized point, including nested cell refinements."""
    if (len(bins) != 2 or any(type(n) is not int or n <= 0 for n in bins)
            or type(depth) is not int or depth <= 0):
        raise ValueError('Grid dimensions and depth must be positive integers')
    x, y = point
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
        raise ValueError('Point coordinates must be finite and in [0, 1]')
    nx, ny = bins
    cells = []
    for _ in range(depth):
        c, r = min(int(x * nx), nx - 1), min(int(y * ny), ny - 1)
        cells.append((r, c))
        x, y = x * nx - c, y * ny - r
    return cells


def grid_center(cells, bins=(7, 7)):
    """Decode a cell path to its center in the original normalized image."""
    grid_labels((0, 0), bins, len(cells))  # validate grid and depth
    x = y = 0.5
    nx, ny = bins
    for r, c in reversed(cells):
        if not (type(r) is int and type(c) is int and 0 <= r < ny and 0 <= c < nx):
            raise ValueError('Cell outside grid')
        x, y = (c + x) / nx, (r + y) / ny
    return x, y


def prepare_mind2web(dataset):
    """One eligible action pool for all views; boxes supervise element grounding."""
    def decode(value):
        return json.loads(value) if isinstance(value, str) else value

    def candidate(value, width, height):
        item = decode(value)
        attrs = decode(item['attributes'])
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

    def rows(batch):
        output = {k: [] for k in ('images', 'inputs', 'action', 'x10', 'y10', 'grid5', 'grid7',
                                  'choices_list', 'target_index', 'metadata')}
        for source in (dict(zip(batch, values)) for values in zip(*batch.values())):
            operation = decode(source['operation'])
            if operation['op'] not in ('CLICK', 'TYPE', 'SELECT'):
                continue
            image = source['screenshot']
            # Read the encoded header for dimensions; never decode the pixel array.
            with PILImage.open(io.BytesIO(image['bytes']) if image.get('bytes') is not None else image['path']) as header:
                width, height = header.size
            positives = [decode(v) for v in source['pos_candidates']]
            originals = [v for v in positives if v.get('is_original_target') is True]
            if len(originals) != 1:
                continue
            try:
                gold = candidate(originals[0], width, height)
            except (ValueError, KeyError, TypeError):
                continue
            if gold is None:
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
                continue
            identity = f"{source['annotation_id']}:{source['action_uid']}"
            rng = random.Random(int(hashlib.sha256(identity.encode()).hexdigest(), 16))
            options = [gold] + rng.sample(sorted(negatives.values(), key=lambda v: v[1]), min(23, len(negatives)))
            rng.shuffle(options)
            target = next(i for i, option in enumerate(options) if option[1] == gold[1])
            index = int(source['target_action_index'])
            history = source['action_reprs']
            if not 0 <= index < len(history):
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
        'action': Value('string'), **{k: Value('int64') for k in ('x10', 'y10', 'grid5', 'grid7', 'target_index')},
        'choices_list': Sequence(Value('string')), 'metadata': Value('string')})
    return type(dataset)({split: rows_.map(rows, batched=True, batch_size=16,
        remove_columns=list(rows_.features), features=features)
        for split, rows_ in dataset.items()})


_MIND2WEB = dict(dataset_name='osunlp/Multimodal-Mind2Web', pre_process=prepare_mind2web,
    load_dataset_kwargs={'revision': _MIND2WEB_REVISION})

mind2web_action = VisualClassification(
    **_MIND2WEB, task_id='mind2web/action', labels='action',
    label_values={v: v for v in ('CLICK', 'TYPE', 'SELECT')},
    question='What kind of GUI action should be performed next?',
    metadata=lambda x: {**json.loads(x['metadata']), 'question_id': 'action'},
)

mind2web_element = VisualMultipleChoice(
    **_MIND2WEB, task_id='mind2web/element', choices_list='choices_list', labels='target_index',
    question='Which candidate element should be acted on next?',
    metadata=lambda x: {**json.loads(x['metadata']), 'question_id': 'element'},
)

mind2web_x10 = VisualClassification(
    **_MIND2WEB, task_id='mind2web/x10', labels='x10', ordinal=True, score_only=True,
    label_values={i: f'x{i}' for i in range(10)},
    question='Horizontal target position: ten equal-width bins, left to right.',
    metadata=lambda x: {**json.loads(x['metadata']), 'question_id': 'x10'},
)

mind2web_y10 = VisualClassification(
    **_MIND2WEB, task_id='mind2web/y10', labels='y10', ordinal=True, score_only=True,
    label_values={i: f'y{i}' for i in range(10)},
    question='Vertical target position: ten equal-height bins, top to bottom.',
    metadata=lambda x: {**json.loads(x['metadata']), 'question_id': 'y10'},
)

mind2web_grid5 = VisualClassification(
    **_MIND2WEB, task_id='mind2web/grid5', labels='grid5',
    label_values={i: f'r{i // 5}c{i % 5}' for i in range(25)},
    question='Which cell contains the target center in a 5-row, 5-column grid over the full screenshot?',
    metadata=lambda x: {**json.loads(x['metadata']), 'question_id': 'grid5'},
)

mind2web_grid7 = VisualClassification(
    **_MIND2WEB, task_id='mind2web/grid7', labels='grid7',
    label_values={i: f'r{i // 7}c{i % 7}' for i in range(49)},
    question='Which cell contains the target center in a 7-row, 7-column grid over the full screenshot?',
    metadata=lambda x: {**json.loads(x['metadata']), 'question_id': 'grid7'},
)
