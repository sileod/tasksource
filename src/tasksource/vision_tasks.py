"""Visual tasks, using the shared loader and pinned, data-only Hub sources."""
import hashlib
import json
import math
import re
from datasets import Features, Image, Sequence, Value, IterableDataset
from .preprocess import VisualClassification, VisualMultipleChoice
from .grounding import grid_labels, grid_center


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


def grouped_qa_rows(dataset, view=None, multiple_choice=False):
    """Flatten canonical QAs sharing encoded images, optionally selecting a source view."""
    def flatten(batch):
        rows = [(images, group, qa) for images, group, qas in zip(
            batch['images'], batch['image_group_id'], batch['qa']) for qa in qas if view is None or qa['view'] == view]
        return {'images': [images for images, group, qa in rows],
                **{key: [qa[key] for images, group, qa in rows] for key in (('inputs', 'choices_list', 'labels') if multiple_choice else ('inputs', 'labels'))},
                'metadata': [json.dumps({**json.loads(qa['metadata']), 'image_group_id': group},
                    ensure_ascii=False, sort_keys=True) for images, group, qa in rows]}
    features = Features({'images': Sequence(Image(decode=False)), 'inputs': Value('string'),
        'labels': Value('int64' if multiple_choice else 'string'), 'metadata': Value('string')})
    if multiple_choice:
        features['choices_list'] = Sequence(Value('string'))
    return type(dataset)({split: rows.map(flatten, batched=True, batch_size=16,
        remove_columns=list(rows.features), features=features) for split, rows in dataset.items()})


def grouped_mc_rows(dataset):
    return grouped_qa_rows(dataset, multiple_choice=True)


def normalize_cauldron_answer(answer):
    return re.sub(r'^Answer:\s*', '', answer.strip(), flags=re.I).casefold().rstrip('.').strip()


def parse_cauldron_mc(qa):
    """Native prompt-encoded options only; return a canonical QA or an audit reason."""
    if '\nChoices:\n' not in qa['user']:
        return None, 'no_explicit_choices'
    question, options = qa['user'].rsplit('\nChoices:\n', 1)
    if not options.endswith('\nAnswer with the letter.'):
        return None, 'invalid_option_format'
    options = options.removesuffix('\nAnswer with the letter.')
    markers = list(re.finditer(r'^([A-Z])\.\s+', options, re.M))
    letters = [marker[1] for marker in markers]
    choices = [options[marker.end():markers[i + 1].start() if i + 1 < len(markers) else None].strip()
               for i, marker in enumerate(markers)]
    if not 2 <= len(choices) <= 26 or letters != list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(choices)]):
        return None, 'invalid_option_format'
    if not all(choices):
        return None, 'empty_option'
    if len(set(choice.casefold() for choice in choices)) != len(choices):
        return None, 'duplicate_options'
    answer = normalize_cauldron_answer(qa['assistant']).upper()
    if answer not in letters:
        return None, 'invalid_gold'
    inputs = question.removeprefix('Question: ').strip()
    if not inputs:
        return None, 'empty_question'
    return {'inputs': inputs, 'choices_list': choices, 'labels': letters.index(answer)}, None


# Original data terms are separate from repository software licenses.
_CAULDRON_LICENSE_EVIDENCE = {
    'clevr': {'source_license': 'cc-by-4.0', 'source_license_url': 'https://cs.stanford.edu/people/jcjohns/clevr/'},
    'mapqa': {'source_license': 'cc-by-sa-4.0', 'source_license_url': 'https://github.com/OSU-slatelab/MapQA#citation'},
    'tqa': {'source_license': 'cc-by-sa-4.0', 'source_license_url': 'https://registry.opendata.aws/allenai-tqa/'},
    'visual7w': {'source_license': 'unspecified', 'source_license_url': 'https://ai.stanford.edu/~yukez/visual7w/',
                 'source_code_license': 'mit'},
    'intergps': {'source_license': 'unspecified', 'source_license_url': 'https://github.com/lupantech/InterGPS',
                'source_code_license': 'mit'},
    'tallyqa': {'source_license': 'unspecified', 'source_license_url': 'https://github.com/manoja328/TallyQA_dataset'},
    'vsr': {'source_license': 'apache-2.0', 'source_license_url': 'https://github.com/cambridgeltl/visual-spatial-reasoning',
            'image_license_note': 'COCO images retain their individual original licenses.'},
}


def _cauldron_rows(dataset, allowed_answers=None):
    """Flatten one native source representation, retaining ordered image identities."""
    from .preprocess import disable_image_decoding
    allowed = set(allowed_answers) if allowed_answers is not None else None
    def flatten(batch):
        output = {key: [] for key in ('images', 'inputs', 'labels', 'metadata')}
        if allowed is None:
            output['choices_list'] = []
        for images, texts in zip(batch['images'], batch['texts']):
            if not images or any(image is None or not (image.get('bytes') or image.get('path')) for image in images):
                continue
            identities = [hashlib.sha256(image['bytes']).hexdigest() if image.get('bytes') is not None else image['path'] for image in images]
            group = hashlib.sha256(json.dumps(identities).encode()).hexdigest()
            for index, qa in enumerate(texts):
                if allowed is None:
                    record, reason = parse_cauldron_mc(qa)
                    if reason:
                        continue
                else:
                    answer = normalize_cauldron_answer(qa['assistant'])
                    if answer not in allowed:
                        continue
                    record = {'inputs': qa['user'].removeprefix('Question: ').strip(), 'labels': answer}
                record['images'] = images
                record['metadata'] = json.dumps({'image_group_id': group, 'id': f'{group}:{index}',
                    'source_dataset': qa['source'], 'source_answer': qa['assistant'], 'source_question': qa['user'],
                    **_CAULDRON_LICENSE_EVIDENCE.get(qa['source'].casefold(), {})},
                    ensure_ascii=False, sort_keys=True)
                for key in output:
                    output[key].append(record[key])
        return output
    features = Features({'images': Sequence(Image(decode=False)), 'inputs': Value('string'),
                         'labels': Value('string' if allowed is not None else 'int64'), 'metadata': Value('string')})
    if allowed is None:
        features['choices_list'] = Sequence(Value('string'))
    dataset = disable_image_decoding(dataset)
    return type(dataset)({split: rows.map(flatten, batched=True, batch_size=16,
        remove_columns=list(rows.features), features=features) for split, rows in dataset.items()})


def cauldron_mc_rows(dataset):
    return _cauldron_rows(dataset)


def cauldron_closed_rows(dataset, allowed_answers):
    return _cauldron_rows(dataset, allowed_answers)


nlvr2 = VisualClassification(
    images=lambda x: [x['image0'], x['image1']], inputs='sentence', labels='label',
    dataset_name='pingzhili/nlvr2', task_id='nlvr2', metadata=lambda x: {'image_group': x['identifier'].rsplit('-', 1)[0]}, label_values={'False': 'False', 'True': 'True'},
    question='Is the statement true for these two images?',
    load_dataset_kwargs={'revision': '6ad9994db49bf1162feaecacc5554e5ca4a36487'},
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
    choices_list='choices_list', metadata='metadata', dataset_name='tasksource/ai2d', task_id='ai2d',
    pre_process=grouped_mc_rows,
    load_dataset_kwargs={'revision': '24013c0e6bb313a1fc4dfa79f9db936072f6a3af', 'streaming': True},
)

figureqa = VisualClassification(
    dataset_name='vikhyatk/figureqa', task_id='figureqa', pre_process=figureqa_rows,
    label_values={'No': 'No', 'Yes': 'Yes'}, question='Answer the question about the chart.',
    load_dataset_kwargs={'revision': '1afe55949decfebe4ccaaa0ce175ff4bce28d24c'},
)


_MIND2WEB = dict(dataset_name='tasksource/multimodal-mind2web',
    load_dataset_kwargs={'revision': 'afa19ed8797c18996d02e06a52fffe2401ba6dbc'})

def mind2web_element_metadata(row):
    """Recover native candidate geometry already preserved in the data-only mirror."""
    metadata = json.loads(row['metadata'])
    width, height = metadata['image_size']
    def box(rect):
        x, y, w, h = rect
        return [x/width, y/height, (x+w)/width, (y+h)/height]
    return {**metadata, 'question_id': 'element',
            'source_dataset': 'osunlp/Multimodal-Mind2Web',
            'target_bbox': box(metadata['bbox']), 'bbox_units': 'normalized_xyxy',
            'candidate_boxes': [box(json.loads(choice)['bbox']) for choice in row['choices_list']]}

mind2web_action = VisualClassification(
    **_MIND2WEB, task_id='mind2web/action', labels='action',
    label_values={v: v for v in ('CLICK', 'TYPE', 'SELECT')},
    question='What kind of GUI action should be performed next?',
    metadata=lambda x: {**json.loads(x['metadata']), 'question_id': 'action'},
)

mind2web_element = VisualMultipleChoice(
    **_MIND2WEB, task_id='mind2web/element', choices_list='choices_list', labels='labels',
    question='Which candidate element should be acted on next?',
    metadata=mind2web_element_metadata,
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


m3cot = VisualMultipleChoice(
    images=lambda x: [x['image']], inputs=lambda x: '\n'.join(filter(None, [x['context'], x['question']])),
    choices_list='choices', labels=lambda x: ord(x['answer']) - ord('A'),
    dataset_name='LightChen2333/M3CoT', task_id='m3cot',
    pre_process=lambda ds: ds.filter(lambda x: x['image'] is not None),
    metadata=lambda x: {'id': x['id'], 'image_id': x['image_id'], 'rationale': x['rationale'],
                           'domain': x['domain'], 'topic': x['topic']},
    load_dataset_kwargs={'revision': '48cf35001d595a6b0290c82c897a4b4563390821'},
)

def exam_option_position(answer):
    """Native Latin/Cyrillic letters and one-based numbers denote option positions."""
    answer = str(answer).strip().upper()
    if answer in 'ABCDE' and len(answer) == 1:
        return ord(answer) - ord('A')
    if answer in 'АБВГД' and len(answer) == 1:
        return 'АБВГД'.index(answer)
    return int(answer) - 1 if answer in ('1', '2', '3', '4', '5') else None


exams_v = VisualClassification(
    images=lambda x: [x['image']], inputs=lambda x: 'Answer the exam question in the image. Choose the position of the correct option among those displayed.',
    labels=lambda x: exam_option_position(x['answer_key']),
    label_values={i: name + ' option' for i, name in enumerate(('first', 'second', 'third', 'fourth', 'fifth'))},
    dataset_name='MBZUAI/EXAMS-V', task_id='exams-v',
    pre_process=lambda ds: ds.filter(lambda x: x['image'] is not None and exam_option_position(x['answer_key']) is not None),
    metadata=lambda x: {'sample_id': x['sample_id'], 'language': x['language'], 'subject': x['subject'], 'grade': x['grade'],
                        'source_answer_key': x['answer_key']},
    load_dataset_kwargs={'revision': '7594b37a10e87fcfbc4def0fa61809ad7114b7f2'},
)

visualsphinx = VisualMultipleChoice(
    inputs=lambda x: x['problem'].replace('<image>', '').strip(),
    choices_list=lambda x: list(json.loads(x['choice']).values()),
    labels=lambda x: list(json.loads(x['choice'])).index(x['answer']),
    dataset_name='VisualSphinx/VisualSphinx-V1-RL-20K', task_id='visualsphinx',
    metadata=lambda x: {'id': x['id'], 'explanation': x['explanation'], 'readability': x['readability'],
                           'reasonableness': x['reasonableness'], 'has_duplicate': x['has_duplicate']},
    load_dataset_kwargs={'revision': '2d6dccaef5e72ac12569fd977dc37dc048870176'},
)

muslr_tfu = VisualClassification(
    images=lambda x: [x['image']], inputs=lambda x: x['full_context'] + '\n\n' + x['question'], labels='answer',
    label_values={answer: answer for answer in ('True', 'False', 'Unknown')},
    dataset_name='Aiden0526/MuSLR', task_id='muslr/tfu',
    pre_process=lambda ds: ds.filter(lambda x: x['choices'] is None),
    metadata=lambda x: {'id': x['id'], 'domain': x['domain'], 'symbol': x['symbol'], 'depth': x['depth'], 'reasoning': x['reasoning']},
    load_dataset_kwargs={'revision': '16dbb73e00bfc49011f645bcb97510f0ad7de962'},
)

muslr_mc = VisualMultipleChoice(
    images=lambda x: [x['image']], inputs=lambda x: x['full_context'] + '\n\n' + x['question'],
    choices_list=lambda x: [re.sub(r'^[A-Z]\.\s*', '', option) for option in json.loads(x['choices'])],
    labels=lambda x: ord(x['answer']) - ord('A'),
    dataset_name='Aiden0526/MuSLR', task_id='muslr/mc',
    pre_process=lambda ds: ds.filter(lambda x: x['choices'] is not None),
    metadata=lambda x: {'id': x['id'], 'domain': x['domain'], 'symbol': x['symbol'], 'depth': x['depth'], 'reasoning': x['reasoning']},
    load_dataset_kwargs={'revision': '16dbb73e00bfc49011f645bcb97510f0ad7de962'},
)

iconqa_text = VisualMultipleChoice(
    choices_list='choices_list', metadata='metadata', dataset_name='tasksource/iconqa-text', task_id='iconqa/text',
    pre_process=grouped_mc_rows,
    load_dataset_kwargs={'revision': '18c952834f240750908d849c2e4455067a7bf5cd', 'streaming': True},
)


view2space_mcq = VisualMultipleChoice(
    choices_list='choices_list', metadata='metadata', pre_process=grouped_mc_rows,
    dataset_name='tasksource/view2space', task_id='view2space/mcq',
    load_dataset_kwargs={'revision': '061e4b2bd587adba48c75a5fbbbac9d6c781bf61'},
)


_CAULDRON = dict(dataset_name='HuggingFaceM4/the_cauldron', metadata='metadata',
    splits=('train', None, None),
    load_dataset_kwargs={'revision': '847a98a779b1652d65111daf20c972dfcd333605', 'streaming': True})
_YESNO = {'no': 'No', 'yes': 'Yes'}
_COLOR = {v: v for v in ('gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow')}
_SHAPE = {v: v for v in ('cube', 'sphere', 'cylinder')}
_SIZE = {v: v for v in ('small', 'large')}
_MATERIAL = {v: v for v in ('rubber', 'metal')}

visual7w = VisualMultipleChoice(
    **_CAULDRON, config_name='visual7w', task_id='visual7w',
    choices_list='choices_list', pre_process=cauldron_mc_rows,
)

clevr_yesno = VisualClassification(
    **_CAULDRON, config_name='clevr', task_id='clevr/yesno', label_values=_YESNO,
    pre_process=lambda ds: cauldron_closed_rows(ds, _YESNO),
)

mapqa_yesno = VisualClassification(
    **_CAULDRON, config_name='mapqa', task_id='mapqa/yesno', label_values=_YESNO,
    pre_process=lambda ds: cauldron_closed_rows(ds, _YESNO),
)

tqa = VisualMultipleChoice(
    **_CAULDRON, config_name='tqa', task_id='tqa', choices_list='choices_list',
    pre_process=cauldron_mc_rows,
)

_HATEFUL_TERMS = 'https://huggingface.co/datasets/emily49/hateful-memes/blob/390eaf2f1a31eed27275b49c9bafcfc8ae721733/LICENSE.txt'
hateful_memes = VisualClassification(
    **{**_CAULDRON, 'metadata': lambda x: {**json.loads(x['metadata']),
        'source_license': 'Hateful Memes Dataset License Agreement',
        'source_license_url': _HATEFUL_TERMS, 'source_redistribution': 'restricted'}},
    config_name='hateful_memes', task_id='hateful-memes', label_values=_YESNO,
    pre_process=lambda ds: cauldron_closed_rows(ds, _YESNO),
)

clevr_color = VisualClassification(
    **_CAULDRON, config_name='clevr', task_id='clevr/color', label_values=_COLOR,
    pre_process=lambda ds: cauldron_closed_rows(ds, _COLOR),
)

clevr_shape = VisualClassification(
    **_CAULDRON, config_name='clevr', task_id='clevr/shape', label_values=_SHAPE,
    pre_process=lambda ds: cauldron_closed_rows(ds, _SHAPE),
)

clevr_size = VisualClassification(
    **_CAULDRON, config_name='clevr', task_id='clevr/size', label_values=_SIZE,
    pre_process=lambda ds: cauldron_closed_rows(ds, _SIZE),
)

clevr_material = VisualClassification(
    **_CAULDRON, config_name='clevr', task_id='clevr/material', label_values=_MATERIAL,
    pre_process=lambda ds: cauldron_closed_rows(ds, _MATERIAL),
)

intergps = VisualMultipleChoice(
    **_CAULDRON, config_name='intergps', task_id='intergps', choices_list='choices_list',
    pre_process=cauldron_mc_rows,
)

clevr_count = VisualClassification(
    **_CAULDRON, config_name='clevr', task_id='clevr/count',
    label_values={str(i): str(i) for i in range(11)},
    pre_process=lambda ds: cauldron_closed_rows(ds, tuple(str(i) for i in range(11))),
    question='How many? Choose the count that answers the question about the image.',
)

tallyqa_count = VisualClassification(
    **_CAULDRON, config_name='tallyqa', task_id='tallyqa/count',
    label_values={str(i): str(i) for i in range(16)},
    pre_process=lambda ds: cauldron_closed_rows(ds, tuple(str(i) for i in range(16))),
    question='How many? Choose the count that answers the question about the image.',
)

vsr_yesno = VisualClassification(
    **_CAULDRON, config_name='vsr', task_id='vsr/yesno', label_values=_YESNO,
    pre_process=lambda ds: cauldron_closed_rows(ds, _YESNO),
    question='Is the spatial statement true for the image?',
)


def rico_candidates(value):
    """Native semantic leaves, scaled by the hierarchy canvas, never by the target."""
    from .grounding import normalized_box
    root = json.loads(value)
    x0, y0, x1, y1 = root['bounds']
    if x0 != 0 or y0 != 0 or x1 <= 0 or y1 <= 0:
        raise ValueError('Unsupported RICO annotation canvas')
    elements = {}
    def visit(node):
        children = node.get('children', [])
        if not children and node.get('componentLabel') and 'bounds' in node:
            try:
                box = normalized_box([node['bounds'][0]/x1, node['bounds'][1]/y1,
                                      node['bounds'][2]/x1, node['bounds'][3]/y1])
            except (ValueError, TypeError):
                return
            text = ' '.join(str(node[k]) for k in ('componentLabel', 'iconClass', 'textButtonClass', 'text') if node.get(k))
            elements.setdefault(tuple(box), {'bbox': box, 'text': text})
        for child in children:
            visit(child)
    visit(root)
    return list(elements.values())


def rico_widget_rows(dataset, elements=False):
    """Human widget descriptions and normalized boxes; retain native splits/images."""
    def flatten(batch, indices, source_split):
        from .grounding import grounding_row
        output = {key: [] for key in ('images', 'inputs', 'labels', 'metadata')}
        if elements:
            output['choices_list'] = []
        for offset, (source_index, screen, image, box, captions) in enumerate(zip(
                indices, batch['screenId'], batch['image'], batch['bbox'], batch['captions'])):
            if image is None or len(box) != 4 or not all(math.isfinite(v) for v in box):
                continue
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                continue
            candidates = None
            if elements:
                try:
                    candidates = rico_candidates(batch['semantic_annotations'][offset])
                except (ValueError, TypeError, KeyError):
                    continue
            widget = f'{screen}:' + hashlib.sha256(json.dumps(box).encode()).hexdigest()
            # Different human captions are independent descriptions of the same widget.
            for index, caption in enumerate(captions):
                if not caption.strip():
                    continue
                try:
                    record = grounding_row([image], caption.strip(), box, candidates=candidates,
                        metadata={'screenId': screen, 'source_widget': widget,
                            'source_row': f'{source_split}:{source_index}:{index}', 'image_group_id': f'rico:{screen}',
                            'source_dataset': 'bevaya/RICO-WidgetCaptioning',
                            'source_revision': '6ec57b56bebd722b9c646c78d0f34e1199b6d7a9',
                            'coordinate_source': 'widget_bbox_center', 'source_license': 'cc-by-4.0',
                            'source_license_url': 'https://github.com/google-research-datasets/widget-caption'})
                except ValueError:
                    continue  # No target insertion, no generated negatives, no target-aware pruning.
                for key in output:
                    output[key].append(record[key])
        return output
    features = Features({'images': Sequence(Image(decode=False)), 'inputs': Value('string'),
                         'labels': Value('int64'), 'metadata': Value('string')})
    if elements:
        features['choices_list'] = Sequence(Value('string'))
    return type(dataset)({split: rows.map(flatten, batched=True, batch_size=16, with_indices=True,
        fn_kwargs={'source_split': split},
        remove_columns=list(rows.features), features=features) for split, rows in dataset.items()})


rico_widget_grid7 = VisualClassification(
    dataset_name='bevaya/RICO-WidgetCaptioning', task_id='rico-widget/grid7',
    splits=('train', 'val', 'test'), pre_process=rico_widget_rows, metadata='metadata',
    label_values={i: f'r{i // 7}c{i % 7}' for i in range(49)},
    question='Which cell contains the described widget center in a 7-row, 7-column grid over the displayed image?',
    load_dataset_kwargs={'revision': '6ec57b56bebd722b9c646c78d0f34e1199b6d7a9', 'streaming': True,
                         'columns': ['screenId', 'image', 'bbox', 'captions']},
)


rico_widget_element = VisualMultipleChoice(
    dataset_name='bevaya/RICO-WidgetCaptioning', task_id='rico-widget/element',
    splits=('train', 'val', 'test'), pre_process=lambda ds: rico_widget_rows(ds, elements=True), metadata='metadata',
    choices_list='choices_list', question='Which element in the displayed image matches the instruction?',
    load_dataset_kwargs={'revision': '6ec57b56bebd722b9c646c78d0f34e1199b6d7a9', 'streaming': True,
                         'columns': ['screenId', 'image', 'bbox', 'captions', 'semantic_annotations']},
)


lvis_region = VisualClassification(
    dataset_name='tasksource/coco-regions', config_name='lvis', task_id='lvis/region',
    metadata='metadata', question='What object category is highlighted in red?',
    load_dataset_kwargs={'revision': '0ab5c2f908e4a4db268515869bcde3913de2bc91'},
)

coco_panoptic_region = VisualClassification(
    dataset_name='tasksource/coco-regions', config_name='panoptic', task_id='coco/panoptic-region',
    metadata='metadata', question='What category is the image segment highlighted in red?',
    load_dataset_kwargs={'revision': '0ab5c2f908e4a4db268515869bcde3913de2bc91'},
)

doclaynet_region = VisualClassification(
    dataset_name='tasksource/doclaynet-region', task_id='doclaynet/region',
    metadata='metadata', question='What document layout category is highlighted in red?',
    load_dataset_kwargs={'revision': 'ad3465ea6385871b9102308865554150358d5656'},
)

bapps_preference = VisualMultipleChoice(
    dataset_name='tasksource/bapps', task_id='bapps/preference', choices_list='choices_list',
    metadata='metadata', question='Which alternative image is closer to the reference?',
    load_dataset_kwargs={'revision': 'cb0940d9a9dc43e6a4a03bf49627ad8bd18eb418'},
)

spair71k_grid7 = VisualClassification(
    dataset_name='tasksource/spair71k-grid', task_id='spair71k/grid7',
    metadata='metadata',
    question='Which cell of the second image contains the corresponding point? '
             'Rows r0–r6 run top to bottom; columns c0–c6 run left to right.',
    load_dataset_kwargs={'revision': 'bb618c093d81057971b6f6ea49833a505ceb1228'},
)


_SUPERCLEVR = dict(dataset_name='tasksource/superclevr', metadata='metadata',
    load_dataset_kwargs={'revision': 'b1dcf17611d997e227ff13fd5692d3705fbf35a1'})
_SUPERCLEVR_SHAPE = {v: v for v in ('airliner', 'articulated bus', 'biplane', 'chopper',
    'cruiser', 'dirtbike', 'double bus', 'fighter', 'jet', 'minivan', 'mountain bike',
    'regular bus', 'road bike', 'school bus', 'scooter', 'sedan', 'suv', 'tandem bike',
    'truck', 'utility bike', 'wagon')}

superclevr__yesno = VisualClassification(**_SUPERCLEVR, task_id='superclevr/yesno',
    pre_process=lambda ds: grouped_qa_rows(ds, 'yesno'), label_values=_YESNO)
superclevr__count = VisualClassification(**_SUPERCLEVR, task_id='superclevr/count',
    pre_process=lambda ds: grouped_qa_rows(ds, 'count'), label_values={str(i): str(i) for i in range(11)})
superclevr__color = VisualClassification(**_SUPERCLEVR, task_id='superclevr/color',
    pre_process=lambda ds: grouped_qa_rows(ds, 'color'), label_values=_COLOR)
superclevr__shape = VisualClassification(**_SUPERCLEVR, task_id='superclevr/shape',
    pre_process=lambda ds: grouped_qa_rows(ds, 'shape'), label_values=_SUPERCLEVR_SHAPE)
superclevr__size = VisualClassification(**_SUPERCLEVR, task_id='superclevr/size',
    pre_process=lambda ds: grouped_qa_rows(ds, 'size'), label_values=_SIZE)
superclevr__material = VisualClassification(**_SUPERCLEVR, task_id='superclevr/material',
    pre_process=lambda ds: grouped_qa_rows(ds, 'material'), label_values=_MATERIAL)
