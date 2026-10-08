"""Visual tasks, using the shared loader and pinned, data-only Hub sources."""
import json
import math
import re
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


def grouped_mc_rows(dataset):
    """Flatten canonical QAs that share ordered encoded images in a visual mirror."""
    def flatten(batch):
        rows = [(images, group, qa) for images, group, qas in zip(
            batch['images'], batch['image_group_id'], batch['qa']) for qa in qas]
        return {'images': [images for images, group, qa in rows],
                **{key: [qa[key] for images, group, qa in rows] for key in ('inputs', 'choices_list', 'labels')},
                'metadata': [json.dumps({**json.loads(qa['metadata']), 'image_group_id': group},
                    ensure_ascii=False, sort_keys=True) for images, group, qa in rows]}
    features = Features({'images': Sequence(Image(decode=False)), 'inputs': Value('string'),
        'choices_list': Sequence(Value('string')), 'labels': Value('int64'), 'metadata': Value('string')})
    return type(dataset)({split: rows.map(flatten, batched=True, batch_size=16,
        remove_columns=list(rows.features), features=features) for split, rows in dataset.items()})


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
    load_dataset_kwargs={'revision': '24013c0e6bb313a1fc4dfa79f9db936072f6a3af'},
)

figureqa = VisualClassification(
    dataset_name='vikhyatk/figureqa', task_id='figureqa', pre_process=figureqa_rows,
    label_values={'No': 'No', 'Yes': 'Yes'}, question='Answer the question about the chart.',
    load_dataset_kwargs={'revision': '1afe55949decfebe4ccaaa0ce175ff4bce28d24c'},
)


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


_MIND2WEB = dict(dataset_name='tasksource/multimodal-mind2web',
    load_dataset_kwargs={'revision': 'afa19ed8797c18996d02e06a52fffe2401ba6dbc'})

mind2web_action = VisualClassification(
    **_MIND2WEB, task_id='mind2web/action', labels='action',
    label_values={v: v for v in ('CLICK', 'TYPE', 'SELECT')},
    question='What kind of GUI action should be performed next?',
    metadata=lambda x: {**json.loads(x['metadata']), 'question_id': 'action'},
)

mind2web_element = VisualMultipleChoice(
    **_MIND2WEB, task_id='mind2web/element', choices_list='choices_list', labels='labels',
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
    load_dataset_kwargs={'revision': '18c952834f240750908d849c2e4455067a7bf5cd'},
)


view2space_mcq = VisualMultipleChoice(
    choices_list='choices_list', metadata='metadata', pre_process=grouped_mc_rows,
    dataset_name='tasksource/view2space', task_id='view2space/mcq',
    load_dataset_kwargs={'revision': '061e4b2bd587adba48c75a5fbbbac9d6c781bf61'},
)
