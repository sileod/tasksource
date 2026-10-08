"""Visual tasks, using the shared loader and pinned, data-only Hub sources."""
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
