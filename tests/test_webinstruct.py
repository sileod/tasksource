import unittest

from datasets import Dataset, DatasetDict

from tasksource import list_tasks
from scripts.build_webinstruct import _webinstruct_choices, prepare
from scripts.audit_webinstruct import batches, render, validated
from tasksource.tasks import (
    webinstruct___binary,
    webinstruct___mc,
)


class WebInstructTest(unittest.TestCase):
    def test_options_follow_source_order(self):
        row = _webinstruct_choices({'question': 'Species?\na. Australopithecus\nc. Homo habilis\nb. Homo erectus\nd. Homo sapiens', 'answer': '(D)'})
        self.assertEqual(row['prompt'], 'Species?')
        self.assertEqual(row['options'][row['gold']], 'Homo sapiens')
        self.assertEqual(row['options'][1], 'Homo habilis')

    def test_ambiguous_options_and_answers_are_excluded(self):
        for question, answer in [('Pick? (a) one (b) two', 'a, b'),
                                 ('Pick? A. one A. two', 'A'),
                                 ('Pick? A. one C. three', 'C'),
                                 ('Pick? one or two', 'B'),
                                 ('Pick? A. one B. two', 'C')]:
            self.assertIsNone(_webinstruct_choices({'question': question, 'answer': answer})['gold'])

    def test_tasks_filter_and_preserve_gold(self):
        rows = [
            {'question': 'Pick? A. wrong B. correct', 'answer': 'b', 'answer_type': 'Multiple Choice'},
            {'question': 'Pick? (a) wrong (b) correct', 'answer': 'B', 'answer_type': 'Multiple Choice'},
            {'question': 'Pick? a) one b) two', 'answer': 'a, b', 'answer_type': 'Multiple Choice'},
            *[{'question': answer, 'answer': answer, 'answer_type': 'Boolean'}
              for answer in ['Yes', ' NO ', 'true', 'FALSE', 'a', 'Yes, perhaps']],
            {'question': 'Other type', 'answer': 'Yes', 'answer_type': 'String'},
        ] * 20
        def source(config):
            return DatasetDict({split: Dataset.from_list(prepare(rows)[config]) for split in ['train', 'test']})
        mc = webinstruct___mc(source('mc'))
        self.assertEqual(sum(len(split) for split in mc.values()), 80)
        for split in mc.values():
            for row in split:
                self.assertEqual(row[f"choice{row['labels']}"], 'correct')
                self.assertEqual(row['inputs'], 'Pick?')
        binary = webinstruct___binary(source('binary'))
        self.assertEqual(sum(len(split) for split in binary.values()), 160)
        for split in binary.values():
            self.assertEqual(split.features['labels'].names, ['no / false', 'yes / true'])
            for row in split:
                self.assertEqual(row['labels'], int(row['sentence1'].strip().lower() in {'yes', 'true'}))

    def test_registration(self):
        tasks = list_tasks()
        tasks = tasks[tasks.dataset_name == 'tasksource/webinstruct']
        self.assertEqual(set(tasks.task_type), {'MultipleChoice', 'Classification'})
        self.assertEqual(len(tasks), 2)

    def test_audit_hides_gold_and_compares_answer_in_python(self):
        row = {'key': 'mc/train/1', 'question': 'Pick? A. wrong B. correct',
               'prompt': 'Pick?', 'options': ['wrong', 'correct'], 'gold': 1}
        self.assertNotIn('gold', render(row))
        record = {'data': {'items': [{'key': row['key'], 'verdict': 'answerable', 'answer': 0, 'reason': 'evidence'}]}}
        self.assertEqual(validated(record, [row])[0]['verdict'], 'wrong')
        record['data']['items'][0]['answer'] = 1
        self.assertEqual(validated(record, [row])[0]['verdict'], 'ok')
        record['data']['items'][0]['answer'] = 2
        self.assertIsNone(validated(record, [row]))
        self.assertIsNone(validated({'data': {'items': []}}, [row]))

    def test_uncertain_audit_is_kept_and_long_inputs_are_complete(self):
        row = {'key': 'binary/train/1', 'question': 'q' * 50000, 'label': 0}
        record = {'data': {'items': [{'key': row['key'], 'verdict': 'uncertain', 'answer': None, 'reason': 'cannot settle'}]}}
        self.assertEqual(validated(record, [row])[0]['verdict'], 'uncertain')
        grouped = list(batches([row, {**row, 'key': 'binary/train/2'}]))
        self.assertEqual([len(group) for group in grouped], [1, 1])
        self.assertEqual(len(render(grouped[0][0])['prompt']), 50000)
