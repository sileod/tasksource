import unittest

from tasksource import list_tasks, parked, eval_only
from tasksource.preprocess import Preprocessing


class ParkedTasksTest(unittest.TestCase):
    def test_every_parked_task_has_a_reason_and_is_unlisted(self):
        names = {key for key, value in vars(parked).items() if isinstance(value, Preprocessing)}
        self.assertEqual(names, set(parked.PARKED))
        self.assertTrue(all(kind in parked.KINDS for kind, _ in parked.PARKED.values()))
        self.assertTrue(all(parked.REASONS.values()))
        self.assertFalse(names & set(list_tasks().preprocessing_name))

    def test_parked_tasks_know_their_source(self):
        self.assertEqual((parked.super_glue___rte.dataset_name, parked.super_glue___rte.config_name),
                         ("super_glue", "rte"))
        self.assertIn("quora", parked.by_kind("duplicate"))
        self.assertNotIn("evaluation", parked.KINDS)

    def test_evaluation_annotations_are_separate_and_unlisted(self):
        names = {key for key, value in vars(eval_only).items() if isinstance(value, Preprocessing)}
        self.assertEqual(names, set(eval_only.REASONS))
        self.assertTrue(all(eval_only.REASONS.values()))
        self.assertFalse(names & set(parked.PARKED))
        for flags in ({}, {"multilingual": True}, {"vision": True}):
            listed = set(list_tasks(**flags).preprocessing_name)
            self.assertFalse(listed & (names | set(parked.PARKED)))
        self.assertIn("mmlu", names)
        self.assertNotIn("quora", names)
        self.assertEqual((eval_only.glue___ax.dataset_name, eval_only.glue___ax.config_name),
                         ("glue", "ax"))
        self.assertEqual(eval_only.bigbench.dataset_name, "tasksource/bigbench")
        self.assertEqual(eval_only.mmlu.splits, ["validation", "dev", "test"])
        self.assertIn("demelin/wino_x", eval_only.NOT_ANNOTATED)
        self.assertFalse(any(kind == "evaluation" for kind, _ in parked.NOT_ANNOTATED.values()))


if __name__ == "__main__":
    unittest.main()
