import unittest

from tasksource import list_tasks, parked
from tasksource.preprocess import Preprocessing


class ParkedTasksTest(unittest.TestCase):
    def test_every_parked_task_has_a_reason_and_is_unlisted(self):
        names = {key for key, value in vars(parked).items() if isinstance(value, Preprocessing)}
        self.assertEqual(names, set(parked.PARKED))
        self.assertTrue(all(kind in parked.KINDS for kind, _ in parked.PARKED.values()))
        self.assertTrue(all(parked.REASONS.values()))
        self.assertFalse(names & set(list_tasks().preprocessing_name))

    def test_parked_tasks_know_their_source(self):
        self.assertEqual((parked.glue___ax.dataset_name, parked.glue___ax.config_name), ("glue", "ax"))
        self.assertEqual(parked.bigbench.dataset_name, "tasksource/bigbench")

    def test_evaluation_benchmarks_are_grouped(self):
        benchmarks = parked.by_kind("evaluation")
        self.assertIn("mmlu", benchmarks)
        self.assertNotIn("quora", benchmarks)


if __name__ == "__main__":
    unittest.main()
