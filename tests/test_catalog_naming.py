"""Unit tests for the task-catalog naming helpers in ``tasksource.access``.

``parse_var_name`` turns a task module variable name into
``(dataset_name, config_name, task_name)`` following the ``dataset``,
``dataset__task`` and ``dataset___config`` conventions used throughout
``tasks.py``. ``pretty_name`` renders a human-readable name from a
preprocessing mapping. These pure helpers had no dedicated coverage before,
so regressions in the (surprisingly branchy) name parsing would go unnoticed.
"""

import unittest
from types import SimpleNamespace

from tasksource import list_tasks, parse_var_name, pretty_name


class ParseVarNameTest(unittest.TestCase):
    def test_dataset_only(self):
        self.assertEqual(parse_var_name("glue"), ("glue", None, None))

    def test_dataset_and_task(self):
        # ``dataset__task`` style (exactly two underscores)
        self.assertEqual(parse_var_name("anli__a1"), ("anli", None, "a1"))

    def test_dataset_and_config(self):
        # ``dataset___config`` style (exactly three underscores)
        self.assertEqual(parse_var_name("glue___mnli"), ("glue", "mnli", None))
        self.assertEqual(parse_var_name("super_glue___cb"), ("super_glue", "cb", None))

    def test_dataset_config_and_task(self):
        # ``dataset___config__task`` style (three, then two underscores)
        self.assertEqual(parse_var_name("a___b__c"), ("a", "b", "c"))

    def test_three_underscores_are_not_misread_as_two(self):
        # The ``___`` in ``glue___mnli`` contains ``__`` as a substring; the
        # parser must still treat it as dataset___config, not dataset__task.
        self.assertEqual(parse_var_name("glue___mnli"), ("glue", "mnli", None))


class PrettyNameTest(unittest.TestCase):
    def _mapping(self, dataset_name, config_name=None, task_name=None):
        return SimpleNamespace(
            dataset_name=dataset_name, config_name=config_name, task_name=task_name
        )

    def test_dataset_only(self):
        self.assertEqual(pretty_name(self._mapping("glue/mnli")), "mnli")

    def test_dataset_and_config(self):
        self.assertEqual(pretty_name(self._mapping("glue/mnli", "mnli")), "mnli/mnli")

    def test_dataset_and_task(self):
        self.assertEqual(pretty_name(self._mapping("glue/mnli", None, "rte")), "mnli/rte")

    def test_dataset_config_and_task(self):
        self.assertEqual(
            pretty_name(self._mapping("glue/mnli", "mnli", "rte")), "mnli/mnli/rte"
        )


class ListTasksCatalogTest(unittest.TestCase):
    def test_returns_a_dataframe_with_unique_ids(self):
        df = list_tasks()
        self.assertIn("id", df.columns)
        self.assertEqual(len(df), df["id"].nunique())

    def test_calls_return_independent_copies(self):
        # The docstring promises a fresh copy on every call.
        first = list_tasks()
        second = list_tasks()
        self.assertIsNot(first, second)
        before = list(second["id"])
        first["id"] = "mutated"
        self.assertEqual(list(second["id"]), before)

    def test_multilingual_catalog_is_queryable(self):
        ml = list_tasks(multilingual=True)
        self.assertIn("id", ml.columns)
        self.assertGreater(len(ml.columns), 0)


if __name__ == "__main__":
    unittest.main()
