import unittest

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from tasksource.recast import recast_jev, render_systemone
from scripts.build_jev_dataset import pretty_order, to_training_row


class RecastJevTest(unittest.TestCase):
    def test_classification(self):
        features = Features({
            "sentence1": Value("string"),
            "sentence2": Value("string"),
            "labels": ClassLabel(names=["entailment", "contradiction"]),
        })
        source = DatasetDict({"train": Dataset.from_dict({
            "sentence1": ["A"], "sentence2": ["B"], "labels": [1]
        }, features=features)})
        row = recast_jev(source, task="demo")["train"][0]
        self.assertEqual(row["criteria"], ["entailment", "contradiction"])
        self.assertEqual(row["answer"], "contradiction")
        self.assertEqual(row["task"], "demo")
        self.assertIn("text_A: A", row["state"])

    def test_multiple_choice_and_renderer(self):
        source = DatasetDict({"train": Dataset.from_dict({
            "inputs": ["Question"], "choice0": ["zero"],
            "choice1": ["one"], "labels": [0]
        })})
        row = recast_jev(source)["train"][0]
        request = render_systemone(row, model="openjev")
        self.assertEqual(row["answer"], "zero")
        self.assertEqual(
            list(request["questions"]["decision"]["criteria"]),
            ["zero", "one"],
        )

    def test_common_jev_training_schema(self):
        row = to_training_row({
            "state": "Question",
            "instructions": "Choose.",
            "criteria": ["zero", "one"],
            "label": 1,
        }, index=3, task_id="demo/task", split="train")
        self.assertEqual(row["kind"], "choice")
        self.assertEqual(row["options"], ["zero", "one"])
        self.assertEqual(row["target"], [0.0, 1.0])
        self.assertEqual(row["source"], "demo/task")

    def test_pretty_order_only_changes_prefix(self):
        dataset = Dataset.from_dict({
            "source": ["b", "b", "b", "a", "a", "c"],
            "value": list(range(6)),
        })
        ordered = pretty_order(dataset, first_rows=4)
        self.assertEqual(ordered["source"][:4], ["a", "b", "c", "a"])
        self.assertEqual(ordered["value"], [3, 0, 5, 4, 1, 2])


if __name__ == "__main__":
    unittest.main()
