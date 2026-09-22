import unittest

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from tasksource.recast import recast_jev, render_systemone


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


if __name__ == "__main__":
    unittest.main()
