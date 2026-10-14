import unittest

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from scripts.build_label_nli import balance, eligible
from tasksource.recast import recast_label_nli


def classification(labels, names=("negative", "positive")):
    features = Features({"sentence1": Value("string"), "sentence2": Value("string"), "labels": ClassLabel(names=list(names))})
    rows = [{"sentence1": f"text {i}", "sentence2": "pair" if i % 2 else "", "labels": label} for i, label in enumerate(labels)]
    return DatasetDict({"train": Dataset.from_list(rows, features=features)})


class LabelNliTest(unittest.TestCase):
    def test_hypotheses_match_gold(self):
        rows = recast_label_nli(classification([0, 1] * 50), "toy")["train"]
        self.assertEqual(rows.features["labels"].names, ["entailment", "neutral", "contradiction"])
        for row, gold in zip(rows, [0, 1] * 50):
            name = ["negative", "positive"][gold]
            self.assertEqual(row["positive"], f"This example is {name}.")
            self.assertNotEqual(row["negative"], row["positive"])
            self.assertEqual(row["hypothesis"], row["positive"] if row["labels"] == 0 else row["negative"])
        self.assertEqual(rows[1]["premise"], "text 1\npair")
        self.assertEqual(rows[0]["premise"], "text 0")
        self.assertTrue(10 < sum(row["labels"] == 0 for row in rows) < 90)

    def test_reproducible(self):
        dataset = classification([0, 1] * 20)
        self.assertEqual(recast_label_nli(dataset, "toy")["train"].to_list(), recast_label_nli(dataset, "toy")["train"].to_list())

    def test_eligible(self):
        self.assertIsNone(eligible(["negative", "positive"]))
        self.assertTrue(eligible(["entailment", "not_entailment"]))
        self.assertTrue(eligible(["1", "2"]))
        self.assertTrue(eligible(["0: not helpful", "1", "2", "3", "4: extremely helpful"]))
        self.assertTrue(eligible(["annotators agree", "annotators disagree"]))

    def test_balance(self):
        skewed = recast_label_nli(classification([0] * 95 + [1] * 5), "toy")["train"]
        self.assertEqual(len(balance(skewed, "toy")), 20)  # majority at most 3x the minority
        generic = recast_label_nli(classification([0, 1] * 600, names=("no", "yes")), "toy")["train"]
        self.assertEqual(len(balance(generic, "toy")), 1000)


if __name__ == "__main__":
    unittest.main()
