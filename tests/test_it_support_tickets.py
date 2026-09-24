import unittest

import pandas as pd
from datasets import ClassLabel

from scripts.build_it_support_tickets import build_dataset, merge_source_split


class ItSupportTicketsIngestionTest(unittest.TestCase):
    def test_join_uses_source_ids_and_preserves_text(self):
        x = pd.DataFrame({
            "Unnamed: 0": [0, 1],
            "id": [17, 12],
            "text": [" ticket text ", "Second ticket"],
        })
        y = pd.DataFrame({
            "id": [12, 17],
            "category_truth": ["Software", "Fileservice"],
        })
        joined = merge_source_split(x, y)
        self.assertEqual(joined["text"].tolist(), [" ticket text ", "Second ticket"])
        self.assertEqual(
            joined["category_truth"].tolist(), ["Fileservice", "Software"]
        )

    def test_train_test_and_seven_class_feature_are_preserved(self):
        x_train = pd.DataFrame({"id": [1, 2], "text": ["a", "b"]})
        y_train = pd.DataFrame({"id": [1, 2], "category_truth": ["Fileservice", "O365"]})
        x_test = pd.DataFrame({"id": [3], "text": ["c"]})
        y_test = pd.DataFrame({"id": [3], "category_truth": ["Software"]})
        data = build_dataset({"train": (x_train, y_train), "test": (x_test, y_test)})
        self.assertEqual(set(data), {"train", "test"})
        self.assertEqual(data["train"]["text"], ["a", "b"])
        self.assertEqual(data["test"]["text"], ["c"])
        self.assertIsInstance(data["train"].features["label"], ClassLabel)
        self.assertEqual(
            data["train"].features["label"].names,
            ["Fileservice", "O365", "Software"],
        )

    def test_mismatched_ids_are_rejected(self):
        x = pd.DataFrame({"id": [1], "text": ["x"]})
        y = pd.DataFrame({"id": [2], "category_truth": ["Software"]})
        with self.assertRaisesRegex(ValueError, "ID sets do not match"):
            merge_source_split(x, y)


if __name__ == "__main__":
    unittest.main()
