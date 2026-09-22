import unittest

from datasets import Dataset, DatasetDict

from tasksource.mtasks import _helpsteer3_context
from tasksource.tasks import (
    _chemprot_relations,
    _docred_relations,
    _fewrel_relation_match,
)


class ModernTaskPreprocessingTest(unittest.TestCase):
    def test_fewrel_positive_and_negative_pairs(self):
        rows = Dataset.from_list([
            {"tokens": ["Alice", "works", "at", "Acme"], "relation": "P1", "names": ["employer"]},
            {"tokens": ["Paris", "is", "in", "France"], "relation": "P2", "names": ["country"]},
        ])
        source = DatasetDict({split: rows for split in ("train_wiki", "val_wiki", "val_nyt")})
        result = _fewrel_relation_match(source)["train_wiki"]
        self.assertEqual(result.num_rows, 4)
        self.assertEqual(result[0], {"text": "Alice works at Acme", "relation": "employer", "label": 1})
        self.assertEqual(result[1], {"text": "Alice works at Acme", "relation": "country", "label": 0})

    def test_docred_entity_pair_and_relation(self):
        row = {
            "sents": [["Alice", "works", "at", "Acme", "."]],
            "vertexSet": [
                [{"name": "Alice"}],
                [{"name": "Acme"}, {"name": "Acme Corp"}],
            ],
            "labels": {"head": [0], "tail": [1], "relation_id": ["P108"], "relation_text": ["employer"]},
        }
        source = DatasetDict({"train_annotated": Dataset.from_list([row]), "validation": Dataset.from_list([row])})
        result = _docred_relations(source)["train_annotated"][0]
        self.assertEqual(result["entity_pair"], "Alice -> Acme / Acme Corp")
        self.assertEqual(result["relation"], "employer")

    def test_chemprot_columnar_entities_and_relations(self):
        row = {
            "text": "Drug inhibits protein.",
            "entities": {"id": ["T1", "T2"], "text": ["Drug", "protein"]},
            "relations": {"type": ["CPR:4"], "arg1": ["T1"], "arg2": ["T2"]},
        }
        source = DatasetDict({split: Dataset.from_list([row]) for split in ("train", "validation", "test")})
        result = _chemprot_relations(source)["train"][0]
        self.assertEqual(result["entity_pair"], "Drug -> protein")
        self.assertEqual(result["relation"], "CPR:4")

    def test_helpsteer_context_preserves_roles(self):
        context = {"context": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}
        self.assertEqual(_helpsteer3_context(context), "user: Hello\nassistant: Hi")


if __name__ == "__main__":
    unittest.main()
