import re
import unittest

import pandas as pd
from datasets import Dataset, DatasetDict

from scripts.upload_repackaged import _wikipedia_detox_votes
from tasksource import list_tasks
from tasksource.metadata.jev_mixes import BUCKETS, PICKED


class WikipediaDetoxTest(unittest.TestCase):
    def setUp(self):
        self.comments = pd.DataFrame([
            dict(rev_id=i, comment="helloNEWLINE_TOKENworldTAB_TOKEN!", sample="random",
                 year=2015, ns=1, split=split)
            for i, split in [(3, "test"), (1, "train"), (2, "dev")]
        ])

    def test_join_vote_order_splits_and_provenance(self):
        for dimension, levels in [("attack", None), ("aggression", range(-3, 4)),
                                  ("toxicity", range(-2, 3))]:
            rows = []
            for rev_id in [2, 3, 1]:
                for score in (list(levels) if levels else [0, 0, 1]):
                    row = dict(rev_id=float(rev_id), worker_id=42)
                    row[dimension] = int(score < 0) if levels else score
                    if levels:
                        row[f"{dimension}_score"] = score
                    rows.append(row)
            result = _wikipedia_detox_votes(self.comments, pd.DataFrame(rows), dimension)
            self.assertEqual(list(result), ["train", "validation", "test"])
            for split, rev_id in [("train", 1), ("validation", 2), ("test", 3)]:
                row = result[split][0]
                self.assertEqual(row["rev_id"], rev_id)
                self.assertEqual(row["text"], "hello\nworld\t!")
                self.assertNotIn("worker_id", row)
                self.assertNotIn("split", row)
                self.assertEqual(row["sample"], "random")
                self.assertEqual(row[f"{dimension}_votes"],
                                 [len(levels) // 2 + 1, len(levels) // 2] if levels else [2, 1])
                self.assertEqual(sum(row[f"{dimension}_votes"]), row["annotators"])
                if levels:
                    self.assertEqual(row["score_votes"], [1] * len(levels))

    def test_reject_unmatched_ids_and_inconsistent_scores(self):
        annotations = pd.DataFrame([dict(rev_id=i, toxicity=0, toxicity_score=-1) for i in [1, 2, 3]])
        with self.assertRaisesRegex(ValueError, "disagrees"):
            _wikipedia_detox_votes(self.comments, annotations, "toxicity")
        with self.assertRaisesRegex(ValueError, "unmatched"):
            _wikipedia_detox_votes(self.comments, annotations.iloc[:2], "toxicity")

    def test_five_soft_tasks_and_first_match_bucket(self):
        tasks = list_tasks(soft=True)
        tasks = tasks[tasks.id.str.startswith("wikipedia-detox/")]
        self.assertEqual(set(tasks.id), {f"wikipedia-detox/{name}" for name in
                         ["attack", "aggression", "aggression_score", "toxicity", "toxicity_score"]})
        for task in tasks.itertuples():
            self.assertEqual(task.dataset_name, "tasksource/wikipedia-detox-votes")
            self.assertEqual(task.mapping.count, "annotators")
            bucket = next(name for name, (pattern, _) in BUCKETS.items() if re.search(pattern, task.id))
            self.assertEqual(bucket, "graded_calibration")
            self.assertRegex(task.id, PICKED)

    def test_tasks_normalize_counts_and_use_actual_annotators(self):
        tasks = list_tasks(soft=True)
        for task in tasks[tasks.id.str.startswith("wikipedia-detox/")].itertuples():
            counts = [1] * (len(task.mapping.options) - 1) + [4]
            rows = Dataset.from_list([
                {"text": "comment", task.mapping.labels: counts, "annotators": sum(counts)},
                {"text": "too few raters", task.mapping.labels: [1] + [0] * (len(counts) - 1),
                 "annotators": 1},
            ])
            dataset = DatasetDict({split: rows for split in ["train", "validation", "test"]})
            result = task.mapping(dataset, soft=True, min_annotators=5)["train"]
            self.assertEqual(result["sentence1"], ["comment"])
            for actual, expected in zip(result[0]["labels"], counts):
                self.assertAlmostEqual(actual, expected / sum(counts))
