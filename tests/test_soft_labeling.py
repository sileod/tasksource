"""SoftLabeling annotations: soft and hard views, catalog listing, and the Jev recast."""

import unittest

from datasets import Dataset, DatasetDict

from tasksource import list_tasks
from tasksource.jev.recast import recast_jev
from tasksource.preprocess import SoftLabeling, soft_target


def _rows(**columns):
    n = len(next(iter(columns.values())))
    return DatasetDict(train=Dataset.from_dict(columns), validation=Dataset.from_dict(columns),
                       test=Dataset.from_dict(columns)) if n else None


class SoftTargetTest(unittest.TestCase):
    def test_raw_values(self):
        self.assertEqual(soft_target([1, 3], "choice", 2), [0.25, 0.75])
        self.assertEqual(soft_target(0.4, "noul"), [0.4])
        self.assertEqual(soft_target(3, "noul", low=1, high=5), [0.5])
        self.assertEqual(soft_target(-1, "score", 3, low=-1), [1.0, 0.0, 0.0])
        self.assertIsNone(soft_target(7, "score", 3))
        self.assertIsNone(soft_target([0, 0], "choice", 2))
        self.assertIsNone(soft_target(float("nan"), "noul"))

    def test_score_mean_splits_between_nearest_levels(self):
        target = soft_target(3.4, "score", 5, low=1)
        self.assertEqual([round(x, 6) for x in target], [0, 0, 0.6, 0.4, 0])
        self.assertAlmostEqual(sum(i * p for i, p in enumerate(target, start=1)), 3.4)  # the mean is kept
        self.assertEqual(soft_target(5, "score", 5, low=1), [0, 0, 0, 0, 1.0])
        self.assertEqual(soft_target(0.5, "score", 5, step=0.25), [0, 0, 1.0, 0, 0])
        self.assertIsNone(soft_target(5.2, "score", 5, low=1))
        self.assertEqual(soft_target(1.6, "choice", 3), [0, 0, 1.0])  # choice levels stay one-hot


class ViewTest(unittest.TestCase):
    RATINGS = _rows(text=["a", "b", "c", "d"], rating=[0.0, 0.2, 0.6, 1.0])

    def test_noul_views(self):
        task = SoftLabeling("text", labels="rating", kind="noul", options=["no", "yes"], hard=0.8)
        self.assertEqual(task.hard_type, "Classification")
        soft = task(self.RATINGS, soft=True)["train"]
        self.assertEqual(soft["labels"], [[0.0], [0.2], [0.6], [1.0]])
        self.assertEqual(soft["options"], [[]] * 4)
        hard = task(self.RATINGS)["train"]
        self.assertEqual(hard.features["labels"].names, ["no", "yes"])
        self.assertEqual(sorted(zip(hard["sentence1"], hard["labels"])), [("a", 0), ("b", 0), ("d", 1)])  # 1 - 0.8 kept

    def test_per_row_choice_hard_view_is_multiple_choice(self):
        polls = _rows(a=["x", "p", "m"], b=["y", "q", "n"], va=[10, 50, 1], vb=[30, 50, 0])
        task = SoftLabeling("a", labels=lambda x: [x["va"], x["vb"]], options=lambda x: [x["a"], x["b"]], hard=2/3)
        self.assertEqual(task.hard_type, "MultipleChoice")
        hard = task(polls)["train"]
        gold = {row["inputs"]: row[f"choice{row['labels']}"] for row in hard}
        self.assertEqual(gold, {"x": "y", "m": "m"})  # 50/50 dropped
        soft = task(polls, soft=True)["train"]
        self.assertEqual(soft["options"][0], ["x", "y"])
        self.assertEqual(soft["labels"][0], [0.25, 0.75])

    def test_min_annotators_filters_vote_rows(self):
        votes = _rows(text=["a", "b"], v=[[1, 2], [4, 6]])
        task = SoftLabeling("text", labels="v", options=["x", "y"], annotators=5)
        self.assertEqual(task(votes, soft=True, min_annotators=5)["train"]["sentence1"], ["b"])
        rated = SoftLabeling("text", labels="v", options=["x", "y"], aggregation="mean")
        self.assertEqual(len(rated(votes, soft=True, min_annotators=5)["train"]), 2)

    def test_regression_view_keeps_raw_value(self):
        task = SoftLabeling("text", labels="rating", kind="noul", high=5, aggregation="mean", regression=True)
        rows = _rows(text=["a", "b"], rating=[1.0, 4.0])
        self.assertEqual(task(rows)["train"]["labels"], [1.0, 4.0])
        self.assertEqual(task(rows, soft=True)["train"]["labels"], [[0.2], [0.8]])

    def test_soft_only(self):
        task = SoftLabeling("text", labels="rating", kind="noul")
        self.assertIsNone(task.hard_type)
        with self.assertRaises(ValueError):
            task(self.RATINGS)
        with self.assertRaises(ValueError):
            SoftLabeling("text", kind="noul", hard=0.8)  # no names for the hard classes
        with self.assertRaises(ValueError):
            SoftLabeling("text", options=["a", "b"], hard=0.5)  # no majority


class CatalogTest(unittest.TestCase):
    def test_views(self):
        hard, soft = list_tasks(), list_tasks(soft=True)
        self.assertNotIn("proto_qa/proto_qa", set(hard.id))  # soft labels only
        self.assertIn("proto_qa/proto_qa", set(soft.id))
        self.assertIn("hate_speech_offensive", set(hard.id))
        self.assertNotIn("hate_speech_offensive", set(soft.id))  # replaced by its vote shares
        enough = list_tasks(soft=True, min_annotators=5)
        # three annotators: the votes are left out and replace nothing; mean ratings are kept
        self.assertNotIn("hate_speech_offensive/votes", set(enough.id))
        self.assertIn("hate_speech_offensive", set(enough.id))
        self.assertIn("civil_comments/toxicity_share", set(enough.id))
        self.assertNotIn("civil_comments/toxicity", set(enough.id))
        self.assertIn("UNLI", set(enough.id))
        feedback = enough[enough.id == "HelpSteer3/feedback"].iloc[0]
        self.assertEqual(feedback.task_type, "Classification")  # by its hard view
        stsb = hard[hard.id == "glue/stsb"].iloc[0]
        self.assertEqual(stsb.task_type, "Classification")  # the regression view, as before
        row = hard[hard.id == "google_wellformed_query"].iloc[0]
        self.assertEqual((row.task_type, row.soft_labels), ("Classification", True))
        self.assertEqual(soft[soft.id == "google_wellformed_query"].iloc[0].task_type, "SoftLabeling")


class JevTest(unittest.TestCase):
    def test_sibling_annotations_share_a_group_key(self):
        rows = _rows(sentence1=["same text"], options=[[]], labels=[[0.4]])
        a = recast_jev(rows, task="d/a", kind="noul", group="d")["train"][0]
        b = recast_jev(rows, task="d/b", kind="noul", group="d")["train"][0]
        self.assertEqual((a["group"], a["source_row"], a["state"]), (b["group"], b["source_row"], b["state"]))
        self.assertNotEqual(a["question_id"], b["question_id"])

    def test_rows_with_the_same_text_and_other_options_are_kept(self):
        rows = _rows(sentence1=["", ""], options=[["a", "b"], ["c", "d"]], labels=[[0.3, 0.7], [0.6, 0.4]])
        self.assertEqual(len(recast_jev(rows, task="t", kind="choice", row_options=True)["train"]), 2)

    def test_per_row_options_are_permuted_with_their_target(self):
        options = [[f"answer {i}-{j}" for j in range(4)] for i in range(40)]
        rows = _rows(sentence1=[f"q{i}" for i in range(40)], options=options, labels=[[0.7, 0.2, 0.1, 0.0]] * 40)
        jev = recast_jev(rows, task="t", question="Q?", kind="choice", row_options=True)["train"]
        slots = set()
        for row, source in zip(jev, options):
            self.assertEqual(dict(zip(row["criteria"], row["target"])), dict(zip(source, [0.7, 0.2, 0.1, 0.0])))
            slots.add(row["target"].index(0.7))
        self.assertGreater(len(slots), 1)
        fixed = recast_jev(rows, task="t", question="Q?", kind="score")["train"][0]
        self.assertEqual(fixed["target"], [0.7, 0.2, 0.1, 0.0])  # ordered levels keep their order
        self.assertEqual(fixed["kind"], "score")


if __name__ == "__main__":
    unittest.main()
