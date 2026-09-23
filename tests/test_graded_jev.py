import unittest

from datasets import Dataset

from tasksource.jev.augmentations import augment_jev_internal
from tasksource.jev.graded import FAMILIES, jev_rows, likert, noul, score
from scripts.build_jev_dataset import TRAINING_FEATURES, source_row_group


class GradedJevTest(unittest.TestCase):
    def test_targets(self):
        half = score("x", "", likert("low", "high", n=9, start=1, step=0.5), low=1, step=0.5)
        self.assertEqual(half.target(3.5).index(1.0), 5)
        self.assertIsNone(half.target(0))
        similarity = noul("x", "", high=5)
        self.assertEqual(similarity.target(4.0), [0.8])
        self.assertIsNone(similarity.target(-1.0))  # hidden test label
        self.assertIsNone(similarity.target(float("nan")))
        relation = FAMILIES["chaos_mnli"].questions["relation"]
        self.assertEqual(relation.target([12, 68, 20]), [0.12, 0.68, 0.2])

    def test_helpsteer_is_one_state_with_five_scores(self):
        example = {"prompt": "P", "response": "R", "helpfulness": 3, "correctness": 4,
                   "coherence": 4, "complexity": 1, "verbosity": 2}
        rows = jev_rows(example, FAMILIES["helpsteer"], "graded/helpsteer", "train", 0)
        self.assertEqual(len(rows), 5)
        self.assertEqual({row["kind"] for row in rows}, {"score"})
        self.assertEqual(len({source_row_group(row["id"]) for row in rows}), 1)
        self.assertEqual(rows[0]["state"], "Prompt:\nP\n\nResponse:\nR")
        shard = Dataset.from_list(rows, features=TRAINING_FEATURES)
        self.assertEqual(len(augment_jev_internal(shard, 1.0, 1.0, 1.0, 1.0, 1.0)), len(shard))

    def test_unannotated_questions_are_dropped(self):
        example = {"parent_text": "P", "text": "R", **{q.column: float("nan")
                   for q in FAMILIES["oasst2"].questions.values()}, "quality": 0.75}
        rows = jev_rows(example, FAMILIES["oasst2"], "graded/oasst2", "train", 0)
        self.assertEqual([(row["id"].rsplit(":", 1)[1], row["target"]) for row in rows], [("quality", [0.75])])


if __name__ == "__main__":
    unittest.main()
