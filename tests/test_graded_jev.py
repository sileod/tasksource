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

    def _rows(self, name, example):
        family = FAMILIES[name]
        return jev_rows({**example, **(family.prepare(example) if family.prepare else {})},
                        family, f"graded/{name}", "train", 0)

    def test_annotator_distributions(self):
        votes = "{'positive': ['w1', 'w2', 'w3'], 'negative': [], 'neutral': ['w4'], 'mixed': ['w5']}"
        for distribution in (votes, eval(votes)):  # a repr string or an already-parsed dict
            [row] = self._rows("dynasent_r1", {"sentence": "S", "label_distribution": distribution})
            self.assertEqual(row["target"], [0.6, 0.0, 0.2, 0.2])
        [row] = self._rows("hatexplain", {"post_tokens": ["a", "b"], "annotators": {"label": [0, 2, 2]}})
        self.assertEqual((row["state"], row["options"]), ("Post:\na b", ["hate speech", "normal", "offensive"]))
        self.assertEqual(row["target"], [1 / 3, 0.0, 2 / 3])
        [row] = self._rows("unli", {"premise": "P", "hypothesis": "H", "label": 0.74})
        self.assertEqual((row["kind"], row["target"]), ("noul", [0.74]))

    def test_measuring_hate_speech_items(self):
        counts = [0, 0, 1, 2, 0]
        rows = self._rows("measuring_hate_speech", {"text": "T", **{item: counts for item in (
            "sentiment", "respect", "insult", "humiliate", "status", "dehumanize", "violence", "genocide",
            "attack_defend")}, "hatespeech": [2, 1, 0]})
        self.assertEqual(len(rows), 10)
        self.assertEqual({row["kind"] for row in rows}, {"score", "choice"})
        insult = next(row for row in rows if row["id"].endswith(":insult"))
        self.assertEqual((insult["options"][-1], insult["target"]), ("strongly agree", [0, 0, 1 / 3, 2 / 3, 0]))

    def test_lewidi(self):
        [row] = self._rows("lewidi_md_agreement", {"text": "T", "soft_label": [0.4, 0.6]})
        self.assertEqual((row["kind"], row["target"]), ("noul", [0.6]))
        [row] = self._rows("lewidi_conv_abuse", {"prev_agent": "_", "prev_user": "_", "agent": "Hi",
                                                  "user": "go away", "soft_label": [1.0, 0.0]})
        self.assertEqual(row["state"], "Conversation:\nAgent: Hi\nUser: go away")  # "_" marks no turn
        [row] = self._rows("lewidi_csc", {"context": "C", "response": "R", "soft_label": [0.5, 0.5, 0, 0, 0, 0]})
        self.assertEqual((row["kind"], len(row["options"]), row["target"]), ("score", 6, [0.5, 0.5, 0, 0, 0, 0]))
        rows = self._rows("lewidi_varierrnli", {"context": "C", "statement": "S", "entailment": 0.5,
                                                "neutral": 0.75, "contradiction": 0.0})
        self.assertEqual([row["target"] for row in rows], [[0.5], [0.75], [0.0]])


if __name__ == "__main__":
    unittest.main()
