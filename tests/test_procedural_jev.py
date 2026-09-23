import json
import random
import unittest
from collections import Counter

from datasets import Dataset

from tasksource.jev.augmentations import augment_jev_internal
from tasksource.jev.procedural import TASKS, jev_rows
from tasksource.jev.recast import render_systemone_group
from scripts.build_jev_dataset import TRAINING_FEATURES, share_cap, source_row_group
from scripts.build_procedural_jev import build_task


def _intent(state):
    if any(state["billing"].values()):
        return "billing"
    account, telemetry = state["account"], state["telemetry"]
    if account["active"] and (account["auth_failures"] >= 2 or account["permission_mismatch"]):
        return "access"
    if telemetry["integration_failures"] >= 2 or telemetry["error_rate_percent"] >= 20:
        return "technical"
    return "other"


def _impact(workflow):
    if workflow["core_blocked"] and not workflow["workaround_available"]:
        return 2
    return int(workflow["degraded"] or workflow["core_blocked"])


def _risk(record, rule):
    return (
        (0 if record["authorized"] else rule["unauthorized_points"])
        + (rule["blocked_status_points"] if record["status"] == "blocked" else 0)
        + (rule["amount_threshold_points"] if record["amount"] >= rule["amount_threshold"] else 0)
        + (rule["unassigned_owner_points"] if record["owner"] == "unassigned" else 0)
    )


class ProceduralJevTest(unittest.TestCase):
    def samples(self, task, n=200):
        rng = random.Random(task)
        return [TASKS[task].generate(rng, rng.randrange(5)) for _ in range(n)]

    def test_answers_are_well_typed(self):
        for task in TASKS:
            for problem in self.samples(task, 50):
                self.assertEqual(set(problem.questions), set(problem.answers))
                for qid, spec in problem.questions.items():
                    answer = problem.answers[qid]
                    self.assertEqual(spec["type"], answer["type"])
                    if spec["type"] == "choice":
                        self.assertIn(answer["choice"], spec["criteria"])
                    elif spec["type"] == "noul":
                        self.assertTrue(0.0 <= answer["noul"] <= 1.0)

    def test_generation_is_deterministic(self):
        for task in TASKS:
            a = TASKS[task].generate(random.Random(1), 2)
            b = TASKS[task].generate(random.Random(1), 2)
            self.assertEqual(a, b)

    def test_multi_view_gold_follows_stated_rules(self):
        for problem in self.samples("multi_view_adjudication"):
            state, answers = problem.state, problem.answers
            timeline = state["timeline"]
            deadline = timeline["deadline_hours"]
            urgent = timeline["executive_escalation"] or (deadline is not None and deadline <= 24)
            self.assertEqual(answers["intent"]["choice"], _intent(state))
            self.assertEqual(bool(answers["is_urgent"]["noul"]), urgent)
            self.assertEqual(answers["workflow_impact"]["score"], _impact(state["workflow"]))

    def test_state_perturbation_gold_follows_risk_rule(self):
        for problem in self.samples("state_perturbation"):
            rule = problem.state["risk_rule"]
            delta = _risk(problem.state["after"], rule) - _risk(problem.state["before"], rule)
            expected = 0 if delta < 0 else 2 if delta > 0 else 1
            self.assertEqual(problem.answers["risk_direction"]["score"], expected)

    def test_strongest_support_origin_is_not_constant(self):
        answers = {p.answers["strongest_support_origin"]["choice"]
                   for p in self.samples("evidence_sufficiency")}
        self.assertGreater(len(answers), 3)

    def test_built_splits_are_disjoint_and_grouped(self):
        dataset = build_task("state_perturbation", {"train": 60, "validation": 10, "test": 10}, [0, 1])
        train = set(dataset["train"]["state"])
        for split in ("validation", "test"):
            self.assertFalse(train & set(dataset[split]["state"]))
        example = dataset["train"][0]
        answers = json.loads(example["answers"])
        self.assertEqual(
            dataset["train"].features["changed_dimension"].int2str(example["changed_dimension"]),
            answers["changed_dimension"]["choice"],
        )

        rows = jev_rows(example, "procedural-jev/state_perturbation", "train", 0)
        Dataset.from_list(rows, features=TRAINING_FEATURES)
        self.assertEqual({row["kind"] for row in rows}, {"choice", "noul", "score"})
        self.assertEqual(len({source_row_group(row["id"]) for row in rows}), 1)
        request = render_systemone_group([
            {**row, "criteria": row["options"], "instructions": row["question"],
             "question_id": row["id"].split(":", 3)[3]}
            for row in rows if row["kind"] == "choice"
        ])
        self.assertIn("changed_dimension", request["questions"])

    def test_augmentation_leaves_procedural_rows_alone(self):
        dataset = build_task("policy_applicability", {"train": 20, "validation": 2, "test": 2}, [0])
        rows = [row for i, example in enumerate(dataset["train"])
                for row in jev_rows(example, "procedural-jev/policy_applicability", "train", i)]
        shard = Dataset.from_list(rows, features=TRAINING_FEATURES)
        self.assertEqual(len(augment_jev_internal(shard, 1.0, 1.0, 1.0, 1.0, 1.0)), len(shard))

    def test_share_cap_reserves_procedural_rows_whole_groups(self):
        dataset = build_task("state_perturbation", {"train": 40, "validation": 1, "test": 1}, [0])
        procedural = [row for i, example in enumerate(dataset["train"])
                      for row in jev_rows(example, "procedural-jev/state_perturbation", "train", i)]
        other = [{**procedural[0], "id": f"src-{i % 4}:train:{i}", "source": f"src/{i % 4}",
                  "kind": "choice", "options": ["a", "b"], "target": [1.0, 0.0]} for i in range(400)]
        capped = share_cap(Dataset.from_list(other + procedural, features=TRAINING_FEATURES), 100, 0.3)
        ids = [row["id"] for row in capped if row["source"].startswith("procedural-jev/")]
        self.assertEqual(len(capped), 100)
        self.assertEqual(len(ids), 30)
        groups = Counter(source_row_group(i) for i in ids)
        self.assertEqual(set(groups.values()), {3})


if __name__ == "__main__":
    unittest.main()
