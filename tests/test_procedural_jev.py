import json
import random
import re
import unittest
from collections import Counter
from fractions import Fraction
from itertools import product

from datasets import Dataset

from tasksource.jev.augmentations import augment_jev_internal
from tasksource.jev.procedural import TASKS, jev_rows
from tasksource.jev.procedural.policy_applicability import SENSITIVITY
from tasksource.jev.recast import render_typed_decision_group
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
    down = {e["feature"] for e in workflow["affected_features"] if e["status"] == "down"}
    slow = {e["feature"] for e in workflow["affected_features"] if e["status"] == "slow"}
    if down & set(workflow["core_features"]) - set(workflow["workarounds"]):
        return 2
    return int(bool(down | slow))


def _risk(record, rule):
    return (
        (0 if record["authorized"] else rule["unauthorized_points"])
        + (rule["blocked_status_points"] if record["status"] == "blocked" else 0)
        + (rule["amount_threshold_points"] if record["amount"] >= rule["amount_threshold"] else 0)
        + (rule["unassigned_owner_points"] if record["owner"] == "unassigned" else 0)
    )


def _uncertain_policy_answers(state):
    """Brute force from the rendered state: every (role, report outcomes) world, weighted and conditioned."""
    history = dict((plural[:-1], int(n)) for n, plural in re.findall(r"(\d+) by (\w+)", state["role_evidence"]["history"]))
    reports = [(r["says"], Fraction(*map(int, re.findall(r"\d+", r["reliability"]))))
               for r in state["role_evidence"]["reports"]]
    roles = list(history)
    joint = {}
    for role in roles:
        for outcomes in product(roles, repeat=len(reports)):
            weight = Fraction(history[role], sum(history.values()))
            for said, (_, reliability) in zip(outcomes, reports):
                weight *= reliability if said == role else (1 - reliability) / (len(roles) - 1)
            if list(outcomes) == [said for said, _ in reports]:
                joint[role] = joint.get(role, 0) + weight
    total = sum(joint.values())
    allowed, governing = Fraction(0), {}
    for role, weight in joint.items():
        request = state["request"]
        subject = {**request["subject"], "role": role}
        matching = [p for p in state["policies"] if role in p["roles"] and subject["team"] in p["teams"]
                    and subject["clearance"] >= p["min_clearance"] and request["action"] in p["actions"]
                    and SENSITIVITY.index(request["resource"]["sensitivity"]) <= p["max_sensitivity"]]
        top = max(matching, key=lambda p: p["priority"]) if matching else None
        key = top["id"] if top else "none (default deny)"
        governing[key] = governing.get(key, 0) + weight / total
        allowed += weight / total if top and top["effect"] == "allow" else 0
    return {role: weight / total for role, weight in joint.items()}, governing, allowed


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

        rows = jev_rows(example, "procedural-typed-decisions/state_perturbation", "train", 0)
        Dataset.from_list(rows, features=TRAINING_FEATURES)
        self.assertEqual({row["kind"] for row in rows}, {"choice", "noul", "score"})
        self.assertEqual(len({source_row_group(row["id"]) for row in rows}), 1)
        request = render_typed_decision_group([
            {**row, "criteria": row["options"], "instructions": row["question"],
             "question_id": row["id"].split(":", 3)[3]}
            for row in rows if row["kind"] == "choice"
        ])
        self.assertIn("changed_dimension", request["questions"])

    def test_augmentation_leaves_procedural_rows_alone(self):
        dataset = build_task("policy_applicability", {"train": 20, "validation": 2, "test": 2}, [0])
        rows = [row for i, example in enumerate(dataset["train"])
                for row in jev_rows(example, "procedural-typed-decisions/policy_applicability", "train", i)]
        shard = Dataset.from_list(rows, features=TRAINING_FEATURES)
        self.assertEqual(len(augment_jev_internal(shard, 1.0, 1.0, 1.0, 1.0, 1.0)), len(shard))

    def test_share_cap_reserves_procedural_rows_whole_groups(self):
        dataset = build_task("state_perturbation", {"train": 40, "validation": 1, "test": 1}, [0])
        procedural = [row for i, example in enumerate(dataset["train"])
                      for row in jev_rows(example, "procedural-typed-decisions/state_perturbation", "train", i)]
        other = [{**procedural[0], "id": f"src-{i % 4}:train:{i}", "source": f"src/{i % 4}",
                  "kind": "choice", "options": ["a", "b"], "target": [1.0, 0.0]} for i in range(400)]
        capped = share_cap(Dataset.from_list(other + procedural, features=TRAINING_FEATURES), 100, 0.3)
        ids = [row["id"] for row in capped if row["source"].startswith("procedural-typed-decisions/")]
        self.assertEqual(len(capped), 100)
        self.assertEqual(len(ids), 30)
        groups = Counter(source_row_group(i) for i in ids)
        self.assertEqual(set(groups.values()), {3})

    def test_share_cap_balances_formats(self):
        row = {"kind": "choice", "options": ["a", "b"], "target": [1.0, 0.0], "state": "s", "question": "q",
               "variant": "direct", "split": "train"}
        rows = [{**row, "id": f"{kind}-{i % 5}:train:{i}", "source": f"{kind}/{i % 5}"}
                for kind, n in [("cls", 1000), ("mc", 1000), ("tok", 5)] for i in range(n)]
        formats = {f"mc/{i}": "MultipleChoice" for i in range(5)} | {f"tok/{i}": "TokenClassification" for i in range(5)}
        capped = share_cap(Dataset.from_list(rows, features=TRAINING_FEATURES), 100, formats=formats)
        counts = Counter(source.split("/")[0] for source in capped["source"])
        self.assertEqual(len(capped), 100)
        self.assertEqual(counts["tok"], 3)  # 0.03 of the rows the non-reserved formats share
        self.assertTrue(35 <= counts["mc"] <= 39, counts)

    def test_needle_retrieval_gold_follows_records(self):
        for problem in self.samples("needle_retrieval", 300):
            d, a = problem.data, problem.answers
            value = {r["id"]: r[d["field"]] for r in d["records"]}
            self.assertEqual(a["value_of_id"]["choice"], value[d["key"]])
            self.assertEqual(a["id_has_value"]["noul"], float(value[d["key"]] == d["proposed"]))
            self.assertEqual(a["id_listed"]["noul"], float(d["probe"] in value))
            for record in d["records"]:  # every rendering keeps every record
                self.assertIn(record["id"], problem.state)

    def test_record_aggregation_gold_follows_items(self):
        for problem in self.samples("record_aggregation", 300):
            d, a = problem.data, problem.answers
            members = [i for i in d["items"] if i["category"] == d["category"]]
            self.assertEqual(a["count_in_category"]["score"], len(members))
            self.assertEqual(a["any_out_of_stock"]["noul"], float(any(not i["in_stock"] for i in members)))
            self.assertEqual(a["total_above"]["noul"],
                             float(sum(i["quantity"] for i in members) > d["threshold"]))
            pool = members if d["scope_category"] else d["items"]
            top = max(i["quantity"] for i in pool)
            leaders = [i["item"] for i in pool if i["quantity"] == top]
            self.assertEqual(leaders, [a["largest_quantity"]["choice"]])

    def test_uncertain_policy_gold_is_the_exact_posterior(self):
        errors = {"prior": [], "report": []}
        for problem in self.samples("policy_under_uncertainty", 300):
            roles, governing, allowed = _uncertain_policy_answers(problem.state)
            answers = problem.answers
            self.assertAlmostEqual(answers["access_allowed"]["noul"], float(allowed), places=5)
            for question, exact in (("requester_role", roles), ("governing_policy", governing)):
                for option, p in answers[question]["probabilities"].items():
                    self.assertAlmostEqual(p, float(exact.get(option, 0)), places=9)
            # plugging a single role in, the most common one or the first report's, is clearly worse
            history = problem.data["prior"]
            for name, role in (("prior", max(history, key=history.get)), ("report", problem.data["reports"][0][1])):
                plugged = _uncertain_policy_answers({**problem.state, "role_evidence": {
                    "history": f"1 by {role}s", "reports": []}})[2]
                errors[name].append(abs(float(plugged) - float(allowed)))
        self.assertGreater(sum(errors["prior"]) / 300, 0.1)
        self.assertGreater(sum(errors["report"]) / 300, 0.05)
        sharp = sum(problem.answers["access_allowed"]["noul"] in (0.0, 1.0)
                    for problem in self.samples("policy_under_uncertainty", 300))
        self.assertTrue(40 < sharp < 150)  # the role is irrelevant in some problems, not most

    def test_table_lookup_gold_follows_tables(self):
        for problem in self.samples("table_lookup", 300):
            d, a = problem.data, problem.answers
            people = {p["name"]: p for p in d["people"]}
            target = people[d["person"]]
            matching = [p["name"] for p in d["people"]
                        if (p["team"], p["city"]) == (target["team"], target["city"])]
            self.assertEqual(matching, [a["find_person"]["choice"]])
            managers = {t["team"]: t["manager"] for t in d["teams"]}
            subject = people[d["subject"]]
            self.assertEqual(a["manager_of"]["choice"], managers[subject["team"]])
            self.assertEqual(a["started_before"]["noul"], float(subject["start_year"] < d["year"]))
            self.assertEqual(a["count_matching"]["score"], sum(
                p["city"] == d["city"] and p["start_year"] >= d["since"] for p in d["people"]))

    def test_arithmetic_gold_follows_state(self):
        def value(answer):
            return answer.get("choice", answer.get("noul", answer.get("score")))

        sizes, ranks = Counter(), Counter()
        for problem in self.samples("arithmetic", 600):
            d, got = problem.data, {q: value(a) for q, a in problem.answers.items()}
            sizes[len(got)] += 1
            if d["scenario"] == "order":
                cost = [l["unit_price"] * l["quantity"] for l in d["lines"]]
                total = sum(cost)
                due = total - d["discount"] if total >= d["threshold"] else total + d["shipping"]
                expected = {
                    "amount_due": due,
                    "largest_line": d["lines"][cost.index(max(cost))]["item"],
                    "lines_above": sum(c > d["line_cut"] for c in cost),
                    "random_line_bulk": round(sum(l["quantity"] >= d["units"] for l in d["lines"]) / len(cost), 6),
                    "within_budget": float(due <= d["budget"]),
                    "budget_use": 2 if due > d["budget"] else 0 if 2 * due <= d["budget"] else 1,
                }
                self.assertEqual(cost.count(max(cost)), 1)
            elif d["scenario"] == "ledger":
                balance, balances = d["start"], []
                for t in d["transactions"]:
                    balance += t["amount"] if t["type"] == "deposit" else -t["amount"]
                    balances.append(balance)
                n, deposits, change = len(balances), sum(t["type"] == "deposit" for t in d["transactions"]), balance - d["start"]
                expected = {
                    "final_balance": balance,
                    "went_negative": float(min(balances) < 0),
                    "lowest_day": f"day {d['transactions'][balances.index(min(balances))]['day']}",
                    "withdrawal_count": n - deposits,
                    "random_is_deposit": round(deposits / n, 6),
                    "net_change": 0 if change < -50 else 2 if change > 50 else 1,
                }
            else:
                clock, starts = d["start"], []
                for t in d["tasks"]:
                    starts.append(clock)
                    clock += t["minutes"] + d["gap"]
                finish, minutes = clock - d["gap"], [t["minutes"] for t in d["tasks"]]
                expected = {
                    "finish_time": f"{finish // 60:02d}:{finish % 60:02d}",
                    "done_by_deadline": float(finish <= d["deadline"]),
                    "longest_task": d["tasks"][minutes.index(max(minutes))]["task"],
                    "starts_before_noon": sum(s < 720 for s in starts),
                    "random_is_long": round(sum(m > d["long"] for m in minutes) / len(minutes), 6),
                }
            for qid, answer in got.items():
                want = expected[qid]
                if qid in ("amount_due", "final_balance"):  # formatted with a currency symbol
                    self.assertEqual(answer.lstrip("$€£"), str(want), qid)
                else:
                    self.assertEqual(answer, want, qid)
                spec = problem.questions[qid]
                if qid in ("amount_due", "final_balance", "finish_time"):
                    self.assertEqual(len(spec["criteria"]), 5)
                    ranks[sorted(spec["criteria"], key=lambda o: (len(o), o)).index(answer)] += 1
        self.assertEqual(set(sizes), {2, 3, 4, 5})
        self.assertGreater(min(ranks.values()), max(ranks.values()) / 2)  # gold rank is not a tell

    def test_partial_question_sets_build_with_null_labels(self):
        dataset = build_task("arithmetic", {"train": 300, "validation": 30, "test": 30}, [0, 2, 4])
        features = dataset["train"].features
        self.assertEqual(features["amount_due"].dtype, "string")  # open numeric vocabulary
        self.assertEqual(features["random_is_deposit"].dtype, "float32")  # graded probability
        for row in dataset["test"]:
            asked = json.loads(row["questions"])
            for qid in features:
                if qid in ("id", "level", "state", "questions", "answers"):
                    continue
                self.assertEqual(row[qid] is None, qid not in asked, qid)

    def test_rendered_tasks_vary_wording_and_build(self):
        for task in ("needle_retrieval", "record_aggregation", "table_lookup"):
            problems = self.samples(task, 100)
            for qid in problems[0].questions:
                wordings = {tuple(p.questions[qid]["instructions"].split()[:2]) for p in problems}
                self.assertGreater(len(wordings), 1, (task, qid))
            dataset = build_task(task, {"train": 300, "validation": 30, "test": 30}, [0, 1, 2])
            self.assertIsInstance(dataset["test"][0]["state"], str)
            self.assertFalse(dataset["test"][0]["state"].startswith('"'))


if __name__ == "__main__":
    unittest.main()
