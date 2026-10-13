"""Independent truth-table, prose fidelity, difficulty and dataset regression tests."""

import json
import random
import unittest
from collections import Counter

from tasksource.jev.procedural import record_eligibility as task


def _oracle(expr, facts):
    """Second interpreter with independently expressed Kleene logic."""
    op = expr["op"]
    if op in ("eq", "ge", "le", "in", "sum_le"):
        if op == "sum_le":
            operands = [facts.get(f) for f in expr["fields"]]
            return None if any(x is None for x in operands) else sum(operands) <= expr["value"]
        candidate = facts.get(expr["field"])
        if candidate is None:
            return None
        if op == "eq":
            return candidate == expr["value"]
        if op == "ge":
            return candidate >= expr["value"]
        if op == "le":
            return candidate <= expr["value"]
        return candidate in expr["values"]
    if op == "not":
        return {True: False, False: True, None: None}[_oracle(expr["arg"], facts)]
    if op == "implies":
        p, q = (_oracle(expr[k], facts) for k in ("if", "then"))
        return True if p is False or q is True else False if p is True and q is False else None
    values = [_oracle(child, facts) for child in expr["args"]]
    if op == "and":
        return False if any(v is False for v in values) else True if all(v is True for v in values) else None
    if op == "or":
        return True if any(v is True for v in values) else False if all(v is False for v in values) else None
    raise ValueError(op)


def _classify(values):
    if any(value is False for value in values):
        return "ineligible"
    return "eligible" if all(value is True for value in values) else "undetermined"


def _samples(level, count=140):
    return (task.generate(random.Random(f"record_eligibility:{level}:{i}"), level)
            for i in range(count))


class RecordEligibilityTest(unittest.TestCase):
    def test_independent_labels_and_counterfactuals(self):
        for level in range(5):
            for problem in _samples(level, 230):
                state, hidden, answers = problem.state, problem.data, problem.answers
                predicates = hidden["requirements"]
                records = {r["id"]: r for r in hidden["source"]}
                classifications = {}
                for record_id, source in records.items():
                    results = [_oracle(rule["expr"], source["facts"]) for rule in predicates]
                    self.assertEqual(results, source["results"])
                    classifications[record_id] = _classify(results)
                    self.assertEqual(classifications[record_id], source["status"])
                probe = records[hidden["probe"]]
                statuses = [_oracle(rule["expr"], probe["facts"]) for rule in predicates]
                selected = hidden["constraint_index"]
                failures = [i for i, x in enumerate(statuses) if x is False]
                unknowns = [i for i, x in enumerate(statuses) if x is None]
                deciding = f"C{(failures or unknowns)[0] + 1}" if failures or unknowns else "none"
                displayed = [r["id"] for r in state["records"]]
                first = next((r for r in displayed if classifications[r] == "eligible"), "none")
                self.assertEqual(answers["probe_eligibility"]["choice"], classifications[hidden["probe"]])
                self.assertEqual(answers["requirement_status"]["choice"],
                                 {True: "met", False: "violated", None: "unknown"}[statuses[selected]])
                self.assertEqual(answers["eligible_count"]["score"],
                                 min(sum(x == "eligible" for x in classifications.values()), 9))
                self.assertEqual(answers["first_eligible"]["choice"], first)
                self.assertEqual(answers["decisive_constraint"]["choice"], deciding)
                self.assertEqual(answers["eligible_if_relaxed"]["noul"],
                                 float(_classify([v for i, v in enumerate(statuses) if i != selected]) == "eligible"))
                self.assertEqual(answers["probe_fully_eligible"]["noul"],
                                 float(classifications[hidden["probe"]] == "eligible"))

    def test_surface_represents_every_source_fact(self):
        for level in range(5):
            for problem in _samples(level, 100):
                state, hidden = problem.state, problem.data
                self.assertEqual(set(state), {"user_request", "records", "semantics"})
                self.assertEqual(len(state["records"]), len(hidden["source"]))
                self.assertEqual(len(set(r["id"] for r in state["records"])), len(state["records"]))
                descriptions = {r["id"]: r["description"] for r in state["records"]}
                for i, requirement in enumerate(hidden["requirements"], 1):
                    self.assertIn(f"C{i}. {requirement['text']}", state["user_request"])
                for entry in hidden["source"]:
                    description = descriptions[entry["id"]]
                    for field, value in entry["facts"].items():
                        if value is None:
                            name = task.NAMES[field]
                            self.assertTrue(any(text in description for text in
                                                (f"does not specify {name}", f"No information about {name}",
                                                 f"sources conflict about {name}")), (level, field, description))
                        elif value is False and field in task.NEGATIVE:
                            self.assertIn(task.NEGATIVE[field], description)
                        else:
                            unit = "day" if field == "lead_days" and value == 1 else "days"
                            variants = [pattern.format(v=value, unit=unit) for pattern in task.DESCRIPTIONS[field]]
                            self.assertTrue(any(text in description for text in variants),
                                            (level, field, value, description))

    def test_typed_answers_and_seed_reproducibility(self):
        for level in range(5):
            for i, problem in enumerate(_samples(level, 150)):
                self.assertEqual(set(problem.questions), set(problem.answers))
                json.dumps([problem.state, problem.questions, problem.answers], allow_nan=False)
                for key, spec in problem.questions.items():
                    answer = problem.answers[key]
                    self.assertEqual(spec["type"], answer["type"])
                    if spec["type"] == "noul":
                        self.assertTrue(0 <= answer["noul"] <= 1)
                    elif spec["type"] == "score":
                        self.assertLessEqual(len(spec["criteria"]), 10)
                        self.assertIn(answer["score"], range(len(spec["criteria"])))
                    else:
                        self.assertIn(answer["choice"], spec["criteria"])
                        self.assertEqual(set(answer["probabilities"]), set(spec["criteria"]))
                        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1)
                if i == 0:
                    a = task.generate(random.Random(301 + level), level)
                    b = task.generate(random.Random(301 + level), level)
                    self.assertEqual(a, b)

    def test_progression_and_nonautomatic_shortcuts(self):
        mean_lengths, total_sizes = [], []
        for level in range(5):
            problems = list(_samples(level, 250))
            state_ids = [json.dumps(p.state, sort_keys=True) for p in problems]
            self.assertEqual(len(set(state_ids)), len(state_ids))
            counts = Counter(p.answers["eligible_count"]["score"] for p in problems)
            self.assertGreaterEqual(len(counts), 5)
            self.assertTrue(all(len(p.data["requirements"]) == task.N_CONSTRAINTS[level] for p in problems))
            total_sizes.append(sum(len(p.state["records"]) for p in problems) / len(problems))
            mean_lengths.append(sum(len(json.dumps(p.state)) for p in problems) / len(problems))
            if level == 2:
                self.assertTrue(all(any(r["kind"] in ("mode", "support", "total")
                                        for r in p.data["requirements"]) for p in problems))
            if level >= 3:
                self.assertTrue(all(any(r["kind"] == "conditional" for r in p.data["requirements"])
                                    for p in problems))
            if level == 4:
                simple_first = 0
                for p in problems:
                    probe = next(r for r in p.data["source"] if r["id"] == p.data["probe"])
                    simple_first += (probe["results"][0] is True) == (probe["status"] == "eligible")
                self.assertLess(simple_first / len(problems), 0.7)
        self.assertEqual(sorted(total_sizes), total_sizes)
        self.assertEqual(sorted(mean_lengths), mean_lengths)

    def test_dataset_and_jev_conversion(self):
        from scripts.build_procedural_jev import build_task
        from tasksource.jev.procedural import jev_rows
        ds = build_task("record_eligibility", {"train": 350, "validation": 40, "test": 40}, range(5))
        states = [set(ds[split]["state"]) for split in ("train", "validation", "test")]
        self.assertFalse(states[0] & states[1] or states[0] & states[2] or states[1] & states[2])
        for row in ds["validation"].select(range(15)):
            answers = json.loads(row["answers"])
            questions = json.loads(row["questions"])
            self.assertEqual(set(answers), set(questions))
            jev = jev_rows(row, "procedural-typed-decisions/record_eligibility", "validation", 0)
            self.assertEqual(len(jev), len(questions))
            for item in jev:
                self.assertEqual(len(item["target"]),
                                 1 if item["kind"] == "noul" else len(item["options"]))


if __name__ == "__main__":
    unittest.main()
