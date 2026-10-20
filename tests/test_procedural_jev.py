"""tasksource side of the procedural Jev data: published schema, splits, balance and Jev conversion. The generators
and their oracle tests live in reasoning-core (reasoning_core/tasks/jev/)."""
import json
import random
import unittest
from collections import Counter

from datasets import Dataset

from tasksource.jev.augmentations import augment_jev_internal
from tasksource.jev.procedural import TASKS, jev_rows
from tasksource.jev.recast import render_typed_decision_group
from scripts.build_jev_dataset import TRAINING_FEATURES, share_cap, source_row_group
from scripts.build_procedural_jev import MAX_LABEL_SHARE, build_task, generate_split

try:  # the generators live in reasoning-core, a build-time dependency
    import reasoning_core
except ImportError:
    reasoning_core = None
needs_generators = unittest.skipUnless(reasoning_core, "reasoning-core is not installed")


@needs_generators
class ProceduralJevTest(unittest.TestCase):
    def samples(self, task, n=200):
        rng = random.Random(task)
        return [TASKS[task].generate(rng, rng.randrange(5)) for _ in range(n)]

    def test_answers_are_well_typed(self):
        for task in TASKS:
            rng = random.Random(task)
            for problem in (TASKS[task].generate(rng, level) for level in range(5) for _ in range(40)):
                self.assertEqual(set(problem.questions), set(problem.answers))
                for qid, spec in problem.questions.items():
                    answer = problem.answers[qid]
                    self.assertEqual(spec["type"], answer["type"])
                    if spec["type"] == "choice":
                        self.assertIn(answer["choice"], spec["criteria"])
                    elif spec["type"] == "noul":
                        self.assertTrue(0.0 <= answer["noul"] <= 1.0)
                    elif spec["type"] == "score":
                        self.assertLessEqual(len(spec["criteria"]), 10, (task, qid))  # the Jev API limit
                        self.assertIn(answer["score"], range(len(spec["criteria"])), (task, qid))

    def test_generation_is_deterministic(self):
        for task in TASKS:
            a = TASKS[task].generate(random.Random(1), 2)
            b = TASKS[task].generate(random.Random(1), 2)
            self.assertEqual(a, b)

    def test_generation_does_not_depend_on_the_hash_seed(self):
        import os, subprocess, sys
        script = ("import hashlib, json, random\nfrom tasksource.jev.procedural import TASKS\n"
                  "for t in sorted(TASKS):\n    h = hashlib.sha1()\n    for i in range(20):\n"
                  "        p = TASKS[t].generate(random.Random(f'x{i}'), i % 5)\n"
                  "        h.update(json.dumps([p.state, p.questions, p.answers], sort_keys=True, default=str).encode())\n"
                  "    print(t, h.hexdigest())")
        outputs = [subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True,
                                  env={**os.environ, "PYTHONHASHSEED": seed}).stdout for seed in ("1", "2")]
        self.assertEqual(*outputs)

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
        self.assertLessEqual(len(ids), 30)
        groups = Counter(source_row_group(i) for i in ids)
        sizes = Counter(source_row_group(row["id"]) for row in procedural)
        self.assertTrue(groups and all(sizes[group] == n for group, n in groups.items()))

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

    def test_pretty_order_cycles_levels_then_shuffles(self):
        from scripts.build_procedural_jev import pretty_order
        rows = [{"level": i % 5, "i": i} for i in range(100)]
        rows.sort(key=lambda r: r["level"])
        ordered = pretty_order(rows, first_rows=20)
        self.assertEqual([r["level"] for r in ordered[:10]], [0, 1, 2, 3, 4] * 2)
        self.assertEqual(sorted(r["i"] for r in ordered), list(range(100)))
        self.assertNotEqual([r["level"] for r in ordered[20:]], sorted(r["level"] for r in ordered[20:]))
        mixed = pretty_order([{"level": i % 5, "task": "ab"[i % 2]} for i in range(100)], 20,
                             key=lambda r: (r["level"], r["task"]))
        self.assertEqual([(r["level"], r["task"]) for r in mixed[:4]], [(0, "a"), (0, "b"), (1, "a"), (1, "b")])

    def test_option_counts_vary(self):
        for task, qid, low, high in (("needle_retrieval", "value_of_id", 6, 20), ("table_lookup", "find_person", 5, 30),
                                     ("entity_belief_tracking", "world_location", 4, 16),
                                     ("taxonomy_routing", "route", 4, 50)):
            sizes = Counter(len(p.questions[qid]["criteria"]) for p in self.samples(task, 300))
            self.assertLessEqual(min(sizes), low, task)
            self.assertGreaterEqual(max(sizes), high, task)
            self.assertLess(max(sizes.values()) / 300, 0.35, (task, sizes))  # no single count dominates

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
            for qid in {q for p in problems for q in p.questions}:  # some questions are asked from a level on
                wordings = {tuple(p.questions[qid]["instructions"].split()[:2]) for p in problems if qid in p.questions}
                self.assertGreater(len(wordings), 1, (task, qid))
            dataset = build_task(task, {"train": 300, "validation": 30, "test": 30}, [0, 1, 2])
            self.assertIsInstance(dataset["test"][0]["state"], str)
            self.assertFalse(dataset["test"][0]["state"].startswith('"'))


@needs_generators
class LabelBalanceTest(unittest.TestCase):
    def test_no_yes_no_or_choice_label_dominates_a_question(self):
        for task in ("event_triggered_execution", "instruction_execution", "graph_navigation"):
            rows, _ = generate_split(task, "train", 400, [0, 1, 2, 3, 4], set())
            counts = {}
            for row in rows:
                for qid, answer in json.loads(row["answers"]).items():
                    if answer["type"] == "noul":
                        counts.setdefault(qid, Counter())[answer["noul"] >= 0.5] += 1
                    elif answer["type"] == "choice":
                        counts.setdefault(qid, Counter())[answer["choice"]] += 1
            for qid, labels in counts.items():
                share = labels.most_common(1)[0][1] / sum(labels.values())
                self.assertLessEqual(share, MAX_LABEL_SHARE + 0.01, (task, qid))


if __name__ == "__main__":
    unittest.main()
