import unittest
from collections import Counter

from datasets import Dataset, DatasetDict, concatenate_datasets

from tasksource.preprocess import MultipleChoice
from tasksource.recast import recast_jev
from tasksource.jev.augmentations import augment_jev_internal
from tasksource.jev.derived import (
    VARIANT, add_packed_classification, derive_target, render_state,
    verify_packed_rows, InSubset, MostCommon, OPERATORS, _select_questions,
)
from tasksource.jev.length import LengthBudget, render_request
from tasksource.jev.options import choice_permutation, gold_position_violations, permute_choices
from scripts.build_jev_dataset import (
    TRAINING_FEATURES, add_question_groups, diverse_cap, drop_train_overlap,
    content_keys, to_training_row, validate_decisions,
)
from datasets import ClassLabel, Features, Value
from tasksource.tasks import _copa_input, _esci_product


def mc_source(rows=4000, options=4):
    """Legacy-shaped multiple choice: gold always in slot 0."""
    return DatasetDict({"train": Dataset.from_dict({
        "inputs": [f"Question {i}" for i in range(rows)],
        **{f"choice{c}": [f"q{i} candidate {'wxyz'[c]}" for i in range(rows)] for c in range(options)},
        "labels": [0] * rows,
    })})


def classification_rows(n=300, names=("negative", "positive"), task="demo/cls",
                        split="train", label=lambda i: int(i % 3 == 0), text=None):
    rows = [to_training_row({
        "state": text(i) if text else f"{task} example {i}",
        "instructions": "Choose the criterion that best describes the state.",
        "criteria": list(names), "label": label(i),
    }, index=i, task_id=task, split=split) for i in range(n)]
    return Dataset.from_list(rows, features=TRAINING_FEATURES)


def packed(dataset, **kwargs):
    audit = []
    out = add_packed_classification(dataset, "Classification", audit=audit, **kwargs)
    return out, [row for row in out if row["variant"] == VARIANT], audit


class MultipleChoicePermutationTest(unittest.TestCase):
    def test_deterministic_and_label_preserving(self):
        first = recast_jev(mc_source(200), task="demo/mc")["train"]
        second = recast_jev(mc_source(200), task="demo/mc")["train"]
        self.assertEqual(first["criteria"], second["criteria"])
        for index, row in enumerate(first):
            self.assertEqual(row["answer"], f"q{index} candidate w")
            self.assertEqual(row["criteria"][row["label"]], row["answer"])
            self.assertEqual(sorted(row["criteria"]), [f"q{index} candidate {c}" for c in "wxyz"])
        self.assertNotEqual(set(first["label"]), {0})

    def test_gold_position_is_near_uniform(self):
        labels = recast_jev(mc_source(4000), task="demo/mc")["train"]["label"]
        for slot, count in Counter(labels).items():
            self.assertAlmostEqual(count / 4000, 0.25, delta=0.03, msg=slot)

    def test_positional_options(self):
        criteria = ["red", "blue", "green", "All of the above"]
        slots = set()
        for i in range(20):
            permuted, label = permute_choices(criteria, 3, f"row{i}")
            self.assertEqual(permuted[label], "All of the other options")
            slots.add(label)
        self.assertGreater(len(slots), 1)  # reworded, so it moves freely
        for fixed in (["red", "All of the above", "green"], ["red", "blue", "Both of the above"],
                      ["red", "blue", "A and B"], ["I only", "II only", "I and III only"],
                      ["x", "y", "statements 1, 2 and 4"], ["x", "y", "A, B, and C"]):
            self.assertEqual(permute_choices(fixed, 0, "x"), (fixed, 0), fixed)
        for free in (["Vitamin A and B12", "iron", "zinc"], ["Plan B", "Plan C", "none of them"]):
            self.assertIsNotNone(choice_permutation(free, "x"), free)
        long_list = ",\n    ".join("63294342545525455533") + ",\n\nAre you seeking an essay?"
        self.assertIsNotNone(choice_permutation([long_list, "no"], "x"))  # no catastrophic backtracking

    def test_gold_position_validation(self):
        sources = ["biased"] * 200 + ["fine"] * 200
        targets = [[1.0, 0.0]] * 200 + [[1.0, 0.0], [0.0, 1.0]] * 100
        self.assertEqual(set(gold_position_violations(sources, targets)), {"biased"})
        self.assertEqual(gold_position_violations(["small"] * 10, [[1.0, 0.0]] * 10), {})

    def test_jev_preprocessing_keeps_order_and_all_options(self):
        source = DatasetDict({"train": Dataset.from_dict({
            "q": ["a", "b"] * 10,
            "opts": [["w", "x", "y", "z", "v", "u"], ["p", "q", "r"]] * 10,
            "label": [2, 1] * 10,
        })})
        preprocessing = MultipleChoice("q", choices_list="opts", labels="label")
        legacy = preprocessing(DatasetDict(source), gold_first=True)["train"]
        self.assertEqual(set(legacy["labels"]), {0})
        jev = preprocessing(DatasetDict(source), gold_first=False, max_options=None)["train"]
        jev = jev.sort("inputs")
        self.assertEqual(jev["labels"][0], 2)
        jev = jev.select([0, len(jev) - 1])
        self.assertEqual([jev[0][f"choice{i}"] for i in range(6)], ["w", "x", "y", "z", "v", "u"])
        self.assertIsNone(jev[1]["choice3"])
        row = recast_jev(DatasetDict({"train": jev}), task="t")["train"][1]
        self.assertEqual(sorted(row["criteria"]), ["p", "q", "r"])
        self.assertEqual(row["answer"], "q")


class PackedClassificationTest(unittest.TestCase):
    def test_rate_uniqueness_and_additive(self):
        dataset = classification_rows(300)
        out, rows, audit = packed(dataset, rate=0.10)
        members = [m["id"] for record in audit for m in record["members"]]
        self.assertLessEqual(len(members), 30)
        self.assertGreater(len(members), 20)
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual(out.select(range(300))["id"], dataset["id"])
        for record in audit:
            self.assertTrue(2 <= len(record["members"]) <= 4)
        for group in Counter(r["id"].rsplit(":", 1)[0] for r in rows).values():
            self.assertTrue(3 <= group <= 4)

    def test_never_crosses_source_or_split(self):
        dataset = concatenate_datasets([
            classification_rows(100, task="a/x"), classification_rows(100, task="b/y"),
            classification_rows(100, task="a/x", split="validation"),
        ])
        _, _, audit = packed(dataset, rate=0.2)
        by_id = {row["id"]: row for row in dataset}
        for record in audit:
            keys = {(by_id[m["id"]]["source"], by_id[m["id"]]["split"]) for m in record["members"]}
            self.assertEqual(keys, {(record["source"], record["split"])})
        self.assertEqual(len({(r["source"], r["split"]) for r in audit}), 3)

    def test_label_patterns_cover_mixed_and_uniform_packs(self):
        _, _, audit = packed(classification_rows(600), rate=0.10)
        patterns = Counter(
            "".join(sorted(m["label"][0] for m in record["members"])) for record in audit
        )
        self.assertTrue({"nn", "np", "pp", "nnp", "npp"} <= set(patterns), patterns)

    def test_multiclass_packs_mix_labels(self):
        names = ("a", "b", "c", "d", "e")
        _, _, audit = packed(classification_rows(400, names=names, label=lambda i: i % 5), rate=0.10)
        for record in audit:
            self.assertIn(len({m["label"] for m in record["members"]}), (1, 2, 3))
        sizes = Counter(len({m["label"] for m in r["members"]}) for r in audit)
        self.assertTrue({1, 2, 3} <= set(sizes), sizes)  # so "all the same" has both answers

    def test_question_choice_ignores_answers(self):
        # the same label layout seen through different targets asks the same questions
        a = _select_questions([{"question_id": q, "kind": "noul", "target": [1.0], "_family": f}
                               for q, f in (("same-A-B", "relation"), ("exists-0", "aggregate"), ("in-A-1", "relation"))], "s", 2)
        b = _select_questions([{"question_id": q, "kind": "noul", "target": [0.0], "_family": f}
                               for q, f in (("same-A-B", "relation"), ("exists-0", "aggregate"), ("in-A-1", "relation"))], "s", 2)
        self.assertEqual([r["question_id"] for r in a], [r["question_id"] for r in b])
        for labels in ([0, 1], [1, 0], [2, 2]):
            self.assertEqual(InSubset().params(labels, 3, "pack"), InSubset().params([0, 0], 3, "pack"))

    def test_targets_recompute_independently(self):
        dataset = classification_rows(300, names=("x", "y", "z"), label=lambda i: i % 3)
        out, rows, audit = packed(dataset, rate=0.10)
        by_id = {row["id"]: row for row in dataset}
        for record in audit:
            labels = [by_id[m["id"]]["target"].index(1.0) for m in record["members"]]
            for row in rows:
                if row["id"].startswith(record["group_id"] + ":"):
                    kind, options, target = derive_target(
                        row["id"].rsplit(":", 1)[1], labels, ["x", "y", "z"]
                    )
                    self.assertEqual((row["kind"], row["options"], row["target"]),
                                     (kind, options, target))

    def test_verifier_rejects_tampered_targets(self):
        dataset = classification_rows(200)
        _, rows, audit = packed(dataset, rate=0.10)
        tampered = [dict(row) for row in rows]
        noul = next(row for row in tampered if row["kind"] == "noul")
        noul["target"] = [1.0 - noul["target"][0]]
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_packed_rows(dataset.to_list(), tampered, audit, 0.10, LengthBudget())
        swapped = [dict(m) for m in audit[0]["members"]]
        swapped[0]["id"], swapped[1]["id"] = swapped[1]["id"], swapped[0]["id"]
        with self.assertRaisesRegex(ValueError, "state does not match"):
            verify_packed_rows(dataset.to_list(), rows, [{**audit[0], "members": swapped}, *audit[1:]],
                               0.10, LengthBudget())

    def test_ties_skip_most_common(self):
        self.assertFalse(MostCommon().is_valid([0, 1], ("common",)))
        self.assertFalse(MostCommon().is_valid([0, 0, 1, 1], ("common",)))
        self.assertTrue(MostCommon().is_valid([0, 1, 1], ("common",)))
        with self.assertRaises(ValueError):
            derive_target("most-common", [0, 1], ["a", "b"])
        _, rows, audit = packed(classification_rows(600), rate=0.10)
        by_group = {r["group_id"]: [m["label"] for m in r["members"]] for r in audit}
        for row in rows:
            if row["id"].endswith(":most-common"):
                top = Counter(by_group[row["id"].rsplit(":", 1)[0]]).most_common()
                self.assertTrue(len(top) == 1 or top[0][1] > top[1][1])

    def test_item_order_does_not_change_aggregates(self):
        names = ["a", "b", "c"]
        labels = [0, 2, 2, 1]
        for question_id in ("exists-2", "all-same", "count-2", "most-common", "exists-1"):
            expected = derive_target(question_id, labels, names)
            for operator in OPERATORS:
                if question_id.startswith(operator.name):
                    params = next(p for p in operator.params(labels, 3, "s")
                                  if operator.question_id(p) == question_id)
                    for order in ([3, 2, 1, 0], [1, 0, 3, 2]):
                        permuted = [labels[i] for i in order]
                        self.assertEqual(operator.target(permuted, params, 3), expected[2])
                        self.assertEqual(derive_target(question_id, permuted, names), expected)

    def test_length_budget_uses_full_request(self):
        dataset = classification_rows(200, text=lambda i: "word " * (40 + i))
        budget = LengthBudget(max_tokens=1500)
        _, rows, audit = packed(dataset, rate=0.10, budget=budget)
        self.assertTrue(audit)
        groups = {}
        for row in rows:
            groups.setdefault(row["id"].rsplit(":", 1)[0], []).append(
                {**row, "question_id": row["id"].rsplit(":", 1)[1]})
        for group in groups.values():
            text = render_request(group[0]["state"], group)
            self.assertLessEqual(budget.count(text), 1500)
            self.assertGreater(len(text.encode()), len(group[0]["state"].encode()))
        _, none, _ = packed(dataset, rate=0.10, budget=LengthBudget(max_tokens=200))
        self.assertEqual(none, [])

    def test_state_format_and_item_permutation(self):
        _, rows, audit = packed(classification_rows(300), rate=0.10)
        first_letter_labels = Counter(record["members"][0]["label"] for record in audit)
        self.assertEqual(len(first_letter_labels), 2)
        state = rows[0]["state"]
        self.assertTrue(state.startswith("Item A:\n"))
        self.assertIn("\n\nItem B:\n", state)
        self.assertEqual(render_state(["x", "y"]), "Item A:\nx\n\nItem B:\ny")

    def test_ineligible_rows_are_not_packed(self):
        header = classification_rows(100, text=lambda i: f"Item A:\nnested {i}")
        self.assertEqual(packed(header, rate=0.5)[1], [])
        mc = classification_rows(100)
        self.assertEqual(len(add_packed_classification(mc, "MultipleChoice", rate=0.5)), 100)
        soft = Dataset.from_list(
            [{**row, "target": [0.5, 0.5]} for row in classification_rows(100)],
            features=TRAINING_FEATURES)
        self.assertEqual(packed(soft, rate=0.5)[1], [])

    def test_deterministic(self):
        first = packed(classification_rows(300), rate=0.10)[0]
        second = packed(classification_rows(300), rate=0.10)[0]
        self.assertEqual(first.to_list(), second.to_list())

    def test_groups_survive_diverse_cap(self):
        dataset = concatenate_datasets([
            packed(classification_rows(300, task=f"fam{k}/t"), rate=0.10)[0] for k in range(3)
        ])
        dataset = add_question_groups(dataset)
        full = Counter(dataset["group_id"])
        capped = diverse_cap(dataset, 500)
        for group, count in Counter(capped["group_id"]).items():
            self.assertEqual(count, full[group], group)
        self.assertTrue(any(":pack-" in g for g in capped["group_id"]))

    def test_augmentation_skips_packed_rows(self):
        out = packed(classification_rows(200), rate=0.10)[0]
        augmented = augment_jev_internal(out, noul_rate=1.0, permutation_rate=1.0,
                                         prompt_rate=1.0, paired_format_rate=1.0)
        packed_ids = {row["id"] for row in out if row["variant"] == VARIANT}
        for identifier in augmented["id"]:
            self.assertFalse(any(identifier.startswith(p + ":") for p in packed_ids))


class ReleaseQualityTest(unittest.TestCase):
    def test_invalid_options_are_rejected(self):
        for options in (["a", None], ["a", " "], ["a", "a"]):
            rows = Dataset.from_list([{
                "id": "x:train:0", "kind": "choice", "options": options,
                "target": [1.0, 0.0],
            }])
            with self.assertRaises(ValueError, msg=options):
                validate_decisions(rows, "train")

    def test_recast_drops_bad_mc_rows_and_rejects_bad_ontologies(self):
        source = DatasetDict({"train": Dataset.from_dict({
            "inputs": ["q0", "q1", "q2", "q3"],
            "choice0": ["a", "same", "a", None], "choice1": ["b", "same", "", "b"],
            "choice2": ["c", "other", "c", "c"], "labels": [0, 0, 0, 0],
        })})
        rows = recast_jev(source, task="t")["train"]
        self.assertEqual(rows["state"], ["q0"])
        features = Features({"sentence1": Value("string"),
                             "labels": ClassLabel(names=["yes", "None", "yes "])})
        bad = DatasetDict({"train": Dataset.from_dict(
            {"sentence1": ["s"], "labels": [0]}, features=features)})
        with self.assertRaisesRegex(ValueError, "unique"):
            recast_jev(bad)

    def test_eval_rows_overlapping_train_are_dropped_by_group(self):
        long_text = lambda i: f"A sufficiently long review sentence, number {i}." if i else "okay"
        train = packed(classification_rows(100, split="train", text=long_text), rate=0.10)[0]
        shared = classification_rows(100, split="test", text=long_text)
        fresh = classification_rows(100, split="test", task="other/cls")
        test, _, audit = packed(concatenate_datasets([shared, fresh]), rate=0.10)
        test = augment_jev_internal(test, noul_rate=1.0)
        kept, dropped = drop_train_overlap(test, content_keys(train))
        self.assertTrue(dropped)
        self.assertEqual(set(kept["source"]) - {"demo/cls"}, {"other/cls"})
        self.assertEqual(  # short generic texts recur naturally and are kept
            {s for s, v in zip(kept["state"], kept["variant"]) if v == "direct" and "review" not in s
             and "other/cls" not in s}, {"okay"})
        self.assertIn(VARIANT, kept["variant"])
        self.assertIn("label_verification", kept["variant"])

    def test_source_annotation_fixes(self):
        self.assertEqual(
            _copa_input({"premise": "It rained.", "question": "effect"}),
            "It rained. What happened as a result?")
        self.assertEqual(_esci_product({
            "product_title": "Fan", "product_brand": "Acme", "product_color": None,
            "product_description": "None", "product_bullet_point": "Quiet.",
        }), "Fan\nAcme\nQuiet.")


if __name__ == "__main__":
    unittest.main()
