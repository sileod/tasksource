import unittest
from collections import Counter

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

from tasksource.recast import recast_jev, render_typed_decision, render_typed_decision_group
from tasksource.jev.token_labels import normalize_token_label
from tasksource.jev.prompt_augmentations import (
    published_pair_style, published_question_style,
)
from tasksource.jev.augmentations import augment_jev_internal
from scripts.build_jev_dataset import (
    diverse_cap, diversify_published_prompts, exclude_publish_sources,
    fixed_source_audit, pretty_order, read_completed, slug, source_family,
    to_training_row, validate_decisions,
)


class RecastJevTest(unittest.TestCase):
    def test_classification(self):
        features = Features({
            "sentence1": Value("string"),
            "sentence2": Value("string"),
            "labels": ClassLabel(names=["entailment", "contradiction"]),
        })
        source = DatasetDict({"train": Dataset.from_dict({
            "sentence1": ["A"], "sentence2": ["B"], "labels": [1]
        }, features=features)})
        row = recast_jev(source, task="demo")["train"][0]
        self.assertEqual(row["criteria"], ["entailment", "contradiction"])
        self.assertEqual(row["answer"], "contradiction")
        self.assertEqual(row["task"], "demo")
        self.assertIn("text_A: A", row["state"])

    def test_states_undo_html_escaping(self):
        features = Features({"sentence1": Value("string"), "labels": ClassLabel(names=["a", "b"])})
        source = DatasetDict({"train": Dataset.from_dict({
            "sentence1": ["Tom &amp; Jerry &amp;amp; co &gt; you<br>next<br />line <b>kept</b>"],
            "labels": [0],
        }, features=features)})
        self.assertEqual(recast_jev(source)["train"][0]["state"],
                         "Tom & Jerry & co > you\nnext\nline <b>kept</b>")

    def test_multiple_choice_and_renderer(self):
        source = DatasetDict({"train": Dataset.from_dict({
            "inputs": ["Question"], "choice0": ["zero"],
            "choice1": ["one"], "labels": [0]
        })})
        row = recast_jev(source)["train"][0]
        request = render_typed_decision(row, model="openjev")
        self.assertEqual(row["answer"], "zero")
        self.assertEqual(
            list(request["questions"]["decision"]["criteria"]),
            ["zero", "one"],
        )

    def test_common_jev_training_schema(self):
        row = to_training_row({
            "state": "Question",
            "instructions": "Choose.",
            "criteria": ["zero", "one"],
            "label": 1,
        }, index=3, task_id="demo/task", split="train")
        self.assertEqual(row["kind"], "choice")
        self.assertEqual(row["options"], ["zero", "one"])
        self.assertEqual(row["target"], [0.0, 1.0])
        self.assertEqual(row["source"], "demo/task")
        self.assertEqual(row["variant"], "direct")
        self.assertEqual(row["split"], "train")

    def test_normalized_split_annotation(self):
        row = to_training_row({
            "state": "Question", "instructions": "Choose.",
            "criteria": ["a", "b"], "label": 0,
        }, index=0, task_id="demo/task", split="validation")
        self.assertEqual(row["split"], "dev")

    def test_internal_jev_augmentations_are_typed_and_idempotent(self):
        direct = Dataset.from_list([to_training_row({
            "state": "text_A: Example A\ntext_B: Example B",
            "instructions": "Choose the criterion that best describes the state.",
            "criteria": ["negative", "positive"],
            "label": 1,
        }, index=0, task_id="demo", split="train")])
        augmented = augment_jev_internal(
            direct, noul_rate=1.0, score_rate=1.0, permutation_rate=1.0,
            prompt_rate=1.0, paired_format_rate=1.0,
        )
        self.assertEqual(
            augmented["kind"],
            ["choice", "noul", "score", "choice", "choice", "choice"],
        )
        self.assertEqual(augmented["variant"], [
            "direct", "label_verification", "ordered_rubric",
            "criteria_permutation", "instruction_paraphrase", "paired_text_format",
        ])
        self.assertEqual(augmented[1]["options"], [])
        self.assertEqual(len(augmented[1]["target"]), 1)
        self.assertEqual(augmented[2]["options"], ["negative", "positive"])
        self.assertEqual(augmented[3]["options"], ["positive", "negative"])
        self.assertEqual(augmented[3]["target"], [1.0, 0.0])
        self.assertNotEqual(augmented[4]["question"], augmented[0]["question"])
        self.assertNotEqual(augmented[5]["state"], augmented[0]["state"])
        self.assertEqual(
            len(augment_jev_internal(augmented, 1.0, 1.0, 1.0, 1.0, 1.0)), 6
        )

    def test_task_questions_are_not_paraphrased(self):
        direct = Dataset.from_list([to_training_row({
            "state": "30g butter to cups?", "instructions": "Is this search query a well-formed question?",
            "criteria": ["not well-formed", "well-formed"], "label": 1,
        }, index=0, task_id="demo", split="train")])
        augmented = augment_jev_internal(direct, 0.0, 0.0, 0.0, 1.0, 0.0)
        self.assertEqual(augmented["question"], ["Is this search query a well-formed question?"])

    def test_label_verification_keeps_the_task_question(self):
        from tasksource.jev.augmentations import verification_question
        self.assertEqual(verification_question("Choose the criterion that best describes the state.", "positive"),
                         'Is "positive" the correct label for this example?')
        self.assertEqual(verification_question("Choose the criterion that best answers the question.", "Paris"),
                         'Is "Paris" the correct answer to the question?')
        self.assertEqual(verification_question("What stance does the tweet take on feminism?", "against"),
                         'What stance does the tweet take on feminism? Is "against" the correct answer?')

    def test_pretty_order_prefix_then_shuffled_tail(self):
        dataset = Dataset.from_dict({
            "source": ["b", "b", "b", "a", "a", "c"],
            "value": list(range(6)),
        })
        ordered = pretty_order(dataset, first_rows=4)
        self.assertEqual(ordered["source"][:4], ["a", "b", "c", "a"])
        self.assertEqual(ordered["value"][:4], [3, 0, 5, 4])
        self.assertEqual(sorted(ordered["value"][4:]), [1, 2])

    def test_pretty_order_spreads_trailing_sources(self):
        dataset = Dataset.from_dict({
            "source": ["tasks"] * 900 + ["procedural-typed-decisions/x"] * 100,
            "value": list(range(1000)),
        })
        ordered = pretty_order(dataset, first_rows=10)
        self.assertEqual(ordered["value"], pretty_order(dataset, first_rows=10)["value"])
        self.assertEqual(sorted(ordered["value"]), list(range(1000)))
        halves = [ordered["source"][10:505], ordered["source"][505:]]
        for half in halves:
            self.assertGreater(half.count("procedural-typed-decisions/x"), 25)

    def test_pretty_order_exposes_prompt_variants(self):
        dataset = Dataset.from_dict({
            "source": ["a"] * 3 + ["b"] * 3,
            "variant": ["direct", "instruction_paraphrase", "paired_text_format"] * 2,
            "value": list(range(6)),
        })
        ordered = pretty_order(dataset, first_rows=4)
        self.assertEqual(ordered["source"][:4], ["a", "b", "a", "b"])
        self.assertGreaterEqual(len(set(ordered["variant"][:4])), 2)
        prefix = set(ordered["value"][:4])
        self.assertEqual(sorted(ordered["value"][4:]), [i for i in range(6) if i not in prefix])

    def test_paired_public_style_keeps_related_questions_consistent(self):
        state = "text_A: Rain fell.\ntext_B: The ground is wet."
        styled, question = published_pair_style(
            state, "Classify the relationship between text_A and text_B.", 0.9
        )
        self.assertIn("A: Rain fell.", styled)
        self.assertNotIn("text_A", question)
        dataset = Dataset.from_dict({
            "group_id": ["demo:train:0", "demo:train:0"],
            "state": [state, state],
            "question": ["Choose the criterion.", "Choose the criterion."],
            "options": [["entailment", "contradiction"]] * 2,
        })
        varied = diversify_published_prompts(dataset)
        self.assertEqual(varied["state"][0], varied["state"][1])
        self.assertEqual(varied["question"][0], varied["question"][1])
        self.assertNotEqual(
            published_question_style(
                "Choose the criterion that best describes the state.",
                ["entailment", "contradiction"], state, 0.3,
            ),
            "Choose the criterion that best describes the state.",
        )

    def test_diverse_cap_covers_sources_and_preserves_order(self):
        dataset = Dataset.from_dict({
            "source": ["a"] * 8 + ["b"] * 2 + ["c"] * 2,
            "value": list(range(12)),
        })
        capped = diverse_cap(dataset, 6)
        self.assertEqual(set(capped["source"]), {"a", "b", "c"})
        self.assertEqual(capped["value"], sorted(capped["value"]))
        self.assertEqual(len(capped), 6)

    def test_diverse_cap_follows_family_weights(self):
        # anli/ is weighted 3 and linguisticprobing/ 0.25 in metadata/weights.py
        dataset = Dataset.from_dict({
            "source": ["anli/a1"] * 100 + ["plain"] * 100 + ["linguisticprobing/x"] * 100,
            "id": [f"s:train:{i}" for i in range(300)],
        })
        counts = Counter(diverse_cap(dataset, 85)["source"])
        self.assertGreater(counts["anli/a1"], 2 * counts["plain"])
        self.assertLess(counts["linguisticprobing/x"], counts["plain"])

    def test_diverse_cap_keeps_related_questions_together(self):
        dataset = Dataset.from_dict({
            "source": ["ner"] * 6,
            "id": [
                "ner:train:0:token-0", "ner:train:0:token-1",
                "ner:train:1:token-0", "ner:train:1:token-1",
                "ner:train:2:token-0", "ner:train:2:token-1",
            ],
        })
        capped = diverse_cap(dataset, 4)
        self.assertEqual(len(capped), 4)
        group_counts = {}
        for identifier in capped["id"]:
            group = identifier.rsplit(":", 1)[0]
            group_counts[group] = group_counts.get(group, 0) + 1
        self.assertEqual(sorted(group_counts.values()), [2, 2])

    def test_diverse_cap_balances_families_and_samples_configs(self):
        sources = ["family/a"] * 40 + ["family/b"] * 40 + ["other/task"] * 40
        dataset = Dataset.from_dict({
            "source": sources,
            "id": [f"{source.replace('/', '-')}:train:{index}" for index, source in enumerate(sources)],
        })
        capped = diverse_cap(dataset, 40)
        counts = Counter(capped["source"])
        self.assertEqual(counts["other/task"], 20)
        self.assertEqual(counts["family/a"], 10)
        self.assertEqual(counts["family/b"], 10)
        self.assertEqual(source_family("multilingual/xcsr/fr"), "multilingual/xcsr")
        self.assertEqual(source_family("graded/helpsteer"), "graded/helpsteer")
        self.assertEqual(source_family("procedural-typed-decisions/policy_applicability"),
                         "procedural-typed-decisions/policy_applicability")

    def test_completed_tasks_require_their_parquet_shards(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            (root / "build-report.jsonl").write_text(json.dumps({
                "task": "glue/rte", "status": "ok", "rows": {"train": 2},
            }) + "\n")
            report = root / "build-report.jsonl"
            self.assertEqual(read_completed(report, data_dir), set())
            (data_dir / f"train-{slug('glue/rte')}.parquet").touch()
            self.assertEqual(read_completed(report, data_dir), {"glue/rte"})

    def test_fixed_source_audit_separates_failures_and_renames(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            baseline = Path(directory) / "failures.json"
            baseline.write_text(json.dumps([
                {"task": "fixed"}, {"task": "broken"}, {"task": "old-name"},
            ]))
            audit = fixed_source_audit(
                baseline, {"fixed", "broken"},
                {"fixed": {"status": "ok"}, "broken": {
                    "status": "error", "error": "bad source",
                }},
            )
            self.assertEqual(audit["succeeded"], ["fixed"])
            self.assertEqual(audit["failed"], [
                {"task": "broken", "error": "bad source"},
            ])
            self.assertEqual(audit["not_in_current_catalog"], ["old-name"])

    def test_publish_target_validation_covers_all_three_primitives(self):
        good = Dataset.from_dict({
            "id": ["choice", "noul", "score"],
            "kind": ["choice", "noul", "score"],
            "options": [["a", "b"], [], ["low", "high"]],
            "target": [[0.25, 0.75], [0.4], [0.0, 1.0]],
        })
        validate_decisions(good, "train")
        bad = Dataset.from_dict({
            "id": ["bad"], "kind": ["score"],
            "options": [["low", "high"]], "target": [[0.0, 0.0]],
        })
        with self.assertRaisesRegex(ValueError, "Invalid score target"):
            validate_decisions(bad, "train")

    def test_token_classification_is_readable_bounded_and_deterministic(self):
        features = Features({
            "tokens": Sequence(Value("string")),
            "labels": Sequence(ClassLabel(names=["O", "B-PER", "I-PER"])),
        })
        source = DatasetDict({"train": Dataset.from_dict({
            "tokens": [["Obama", "met", "Obama"]],
            "labels": [[1, 0, 2]],
        }, features=features)})
        first = recast_jev(source, task="conll2003/ner_tags")["train"]
        second = recast_jev(source, task="conll2003/ner_tags")["train"]
        self.assertEqual(len(first), 2)
        self.assertEqual(first[:], second[:])
        self.assertIn("beginning of a person entity", first[0]["criteria"])
        self.assertIn("Target token at position", first[0]["state"])
        self.assertEqual(first[0]["state"].count("[TARGET:"), 1)
        self.assertEqual(first[0]["source_row"], first[1]["source_row"])
        self.assertNotEqual(first[0]["question_id"], first[1]["question_id"])
        grouped = render_typed_decision_group(first)
        self.assertEqual(len(grouped["questions"]), 2)
        self.assertEqual(grouped["state"], "Sentence: Obama met Obama")
        self.assertTrue(all("position" in q["instructions"] for q in grouped["questions"].values()))
        rows = [to_training_row(example, index, "conll2003/ner_tags", "train")
                for index, example in enumerate(first)]
        self.assertEqual(rows[0]["id"].rsplit(":", 1)[0], rows[1]["id"].rsplit(":", 1)[0])

    def test_pos_labels_are_expanded(self):
        features = Features({
            "tokens": Sequence(Value("string")),
            "labels": Sequence(ClassLabel(names=["NOUN", "PROPN", "VERB"])),
        })
        source = DatasetDict({"train": Dataset.from_dict({
            "tokens": [["Ada", "writes"]], "labels": [[1, 2]],
        }, features=features)})
        rows = recast_jev(source)["train"]
        self.assertEqual(rows[0]["criteria"], ["noun", "proper noun", "verb"])
        self.assertEqual(
            normalize_token_label("B-ORG"), "beginning of an organization entity"
        )
        self.assertEqual(
            normalize_token_label("I-creative-work"), "inside a creative work entity"
        )

    def test_anonymous_token_labels_are_rejected(self):
        features = Features({
            "tokens": Sequence(Value("string")),
            "labels": Sequence(ClassLabel(names=["LABEL_0", "LABEL_1"])),
        })
        source = DatasetDict({"train": Dataset.from_dict({
            "tokens": [["x"]], "labels": [[0]],
        }, features=features)})
        with self.assertRaisesRegex(NotImplementedError, "semantically readable"):
            recast_jev(source)

    def test_token_splits_remain_separate(self):
        features = Features({
            "tokens": Sequence(Value("string")),
            "labels": Sequence(ClassLabel(names=["O", "B-PER"])),
        })
        source = DatasetDict({
            "train": Dataset.from_dict({"tokens": [["Ada"]], "labels": [[1]]}, features=features),
            "test": Dataset.from_dict({"tokens": [["Grace"]], "labels": [[1]]}, features=features),
        })
        converted = recast_jev(source)
        self.assertEqual(set(converted), {"train", "test"})
        self.assertIn("Ada", converted["train"][0]["state"])
        self.assertIn("Grace", converted["test"][0]["state"])

    def test_large_token_ontology_is_rejected(self):
        features = Features({
            "tokens": Sequence(Value("string")),
            "labels": Sequence(ClassLabel(names=[f"semantic_{i}" for i in range(33)])),
        })
        source = DatasetDict({"train": Dataset.from_dict({
            "tokens": [["x"]], "labels": [[0]],
        }, features=features)})
        with self.assertRaisesRegex(NotImplementedError, "semantically readable"):
            recast_jev(source)

    def test_misaligned_token_labels_are_rejected(self):
        features = Features({
            "tokens": Sequence(Value("string")),
            "labels": Sequence(ClassLabel(names=["O", "B-PER"])),
        })
        source = DatasetDict({"train": Dataset.from_dict({
            "tokens": [["Ada", "writes"]], "labels": [[1]],
        }, features=features)})
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            recast_jev(source)

    def test_release_excludes_named_benchmark_families(self):
        dataset = Dataset.from_dict({
            "source": ["bigbench/a", "mmlu/b", "blimp/c", "glue/rte", "retired/task"],
            "value": [0, 1, 2, 3, 4],
        })
        kept = exclude_publish_sources(dataset, allowed_sources={"glue/rte"})
        self.assertEqual(kept["source"], ["glue/rte"])


if __name__ == "__main__":
    unittest.main()
