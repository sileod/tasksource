import unittest

from datasets import Dataset, DatasetDict

from tasksource.preprocess import cast_explicit_label_values
from tasksource.tasks import (
    _cogalexv_relations,
    _logiqa_options,
    _parse_jeggers_riddle_choices,
    _strip_option_prefix,
    _support_shift_name,
    _twentyquestions_answers,
    _utilitarianism_comparisons,
)


class MigratedLoaderHelperTest(unittest.TestCase):
    def test_utilitarianism_orientation_is_seeded_and_semantic(self):
        import random

        source = DatasetDict({"train": Dataset.from_dict({
            "baseline": ["better", "pleasant"],
            "less_pleasant": ["worse", "unpleasant"],
        })})
        random.seed(97)
        state = random.getstate()
        first = _utilitarianism_comparisons(source)
        second = _utilitarianism_comparisons(source)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(first["train"]["comparison"],
                         second["train"]["comparison"])
        for row, better, worse in zip(
            first["train"], ["better", "pleasant"], ["worse", "unpleasant"]
        ):
            expected = (
                f'"{better}" is better than "{worse}"'
                if row["label"] == 1 else
                f'"{worse}" is better than "{better}"'
            )
            self.assertEqual(row["comparison"], expected)

    def test_twentyquestions_ontology_is_fixed_before_sampling(self):
        source = DatasetDict({
            "train": Dataset.from_dict({"answer": [None, "always", "never"]}),
            "validation": Dataset.from_dict({"answer": ["irrelevant"]}),
        })
        mapped = _twentyquestions_answers(source)
        self.assertEqual(len(mapped["train"]), 2)
        self.assertIn("irrelevant", mapped["train"].features["answer"].names)
        self.assertEqual(mapped["validation"][0]["answer"], 5)

    def test_cogalexv_relations_are_readable_and_complete(self):
        source = DatasetDict({
            "train": Dataset.from_dict({"relation": ["ANT", "SYN"]}),
            "test": Dataset.from_dict({"relation": ["PART_OF"]}),
        })
        mapped = _cogalexv_relations(source)
        self.assertEqual(mapped["test"].features["relation"].names[2],
                         "part-of relation")

    def test_persuasion_shift_labels_are_signed_and_readable(self):
        self.assertEqual(_support_shift_name(-1), "support decreases by 1 point")
        self.assertEqual(_support_shift_name(0), "no change in support")
        self.assertEqual(_support_shift_name(2), "support increases by 2 points")

    def test_explicit_numeric_label_ontology_preserves_meaning(self):
        source = DatasetDict({
            "train": Dataset.from_dict({"labels": [1, 5]}),
            "test": Dataset.from_dict({"labels": [2]}),
        })
        names = {stars: f"{stars} stars" for stars in range(1, 6)}
        mapped = cast_explicit_label_values(source, names)
        self.assertEqual(mapped["train"].features["labels"].names,
                         list(names.values()))
        self.assertEqual(mapped["train"]["labels"], [0, 4])
        self.assertEqual(mapped["test"]["labels"], [1])

    def test_explicit_numeric_label_ontology_rejects_unmapped_values(self):
        source = DatasetDict({"train": Dataset.from_dict({"labels": [0, 3]})})
        with self.assertRaisesRegex(ValueError, "Unmapped labels"):
            cast_explicit_label_values(source, {0: "no", 1: "yes"})

    def test_aces_ontology_is_fixed_before_sampling(self):
        from tasksource.parked import _aces_phenomena_labels

        source = DatasetDict({
            "train": Dataset.from_dict({"phenomena": ["addition", "deletion"]}),
            "validation": Dataset.from_dict({"phenomena": ["deletion"]}),
        })
        mapped = _aces_phenomena_labels(source)
        self.assertEqual(mapped["train"].features["phenomena"].names,
                         ["addition", "deletion"])
        self.assertEqual(mapped["validation"][0]["phenomena"], 1)

    def test_x_fact_ontology_covers_rare_dev_label(self):
        from tasksource.multilingual_tasks import _x_fact_labels

        source = DatasetDict({
            "train": Dataset.from_dict({"label": ["false", "other"]}),
            "dev": Dataset.from_dict({"label": ["other"]}),
        })
        mapped = _x_fact_labels(source)
        self.assertEqual(mapped["dev"][0]["label"], 1)

    def test_strip_option_prefix(self):
        self.assertEqual(_strip_option_prefix("A.Some text"), "Some text")
        self.assertEqual(_strip_option_prefix("B: other"), "other")
        self.assertEqual(_strip_option_prefix("plain"), "plain")

    def test_logiqa_options_parses_stringified_list_and_letter(self):
        row = {
            "options": "['A.First', 'B.Second', 'C.Third', 'D.Fourth']",
            "answer": "C",
            "question": "Q?",
        }
        out = _logiqa_options(row)
        self.assertEqual(
            out["options"], ["First", "Second", "Third", "Fourth"]
        )
        self.assertEqual(out["correct_option"], 2)
        self.assertEqual(out["query"], "Q?")

    def test_logiqa_options_keeps_list(self):
        row = {
            "options": ["A.First", "B.Second"],
            "answer": "A",
            "question": "Q?",
        }
        out = _logiqa_options(row)
        self.assertEqual(out["options"], ["First", "Second"])
        self.assertEqual(out["correct_option"], 0)

    def test_jeggers_riddle_choices_strips_prefix(self):
        row = {"choices": "['A: water', 'B: fire']"}
        out = _parse_jeggers_riddle_choices(row)
        self.assertEqual(out["choices"]["text"], ["water", "fire"])

    def test_migrated_tasks_use_data_only_sources(self):
        from tasksource import tasks as en_tasks
        from tasksource import multilingual_tasks as ml_tasks

        expected = {
            "piqa": "baber/piqa",
            "cosmos_qa": "Samsoup/cosmos_qa",
            "banking77": "legacy-datasets/banking77",
            "logiqa": "fireworks-ai/logiqa",
            "moral_stories": "LabHC/moral_stories",
            "launch_open_question_type": "Korea-MES/open_question_type",
            "silicone": "tasksource/silicone",
            "turingbench": "csv",
            "contract_nli__seg": "tasksource/contract-nli",
            "contract_nli__full": "tasksource/contract-nli",
            "summarize_from_feedback": "vwxyzjn/summarize_from_feedback_oai_preprocessing",
            "health_fact": "marcov/health_fact_promptsource",
            "mc_taco": "marcov/mc_taco_promptsource",
            "phrase_similarity": "Deehan1866/processed_phrase_similarity",
            "dyna_hate": "tasksource/dynahate",
            "trec": "tasksource/trec",
            "liar": "tasksource/liar",
            "math_qa": "tasksource/math_qa",
            "cluttr": "tasksource/clutrr",
            "docred": "json",
            "dream": "dataset-org/dream",
            "hate_speech18": "tasksource/hate_speech18",
            "ethos": "SetFit/ethos_binary",
            "fewrel": "tasksource/few_rel",
            "propsegment": "json",
            "sharc_classification": "tasksource/sharc",
            "scicite": "tasksource/scicite",
            "relbert_lexical_relation_classification": "json",
            "social_i_qa": "tasksource/social_i_qa",
            "wiqa": "tasksource/wiqa",
            "humicroedit___subtask_2": "tasksource/humicroedit",
            "scifact_entailment": "tasksource/scifact_entailment",
            "head_qa___en": "EleutherAI/headqa",
            "wiki_hop___original": "MoE-UNC/wikihop",
            "prost": "json",
            "discosense": "json",
            "hope_edi": "csv",
            "numer_sense": "tasksource/numer_sense",
            "valueeval_stance": "csv",
            "webgpt_comparisons": "heegyu/webgpt_comparisons_ko",
            "rumoureval_2019": "csv",
            "blog_authorship_corpus__job": "tasksource/blog_authorship_corpus",
            "emo": "oneonlee/cleansed_emocontext",
            "it_support_tickets": "tasksource/it-support-tickets",
            "twentyquestions": "tasksource/twentyquestions",
            "syntactic_augmentation_nli": "tasksource/syntactic-augmentation-nli",
            "scruples": "tasksource/scruples",
            "nli_veridicality_transitivity": "tasksource/nli-veridicality-transitivity",
            "cnli": "tasksource/cnli",
            "ambient": "tasksource/ambient",
            "defeasible_nli": "tasksource/defeasible-nli",
            "reclor": "tasksource/reclor",
            "equate": "tasksource/equate",
            "scidtb": "multilingual-discourse-hub/disrpt",
            "utilitarianism": "csv",
        }
        legacy_scripts = {
            "piqa", "cosmos_qa", "PolyAI/banking77", "lucasmccabe/logiqa",
            "demelin/moral_stories", "launch/open_question_type",
            "silicone",
        }
        for var, repo in expected.items():
            obj = getattr(en_tasks, var)
            self.assertEqual(obj.dataset_name, repo)
            self.assertNotIn(obj.dataset_name, legacy_scripts)

        # jeggers mirror keeps original task id via basename
        self.assertEqual(en_tasks.riddle_sense.dataset_name, "jeggers/riddle_sense")
        self.assertEqual(en_tasks.utilitarianism.task_id, "utilitarianism")

    def test_migrated_namespaces_and_raw_loader_ids(self):
        from tasksource import tasks as en_tasks
        from tasksource import multilingual_tasks as ml_tasks
        from tasksource import list_tasks

        for var in [
            "ethics___commonsense", "ethics___deontology",
            "ethics___justice",
        ]:
            self.assertEqual(
                getattr(en_tasks, var).dataset_name, "csv"
            )
        self.assertEqual(ml_tasks.afrisenti.dataset_name,
                         "mteb/AfriSentiClassification")
        self.assertEqual(ml_tasks.nusax_sentiment.dataset_name,
                         "mteb/NusaX-senti")
        self.assertEqual(ml_tasks.xstance.dataset_name, "michiel/xstance")
        self.assertEqual(ml_tasks.amazon_intent.dataset_name,
                         "mteb/MassiveIntentClassification")
        # Public dataset IDs stay tied to source tasks even when a generic
        # CSV/JSON builder loads raw data files directly.
        for task_id in ("hope_edi/english", "numer_sense", "args_me"):
            self.assertIn(task_id, set(list_tasks().id))
        self.assertIn(
            "multilingual-sentiments/all",
            set(list_tasks(multilingual=True).id),
        )
        self.assertEqual(en_tasks.hope_edi.dataset_name, "csv")
        self.assertEqual(en_tasks.numer_sense.dataset_name, "tasksource/numer_sense")
        self.assertEqual(en_tasks.arg_me.dataset_name, "webis/args_me")
        self.assertEqual(en_tasks.ethics___virtue.dataset_name, "hendrycks/ethics")
        self.assertEqual(ml_tasks.sentiment.dataset_name, "tasksource/multilingual-sentiments")
        self.assertEqual(en_tasks.emo.dataset_name,
                         "oneonlee/cleansed_emocontext")
        self.assertEqual(ml_tasks.xglue___qam.dataset_name, "tasksource/xglue")
        self.assertEqual(ml_tasks.xlwic.dataset_name, "tasksource/xlwic")
        self.assertEqual(ml_tasks.miam.dataset_name, "csv")
        self.assertEqual(ml_tasks.mms_sentiment.dataset_name, "parquet")
        self.assertEqual(ml_tasks.mms_sentiment.task_id, "mms")
        self.assertEqual(ml_tasks.x_fact.dataset_name, "tasksource/x-fact")
        self.assertEqual(ml_tasks.emotion.dataset_name, "tasksource/universal-joy")
        self.assertEqual(
            ml_tasks.review_sentiment.label_values,
            {stars: f"{stars} star{'s' if stars != 1 else ''}"
             for stars in range(1, 6)},
        )
        self.assertEqual(
            ml_tasks.udep__pos.dataset_name,
            "universal-dependencies/universal_dependencies",
        )
        self.assertEqual(
            en_tasks.udep__deprel.dataset_name,
            "universal-dependencies/universal_dependencies",
        )
        self.assertEqual(ml_tasks.sentiment.task_id,
                         "multilingual-sentiments/all")
        self.assertEqual(en_tasks.valueeval_stance.dataset_name, "csv")
        self.assertEqual(en_tasks.rumoureval_2019.dataset_name, "csv")
        self.assertEqual(en_tasks.webgpt_comparisons.dataset_name,
                         "heegyu/webgpt_comparisons_ko")
        self.assertEqual(en_tasks.webgpt_comparisons.task_id,
                         "webgpt_comparisons")
        self.assertEqual(en_tasks.blog_authorship_corpus__job.labels, "topic")
        # pragmeval/crowdflower rebuilt in place (same names, now data-only)
        self.assertEqual(en_tasks.pragmeval_1.dataset_name, "pragmeval")
        self.assertEqual(en_tasks.crowdflower.dataset_name,
                         "tasksource/crowdflower")


if __name__ == "__main__":
    unittest.main()
