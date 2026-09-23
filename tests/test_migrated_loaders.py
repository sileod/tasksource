import unittest

from datasets import Dataset, DatasetDict

from tasksource.tasks import (
    _logiqa_options,
    _parse_jeggers_riddle_choices,
    _strip_option_prefix,
)


class MigratedLoaderHelperTest(unittest.TestCase):
    def test_aces_ontology_is_fixed_before_sampling(self):
        from tasksource.mtasks import _aces_phenomena_labels

        source = DatasetDict({
            "train": Dataset.from_dict({"phenomena": ["addition", "deletion"]}),
            "validation": Dataset.from_dict({"phenomena": ["deletion"]}),
        })
        mapped = _aces_phenomena_labels(source)
        self.assertEqual(mapped["train"].features["phenomena"].names,
                         ["addition", "deletion"])
        self.assertEqual(mapped["validation"][0]["phenomena"], 1)

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
        from tasksource import mtasks as ml_tasks

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
            "cluttr": "json",
            "docred": "json",
            "dream": "json",
            "hate_speech18": "tasksource/hate_speech18",
            "ethos": "SetFit/ethos_binary",
            "fewrel": "tasksource/few_rel",
            "propsegment": "json",
            "sharc_classification": "json",
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
            "numer_sense": "csv",
            "valueeval_stance": "csv",
            "webgpt_comparisons": "heegyu/webgpt_comparisons_ko",
            "rumoureval_2019": "csv",
            "blog_authorship_corpus__job": "tasksource/blog_authorship_corpus",
            "emo": "oneonlee/cleansed_emocontext",
            "it_support_tickets": "tasksource/it-support-tickets",
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

    def test_migrated_namespaces_and_raw_loader_ids(self):
        from tasksource import tasks as en_tasks
        from tasksource import mtasks as ml_tasks
        from tasksource import list_tasks

        for var in [
            "ethics___commonsense", "ethics___deontology",
            "ethics___justice", "ethics___virtue",
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
        self.assertEqual(en_tasks.numer_sense.dataset_name, "csv")
        self.assertEqual(en_tasks.arg_me.dataset_name, "json")
        self.assertEqual(ml_tasks.sentiment.dataset_name, "csv")
        self.assertEqual(en_tasks.emo.dataset_name,
                         "oneonlee/cleansed_emocontext")
        self.assertEqual(ml_tasks.xglue___qam.dataset_name, "tasksource/xglue")
        self.assertEqual(ml_tasks.xlwic.dataset_name, "tasksource/xlwic")
        self.assertEqual(ml_tasks.miam.dataset_name, "csv")
        self.assertEqual(ml_tasks.mms_sentiment.dataset_name, "csv")
        self.assertEqual(ml_tasks.mms_sentiment.task_id, "mms")
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
