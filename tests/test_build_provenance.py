"""Regressions for build resume/provenance, typed rendering, and catalog lookup."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from tasksource import list_tasks, task_provenance
from tasksource.access import load_preprocessing
from tasksource.jev.recast import render_typed_decision, render_typed_decision_group
from tasksource.tasks import _intent_grasp_keep
from scripts.build_jev_dataset import build_fingerprint, read_completed, slug, source_provenance

def _args(**overrides):
    values = dict(max_rows=1000, max_rows_eval=100, noul_rate=0.05, score_rate=0.0, permutation_rate=0.05,
                  prompt_rate=0.05, paired_format_rate=0.05, pack_rate=0.1, pack_max_tokens=4096,
                  pack_tokenizer=None, pack_max_items=4, output=Path("a"), upload=False)
    return argparse.Namespace(**{**values, **overrides})


class ResumeTest(unittest.TestCase):
    def test_fingerprint_tracks_shard_settings_only(self):
        base = build_fingerprint(_args())
        self.assertNotEqual(base, build_fingerprint(_args(max_rows=30000)))
        # where the output goes or whether it uploads does not change shard contents
        self.assertEqual(base, build_fingerprint(_args(output=Path("b"), upload=True)))

    def test_shards_from_another_fingerprint_are_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir, report = Path(tmp), Path(tmp) / "build-report.jsonl"
            for task in ("old", "new", "legacy"):
                (data_dir / f"train-{slug(task)}.parquet").touch()
            records = [{"task": "old", "status": "ok", "rows": {"train": 1}, "fingerprint": "f1"},
                       {"task": "new", "status": "ok", "rows": {"train": 1}, "fingerprint": "f2"},
                       {"task": "legacy", "status": "ok", "rows": {"train": 1}}]  # built before fingerprints
            report.write_text("".join(json.dumps(r) + "\n" for r in records))
            self.assertEqual(read_completed(report, data_dir, "f2"), ({"new"}, {"old", "legacy"}))
            self.assertEqual(read_completed(report, data_dir), {"old", "new", "legacy"})


class ProvenanceTest(unittest.TestCase):
    def test_requested_revisions_are_reported(self):
        self.assertEqual(task_provenance("dream")["revision"], "refs/convert/parquet")
        conll = task_provenance("conll2002/es", multilingual=True)
        self.assertEqual(conll["data_file_revisions"], {"eriktks/conll2002": "refs/convert/parquet"})
        self.assertEqual(source_provenance("graded/hatexplain")["revision"], "refs/convert/parquet")
        self.assertNotIn("revision", task_provenance("glue/mnli"))


class TypedRendererTest(unittest.TestCase):
    ROW = {"state": "S", "instructions": "Q?", "criteria": ["low", "mid", "high"]}

    def test_each_kind_renders_as_itself(self):
        question = lambda kind: render_typed_decision({**self.ROW, "kind": kind})["questions"]["decision"]
        self.assertEqual(question("choice")["criteria"], {"low": None, "mid": None, "high": None})
        self.assertEqual((question("score")["type"], question("score")["criteria"]), ("score", ["low", "mid", "high"]))
        self.assertEqual(question("noul"), {"type": "noul", "instructions": "Q?"})
        self.assertEqual(render_typed_decision(self.ROW)["questions"]["decision"]["type"], "choice")  # default
        with self.assertRaises(ValueError):
            render_typed_decision({**self.ROW, "kind": "free_text"})
        with self.assertRaises(ValueError):  # System One scores take 2 to 10 levels
            render_typed_decision({**self.ROW, "kind": "score", "criteria": [str(i) for i in range(11)]})

    def test_group_keeps_each_kind(self):
        request = render_typed_decision_group([
            {**self.ROW, "kind": "score", "question_id": "a"}, {**self.ROW, "kind": "choice", "question_id": "b"}])
        self.assertEqual({qid: q["type"] for qid, q in request["questions"].items()}, {"a": "score", "b": "choice"})


class CatalogApiTest(unittest.TestCase):
    def test_intent_grasp_answer_bounds(self):
        row = lambda index: {"answer_index": [index], "options": ["a", "b", "c"],
                             "metadata": {"original_task": "x", "original_split": "train"}}
        self.assertEqual([_intent_grasp_keep(row(i), "train") for i in (-1, 0, 2, 3)], [False, True, True, False])

    def test_list_tasks_accepts_lists_and_returns_copies(self):
        self.assertFalse(list_tasks(excluded=["glue/"]).id.str.contains("glue/").any())
        full = len(list_tasks())
        list_tasks().drop(list_tasks().index[:5], inplace=True)
        edited = list_tasks()
        edited["extra"] = 1
        self.assertEqual(len(list_tasks()), full)
        self.assertNotIn("extra", list_tasks().columns)

    def test_lookup_errors(self):
        with self.assertRaises(KeyError):
            load_preprocessing(id="no-such-task")
        with self.assertRaises(ValueError):  # a dataset name alone matches every glue config
            load_preprocessing(dataset_name="nyu-mll/glue")
        self.assertEqual(load_preprocessing(id="glue/rte").config_name, "rte")


class SafetySplitTest(unittest.TestCase):
    def test_grouped_split_keeps_prompts_on_one_side(self):
        from datasets import Dataset, DatasetDict
        from tasksource.tasks import _wildguardmix
        rows = Dataset.from_dict({"prompt": [f"p{i // 2}" for i in range(4000)], "response": ["r"] * 4000,
                                  "prompt_harm_label": ["harmful"] * 4000})
        split = _wildguardmix(DatasetDict(train=rows))
        prompts = {name: set(part["prompt"]) for name, part in split.items()}
        self.assertFalse(prompts["train"] & (prompts["validation"] | prompts["test"]))
        self.assertTrue(all(len(part) for part in split.values()))
        one = _wildguardmix(DatasetDict(train=rows), one_row_per_prompt=True)
        self.assertEqual(sum(map(len, one.values())), 2000)

    def test_beavertails_votes(self):
        from datasets import Dataset, DatasetDict
        from tasksource.tasks import _beavertails_majority
        from tasksource.jev.graded import _beavertails_votes
        rows = Dataset.from_dict({"prompt": ["p"] * 3 + ["q"] * 3, "response": ["r"] * 3 + ["s"] * 3,
                                  "category": [{}] * 6, "is_safe": [True, True, False, False, False, True]})
        majority = _beavertails_majority(DatasetDict({"330k_train": rows}))["330k_train"]
        self.assertEqual(sorted(zip(majority["prompt"], majority["is_safe"])), [("p", True), ("q", False)])
        votes = _beavertails_votes(DatasetDict({"330k_train": rows, "330k_test": rows}))["train"]
        self.assertEqual(sorted(zip(votes["prompt"], votes["unsafe_share"])), [("p", 1 / 3), ("q", 2 / 3)])


if __name__ == "__main__":
    unittest.main()


class ResumeAtomicityTest(unittest.TestCase):
    def test_latest_failure_voids_an_earlier_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir, report = Path(tmp), Path(tmp) / "build-report.jsonl"
            (data_dir / f"train-{slug('t')}.parquet").touch()
            records = [{"task": "t", "status": "ok", "rows": {"train": 1}, "fingerprint": "f"},
                       {"task": "t", "status": "error"}]
            report.write_text("".join(json.dumps(r) + "\n" for r in records))
            self.assertEqual(read_completed(report, data_dir, "f"), (set(), set()))


class PinTest(unittest.TestCase):
    def test_hf_urls_point_at_pinned_commits(self):
        from tasksource.access import pin_hf_urls
        files = {"train": ["hf://datasets/a/b/x.parquet", "hf://datasets/a/b@refs%2Fconvert%2Fparquet/y/*.parquet"],
                 "test": "hf://datasets/c/d/z.jsonl"}
        pinned = pin_hf_urls(files, {"a/b": "abc123"})
        self.assertEqual(pinned["train"], ["hf://datasets/a/b@abc123/x.parquet", "hf://datasets/a/b@abc123/y/*.parquet"])
        self.assertEqual(pinned["test"], "hf://datasets/c/d/z.jsonl")  # no pin: unchanged
