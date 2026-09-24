"""Regressions for invalid supervision, reproducibility and API-correctness fixes."""

import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datasets import ClassLabel, Dataset, DatasetDict

from tasksource import access, hub_datasets, list_tasks, load_task
from tasksource.jev.synthetic import annotate as annot_mod
from tasksource.jev.synthetic import critic as critic_mod
from tasksource.jev.synthetic import run as run_mod
from tasksource.jev.synthetic import specs as specs_mod
from tasksource.jev.synthetic import split as split_mod
from tasksource.jev.synthetic.config import AnnotatorConfig, AppConfig, SplitConfig
from tasksource.jev.synthetic.generate import mock_realization
from tasksource.preprocess import fix_splits
from tasksource.recast import recast_instruct


def _question(fmt, options=None):
    return {"question_id": "q0", "format": fmt, "question": "Q?", "options": options or []}


class ProbabilityValidationTest(unittest.TestCase):
    def test_invalid_choice_distributions_rejected(self):
        for probs in ([-0.1, 1.1], [float("nan"), float("nan")], [float("inf"), -float("inf")], ["x", 1]):
            with self.subTest(probs=probs), self.assertRaises(RuntimeError):
                annot_mod._jev_probabilities(_question("choice", ["a", "b"]),
                                             {"q0": {"probabilities": dict(zip("ab", probs))}})

    def test_invalid_noul_rejected(self):
        for noul in (1.7, -0.2, float("nan"), "high"):
            with self.subTest(noul=noul), self.assertRaises(RuntimeError):
                annot_mod._jev_probabilities(_question("noul"), {"q0": {"noul": noul}})

    def test_valid_values_pass(self):
        self.assertEqual(annot_mod._jev_probabilities(_question("noul"), {"q0": {"noul": 1}})[0], [1.0])
        probs, _ = annot_mod._jev_probabilities(_question("choice", ["a", "b"]),
                                                {"q0": {"probabilities": {"a": 0.3, "b": 0.7}}})
        self.assertEqual(probs, [0.3, 0.7])


class ExportTest(unittest.TestCase):
    def _export(self, drop_annotation):
        cfg = AppConfig()
        bundles = [annot_mod.annotate_bundle(mock_realization(s), cfg.annotator)
                   for s in specs_mod.sample_specs(cfg.sampler, 3)]
        for bundle in bundles:
            bundle["split"] = "train"
        if drop_annotation:
            bundles[1]["annotations"] = bundles[1]["annotations"][:-1]
        with tempfile.TemporaryDirectory() as tmp:
            run_mod._write_bundles(Path(tmp) / "final.jsonl", bundles)
            run_mod._to_frame(bundles).to_parquet(Path(tmp) / "final.parquet", index=False)
            return run_mod.stage_export(cfg, Path(tmp))

    def test_complete_annotations_export(self):
        self.assertGreater(self._export(drop_annotation=False)["decisions"], 0)

    def test_missing_annotation_fails_instead_of_uniform_target(self):
        with self.assertRaises(ValueError):
            self._export(drop_annotation=True)


class SplitFractionTest(unittest.TestCase):
    def test_fractions_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            SplitConfig(train=0.5, validation=0.1, test=0.1)

    def test_test_fraction_is_respected(self):
        cfg = SplitConfig(train=0.5, validation=0.1, test=0.4)
        splits = [split_mod._family_split({"domain": str(i)}, cfg) for i in range(20000)]
        for name, share in [("train", 0.5), ("validation", 0.1), ("test", 0.4)]:
            self.assertAlmostEqual(splits.count(name) / len(splits), share, delta=0.02)


class CacheKeyEndpointTest(unittest.TestCase):
    def test_annotation_key_depends_on_endpoint(self):
        bundle = mock_realization(specs_mod.sample_specs(AppConfig().sampler, 1)[0])
        first = AnnotatorConfig(name="jev", base_url="https://a.example/api", model="m")
        second = AnnotatorConfig(name="jev", base_url="https://b.example/api", model="m")
        self.assertNotEqual(annot_mod.jev_cache_key(bundle, first), annot_mod.jev_cache_key(bundle, second))

    def test_critic_key_depends_on_endpoint(self):
        keys = {critic_mod.critic_cache_key("m", 0.0, "prompt", {"state_id": "s"}, endpoint)
                for endpoint in ("albert@https://a", "openai@https://b")}
        self.assertEqual(len(keys), 2)


def _classification(n=40):
    rows = Dataset.from_dict({"sentence1": [f"text {i}" for i in range(n)], "labels": [i % 5 for i in range(n)]})
    return DatasetDict(train=rows.cast_column("labels", ClassLabel(names=list("abcde"))))


class RecastSeedTest(unittest.TestCase):
    def _recast(self, seed):
        return list(recast_instruct(_classification(), seed=seed)["train"]["inputs"])

    def test_instruct_recast_is_seeded_and_ignores_global_random(self):
        random.seed(1)
        first = self._recast(0)
        random.seed(2)
        self.assertEqual(first, self._recast(0))
        self.assertNotEqual(first, self._recast(1))


class LoadTaskSeedTest(unittest.TestCase):
    def test_seed_reaches_sampling(self):
        source = lambda *args, **kwargs: DatasetDict(  # a fresh copy per call: preprocessing edits it in place
            train=Dataset.from_dict({"text": [f"t{i}" for i in range(200)], "label": [i % 2 for i in range(200)]}),
            test=Dataset.from_dict({"text": ["x", "y"], "label": [0, 1]}))
        with mock.patch.object(access, "load_dataset", side_effect=source):
            sample = lambda seed: list(load_task("imdb", max_rows=20, seed=seed)["train"]["sentence1"])
            self.assertEqual(sample(1), sample(1))
            self.assertNotEqual(sample(1), sample(2))


class FixSplitsTest(unittest.TestCase):
    def _dataset(self, test_labels):
        return DatasetDict(train=Dataset.from_dict({"labels": [0, 1, 0, 1]}),
                           test=Dataset.from_dict({"labels": test_labels}))

    def test_hidden_test_labels_removed(self):
        self.assertNotIn("test", fix_splits(self._dataset([-1, -1])))

    def test_single_class_test_kept(self):
        self.assertIn("test", fix_splits(self._dataset([1, 1])))


class ApiTest(unittest.TestCase):
    def test_excluded_matches_substrings(self):
        ids = set(list_tasks(excluded=("glue/",)).id)
        self.assertFalse(any("glue/" in task_id for task_id in ids))
        self.assertIn("imdb", ids)

    def test_unknown_task_raises(self):
        with self.assertRaises(KeyError):
            hub_datasets(["no-such-task"])


if __name__ == "__main__":
    unittest.main()
