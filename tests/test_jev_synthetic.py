"""Tests for the synthetic Jev dataset-construction package (offline, mock only)."""

import unittest

from tasksource.jev.synthetic import annotate as annot_mod
from tasksource.jev.synthetic import dedup as dedup_mod
from tasksource.jev.synthetic import providers
from tasksource.jev.synthetic import select as select_mod
from tasksource.jev.synthetic import specs as specs_mod
from tasksource.jev.synthetic import split as split_mod
from tasksource.jev.synthetic import validate as validate_mod
from tasksource.jev.synthetic.config import AppConfig, load_config
from tasksource.jev.synthetic.generate import mock_realization
from tasksource.jev.synthetic.schemas import (
    bundle_to_flat_rows,
    flat_to_training_row,
)


def _cfg():
    return AppConfig()


class SamplerTest(unittest.TestCase):
    def test_deterministic(self):
        cfg = _cfg()
        first = specs_mod.sample_specs(cfg.sampler, 50)
        second = specs_mod.sample_specs(cfg.sampler, 50)
        self.assertEqual(first, second)

    def test_state_ids_unique_and_ordered(self):
        specs = specs_mod.sample_specs(_cfg().sampler, 30)
        ids = [s["state_id"] for s in specs]
        self.assertEqual(len(set(ids)), 30)
        self.assertEqual(ids[0], "state_000000")

    def test_questions_per_state_distribution(self):
        specs = specs_mod.sample_specs(_cfg().sampler, 2000)
        counts = [len(s["questions"]) for s in specs]
        mean = sum(counts) / len(counts)
        self.assertTrue(1.4 < mean < 1.8, f"mean={mean}")  # target 1.6

    def test_format_coverage_and_mixing(self):
        specs = specs_mod.sample_specs(_cfg().sampler, 2000)
        from collections import Counter
        formats = Counter(q["format"] for s in specs for q in s["questions"])
        total = sum(formats.values())
        self.assertTrue(abs(formats["choice"] / total - 0.40) < 0.05)
        self.assertTrue(abs(formats["noul"] / total - 0.30) < 0.05)
        multi = [s for s in specs if len(s["questions"]) >= 2]
        mixed = sum(1 for s in multi if len({q["format"] for q in s["questions"]}) >= 2)
        self.assertGreater(mixed / len(multi), 0.70)


class BundleTest(unittest.TestCase):
    def test_mock_bundles_validate(self):
        specs = specs_mod.sample_specs(_cfg().sampler, 20)
        bad = 0
        for spec in specs:
            bundle = mock_realization(spec)
            errors = validate_mod.validate_bundle(bundle, spec)
            if errors:
                bad += 1
        self.assertEqual(bad, 0)

    def test_flat_preserves_grouping(self):
        spec = specs_mod.sample_specs(_cfg().sampler, 5)[0]
        bundle = mock_realization(spec)
        rows = bundle_to_flat_rows(bundle)
        self.assertEqual(len(rows), len(bundle["questions"]))
        self.assertTrue(all(r["state_id"] == bundle["state_id"] for r in rows))
        self.assertTrue(all(r["bundle_size"] == len(rows) for r in rows))

    def test_training_targets_sum_to_one(self):
        specs = specs_mod.sample_specs(_cfg().sampler, 10)
        for spec in specs:
            bundle = mock_realization(spec)
            annotated = annot_mod.annotate_bundle(bundle, "mock-0.1")
            for question, ann in zip(annotated["questions"], annotated["annotations"]):
                flat = bundle_to_flat_rows(annotated)[0]  # grouping check only
                row = flat_to_training_row(
                    {"format": question["format"], "question_id": question["question_id"],
                     "state": "s", "question": "q",
                     "options": question.get("options", [])},
                    ann["probabilities"])
                if question["format"] == "noul":
                    self.assertEqual(len(row["target"]), 1)
                else:
                    self.assertAlmostEqual(sum(row["target"]), 1.0, places=4)
            break
        self.assertTrue(flat["state_id"].startswith("state_"))


class DedupTest(unittest.TestCase):
    def test_exact_duplicates_dropped(self):
        specs = specs_mod.sample_specs(_cfg().sampler, 5)
        bundles = [mock_realization(s) for s in specs]
        doubled = bundles + [dict(b) for b in bundles]
        kept, dropped = dedup_mod.dedup_bundles(doubled)
        self.assertEqual(len(kept), len(bundles))
        self.assertEqual(len(dropped), len(bundles))


class SplitTest(unittest.TestCase):
    def test_family_stays_together(self):
        cfg = _cfg()
        specs = specs_mod.sample_specs(cfg.sampler, 100)
        bundles = [mock_realization(s) for s in specs]
        assigned = split_mod.assign_splits(bundles, cfg.split)
        by_family = {}
        for bundle in assigned:
            by_family.setdefault(split_mod.family_id(bundle), set()).add(bundle["split"])
        for family, splits in by_family.items():
            self.assertEqual(len(splits), 1, f"family {family} split apart: {splits}")

    def test_split_is_stable(self):
        cfg = _cfg()
        specs = specs_mod.sample_specs(cfg.sampler, 30)
        bundles = [mock_realization(s) for s in specs]
        first = [b["split"] for b in split_mod.assign_splits(bundles, cfg.split)]
        second = [b["split"] for b in split_mod.assign_splits(bundles, cfg.split)]
        self.assertEqual(first, second)


class SelectTest(unittest.TestCase):
    def test_plan_sums_to_total(self):
        cfg = _cfg()
        specs = specs_mod.sample_specs(cfg.sampler, 60)
        bundles = [mock_realization(s) for s in specs]
        annotated = annot_mod.annotate_bundles(bundles, "mock-0.1")
        result = select_mod.select_bundles(annotated, cfg.selection)
        self.assertEqual(len(result["selected"]), len(annotated))
        self.assertEqual(sum(result["diagnostics"]["plan"].values()), len(annotated))


class PreflightTest(unittest.TestCase):
    def test_missing_key_raises(self):
        import os
        from tasksource.jev.synthetic.config import ProviderConfig
        provider = ProviderConfig(name="albert", api_key_env="DEFINITELY_UNSET_ENV_VAR_XYZ")
        os.environ.pop("DEFINITELY_UNSET_ENV_VAR_XYZ", None)
        with self.assertRaises(RuntimeError):
            providers.require_api_key(provider)

    def test_mock_needs_no_key(self):
        from tasksource.jev.synthetic.config import ProviderConfig
        provider = ProviderConfig(name="mock", api_key_env="DEFINITELY_UNSET_ENV_VAR_XYZ")
        self.assertEqual(providers.require_api_key(provider), "mock-key")


class ConfigTest(unittest.TestCase):
    def test_albert_and_luna_configs_load(self):
        from pathlib import Path
        base = Path(__file__).resolve().parents[1] / "src" / "tasksource" / "jev" / "synthetic" / "configs"
        # When installed, fall back to the package directory.
        if not base.exists():
            import tasksource.jev.synthetic as pkg
            base = Path(pkg.__file__).parent / "configs"
        for name in ("albert_deepseek_v4_flash.yaml", "openai_luna.yaml", "mock_pilot.yaml"):
            cfg = load_config(str(base / name))
            self.assertTrue(cfg.provider.model)
            self.assertTrue(cfg.provider.api_key_env)


if __name__ == "__main__":
    unittest.main()
