"""Tests for the synthetic Jev dataset-construction package (offline, mock only)."""

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tasksource.jev.synthetic import annotate as annot_mod
from tasksource.jev.synthetic import critic as critic_mod
from tasksource.jev.synthetic import dedup as dedup_mod
from tasksource.jev.synthetic import providers
from tasksource.jev.synthetic import select as select_mod
from tasksource.jev.synthetic import specs as specs_mod
from tasksource.jev.synthetic import split as split_mod
from tasksource.jev.synthetic import validate as validate_mod
from tasksource.jev.synthetic.config import AppConfig, load_config
from tasksource.jev.synthetic.generate import (
    generation_cache_key,
    mock_realization,
)
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
        formats = Counter(q["format"] for s in specs for q in s["questions"])
        total = sum(formats.values())
        self.assertTrue(abs(formats["choice"] / total - 0.40) < 0.05)
        self.assertTrue(abs(formats["noul"] / total - 0.30) < 0.05)
        multi = [s for s in specs if len(s["questions"]) >= 2]
        mixed = sum(1 for s in multi if len({q["format"] for q in s["questions"]}) >= 2)
        self.assertGreater(mixed / len(multi), 0.70)

    def test_all_three_formats_guaranteed(self):
        import random
        weights = {"choice": 0.40, "noul": 0.30, "score": 0.30}
        for n in (3, 4):
            for seed in range(50):
                rng = random.Random(seed)
                sampled = specs_mod.sample_formats(rng, n, weights, 0.85, 1.0)
                self.assertEqual(set(sampled), {"choice", "noul", "score"},
                                 f"n={n} seed={seed}: {sampled}")

    def test_skills_distinct_within_state(self):
        specs = specs_mod.sample_specs(_cfg().sampler, 500)
        for spec in specs:
            skills = [q["skill"] for q in spec["questions"]]
            self.assertEqual(len(skills), len(set(skills)),
                             f"{spec['state_id']}: {skills}")

    def test_taxonomy_scale(self):
        self.assertGreaterEqual(len(specs_mod.DOMAINS), 30)
        self.assertGreaterEqual(len(specs_mod.SKILLS), 20)
        self.assertGreaterEqual(len(specs_mod.STYLES), 10)
        self.assertGreaterEqual(len(specs_mod.EVIDENCE_STRUCTURES), 8)
        self.assertGreaterEqual(len(specs_mod.AMBIGUITY_LEVELS), 5)

    def test_score_specs_have_ordered_criteria(self):
        specs = specs_mod.sample_specs(_cfg().sampler, 500)
        score_qs = [q for s in specs for q in s["questions"] if q["format"] == "score"]
        self.assertTrue(len(score_qs) > 50)
        for q in score_qs:
            self.assertGreaterEqual(len(q["criteria"]), 3)
            self.assertEqual(len(q["criteria"]), q["max"] - q["min"] + 1)
        numeric = sum(1 for q in score_qs if q["criteria"][0] == q["criteria"][0].strip()
                      and q["criteria"][0].lstrip("-").isdigit())
        self.assertGreater(numeric, 0)
        self.assertGreater(len(score_qs) - numeric, 0)  # semantic rubrics too


class BundleTest(unittest.TestCase):
    def test_mock_bundles_validate(self):
        specs = specs_mod.sample_specs(_cfg().sampler, 20)
        for spec in specs:
            bundle = mock_realization(spec)
            errors = validate_mod.validate_bundle(bundle, spec)
            self.assertEqual(errors, [], f"{spec['state_id']}: {errors}")

    def test_noul_requires_probability_for_a_proposition(self):
        valid = [
            "Does this ticket need escalation?",
            "Based on the report, what is the likelihood that the vendor will fail?",
            "How likely is the customer to cancel the contract?",
        ]
        invalid = [
            "What is the most likely root cause of the failure?",
            "Which team should own this ticket?",
            "According to the report, what is the timestamp on the monitor?",
            "List all of the action items assigned to the agent.",
        ]
        for text in valid:
            self.assertTrue(validate_mod.valid_noul_question(text), text)
        for text in invalid:
            self.assertFalse(validate_mod.valid_noul_question(text), text)

    def test_flat_preserves_grouping(self):
        spec = specs_mod.sample_specs(_cfg().sampler, 5)[0]
        bundle = mock_realization(spec)
        rows = bundle_to_flat_rows(bundle)
        self.assertEqual(len(rows), len(bundle["questions"]))
        self.assertTrue(all(r["state_id"] == bundle["state_id"] for r in rows))
        self.assertTrue(all(r["bundle_size"] == len(rows) for r in rows))

    def test_training_targets_aligned(self):
        specs = specs_mod.sample_specs(_cfg().sampler, 30)
        seen_score = False
        for spec in specs:
            bundle = mock_realization(spec)
            annotated = annot_mod.annotate_bundle(bundle, _cfg().annotator)
            for question, ann in zip(annotated["questions"], annotated["annotations"]):
                row = flat_to_training_row(
                    {"format": question["format"], "question_id": question["question_id"],
                     "state": "s", "question": "q",
                     "options": question.get("options", [])},
                    ann["probabilities"])
                if question["format"] == "noul":
                    self.assertEqual(row["options"], [])
                    self.assertEqual(len(row["target"]), 1)
                else:
                    seen_score = seen_score or question["format"] == "score"
                    self.assertTrue(len(row["options"]) >= 2)
                    self.assertEqual(len(row["target"]), len(row["options"]))
                    self.assertAlmostEqual(sum(row["target"]), 1.0, places=4)
        self.assertTrue(seen_score)


class AnnotatorGatingTest(unittest.TestCase):
    def test_mock_is_labeled_mock(self):
        spec = specs_mod.sample_specs(_cfg().sampler, 3)[0]
        bundle = mock_realization(spec)
        annotated = annot_mod.annotate_bundle(bundle, _cfg().annotator)
        for ann in annotated["annotations"]:
            self.assertEqual(ann["annotator"], "mock")
            self.assertNotEqual(ann["annotator"], "jev")

    def test_jev_without_endpoint_fails_loudly(self):
        from tasksource.jev.synthetic.config import AnnotatorConfig
        spec = specs_mod.sample_specs(_cfg().sampler, 1)[0]
        bundle = mock_realization(spec)
        cfg = AnnotatorConfig(name="jev", version="jev-test", base_url="",
                              api_key_env="DEFINITELY_UNSET_XYZ", model="jev-test")
        with self.assertRaises(RuntimeError):
            annot_mod.annotate_bundle(bundle, cfg)

    def test_jev_without_key_fails_loudly(self):
        import os
        from tasksource.jev.synthetic.config import AnnotatorConfig
        os.environ.pop("DEFINITELY_UNSET_XYZ", None)
        spec = specs_mod.sample_specs(_cfg().sampler, 1)[0]
        bundle = mock_realization(spec)
        cfg = AnnotatorConfig(name="jev", version="jev-test",
                              base_url="https://example.invalid",
                              api_key_env="DEFINITELY_UNSET_XYZ", model="jev-test")
        with self.assertRaises(RuntimeError):
            annot_mod.annotate_bundle(bundle, cfg)

    def test_unknown_annotator_rejected(self):
        from tasksource.jev.synthetic.config import AnnotatorConfig
        spec = specs_mod.sample_specs(_cfg().sampler, 1)[0]
        bundle = mock_realization(spec)
        with self.assertRaises(ValueError):
            annot_mod.annotate_bundle(bundle, AnnotatorConfig(name="oracle"))


class CriticTest(unittest.TestCase):
    def test_critic_uses_own_provider(self):
        cfg = _cfg()
        from tasksource.jev.synthetic.config import ProviderConfig
        cfg.provider = ProviderConfig(name="openai", api_key_env="OPENAI_API_KEY",
                                      base_url="https://api.openai.com/v1", model="gpt-6-luna")
        cfg.critic.provider = ProviderConfig(
            name="albert", api_key_env="ALBERT_API_KEY",
            base_url="https://albert.api.etalab.gouv.fr/v1",
            model="deepseek-v4-flash-0731")
        resolved = cfg.critic_provider()
        self.assertEqual(resolved.name, "albert")
        self.assertEqual(resolved.base_url, "https://albert.api.etalab.gouv.fr/v1")

    def test_critic_inherits_generator_provider(self):
        cfg = _cfg()
        cfg.critic.provider = None
        self.assertEqual(cfg.critic_provider().name, cfg.provider.name)

    def test_critic_results_cached_by_content_hash(self):
        cfg = _cfg()
        cfg.provider.name = "mock"
        specs = specs_mod.sample_specs(cfg.sampler, 3)
        bundles = [mock_realization(s) for s in specs]
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            first = critic_mod.critique_bundles(cfg, bundles, raw)
            files_after_first = sorted(p.name for p in raw.glob("*.json"))
            second = critic_mod.critique_bundles(cfg, bundles, raw)
            files_after_second = sorted(p.name for p in raw.glob("*.json"))
        self.assertEqual(first, second)
        self.assertEqual(files_after_first, files_after_second)
        # Cache files are content-hashed, not state_id-named.
        self.assertTrue(all(not p.startswith("state_") for p in files_after_first))


class CacheKeyTest(unittest.TestCase):
    def test_unrelated_settings_do_not_invalidate_generation(self):
        cfg = _cfg()
        before = generation_cache_key(cfg)
        cfg.split.train = 0.5
        cfg.selection.max_per_family = 10
        cfg.annotator.version = "mock-0.2"
        cfg.sampler.seed = 999
        self.assertEqual(generation_cache_key(cfg), before)

    def test_generation_settings_change_the_key(self):
        cfg = _cfg()
        before = generation_cache_key(cfg)
        cfg.generation.temperature = 0.1
        self.assertNotEqual(generation_cache_key(cfg), before)
        cfg2 = _cfg()
        cfg2.provider.model = "other-model"
        self.assertNotEqual(generation_cache_key(cfg2), before)


class DedupTest(unittest.TestCase):
    def test_exact_duplicates_dropped(self):
        specs = specs_mod.sample_specs(_cfg().sampler, 5)
        bundles = [mock_realization(s) for s in specs]
        doubled = bundles + [dict(b) for b in bundles]
        kept, dropped = dedup_mod.dedup_bundles(doubled)
        self.assertEqual(len(kept), len(bundles))
        self.assertEqual(len(dropped), len(bundles))

    def test_near_duplicates_dropped(self):
        specs = specs_mod.sample_specs(_cfg().sampler, 3)
        bundles = [mock_realization(s) for s in specs]
        near = dict(bundles[0])
        near = {**near, "state_id": "state_999999",
                "state": near["state"] + " Extra trailing sentence here."}
        # Jaccard ~0.86: dropped at 0.8, kept at 0.9 (threshold semantics).
        kept, dropped = dedup_mod.dedup_bundles(bundles + [near], threshold=0.8)
        self.assertEqual(len(kept), len(bundles))
        self.assertIn("state_999999", dropped)
        kept, _ = dedup_mod.dedup_bundles(bundles + [near], threshold=0.9)
        self.assertEqual(len(kept), len(bundles) + 1)

    def test_distinct_states_kept(self):
        specs = specs_mod.sample_specs(_cfg().sampler, 50)
        bundles = [mock_realization(s) for s in specs]
        kept, dropped = dedup_mod.dedup_bundles(bundles)
        self.assertEqual(len(kept) + len(dropped), len(bundles))
        by_id = {b["state_id"]: b for b in bundles}
        kept_ids = {b["state_id"] for b in kept}
        for sid in dropped:
            self.assertTrue(
                any(dedup_mod.jaccard(by_id[sid]["state"], by_id[k]["state"]) >= 0.9
                    for k in kept_ids),
                f"{sid} dropped without a >=0.9 near-duplicate")

    def test_scales_to_thousands(self):
        import time
        specs = specs_mod.sample_specs(_cfg().sampler, 2000)
        bundles = [mock_realization(s) for s in specs]
        start = time.time()
        kept, dropped = dedup_mod.dedup_bundles(bundles)
        elapsed = time.time() - start
        # Mock states share boilerplate, so some are genuine near-dups.
        self.assertEqual(len(kept) + len(dropped), len(bundles))
        self.assertLess(elapsed, 30, f"2000-state dedup took {elapsed:.1f}s")


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

    def test_ood_is_genuinely_compositional(self):
        cfg = _cfg()
        specs = specs_mod.sample_specs(cfg.sampler, 300)
        bundles = [mock_realization(s) for s in specs]
        assigned = split_mod.assign_splits(bundles, cfg.split)
        # verify_splits raises on any held-out pair leaking into train.
        report = split_mod.verify_splits(
            assigned, cfg.split.ood_fraction, cfg.split.seed_salt)
        self.assertGreater(report["n_held_out_pairs"], 0)
        self.assertGreater(report["split_counts"].get("ood", 0), 0)
        held = set(report["held_out_pairs"])
        for bundle in assigned:
            pairs = {f"{d}::{s}" for d, s in split_mod.bundle_pairs(bundle)}
            if bundle["split"] == "ood":
                self.assertTrue(pairs & held)
            if bundle["split"] == "train":
                self.assertFalse(pairs & held)


class SelectTest(unittest.TestCase):
    def test_plan_sums_to_total(self):
        cfg = _cfg()
        specs = specs_mod.sample_specs(cfg.sampler, 60)
        bundles = [mock_realization(s) for s in specs]
        annotated = annot_mod.annotate_bundles(bundles, cfg.annotator)
        result = select_mod.select_bundles(annotated, cfg.selection)
        self.assertEqual(len(result["selected"]), len(annotated))
        self.assertEqual(sum(result["diagnostics"]["plan"].values()), len(annotated))


class JevParsingTest(unittest.TestCase):
    def _question(self, fmt, options=None):
        return {"question_id": "q0", "format": fmt, "question": "Q?",
                "options": options or []}

    def test_noul(self):
        probs, entry = annot_mod._jev_probabilities(
            self._question("noul"), {"q0": {"type": "noul", "noul": 0.73}})
        self.assertEqual(probs, [0.73])

    def test_choice_ordered_by_options(self):
        probs, _ = annot_mod._jev_probabilities(
            self._question("choice", ["b", "a"]),
            {"q0": {"type": "choice", "choice": "a",
                    "probabilities": {"a": 0.8, "b": 0.2}, "confidence": 0.7}})
        self.assertEqual(probs, [0.2, 0.8])

    def test_score_index_keyed_with_legend(self):
        probs, _ = annot_mod._jev_probabilities(
            self._question("score", ["low", "high"]),
            {"q0": {"type": "score", "score": 0.9,
                    "legend": {"0": "low", "1": "high"},
                    "probabilities": {"0": 0.1, "1": 0.9}, "confidence": 0.8}})
        self.assertEqual(probs, [0.1, 0.9])

    def test_score_legend_drift_rejected(self):
        with self.assertRaises(RuntimeError):
            annot_mod._jev_probabilities(
                self._question("score", ["low", "high"]),
                {"q0": {"type": "score", "score": 0.9,
                        "legend": {"0": "low", "1": "RENAMED"},
                        "probabilities": {"0": 0.1, "1": 0.9}}})

    def test_missing_entry_rejected(self):
        with self.assertRaises(RuntimeError):
            annot_mod._jev_probabilities(self._question("noul"), {})

    def test_decisions_url(self):
        from tasksource.jev.synthetic.config import AnnotatorConfig
        cfg = AnnotatorConfig(name="jev", base_url="https://openrouter.ai/",
                              api_path="/api/alpha/decisions")
        self.assertEqual(annot_mod.decisions_url(cfg),
                         "https://openrouter.ai/api/alpha/decisions")

    def test_score_criteria_capped_at_ten(self):
        for lo, hi in specs_mod.SCORE_RANGES:
            self.assertLessEqual(hi - lo + 1, 10)
        specs = specs_mod.sample_specs(_cfg().sampler, 500)
        for spec in specs:
            for q in spec["questions"]:
                if q["format"] == "score":
                    self.assertLessEqual(len(q["criteria"]), 10)


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
    def _config_dir(self):
        from pathlib import Path as _Path
        base = _Path(__file__).resolve().parents[1] / "src" / "tasksource" / "jev" / "synthetic" / "configs"
        if not base.exists():
            import tasksource.jev.synthetic as pkg
            base = _Path(pkg.__file__).parent / "configs"
        return base

    def test_all_configs_load(self):
        base = self._config_dir()
        for name in ("albert_deepseek_v4_flash.yaml", "openai_luna.yaml", "mock_pilot.yaml"):
            cfg = load_config(str(base / name))
            self.assertTrue(cfg.provider.model)
            self.assertTrue(cfg.provider.api_key_env)
            # Serializes cleanly into the run manifest.
            json.dumps(cfg.to_dict())

    def test_configs_use_explicit_mock_annotator(self):
        base = self._config_dir()
        for name in ("albert_deepseek_v4_flash.yaml", "openai_luna.yaml", "mock_pilot.yaml"):
            cfg = load_config(str(base / name))
            self.assertEqual(cfg.annotator.name, "mock")

    def test_critic_provider_independent_of_generator(self):
        base = self._config_dir()
        cfg = load_config(str(base / "albert_deepseek_v4_flash.yaml"))
        critic = cfg.critic_provider()
        self.assertEqual(critic.name, "albert")
        self.assertIn("albert", critic.base_url)
        # Generator and critic can diverge: simulate Luna gen + Albert critic.
        other = copy.deepcopy(cfg)
        other.provider.name = "openai"
        self.assertEqual(other.critic_provider().name, "albert")


if __name__ == "__main__":
    unittest.main()
