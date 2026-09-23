"""Family-aware splits + genuine compositional OOD.

Train/validation/test split by state family (never by flattened
questions — all questions/variants of a state stay together).

The OOD split is a genuine domain x skill compositional holdout: a
deterministic band of (domain, skill) pairs is held out, and EVERY
bundle containing a held-out pair goes to ``ood``. Held-out pairs
therefore never appear in train — verified by ``verify_splits`` and
reported in ``split_report.json``.
"""

from __future__ import annotations

import hashlib


def stable_hash(text: str, salt: str = "") -> int:
    return int(hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()[:8], 16)


def family_id(bundle: dict) -> str:
    return f"{bundle.get('domain', '')}::{bundle.get('scenario_type', '')}::{bundle.get('style', '')}"


def bundle_pairs(bundle: dict) -> set[tuple[str, str]]:
    domain = bundle.get("domain", "")
    return {(domain, q.get("skill", "")) for q in bundle.get("questions", [])}


def held_out_pairs(bundles: list[dict], ood_fraction: float, salt: str) -> set[tuple[str, str]]:
    band = int(ood_fraction * 100)
    pairs: set[tuple[str, str]] = set()
    for bundle in bundles:
        pairs.update(bundle_pairs(bundle))
    return {pair for pair in pairs
            if stable_hash(f"ood-pair:{pair[0]}::{pair[1]}", salt) % 100 < band}


def _family_split(bundle: dict, split_cfg) -> str:
    position = stable_hash(family_id(bundle), split_cfg.seed_salt) % 100
    train_cut = int(split_cfg.train * 100)
    val_cut = train_cut + int(split_cfg.validation * 100)
    if position < train_cut:
        return "train"
    if position < val_cut:
        return "validation"
    return "test"


def assign_split(bundle: dict, split_cfg, held_out: set[tuple[str, str]] | None = None) -> str:
    if held_out and bundle_pairs(bundle) & held_out:
        return "ood"
    return _family_split(bundle, split_cfg)


def assign_splits(bundles: list[dict], split_cfg) -> list[dict]:
    held_out = held_out_pairs(bundles, split_cfg.ood_fraction, split_cfg.seed_salt)
    return [{**bundle, "split": assign_split(bundle, split_cfg, held_out)} for bundle in bundles]


def verify_splits(bundles: list[dict], ood_fraction: float, salt: str) -> dict:
    """Check the compositional guarantee; raises on leakage."""
    held_out = held_out_pairs(
        [{k: b.get(k) for k in ("domain", "questions")} for b in bundles],
        ood_fraction, salt)
    train_pairs: set[tuple[str, str]] = set()
    ood_pairs: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    for bundle in bundles:
        counts[bundle.get("split", "?")] = counts.get(bundle.get("split", "?"), 0) + 1
        if bundle.get("split") == "train":
            train_pairs.update(bundle_pairs(bundle))
        if bundle.get("split") == "ood":
            ood_pairs.update(bundle_pairs(bundle))
    leaked = sorted("::".join(pair) for pair in (held_out & train_pairs))
    if leaked:
        raise ValueError(f"Compositional OOD leakage: held-out pairs in train: {leaked}")
    uncovered = sorted("::".join(pair) for pair in (held_out - ood_pairs))
    if uncovered:
        raise ValueError(f"Held-out pairs missing from ood: {uncovered}")
    return {"held_out_pairs": sorted("::".join(p) for p in held_out),
            "n_held_out_pairs": len(held_out),
            "split_counts": counts}
