"""Family-aware splits: never split flattened questions apart.

split = stable_hash(family_id) % 100 keeps all questions/variants of a
state together. An OOD split holds out unseen domain x skill combos.
"""

from __future__ import annotations

import hashlib


def stable_hash(text: str, salt: str = "") -> int:
    return int(hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()[:8], 16)


def family_id(bundle: dict) -> str:
    return f"{bundle.get('domain', '')}::{bundle.get('scenario_type', '')}::{bundle.get('style', '')}"


def assign_split(bundle: dict, split_cfg) -> str:
    position = stable_hash(family_id(bundle), split_cfg.seed_salt) % 100
    train_cut = int(split_cfg.train * 100)
    val_cut = train_cut + int(split_cfg.validation * 100)
    if position < train_cut:
        base = "train"
    elif position < val_cut:
        base = "validation"
    else:
        base = "test"
    # OOD: deterministic held-out slice of families.
    if stable_hash("ood:" + family_id(bundle), split_cfg.seed_salt) % 100 < int(split_cfg.ood_fraction * 100):
        return "ood"
    return base


def assign_splits(bundles: list[dict], split_cfg) -> list[dict]:
    out = []
    for bundle in bundles:
        assigned = assign_split(bundle, split_cfg)
        out.append({**bundle, "split": assigned})
    return out
