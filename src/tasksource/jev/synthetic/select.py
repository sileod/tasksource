"""Quality / distribution selection using observed Jev ambiguity.

When a run is downsampled, selection deliberately has two arms:

1. a deterministic teacher-independent reservoir, chosen only from state_id;
2. confidence-bucket balancing over the remaining examples.

This makes it possible to measure how much Jev-confidence-based curation changes
the training distribution instead of making every retained example conditional
on the teacher's own uncertainty. Every selected bundle records its
`selection_reason` and observed `selection_bucket`.
"""

from __future__ import annotations

import hashlib
from collections import Counter

from .split import family_id


def ambiguity_bucket(max_prob: float) -> str:
    if max_prob >= 0.90:
        return "very_confident"
    if max_prob >= 0.70:
        return "confident"
    if max_prob >= 0.50:
        return "moderately_ambiguous"
    if max_prob >= 0.35:
        return "high_ambiguity"
    return "near_uniform"


def _bundle_score(bundle: dict) -> float:
    """Mean max_prob over questions (lower = more ambiguous)."""
    stats = bundle.get("annotation_stats", [])
    if not stats:
        return 1.0
    return sum(s["max_prob"] for s in stats) / len(stats)


def _hash_key(bundle: dict, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{bundle['state_id']}".encode()).hexdigest()


def _with_selection_metadata(bundle: dict, reason: str) -> dict:
    out = dict(bundle)
    out["selection_reason"] = reason
    out["selection_bucket"] = ambiguity_bucket(_bundle_score(bundle))
    return out


def _allocation(total: int, weights: dict) -> dict[str, int]:
    mass = sum(float(weight) for weight in weights.values())
    plan = {key: int(round(total * float(weight) / mass))
            for key, weight in weights.items()}
    drift = total - sum(plan.values())
    if drift and plan:
        biggest = max(plan, key=lambda key: float(weights[key]))
        plan[biggest] += drift
    return plan


def select_bundles(bundles: list[dict], selection_cfg,
                   n_target: int | None = None) -> dict:
    if not bundles:
        return {"selected": [], "diagnostics": {
            "bucket_counts": {}, "plan": {}, "unfiltered_target": 0,
            "unfiltered_selected": 0, "requested_ambiguity": {}}}

    total = min(len(bundles), n_target if n_target is not None else len(bundles))
    cap = getattr(selection_cfg, "max_per_family", 0)
    salt = getattr(selection_cfg, "seed_salt", "jev-synthetic-selection-v2")
    unfiltered_fraction = float(getattr(selection_cfg, "unfiltered_fraction", 0.0))
    per_family: Counter = Counter()
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def admit(bundle: dict, reason: str) -> bool:
        if bundle["state_id"] in selected_ids:
            return False
        family = family_id(bundle)
        if cap and per_family[family] >= cap:
            return False
        per_family[family] += 1
        selected_ids.add(bundle["state_id"])
        selected.append(_with_selection_metadata(bundle, reason))
        return True

    # Arm 1: teacher-independent reservoir. This ordering never consults the
    # annotation or ambiguity metadata.
    unfiltered_target = int(round(total * unfiltered_fraction))
    for bundle in sorted(bundles, key=lambda b: _hash_key(b, f"{salt}:unfiltered")):
        if len(selected) >= unfiltered_target:
            break
        admit(bundle, "unfiltered_reservoir")
    unfiltered_selected = len(selected)

    # Arm 2: desired Jev ambiguity mixture over the remaining capacity.
    buckets: dict[str, list[dict]] = {}
    for bundle in bundles:
        buckets.setdefault(ambiguity_bucket(_bundle_score(bundle)), []).append(bundle)
    for key in buckets:
        buckets[key].sort(key=lambda b: _hash_key(b, f"{salt}:bucket:{key}"))

    remaining_target = max(0, total - len(selected))
    plan = _allocation(remaining_target, selection_cfg.buckets)
    for bucket_name, count in plan.items():
        taken = 0
        for bundle in buckets.get(bucket_name, []):
            if taken >= count:
                break
            if admit(bundle, f"ambiguity_bucket:{bucket_name}"):
                taken += 1

    # Backfill deterministically when a requested bucket is short or a family
    # cap blocks its quota. The backfill is reported distinctly.
    if len(selected) < total:
        for bundle in sorted(bundles, key=lambda b: _hash_key(b, f"{salt}:backfill")):
            if len(selected) >= total:
                break
            admit(bundle, "backfill")

    selected.sort(key=lambda b: b["state_id"])
    by_requested = Counter(bundle.get("ambiguity", "?") for bundle in selected)
    by_reason = Counter(bundle.get("selection_reason", "?") for bundle in selected)
    final_buckets = Counter(bundle.get("selection_bucket", "?") for bundle in selected)
    return {
        "selected": selected,
        "diagnostics": {
            "input_states": len(bundles),
            "target_states": total,
            "bucket_counts": {key: len(value) for key, value in buckets.items()},
            "plan": plan,
            "unfiltered_target": unfiltered_target,
            "unfiltered_selected": unfiltered_selected,
            "selection_reasons": dict(by_reason),
            "selected_bucket_counts": dict(final_buckets),
            "requested_ambiguity": dict(by_requested),
            "max_per_family": cap,
            "seed_salt": salt,
        },
    }
