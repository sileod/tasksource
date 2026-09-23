"""Quality / distribution selection using observed Jev ambiguity.

Buckets (by max probability, configurable in SelectionConfig):
    very_confident       max_prob >= 0.90   (target 20%)
    confident            max_prob >= 0.70   (target 30%)
    moderately_ambiguous max_prob >= 0.50   (target 30%)
    high_ambiguity       max_prob >= 0.35   (target 15%)
    near_uniform         max_prob <  0.35   (target  5%)

Selection is deterministic (stable hash ordering within buckets) and
also reports requested-vs-observed ambiguity as a quality diagnostic.
"""

from __future__ import annotations

import hashlib


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


def select_bundles(bundles: list[dict], selection_cfg, n_target: int | None = None) -> dict:
    buckets: dict[str, list[dict]] = {}
    for bundle in bundles:
        buckets.setdefault(ambiguity_bucket(_bundle_score(bundle)), []).append(bundle)
    for key in buckets:
        buckets[key].sort(key=lambda b: hashlib.sha256(b["state_id"].encode()).hexdigest())
    total = n_target or len(bundles)
    weights = selection_cfg.buckets
    plan = {k: int(round(total * w)) for k, w in weights.items()}
    # Fix rounding drift on the largest bucket.
    drift = total - sum(plan.values())
    if drift and plan:
        biggest = max(plan, key=plan.get)
        plan[biggest] += drift
    selected: list[dict] = []
    for bucket_name, count in plan.items():
        selected.extend(buckets.get(bucket_name, [])[:max(0, count)])
    # Backfill from any bucket if a bucket is short.
    if len(selected) < total:
        have = {b["state_id"] for b in selected}
        for bucket_bundles in buckets.values():
            for bundle in bucket_bundles:
                if len(selected) >= total:
                    break
                if bundle["state_id"] not in have:
                    selected.append(bundle)
                    have.add(bundle["state_id"])
    selected.sort(key=lambda b: b["state_id"])
    by_requested = {}
    for bundle in selected:
        by_requested[bundle.get("ambiguity", "?")] = by_requested.get(bundle.get("ambiguity", "?"), 0) + 1
    return {"selected": selected,
            "diagnostics": {"bucket_counts": {k: len(v) for k, v in buckets.items()},
                            "plan": plan,
                            "requested_ambiguity": by_requested}}
