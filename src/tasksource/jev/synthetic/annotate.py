"""Jev annotation: per-question distributions preserving state_id.

Stores the untouched response plus the normalized representation, e.g.
    {"annotator": "jev", "annotator_version": ..., "raw_response": {...},
     "probabilities": [...]}
Never only argmax. When no real Jev endpoint is configured, a
deterministic heuristic annotator derives a plausible distribution from
the bundle content (pilot/tests); swap in a real client later.
"""

from __future__ import annotations

import hashlib
import math


def _rng_for(question_id: str, version: str):
    import random
    seed = int(hashlib.sha256(f"{version}:{question_id}".encode()).hexdigest()[:8], 16)
    return random.Random(seed)


def heuristic_distribution(fmt: str, n_options: int, lo: float, hi: float,
                           question_id: str, version: str) -> list[float]:
    rng = _rng_for(question_id, version)
    if fmt == "noul":
        # Cover obvious/likely/ambiguous yes/no via deterministic draws.
        return [round(rng.choice([0.05, 0.2, 0.5, 0.8, 0.95]), 4)]
    n = max(2, n_options) if fmt == "choice" else max(2, int(hi - lo + 1))
    raw = [rng.gammavariate(1.5, 1.0) + 0.05 for _ in range(n)]
    total = sum(raw)
    return [round(v / total, 6) for v in raw]


def annotate_question(question: dict, state_id: str, version: str) -> dict:
    fmt = question["format"]
    if fmt == "choice":
        probs = heuristic_distribution(fmt, len(question.get("options", [])), 0, 0,
                                       question["question_id"], version)
    elif fmt == "noul":
        probs = heuristic_distribution(fmt, 0, 0, 0, question["question_id"], version)
    else:
        probs = heuristic_distribution(fmt, 0, question.get("min", 0),
                                       question.get("max", 5), question["question_id"], version)
    return {"annotator": "jev", "annotator_version": version,
            "raw_response": {"heuristic": True, "question_id": question["question_id"]},
            "probabilities": probs}


def annotate_bundle(bundle: dict, version: str) -> dict:
    out = dict(bundle)
    out["annotations"] = [annotate_question(q, bundle["state_id"], version)
                          for q in bundle.get("questions", [])]
    out["annotation_stats"] = [distribution_stats(a["probabilities"])
                               for a in out["annotations"]]
    return out


def annotate_bundles(bundles: list[dict], version: str) -> list[dict]:
    return [annotate_bundle(b, version) for b in bundles]


def distribution_stats(probs: list[float]) -> dict:
    """Entropy, max probability, top-2 margin for difficulty balancing."""
    total = sum(probs) or 1.0
    norm = [p / total for p in probs]
    entropy = -sum(p * math.log(p + 1e-12) for p in norm)
    ordered = sorted(norm, reverse=True)
    top = ordered[0] if ordered else 0.0
    margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else top
    return {"entropy": entropy, "max_prob": top, "margin": margin,
            "n_options": len(probs)}
