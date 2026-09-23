"""Deduplication: exact request-hash + near-duplicate state detection."""

from __future__ import annotations

import hashlib
import re

_WORD = re.compile(r"[a-z0-9]+")


def normalize_state(state: str) -> str:
    return " ".join(_WORD.findall(state.lower()))


def state_hash(state: str) -> str:
    return hashlib.sha256(normalize_state(state).encode("utf-8")).hexdigest()


def jaccard(a: str, b: str) -> float:
    set_a, set_b = set(normalize_state(a).split()), set(normalize_state(b).split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def dedup_bundles(bundles: list[dict], threshold: float = 0.9) -> tuple[list[dict], list[str]]:
    """Return (kept, dropped_state_ids). Exact duplicates always dropped;
    near-duplicates (Jaccard >= threshold) keep the first occurrence."""
    seen_hashes: set[str] = set()
    kept: list[dict] = []
    dropped: list[str] = []
    for bundle in bundles:
        digest = state_hash(bundle.get("state", ""))
        if digest in seen_hashes:
            dropped.append(bundle["state_id"])
            continue
        if any(jaccard(bundle.get("state", ""), other.get("state", "")) >= threshold
               for other in kept):
            dropped.append(bundle["state_id"])
            continue
        seen_hashes.add(digest)
        kept.append(bundle)
    return kept, dropped
