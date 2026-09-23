"""Deduplication that scales: exact hash + SimHash LSH banding.

The old all-pairs Jaccard loop is O(n^2) — fine for a 1k pilot,
unusable at ~60k states (~10^9 comparisons). This module is O(n)
amortized:

1. exact normalized-state hash (keeps the first occurrence), then
2. SimHash (64-bit, word unigrams, tf-weighted) with LSH banding:
   states sharing any 16-bit band become candidates, and only
   candidates pay the exact Jaccard check.

Pure stdlib, deterministic (fixed tokenization, md5-based token
hashes with a global cache, input-order keeps). Near-duplicates keep
the first occurrence in input order.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9]+")
_TOKEN_HASHES: dict[str, int] = {}
N_BANDS = 4
BAND_BITS = 16


def normalize_state(state: str) -> str:
    return " ".join(_WORD.findall(state.lower()))


def state_hash(state: str) -> str:
    return hashlib.sha256(normalize_state(state).encode("utf-8")).hexdigest()


def tokens(state: str) -> list[str]:
    return _WORD.findall(state.lower())


def jaccard(a: str, b: str) -> float:
    set_a, set_b = set(tokens(a)), set(tokens(b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _token_hash(token: str) -> int:
    hashed = _TOKEN_HASHES.get(token)
    if hashed is None:
        hashed = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:16], 16)
        _TOKEN_HASHES[token] = hashed
    return hashed


def simhash64(counts: Counter) -> int:
    """64-bit SimHash over tf-weighted token counts (deterministic)."""
    acc = [0] * 64
    for token, weight in counts.items():
        digest = _token_hash(token)
        for bit in range(64):
            acc[bit] += weight if (digest >> bit) & 1 else -weight
    fingerprint = 0
    for bit in range(64):
        if acc[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def _bands(fingerprint: int) -> list[tuple[int, int]]:
    mask = (1 << BAND_BITS) - 1
    return [(band, (fingerprint >> (band * BAND_BITS)) & mask)
            for band in range(N_BANDS)]


def dedup_bundles(bundles: list[dict], threshold: float = 0.9) -> tuple[list[dict], list[str]]:
    """Return (kept, dropped_state_ids).

    Exact duplicates always drop; near-duplicates (exact token-set
    Jaccard >= threshold over LSH candidates) keep the first input
    occurrence.
    """
    # Pass 1: exact normalized duplicates.
    seen_hashes: set[str] = set()
    unique: list[dict] = []
    dropped: list[str] = []
    for bundle in bundles:
        digest = state_hash(bundle.get("state", ""))
        if digest in seen_hashes:
            dropped.append(bundle["state_id"])
            continue
        seen_hashes.add(digest)
        unique.append(bundle)

    # Pass 2: SimHash LSH candidates + exact verification.
    fingerprints = [simhash64(Counter(tokens(b.get("state", "")))) for b in unique]
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, fingerprint in enumerate(fingerprints):
        for band_key in _bands(fingerprint):
            buckets.setdefault(band_key, []).append(index)
    candidate_pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) > 1:
            members = sorted(members)
            for pos, left in enumerate(members):
                for right in members[pos + 1:]:
                    candidate_pairs.add((left, right))

    token_sets = [set(tokens(b.get("state", ""))) for b in unique]
    dropped_positions: set[int] = set()
    for left, right in sorted(candidate_pairs):
        if left in dropped_positions or right in dropped_positions:
            continue
        set_a, set_b = token_sets[left], token_sets[right]
        if not set_a or not set_b:
            continue
        if len(set_a & set_b) / len(set_a | set_b) >= threshold:
            dropped_positions.add(right)
    kept = [b for i, b in enumerate(unique) if i not in dropped_positions]
    dropped.extend(unique[i]["state_id"] for i in sorted(dropped_positions))
    return kept, dropped
