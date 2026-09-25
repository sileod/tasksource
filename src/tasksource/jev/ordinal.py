"""Ordinal label sets, asked as Jev score questions on part of their rows.

A score question needs criteria in scale order. Known scales (sentiment
polarity, star ratings) are recognized from the label names; other ordered
label sets are tagged on their annotation with ``ordinal=True``, their names
listed in scale order. Only part of the rows become score questions, so the
same task keeps the choice format too.
"""

import re

from .augmentations import stable_fraction

ORDINAL_SCORE_SHARE = 0.5

SCALES = (
    ("negative", "neutral", "positive"),
    ("very negative", "negative", "neutral", "positive", "very positive"),
)
_STARS = re.compile(r"(\d+) stars?")


def scale_order(names, tagged=False):
    """Indices of ``names`` in scale order, or ``None`` if not ordinal."""
    lowered = [str(name).strip().lower() for name in names]
    for scale in SCALES:
        if sorted(lowered) == sorted(scale):
            return [lowered.index(level) for level in scale]
    stars = [_STARS.fullmatch(name) for name in lowered]
    if len(names) >= 3 and all(stars):
        return sorted(range(len(names)), key=lambda i: int(stars[i].group(1)))
    return list(range(len(names))) if tagged else None


def asks_score(identifier):
    return stable_fraction(identifier, "ordinal-score") < ORDINAL_SCORE_SHARE
