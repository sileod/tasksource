"""Position-neutral multiple-choice criteria for the canonical Jev recast.

Legacy Tasksource preprocessing moves the gold answer to index 0. Jev reads
option probabilities from runtime-supplied criteria, so a positional prior
would be learned as a shortcut. Each MultipleChoice row is therefore
permuted deterministically from ``(task, split, source_row)``, never from
global RNG state.
"""

import re
from collections import Counter

from .augmentations import stable_fraction


# No explicit option cap for Jev: every source option is kept.
JEV_MAX_MC_OPTIONS = None

# Options whose meaning depends on their slot stay where they are.
PINNED_OPTION = re.compile(
    r"^\W*(?:all|none|neither|both) of (?:the )?"
    r"(?:above|these|those|the above|following|the following|options|choices|answers)\W*$",
    re.IGNORECASE,
)
# Options that refer to other options by letter or number make the whole row
# order-dependent, e.g. "A and B" or "only options 1 and 3".
REFERENTIAL_OPTION = re.compile(
    r"^\W*(?:\(?[A-Ea-e1-5]\)?\s*(?:,|and|or|&)\s*)+\(?[A-Ea-e1-5]\)?\W*$"
    r"|\b(?:options?|choices?|answers?)\s+\(?[A-E1-5]\)?(?:\W|$)",
)


def choice_permutation(criteria, identifier):
    """Return a new slot order, or ``None`` when options must keep their order."""
    texts = [str(option) for option in criteria]
    if any(REFERENTIAL_OPTION.search(text) for text in texts):
        return None
    movable = [index for index, text in enumerate(texts) if not PINNED_OPTION.match(text)]
    shuffled = sorted(
        movable, key=lambda index: stable_fraction(identifier, f"mc-option-{index}")
    )
    order = list(range(len(texts)))
    for slot, index in zip(movable, shuffled):
        order[slot] = index
    return order


def permute_choices(criteria, label, identifier):
    """Permute criteria and remap the label; unlabeled rows are unchanged."""
    criteria = list(criteria)
    if not 0 <= label < len(criteria):
        return criteria, label
    order = choice_permutation(criteria, identifier)
    if order is None:
        return criteria, label
    return [criteria[index] for index in order], order.index(label)


def gold_position_violations(sources, targets, min_rows=100, max_share=0.8):
    """Sources whose gold answers pile into one slot.

    Returns ``{source: (share, slot, rows)}`` for sources with at least
    ``min_rows`` one-hot targets and more than ``max_share`` of them in one
    slot. Apply this only to multiple-choice sources: for classification the
    gold index is the label prior, not a position artifact.
    """
    slots = {}
    for source, target in zip(sources, targets):
        if target and max(target) == 1.0 and sum(target) == 1.0:
            slots.setdefault(source, Counter())[target.index(1.0)] += 1
    violations = {}
    for source, counts in slots.items():
        rows = sum(counts.values())
        slot, top = counts.most_common(1)[0]
        if rows >= min_rows and top / rows > max_share:
            violations[source] = (round(top / rows, 4), slot, rows)
    return violations
