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

# A final "all/none of the above" means all/none of the other options, which
# reads the same in any slot once reworded.
TRAILING_ALL_NONE = re.compile(
    r"^(\W*)(all|none) of (?:the )?(?:above|these|those|the above|following|the following)"
    r"(?: (?:choices|options|answers))?(\W*)$",
    re.IGNORECASE,
)
# Any other option about its neighbours ("both of the above", a non-final
# "none of the above") depends on slot order.
POSITIONAL_OPTION = re.compile(
    r"\b(?:all|none|neither|both|either|any|each) of (?:the )?(?:above|below|these|those|following|preceding)\b",
    re.IGNORECASE,
)
# Options that refer to other options by letter, number, or roman numeral:
# "A and B", "I and III only", "statements 1, 2 and 4", "options (a) or (c)".
_REFERENCE = r"(?:\(?[A-H]\)|[A-H]|\([a-h]\)|[IVX]{1,4}|\(?[1-9]\)?)"
REFERENTIAL_OPTION = re.compile(
    rf"^\W*(?:(?:both|neither|either|only|all|none|except|but|not|of|the|statements?|options?|choices?|answers?)\s+)*"
    rf"{_REFERENCE}(?:\s*(?:,\s*(?:and|or)?|&|and|or|nor|/)\s*(?:both\s+|only\s+)?{_REFERENCE})+(?:\s+(?:only|both|are correct|are true))?\W*$"
    rf"|^\W*(?:{_REFERENCE}\s+only|only\s+{_REFERENCE})\W*$"
    rf"|\b(?:options?|choices?|answers?|statements?)\s+{_REFERENCE}(?:\W|$)",
    re.IGNORECASE,
)


def normalize_all_none(criteria):
    """Reword a final "all/none of the above" so it no longer depends on its slot."""
    texts = [str(option) for option in criteria]
    match = TRAILING_ALL_NONE.match(texts[-1]) if texts else None
    if not match:
        return list(criteria)
    lead, word, tail = match.groups()
    word = word.capitalize() if word[0].isupper() else word.lower()
    return list(criteria[:-1]) + [f"{lead}{word} of the other options{tail}"]


def choice_permutation(criteria, identifier):
    """Return a new slot order, or ``None`` when options must keep their order."""
    texts = [str(option) for option in criteria]
    if any(POSITIONAL_OPTION.search(t) or REFERENTIAL_OPTION.search(t) for t in texts):
        return None
    return sorted(range(len(texts)), key=lambda index: stable_fraction(identifier, f"mc-option-{index}"))


def permute_choices(criteria, label, identifier):
    """Permute criteria and remap the label; unlabeled rows are unchanged.

    Rows whose options refer to each other by position keep the source order.
    """
    criteria = list(criteria)
    if not 0 <= label < len(criteria):
        return criteria, label
    reworded = normalize_all_none(criteria)
    order = choice_permutation(reworded, identifier)
    if order is None:
        return criteria, label
    return [reworded[index] for index in order], order.index(label)


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
