"""Packed classification states with exactly derived multi-question targets.

A small share of direct classification rows is packed into one state
(``Item A: ... Item B: ...``) that carries several questions whose answers
follow mechanically from the members' gold labels: one item's label, whether
two items agree, whether some or all items have a label, how many do, and so
on. Packed rows are additional supervision; every member also remains a
direct row.

Scope is deliberately narrow: only one-hot, direct ``choice`` rows of
Tasksource Classification tasks sharing one duplicate-free ontology, packed
within one source and one physical split. Every emitted target is recomputed
by :func:`verify_packed_rows`, which shares no code with the operators.
"""

import hashlib
import math
import re
from collections import Counter, deque
from itertools import zip_longest

from datasets import Dataset, concatenate_datasets

from . import graded, procedural
from .augmentations import stable_fraction
from .length import LengthBudget


VARIANT = "packed_derived"
LETTERS = "ABCDEFGH"
MIN_ITEMS = 2
MIN_QUESTIONS = 3
ITEM_HEADER = re.compile(r"(?m)^Item [A-Z]:$")
ITEM_SPLIT = re.compile(r"(?m)^Item ([A-Z]):\n")


def _one_hot(size, index):
    target = [0.0] * size
    target[index] = 1.0
    return target


def _quoted(names):
    return ", ".join(f'"{name}"' for name in names)


class Operator:
    """An exact question over a pack; ``labels`` are gold indices by item."""

    name = kind = family = None

    def params(self, labels, n_labels, seed):
        return ()

    def is_valid(self, labels, params):
        return True

    def target(self, labels, params, n_labels):
        raise NotImplementedError

    def options(self, labels, names):
        return []

    def question(self, params, names):
        raise NotImplementedError

    def question_id(self, params):
        return "-".join((self.name, *map(str, params)))


class Label(Operator):
    name, kind, family = "label", "choice", "local"

    def params(self, labels, n_labels, seed):
        return [(LETTERS[i],) for i in range(len(labels))]

    def target(self, labels, params, n_labels):
        return _one_hot(n_labels, labels[LETTERS.index(params[0])])

    def options(self, labels, names):
        return list(names)

    def question(self, params, names):
        return f"Choose the criterion that best describes Item {params[0]}."


class SameLabel(Operator):
    name, kind, family = "same", "noul", "relation"

    def params(self, labels, n_labels, seed):
        return [(LETTERS[i], LETTERS[j])
                for i in range(len(labels)) for j in range(i + 1, len(labels))]

    def target(self, labels, params, n_labels):
        i, j = (LETTERS.index(letter) for letter in params)
        return [float(labels[i] == labels[j])]

    def question(self, params, names):
        return (f"Do Item {params[0]} and Item {params[1]} have the same label? "
                f"Possible labels: {_quoted(names)}.")


class InSubset(Operator):
    name, kind, family = "in", "noul", "relation"

    def params(self, labels, n_labels, seed):
        result = []
        for i, gold in enumerate(labels):
            key = f"{seed}:{LETTERS[i]}"
            size = 1 + int(stable_fraction(key, "subset-size") * (n_labels - 1))
            others = sorted((l for l in range(n_labels) if l != gold),
                            key=lambda l: stable_fraction(f"{key}:{l}", "subset"))
            for subset in ([gold, *others[:size - 1]], others[:size]):
                if 0 < len(subset) < n_labels:
                    result.append((LETTERS[i], ".".join(map(str, sorted(subset)))))
        return result

    def target(self, labels, params, n_labels):
        subset = {int(value) for value in params[1].split(".")}
        return [float(labels[LETTERS.index(params[0])] in subset)]

    def question(self, params, names):
        subset = [names[int(value)] for value in params[1].split(".")]
        wanted = f'"{subset[0]}"' if len(subset) == 1 else f"one of {_quoted(subset)}"
        return (f"Is the label of Item {params[0]} {wanted}? "
                f"Possible labels: {_quoted(names)}.")


class Exists(Operator):
    name, kind, family = "exists", "noul", "aggregate"

    def params(self, labels, n_labels, seed):
        return [(label,) for label in range(n_labels)]

    def target(self, labels, params, n_labels):
        return [float(params[0] in labels)]

    def question(self, params, names):
        return (f'Does at least one item have the label "{names[params[0]]}"? '
                f"Possible labels: {_quoted(names)}.")


class ForAll(Operator):
    name, kind, family = "forall", "noul", "aggregate"

    def params(self, labels, n_labels, seed):
        return [(label,) for label in range(n_labels)]

    def target(self, labels, params, n_labels):
        return [float(all(label == params[0] for label in labels))]

    def question(self, params, names):
        return (f'Do all items have the label "{names[params[0]]}"? '
                f"Possible labels: {_quoted(names)}.")


class Count(Operator):
    name, kind, family = "count", "score", "numeric"

    def params(self, labels, n_labels, seed):
        return [(label,) for label in range(n_labels)]

    def target(self, labels, params, n_labels):
        return _one_hot(len(labels) + 1, labels.count(params[0]))

    def options(self, labels, names):
        return [str(count) for count in range(len(labels) + 1)]

    def question(self, params, names):
        return (f'How many items have the label "{names[params[0]]}"? '
                f"Possible labels: {_quoted(names)}.")


class MostCommon(Operator):
    """Emitted only when one label is strictly the most frequent."""

    name, kind, family = "most", "choice", "numeric"

    def params(self, labels, n_labels, seed):
        return [("common",)]

    def is_valid(self, labels, params):
        counts = Counter(labels).most_common()
        return len(counts) == 1 or counts[0][1] > counts[1][1]

    def target(self, labels, params, n_labels):
        return _one_hot(n_labels, Counter(labels).most_common(1)[0][0])

    def options(self, labels, names):
        return list(names)

    def question(self, params, names):
        return "Which label is shared by the most items?"


OPERATORS = (Label(), SameLabel(), InSubset(), Exists(), ForAll(), Count(), MostCommon())


def render_state(item_states):
    """The mechanically explicit packed state; item order is display order."""
    return "\n\n".join(
        f"Item {letter}:\n{state}" for letter, state in zip(LETTERS, item_states)
    )


def _select_questions(candidates, seed, max_questions):
    """One question per family, then fill, keeping Noul answers mixed."""
    ranked = sorted(candidates, key=lambda row: stable_fraction(row["question_id"], seed))
    chosen = []
    for family in ("local", "relation", "aggregate", "numeric"):
        nouls = [row["target"][0] for row in chosen if row["kind"] == "noul"]
        pool = [row for row in ranked if row["_family"] == family]
        # Prefer the Noul answer not yet seen so the set is not all-yes/all-no.
        pool.sort(key=lambda row: row["kind"] == "noul" and row["target"][0] in nouls)
        if pool and len(chosen) < max_questions:
            chosen.append(pool[0])
    for row in ranked:
        if len(chosen) >= max_questions:
            break
        if row in chosen:
            continue
        nouls = Counter(r["target"][0] for r in chosen if r["kind"] == "noul")
        if row["kind"] == "noul" and nouls and row["target"][0] == nouls.most_common(1)[0][0] \
                and len(nouls) == 1:
            continue
        chosen.append(row)
    return chosen


def _build_pack(members, rows, names, prefix, budget, max_questions):
    """Render one candidate pack; ``None`` if degenerate or over budget."""
    ids = [rows[m]["id"] for m in members]
    pack_id = hashlib.sha1("\n".join(sorted(ids)).encode("utf-8")).hexdigest()[:12]
    display = sorted(members, key=lambda m: stable_fraction(rows[m]["id"], f"pack-order:{pack_id}"))
    labels = [rows[m]["target"].index(1.0) for m in display]
    state = render_state([rows[m]["state"] for m in display])
    group_id = f"{prefix}:pack-{pack_id}"
    candidates = []
    for operator in OPERATORS:
        for params in operator.params(labels, len(names), pack_id):
            if not operator.is_valid(labels, params):
                continue
            question_id = operator.question_id(params)
            candidates.append({
                "id": f"{group_id}:{question_id}",
                "question_id": question_id,
                "kind": operator.kind,
                "options": operator.options(labels, names),
                "target": operator.target(labels, params, len(names)),
                "question": operator.question(params, names),
                "_family": operator.family,
            })
    questions = _select_questions(candidates, f"pack-questions:{pack_id}", max_questions)
    nouls = {row["target"][0] for row in questions if row["kind"] == "noul"}
    noul_count = sum(row["kind"] == "noul" for row in questions)
    if len(questions) < MIN_QUESTIONS or (noul_count >= 2 and len(nouls) == 1):
        return None
    if not budget.fits(state, questions):
        return None
    audit = {
        "group_id": group_id,
        "source": rows[members[0]]["source"],
        "split": rows[members[0]]["split"],
        "options": list(names),
        "members": [
            {"item": letter, "id": rows[m]["id"], "label": names[label]}
            for letter, m, label in zip(LETTERS, display, labels)
        ],
        "questions": [row["question_id"] for row in questions],
    }
    return questions, state, audit


def _plan(pack_index, pools, n_labels, max_items, seed):
    """Choose a label pattern first, then members fill it.

    Binary tasks cycle through every size/count pattern (00, 01, 11, 001, ...).
    Multiclass packs mix two or three distinct labels. The first two labels
    differ whenever the pattern allows, so greedy truncation stays mixed.
    """
    available = {label for label, pool in pools.items() if pool}
    if n_labels == 2:
        patterns = [(size, ones) for size in range(MIN_ITEMS, max_items + 1)
                    for ones in range(size + 1)]
        for attempt in range(len(patterns)):
            size, ones = patterns[(pack_index + attempt) % len(patterns)]
            if len(pools[1]) >= ones and len(pools[0]) >= size - ones:
                minority, majority = sorted(([1] * ones, [0] * (size - ones)), key=len)
                return [label for pair in zip_longest(minority, majority)
                        for label in pair if label is not None]
        return None
    if len(available) < 2:
        return None
    span = max_items - MIN_ITEMS + 1
    size = MIN_ITEMS + pack_index % span
    distinct = min(2 + (pack_index // span) % 2, size, len(available))
    chosen = sorted(available, key=lambda l: stable_fraction(f"{seed}:{pack_index}:{l}", "pack-labels"))
    chosen = chosen[:distinct]
    plan, room = list(chosen), {l: len(pools[l]) - 1 for l in chosen}
    while len(plan) < size and any(room.values()):
        for label in chosen:
            if len(plan) < size and room[label]:
                plan.append(label)
                room[label] -= 1
    return plan


def _pack_group(indices, rows, rate, budget, max_items, max_questions):
    """Pack one source/split; returns (derived rows, audit records)."""
    counts = Counter(tuple(rows[i]["options"]) for i in indices)
    ontology = min(counts, key=lambda names: (-counts[names], names))
    names = list(ontology)
    eligible = [
        i for i in indices
        if tuple(rows[i]["options"]) == ontology and len(set(names)) == len(names)
        and len(names) >= 2 and rows[i]["target"].count(1.0) == 1
        and rows[i]["target"].count(0.0) == len(names) - 1
        and len(rows[i]["id"].split(":")) == 3
        and not ITEM_HEADER.search(rows[i]["state"])
    ]
    quota = math.floor(rate * len(indices))
    seed = f"{rows[indices[0]]['source']}:{rows[indices[0]]['split']}"
    pools = {label: deque() for label in range(len(names))}
    for i in sorted(eligible, key=lambda i: stable_fraction(rows[i]["id"], "pack-member")):
        pools[rows[i]["target"].index(1.0)].append(i)
    prefix = rows[indices[0]]["id"].rsplit(":", 1)[0]
    derived, audits, used, pack_index = [], [], 0, 0
    while quota - used >= MIN_ITEMS:
        plan = _plan(pack_index, pools, len(names), max_items, seed)
        pack_index += 1
        if plan is None:
            break
        members = [pools[label].popleft() for label in plan[:quota - used]]
        accepted = None
        for size in range(MIN_ITEMS, len(members) + 1):
            pack = _build_pack(members[:size], rows, names, prefix, budget, max_questions)
            if pack is None:
                break
            accepted, kept = pack, size
        if accepted is None:
            # Drop the longer of the first two for good, so the loop progresses.
            longest = max(members[:2], key=lambda m: len(rows[m]["state"]))
            kept, members = 0, [m for m in members if m != longest]
        for m in reversed(members[kept:]):
            pools[rows[m]["target"].index(1.0)].appendleft(m)
        if accepted is None:
            continue
        questions, state, audit = accepted
        used += kept
        audits.append(audit)
        for question in questions:
            derived.append({
                "id": question["id"], "kind": question["kind"],
                "options": question["options"], "target": question["target"],
                "state": state, "question": question["question"],
                "source": audit["source"], "variant": VARIANT, "split": audit["split"],
            })
    return derived, audits


def add_packed_classification(
    dataset, task_type, rate=0.10, budget=None, max_items=4, max_questions=4,
    audit=None,
):
    """Append verified packed rows to direct Jev training rows.

    At most ``floor(rate * n)`` of the ``n`` direct rows of each source/split
    join a pack, each at most once. Only ``task_type == "Classification"``
    is packed; anything else is returned unchanged. ``audit``, if a list,
    receives one provenance record per pack (member IDs by item letter).
    """
    if task_type != "Classification" or not rate or not len(dataset):
        return dataset
    if not 1 < max_items <= len(LETTERS):
        raise ValueError(f"max_items must be in 2..{len(LETTERS)}")
    budget = budget or LengthBudget()
    rows = dataset.select_columns(
        ["id", "kind", "options", "target", "state", "source", "variant", "split"]
    ).to_list()
    groups = {}
    for index, row in enumerate(rows):
        if (row["kind"] == "choice" and row["variant"] == "direct"
                and not row["source"].startswith((procedural.SOURCE_PREFIX, graded.SOURCE_PREFIX))):
            groups.setdefault((row["source"], row["split"]), []).append(index)
    derived, audits = [], []
    for key in sorted(groups):
        group_rows, group_audits = _pack_group(
            groups[key], rows, rate, budget, max_items, max_questions
        )
        derived.extend(group_rows)
        audits.extend(group_audits)
    if not derived:
        return dataset
    verify_packed_rows(rows, derived, audits, rate, budget)
    if audit is not None:
        audit.extend(audits)
    return concatenate_datasets([dataset, Dataset.from_list(derived, features=dataset.features)])


# --- Independent verification -------------------------------------------------
# Nothing below calls the operators: targets are recomputed from the direct
# rows' gold labels, the parsed packed state, and the parsed question id.

def derive_target(question_id, labels, names):
    """Expected ``(kind, options, target)`` for ``question_id`` over ``labels``."""
    head, *args = question_id.split("-")
    letter = lambda value: "ABCDEFGH".index(value)
    size = len(labels)
    if head == "label":
        return "choice", list(names), [float(l == labels[letter(args[0])]) for l in range(len(names))]
    if head == "same":
        return "noul", [], [1.0 if labels[letter(args[0])] == labels[letter(args[1])] else 0.0]
    if head == "in":
        subset = {int(value) for value in args[1].split(".")}
        if not 0 < len(subset) < len(names):
            raise ValueError(f"Trivial subset in {question_id}")
        return "noul", [], [1.0 if labels[letter(args[0])] in subset else 0.0]
    if head == "exists":
        return "noul", [], [1.0 if int(args[0]) in labels else 0.0]
    if head == "forall":
        return "noul", [], [1.0 if set(labels) == {int(args[0])} else 0.0]
    if head == "count":
        hits = len([l for l in labels if l == int(args[0])])
        return "score", [str(c) for c in range(size + 1)], [float(c == hits) for c in range(size + 1)]
    if head == "most":
        tally = {l: labels.count(l) for l in set(labels)}
        top = max(tally.values())
        winners = [l for l, c in tally.items() if c == top]
        if len(winners) != 1:
            raise ValueError(f"Tied most-common label in {question_id}")
        return "choice", list(names), [float(l == winners[0]) for l in range(len(names))]
    raise ValueError(f"Unknown derived question {question_id}")


def packed_items(state):
    """Item texts of a packed state, in item order."""
    return _parse_state(state)[1]


def _parse_state(state):
    """Split ``Item X:`` blocks; members never contain such header lines."""
    parts = ITEM_SPLIT.split(state)
    if parts[0] != "":
        raise ValueError("Packed state must start with an item header")
    letters, blocks = parts[1::2], parts[2::2]
    if any(not block.endswith("\n\n") for block in blocks[:-1]):
        raise ValueError("Packed items must be separated by a blank line")
    return letters, [block[:-2] for block in blocks[:-1]] + blocks[-1:]


def verify_packed_rows(direct_rows, derived_rows, audits, rate, budget):
    """Raise unless every packed row is exactly recomputable from gold labels."""
    direct = {row["id"]: row for row in direct_rows if row["variant"] == "direct"}
    by_group = {}
    for row in derived_rows:
        by_group.setdefault(row["id"].rsplit(":", 1)[0], []).append(row)
    audited = {record["group_id"]: record for record in audits}
    if set(by_group) != set(audited) or len(audited) != len(audits):
        raise ValueError("Packed rows and audit records disagree")
    seen, participation = set(), Counter()
    for group_id, record in audited.items():
        rows = by_group[group_id]
        member_ids = [member["id"] for member in record["members"]]
        if seen & set(member_ids) or len(set(member_ids)) != len(member_ids):
            raise ValueError(f"Source row reused in {group_id}")
        seen.update(member_ids)
        if not set(member_ids) <= set(direct):
            raise ValueError(f"Pack {group_id} references non-direct rows")
        members = [direct[identifier] for identifier in member_ids]
        names = members[0]["options"]
        for member in members:
            if (member["source"], member["split"]) != (record["source"], record["split"]):
                raise ValueError(f"Pack {group_id} crosses sources or splits")
            if member["options"] != names or len(set(names)) != len(names):
                raise ValueError(f"Pack {group_id} mixes ontologies")
            if sorted(member["target"]) != [0.0] * (len(names) - 1) + [1.0]:
                raise ValueError(f"Pack {group_id} includes a soft label")
        participation[(record["source"], record["split"])] += len(members)
        states = {row["state"] for row in rows}
        if len(states) != 1:
            raise ValueError(f"Pack {group_id} questions do not share a state")
        letters, blocks = _parse_state(states.pop())
        expected_letters = list("ABCDEFGH"[:len(members)])
        if letters != expected_letters or [m["item"] for m in record["members"]] != expected_letters:
            raise ValueError(f"Pack {group_id} has malformed item letters")
        if blocks != [member["state"] for member in members]:
            raise ValueError(f"Pack {group_id} state does not match its members")
        labels = [member["target"].index(1.0) for member in members]
        if not MIN_QUESTIONS <= len(rows) or [r["id"].rsplit(":", 1)[1] for r in rows] != record["questions"]:
            raise ValueError(f"Pack {group_id} questions do not match the audit")
        for row in rows:
            question_id = row["id"].rsplit(":", 1)[1]
            kind, options, target = derive_target(question_id, labels, names)
            if (row["kind"], row["options"], row["target"]) != (kind, options, target):
                raise ValueError(f"Derived target mismatch for {row['id']}")
            if (row["source"], row["split"], row["variant"]) != (record["source"], record["split"], VARIANT):
                raise ValueError(f"Derived provenance mismatch for {row['id']}")
            if kind != "noul" and len(options) < 2:
                raise ValueError(f"Invalid {kind} primitive for {row['id']}")
            head, *args = question_id.split("-")
            referenced = {"label": args[:1], "in": args[:1], "same": args[:2]}.get(head, [])
            for letter in referenced:
                if f"Item {letter}" not in row["question"]:
                    raise ValueError(f"Question text omits Item {letter} in {row['id']}")
        if not budget.fits(rows[0]["state"], [
            {**row, "question_id": row["id"].rsplit(":", 1)[1]} for row in rows
        ]):
            raise ValueError(f"Pack {group_id} exceeds the length budget")
    totals = Counter((row["source"], row["split"]) for row in direct.values())
    for key, count in participation.items():
        if count > math.floor(rate * totals[key]):
            raise ValueError(f"Packing exceeds rate {rate} for {key}")
