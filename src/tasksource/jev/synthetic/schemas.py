"""Canonical bundled-state and flattened-decision schemas.

Bundled (one row per state):
    {state_id, state, questions: [{question_id, format, question,
     options?, min?, max?, skill, ...}], domain, scenario_type, style,
     difficulty, ambiguity, ...}

Flat (one row per decision, tasksource-jev compatible):
    {state_id, question_id, state, format(kind), question, options,
     bundle_size, ...} + training fields (id, kind, options, target,
     state, question, source, variant, split).
"""

from __future__ import annotations

CHOICE = "choice"
NOUL = "noul"
SCORE = "score"
FORMATS = (CHOICE, NOUL, SCORE)

# Flat rows reuse the published Jev training columns where possible.
TRAINING_COLUMNS = ("id", "kind", "options", "target", "state",
                    "question", "source", "variant", "split")


def make_question_id(state_id: str, index: int) -> str:
    return f"{state_id}_q{index}"


def bundle_to_flat_rows(bundle: dict) -> list[dict]:
    """Expand one state bundle into one flat row per question."""
    questions = bundle.get("questions", [])
    rows = []
    for index, question in enumerate(questions):
        qid = question.get("question_id") or make_question_id(bundle["state_id"], index)
        rows.append({
            "state_id": bundle["state_id"],
            "question_id": qid,
            "state": bundle["state"],
            "format": question["format"],
            "kind": question["format"],
            "question": question["question"],
            "options": list(question.get("options") or []),
            "min": question.get("min"),
            "max": question.get("max"),
            "skill": question.get("skill", ""),
            "bundle_size": len(questions),
            "domain": bundle.get("domain", ""),
            "scenario_type": bundle.get("scenario_type", ""),
            "style": bundle.get("style", ""),
            "difficulty": bundle.get("difficulty", 0),
        })
    return rows


def flat_to_training_row(flat: dict, target: list[float] | None = None,
                         source: str = "synthetic/jev", split: str = "train") -> dict:
    """Map a flat decision to the tasksource-jev training schema."""
    kind = flat["format"]
    options = list(flat.get("options") or [])
    if target is None:
        if kind == NOUL:
            target = [0.5]
        else:
            target = [1.0 / len(options)] * len(options) if options else []
    return {
        "id": flat["question_id"],
        "kind": kind,
        "options": options,
        "target": [float(v) for v in target],
        "state": flat["state"],
        "question": flat["question"],
        "source": source,
        "variant": "direct",
        "split": split,
    }


def validate_bundle_shape(bundle: dict) -> list[str]:
    """Lightweight structural check; semantic checks live in validate.py."""
    errors = []
    if not bundle.get("state_id"):
        errors.append("missing state_id")
    if not bundle.get("state"):
        errors.append("missing state")
    questions = bundle.get("questions")
    if not isinstance(questions, list) or not questions:
        errors.append("bundle must contain 1..N questions")
        return errors
    seen = set()
    for question in questions:
        qid = question.get("question_id", "")
        if qid in seen:
            errors.append(f"duplicate question_id: {qid}")
        seen.add(qid)
        if question.get("format") not in FORMATS:
            errors.append(f"bad format for {qid}: {question.get('format')}")
        if not question.get("question"):
            errors.append(f"empty question text for {qid}")
        if question.get("format") == CHOICE:
            options = question.get("options") or []
            if len(options) < 2:
                errors.append(f"choice {qid} needs >=2 options")
            if len(set(options)) != len(options):
                errors.append(f"choice {qid} has duplicate options")
        if question.get("format") == SCORE:
            if question.get("min") is None or question.get("max") is None:
                errors.append(f"score {qid} needs min/max")
            elif not question["min"] < question["max"]:
                errors.append(f"score {qid} needs min<max")
    return errors
