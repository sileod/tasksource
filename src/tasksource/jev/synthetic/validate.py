"""Deterministic validation of state bundles (no LLM needed)."""

from __future__ import annotations

import re

from .schemas import validate_bundle_shape

LEAK_TOKENS = ("difficulty", "ambiguity", "skill", "distractor",
               "state_length", "evidence", "certainty")
ANSWER_LEAK = re.compile(r"correct (answer|option|choice)|answer is\b", re.IGNORECASE)


def validate_bundle(bundle: dict, spec: dict | None = None) -> list[str]:
    errors = validate_bundle_shape(bundle)
    state = bundle.get("state", "") or ""
    if len(state.strip()) < 20:
        errors.append("state too short")
    if len(state) > 8000:
        errors.append("state too long")
    lowered = state.lower()
    for token in LEAK_TOKENS:
        if token in lowered:
            errors.append(f"state leaks sampler metadata: {token}")
            break
    if ANSWER_LEAK.search(state):
        errors.append("state leaks correct answer phrasing")
    texts = [q.get("question", "") for q in bundle.get("questions", [])]
    for text in texts:
        if not text or len(text.strip()) < 10:
            errors.append("empty/truncated question")
        if ANSWER_LEAK.search(text):
            errors.append("question leaks correct answer phrasing")
    if len(set(texts)) != len(texts):
        errors.append("duplicate question texts in bundle")
    options_seen = [tuple(q.get("options", [])) for q in bundle.get("questions", []) if q.get("format") == "choice"]
    for opts in options_seen:
        if any(not o or not o.strip() for o in opts):
            errors.append("empty choice option")
    if spec is not None:
        expected = [q["format"] for q in spec.get("questions", [])]
        got = [q.get("format") for q in bundle.get("questions", [])]
        if expected != got:
            errors.append(f"format mismatch: expected {expected}, got {got}")
        for qspec, q in zip(spec.get("questions", []), bundle.get("questions", [])):
            if qspec.get("format") == "choice" and len(q.get("options", [])) != qspec.get("n_options"):
                errors.append(f"option count mismatch for {q.get('question_id')}")
    return errors


def is_valid(bundle: dict, spec: dict | None = None) -> bool:
    return not validate_bundle(bundle, spec)
