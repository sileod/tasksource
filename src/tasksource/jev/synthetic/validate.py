"""Deterministic validation of state bundles (no LLM needed)."""

from __future__ import annotations

import re

from .schemas import validate_bundle_shape

METADATA_LEAK = re.compile(
    r"\b(?:difficulty|ambiguity|skill|distractors?|state_length|"
    r"evidence_structure|certainty_hint)\s*[:=]", re.IGNORECASE)
ANSWER_LEAK = re.compile(r"correct (answer|option|choice)|answer is\b", re.IGNORECASE)
MODEL_LEAK = re.compile(r"\bjev\b", re.IGNORECASE)
EITHER_OR_LABEL = re.compile(r"\b(?:positive\s+or\s+negative|negative\s+or\s+positive|yes\s+or\s+no|no\s+or\s+yes)\b", re.IGNORECASE)
OPEN_QUESTION = re.compile(r"\b(?:what|which|who|where|when)\b", re.IGNORECASE)
EVENT_PROBABILITY = re.compile(
    r"\b(?:likelihood|probability|chance|risk|confidence)\s+that\b|"
    r"\bhow\s+(?:likely|probable|confident)\b", re.IGNORECASE)


def valid_noul_question(text: str) -> bool:
    """A noul target is a probability for one proposition, not a free-form answer."""
    if EITHER_OR_LABEL.search(text):
        return False
    if EVENT_PROBABILITY.search(text):
        return True
    if OPEN_QUESTION.search(text) or re.search(r"\bhow\b", text, re.IGNORECASE):
        return False
    return bool(re.search(
        r"\b(?:is|are|was|were|do|does|did|can|could|should|would|will|"
        r"has|have|had|may|might|must)\b", text, re.IGNORECASE))


def validate_bundle(bundle: dict, spec: dict | None = None) -> list[str]:
    errors = validate_bundle_shape(bundle)
    state = bundle.get("state", "") or ""
    if len(state.strip()) < 20:
        errors.append("state too short")
    if len(state) > 8000:
        errors.append("state too long")
    if METADATA_LEAK.search(state):
        errors.append("state leaks sampler metadata field")
    if ANSWER_LEAK.search(state):
        errors.append("state leaks correct answer phrasing")
    if MODEL_LEAK.search(state):
        errors.append("state mentions the target model")
    texts = [q.get("question", "") for q in bundle.get("questions", [])]
    for text in texts:
        if not text or len(text.strip()) < 10:
            errors.append("empty/truncated question")
        if ANSWER_LEAK.search(text):
            errors.append("question leaks correct answer phrasing")
        if MODEL_LEAK.search(text):
            errors.append("question mentions the target model")
    for q in bundle.get("questions", []):
        if q.get("format") == "noul" and not valid_noul_question(q.get("question", "")):
            errors.append(f"noul {q.get('question_id')} must ask about one yes/no proposition")
        if (q.get("format") == "score"
                and list(q.get("options", [])) ==
                ["very unlikely", "unlikely", "possible", "likely", "very likely"]
                and re.search(r"\burgen(?:t|cy)\b", q.get("question", ""), re.IGNORECASE)):
            errors.append(f"score {q.get('question_id')} uses likelihood levels for urgency")
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
            if qspec.get("format") == "choice":
                if not 2 <= len(q.get("options", [])) <= 8:
                    errors.append(f"choice {q.get('question_id')} needs 2..8 options")
                if len(q.get("options", [])) != qspec.get("n_options"):
                    errors.append(f"option count mismatch for {q.get('question_id')}")
            if qspec.get("format") == "score":
                options = q.get("options", [])
                if not 3 <= len(options) <= 10:
                    errors.append(f"score {q.get('question_id')} needs 3..10 criteria")
                if qspec.get("criteria") and list(options) != list(qspec["criteria"]):
                    errors.append(
                        f"score {q.get('question_id')} must reuse the spec criteria verbatim")
    return errors


def is_valid(bundle: dict, spec: dict | None = None) -> bool:
    return not validate_bundle(bundle, spec)
