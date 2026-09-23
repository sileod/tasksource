"""Jev annotation: per-question distributions preserving state_id.

Two explicit modes (see AnnotatorConfig):

- ``name: mock`` — deterministic heuristic placeholder for offline
  pilot/tests. Outputs are labeled ``annotator: mock`` and must never
  be presented as Jev judgments.
- ``name: jev`` — real Jev endpoint. FAILS LOUDLY (RuntimeError) when
  the client is unavailable: missing base_url, missing API key, HTTP
  error, or unexpected response shape. Heuristic output must never
  masquerade as Jev judgments.

Stores the untouched response plus the normalized representation, e.g.
    {"annotator": "jev", "annotator_version": ..., "raw_response": {...},
     "probabilities": [...]}
Never only argmax.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request


def _rng_for(question_id: str, version: str):
    import random
    seed = int(hashlib.sha256(f"{version}:{question_id}".encode()).hexdigest()[:8], 16)
    return random.Random(seed)


def heuristic_distribution(fmt: str, n_options: int, question_id: str, version: str) -> list[float]:
    rng = _rng_for(question_id, version)
    if fmt == "noul":
        # Cover obvious/likely/ambiguous yes/no via deterministic draws.
        return [round(rng.choice([0.05, 0.2, 0.5, 0.8, 0.95]), 4)]
    n = max(2, n_options)
    raw = [rng.gammavariate(1.5, 1.0) + 0.05 for _ in range(n)]
    total = sum(raw)
    return [round(v / total, 6) for v in raw]


def annotate_question_mock(question: dict, version: str) -> dict:
    fmt = question["format"]
    n_options = len(question.get("options", [])) if fmt in ("choice", "score") else 0
    probs = heuristic_distribution(fmt, n_options, question["question_id"], version)
    return {"annotator": "mock", "annotator_version": version,
            "raw_response": {"heuristic": True, "question_id": question["question_id"]},
            "probabilities": probs}


def _jev_question_payload(question: dict) -> dict:
    """TypeSafe-shaped question payload (state travels alongside)."""
    fmt = question["format"]
    if fmt == "choice":
        return {"type": "choice", "instructions": question["question"],
                "criteria": {option: None for option in question.get("options", [])}}
    if fmt == "score":
        return {"type": "score", "instructions": question["question"],
                "criteria": {option: None for option in question.get("options", [])}}
    return {"type": "noul", "instructions": question["question"]}


def _jev_post(base_url: str, api_key: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/v1/decisions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Jev annotation request failed: {exc}") from exc


def _jev_probabilities(question: dict, response: dict) -> tuple[list[float], dict]:
    """Extract ordered probabilities; raise on any unexpected shape."""
    questions = response.get("questions", response)
    entry = questions.get(question["question_id"]) if isinstance(questions, dict) else None
    if not isinstance(entry, dict):
        raise RuntimeError(
            f"Jev response has no entry for {question['question_id']}: "
            f"{str(response)[:300]}")
    fmt = question["format"]
    if fmt == "noul":
        if "noul" not in entry:
            raise RuntimeError(f"Jev noul response missing 'noul': {entry}")
        return [float(entry["noul"])], entry
    options = list(question.get("options", []))
    probs = entry.get("probabilities")
    if isinstance(probs, dict):
        try:
            ordered = [float(probs[option]) for option in options]
        except KeyError as exc:
            raise RuntimeError(f"Jev probabilities missing option {exc}: {entry}") from exc
    elif isinstance(probs, list) and len(probs) == len(options):
        ordered = [float(v) for v in probs]
    else:
        raise RuntimeError(
            f"Jev {fmt} response has unusable probabilities: {entry}")
    if not options or abs(sum(ordered) - 1.0) > 0.02:
        raise RuntimeError(f"Jev probabilities do not form a distribution: {entry}")
    return ordered, entry


def annotate_bundle_jev(bundle: dict, annotator_cfg, api_key: str) -> dict:
    """Annotate one bundle against the real Jev endpoint (fail loudly)."""
    payload = {"model": annotator_cfg.model, "state": bundle["state"],
               "questions": {q["question_id"]: _jev_question_payload(q)
                             for q in bundle.get("questions", [])}}
    response = _jev_post(annotator_cfg.base_url, api_key, payload)
    annotations = []
    for question in bundle.get("questions", []):
        probs, entry = _jev_probabilities(question, response)
        annotations.append({"annotator": "jev",
                            "annotator_version": annotator_cfg.version,
                            "raw_response": entry, "probabilities": probs})
    out = dict(bundle)
    out["annotations"] = annotations
    out["annotation_stats"] = [distribution_stats(a["probabilities"])
                               for a in annotations]
    return out


def annotate_bundle(bundle: dict, annotator_cfg) -> dict:
    name = getattr(annotator_cfg, "name", "mock")
    if name == "mock":
        out = dict(bundle)
        out["annotations"] = [annotate_question_mock(q, annotator_cfg.version)
                              for q in bundle.get("questions", [])]
        out["annotation_stats"] = [distribution_stats(a["probabilities"])
                                   for a in out["annotations"]]
        return out
    if name == "jev":
        if not getattr(annotator_cfg, "base_url", ""):
            raise RuntimeError(
                "annotator.name is 'jev' but annotator.base_url is unset — "
                "refusing to substitute heuristic output for Jev judgments.")
        api_key = os.getenv(getattr(annotator_cfg, "api_key_env", "JEV_API_KEY"), "")
        if not api_key:
            raise RuntimeError(
                f"annotator.name is 'jev' but {annotator_cfg.api_key_env} is not set — "
                "refusing to substitute heuristic output for Jev judgments.")
        return annotate_bundle_jev(bundle, annotator_cfg, api_key)
    raise ValueError(f"Unknown annotator name: {name!r} (expected 'mock' or 'jev')")


def annotate_bundles(bundles: list[dict], annotator_cfg) -> list[dict]:
    return [annotate_bundle(b, annotator_cfg) for b in bundles]


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
