"""Jev annotation: per-question distributions preserving state_id.

Two explicit modes (see AnnotatorConfig):

- ``name: mock`` — deterministic heuristic placeholder for offline
  pilot/tests. Outputs are labeled ``annotator: mock`` and must never
  be presented as Jev judgments.
- ``name: jev`` — real Jev endpoint via the Decisions API
  (``base_url + api_path``, e.g. OpenRouter
  ``https://openrouter.ai/api`` + ``/api/alpha/decisions`` with model
  ``~typesafe/jev-latest``). FAILS LOUDLY (RuntimeError) when the
  client is unavailable: missing base_url, missing API key, HTTP
  error, or unexpected response shape. Heuristic output must never
  masquerade as Jev judgments. The API key travels via the environment
  variable named in ``api_key_env`` and is never written to disk.

Real annotations are cached by
``hash(model + version + canonical(state + questions))`` so reruns
reuse paid responses instead of re-calling. All questions over one
state travel in a single request.

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
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


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


def decisions_url(annotator_cfg) -> str:
    return annotator_cfg.base_url.rstrip("/") + annotator_cfg.api_path


def canonical_bundle_payload(bundle: dict) -> str:
    return json.dumps(
        {"state_id": bundle.get("state_id"), "state": bundle.get("state"),
         "questions": bundle.get("questions")},
        sort_keys=True, ensure_ascii=False)


def jev_cache_key(bundle: dict, annotator_cfg) -> str:
    return hashlib.sha256(
        f"{annotator_cfg.model}\n{annotator_cfg.version}\n"
        f"{canonical_bundle_payload(bundle)}".encode("utf-8")).hexdigest()


def _jev_question_payload(question: dict) -> dict:
    fmt = question["format"]
    if fmt == "choice":
        return {"type": "choice", "instructions": question["question"],
                "criteria": {option: None for option in question.get("options", [])}}
    if fmt == "score":
        # Ordered level array; the API echoes a legend + index-keyed probs.
        return {"type": "score", "instructions": question["question"],
                "criteria": list(question.get("options", []))}
    return {"type": "noul", "instructions": question["question"]}


def _jev_post(url: str, api_key: str, payload: dict, max_retries: int = 4) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(max_retries):
        request = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}",
                     "HTTP-Referer": "https://github.com/sileod/tasksource",
                     "X-Title": "tasksource-jev-synthetic"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")[:500]
            except Exception:
                pass
            last_error = RuntimeError(f"Jev HTTP {exc.code}: {body}")
            if exc.code not in (429, 500, 502, 503, 504):
                raise last_error from exc
        except Exception as exc:  # noqa: BLE001 — transient network errors
            last_error = RuntimeError(f"Jev request failed: {exc}")
        time.sleep(min(2.0 ** attempt, 30.0))
    raise last_error  # type: ignore[misc]


def _check_distribution(ordered: list[float], entry: dict) -> list[float]:
    if not ordered or abs(sum(ordered) - 1.0) > 0.02:
        raise RuntimeError(f"Jev probabilities do not form a distribution: {entry}")
    return [float(v) for v in ordered]


def _jev_probabilities(question: dict, answers: dict) -> tuple[list[float], dict]:
    """Extract ordered probabilities; raise on any unexpected shape."""
    entry = answers.get(question["question_id"]) if isinstance(answers, dict) else None
    if not isinstance(entry, dict):
        raise RuntimeError(
            f"Jev response has no entry for {question['question_id']}")
    fmt = question["format"]
    if fmt == "noul":
        if "noul" not in entry:
            raise RuntimeError(f"Jev noul answer missing 'noul': {entry}")
        return [float(entry["noul"])], entry
    options = list(question.get("options", []))
    probs = entry.get("probabilities")
    if fmt == "choice":
        if not isinstance(probs, dict):
            raise RuntimeError(f"Jev choice answer has unusable probabilities: {entry}")
        try:
            ordered = [probs[option] for option in options]
        except KeyError as exc:
            raise RuntimeError(f"Jev probabilities missing option {exc}: {entry}") from exc
        return _check_distribution(ordered, entry), entry
    # Score: index-keyed probabilities + legend echoing our criteria.
    if not isinstance(probs, dict):
        raise RuntimeError(f"Jev score answer has unusable probabilities: {entry}")
    legend = entry.get("legend", {})
    for index, option in enumerate(options):
        if legend and legend.get(str(index), option) != option:
            raise RuntimeError(
                f"Jev legend drifts from spec criteria at {index}: {legend} vs {options}")
    try:
        ordered = [probs[str(index)] for index in range(len(options))]
    except KeyError as exc:
        raise RuntimeError(f"Jev score probabilities missing index {exc}: {entry}") from exc
    return _check_distribution(ordered, entry), entry


def annotate_bundle_jev(bundle: dict, annotator_cfg, api_key: str,
                        cache_dir: Path | None = None) -> dict:
    """Annotate one bundle against the real Jev endpoint (fail loudly)."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    key = jev_cache_key(bundle, annotator_cfg)
    cached_path = cache_dir / f"{key}.json" if cache_dir is not None else None
    if cached_path is not None and cached_path.exists():
        record = json.loads(cached_path.read_text(encoding="utf-8"))
        answers, returned_model = record["answers"], record["returned_model"]
    else:
        payload = {"model": annotator_cfg.model, "state": bundle["state"],
                   "questions": {q["question_id"]: _jev_question_payload(q)
                                 for q in bundle.get("questions", [])}}
        response = _jev_post(decisions_url(annotator_cfg), api_key, payload)
        if not isinstance(response.get("answers"), dict):
            raise RuntimeError(
                f"Jev response has no 'answers' map: {str(response)[:300]}")
        answers, returned_model = response["answers"], response.get("model", "")
        if cached_path is not None:
            cached_path.write_text(json.dumps(
                {"cache_key": key, "state_id": bundle.get("state_id"),
                 "requested_model": annotator_cfg.model,
                 "returned_model": returned_model,
                 "usage": response.get("usage", {}),
                 "answers": answers},
                ensure_ascii=False, indent=2), encoding="utf-8")
    annotations = []
    for question in bundle.get("questions", []):
        probs, entry = _jev_probabilities(question, answers)
        annotations.append({"annotator": "jev",
                            "annotator_version": annotator_cfg.version,
                            "returned_model": returned_model,
                            "raw_response": entry, "probabilities": probs})
    out = dict(bundle)
    out["annotations"] = annotations
    out["annotation_stats"] = [distribution_stats(a["probabilities"])
                               for a in annotations]
    return out


def require_jev_key(annotator_cfg) -> str:
    api_key = os.getenv(getattr(annotator_cfg, "api_key_env", "JEV_API_KEY"), "")
    if not api_key:
        raise RuntimeError(
            f"annotator.name is 'jev' but {annotator_cfg.api_key_env} is not set — "
            "refusing to substitute heuristic output for Jev judgments.")
    return api_key


def annotate_bundle(bundle: dict, annotator_cfg, cache_dir: Path | None = None) -> dict:
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
        return annotate_bundle_jev(bundle, annotator_cfg, require_jev_key(annotator_cfg),
                                   cache_dir)
    raise ValueError(f"Unknown annotator name: {name!r} (expected 'mock' or 'jev')")


def annotate_bundles(bundles: list[dict], annotator_cfg,
                     cache_dir: Path | None = None, max_workers: int = 8) -> list[dict]:
    if getattr(annotator_cfg, "name", "mock") != "jev":
        return [annotate_bundle(b, annotator_cfg) for b in bundles]
    # Fail fast on config errors before spending on N requests.
    api_key = require_jev_key(annotator_cfg)
    if not getattr(annotator_cfg, "base_url", ""):
        raise RuntimeError("annotator.name is 'jev' but annotator.base_url is unset.")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _one(bundle: dict) -> dict:
        return annotate_bundle_jev(bundle, annotator_cfg, api_key, cache_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_one, bundles))
    failed = [b.get("state_id") for b, r in zip(bundles, results) if r is None]
    if failed:  # pragma: no cover — pool.map raises before this
        raise RuntimeError(f"Jev annotation failed for: {failed}")
    return results


def distribution_stats(probs: list[float]) -> dict:
    """Entropy, max probability, top-2 margin for difficulty balancing.

    Single-element (noul) probabilities are expanded to the binary
    distribution [p, 1-p] so ambiguity buckets stay meaningful.
    """
    if len(probs) == 1:
        probs = [probs[0], 1.0 - probs[0]]
    total = sum(probs) or 1.0
    norm = [p / total for p in probs]
    entropy = -sum(p * math.log(p + 1e-12) for p in norm)
    ordered = sorted(norm, reverse=True)
    top = ordered[0] if ordered else 0.0
    margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else top
    return {"entropy": entropy, "max_prob": top, "margin": margin,
            "n_options": len(probs) if len(probs) > 1 else 2}
