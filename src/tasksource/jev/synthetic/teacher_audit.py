"""Independent post-teacher audit for Jev-labelled synthetic bundles.

The auditor is deliberately not shown Jev's probabilities. It answers the
state/questions independently, then this module compares that answer with the
stored teacher distribution. This catches the specific failure mode where a
teacher is highly confident on a generated example that another strong model
reads differently, without replacing Jev as the training target.

Auditor calls are content-addressed and cached. The cache contains the raw
auditor response and its independent answers; comparison with Jev is recomputed
from the current annotations so the same audit can be reused across teacher
versions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from . import providers
from .critic import RequestPacer
from .generate import PROMPTS_DIR, extract_json_object, prompt_hash


def load_teacher_audit_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.txt").read_text(encoding="utf-8")


def teacher_audit_cache_key(model: str, temperature: float, prompt: str,
                            bundle: dict, endpoint: str = "") -> str:
    canonical = json.dumps(
        {"state_id": bundle.get("state_id"), "state": bundle.get("state"),
         "questions": bundle.get("questions")},
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(
        f"{endpoint}\n{model}\n{temperature}\n{prompt_hash(prompt)}\n{canonical}".encode("utf-8")
    ).hexdigest()


def _mock_independent_answers(bundle: dict) -> list[dict]:
    """Deterministic offline answers for tests; never used as training targets."""
    out = []
    for question in bundle.get("questions", []):
        fmt = question["format"]
        if fmt == "noul":
            answer = False
        elif fmt == "score":
            answer = 0
        else:
            options = list(question.get("options", []))
            answer = options[0] if options else None
        out.append({"question_id": question["question_id"], "answer": answer,
                    "confidence": 0.9, "issues": []})
    return out


def _parse_independent_answers(raw: dict, bundle: dict) -> list[dict]:
    answers = raw.get("answers")
    if not isinstance(answers, list):
        raise ValueError("teacher-audit response needs an 'answers' list")
    known = {q["question_id"] for q in bundle.get("questions", [])}
    parsed = []
    for item in answers:
        if not isinstance(item, dict) or item.get("question_id") not in known:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        parsed.append({
            "question_id": item["question_id"],
            "answer": item.get("answer"),
            "confidence": max(0.0, min(1.0, confidence)),
            "issues": list(item.get("issues", [])) if isinstance(item.get("issues", []), list) else [],
        })
    return parsed


def _teacher_answer(question: dict, annotation: dict) -> tuple[object, float]:
    probs = [float(p) for p in annotation.get("probabilities", [])]
    if question["format"] == "noul":
        p_yes = probs[0]
        return p_yes >= 0.5, max(p_yes, 1.0 - p_yes)
    index = max(range(len(probs)), key=probs.__getitem__)
    if question["format"] == "score":
        return index, probs[index]
    return list(question.get("options", []))[index], probs[index]


def _normalize_auditor_answer(question: dict, answer) -> object | None:
    fmt = question["format"]
    if fmt == "noul":
        if isinstance(answer, bool):
            return answer
        if isinstance(answer, str):
            lowered = answer.strip().lower()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        return None
    options = list(question.get("options", []))
    if fmt == "choice":
        return answer if answer in options else None
    if isinstance(answer, int) and not isinstance(answer, bool) and 0 <= answer < len(options):
        return answer
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped in options:
            return options.index(stripped)
        try:
            index = int(stripped)
        except ValueError:
            return None
        return index if 0 <= index < len(options) else None
    return None


def compare_with_teacher(bundle: dict, independent_answers: list[dict], audit_cfg,
                         auditor_meta: dict | None = None) -> dict:
    by_id = {a["question_id"]: a for a in independent_answers}
    annotations = list(bundle.get("annotations", []))
    questions = list(bundle.get("questions", []))
    rows = []
    complete = len(annotations) == len(questions)
    for index, question in enumerate(questions):
        independent = by_id.get(question["question_id"])
        annotation = annotations[index] if index < len(annotations) else None
        if independent is None or annotation is None:
            complete = False
            rows.append({"question_id": question["question_id"], "agrees": None,
                         "confident_disagreement": False,
                         "issues": ["missing auditor answer or teacher annotation"]})
            continue
        teacher_answer, teacher_confidence = _teacher_answer(question, annotation)
        auditor_answer = _normalize_auditor_answer(question, independent.get("answer"))
        auditor_confidence = float(independent.get("confidence", 0.0))
        agrees = auditor_answer is not None and auditor_answer == teacher_answer
        confident_disagreement = (
            auditor_answer is not None
            and not agrees
            and teacher_confidence >= audit_cfg.min_teacher_confidence
            and auditor_confidence >= audit_cfg.min_auditor_confidence
        )
        rows.append({
            "question_id": question["question_id"],
            "teacher_answer": teacher_answer,
            "teacher_confidence": teacher_confidence,
            "auditor_answer": auditor_answer,
            "auditor_confidence": auditor_confidence,
            "agrees": agrees,
            "confident_disagreement": confident_disagreement,
            "issues": independent.get("issues", []),
        })
    disagreements = sum(bool(row.get("confident_disagreement")) for row in rows)
    return {
        "enabled": True,
        "pass": complete and disagreements == 0,
        "complete": complete,
        "confident_disagreements": disagreements,
        "min_teacher_confidence": audit_cfg.min_teacher_confidence,
        "min_auditor_confidence": audit_cfg.min_auditor_confidence,
        "auditor": auditor_meta or {},
        "questions": rows,
    }


async def _audit_one(sem, client, model: str, temperature: float, template: str,
                     bundle: dict, raw_dir: Path, audit_cfg,
                     pacer: RequestPacer | None = None, endpoint: str = "") -> dict:
    key = teacher_audit_cache_key(model, temperature, template, bundle, endpoint)
    cached = raw_dir / f"{key}.json"
    auditor_meta = {"model": model, "endpoint": endpoint}
    if cached.exists():
        record = json.loads(cached.read_text(encoding="utf-8"))
        independent = record["independent_answers"]
        auditor_meta.update({
            "provider": record.get("provider", ""),
            "requested_model": record.get("requested_model", model),
            "returned_model": record.get("returned_model", ""),
            "cache_key": key,
        })
    elif client is None:
        independent = _mock_independent_answers(bundle)
        cached.write_text(json.dumps(
            {"cache_key": key, "state_id": bundle["state_id"], "provider": "mock",
             "requested_model": model, "returned_model": model,
             "independent_answers": independent},
            ensure_ascii=False, indent=2), encoding="utf-8")
        auditor_meta.update({"provider": "mock", "requested_model": model,
                             "returned_model": model, "cache_key": key})
    else:
        payload = {"state_id": bundle["state_id"], "state": bundle["state"],
                   "questions": bundle.get("questions", [])}
        prompt = template.replace(
            "{{BUNDLE_JSON}}", json.dumps(payload, ensure_ascii=False, indent=2))
        async with sem:
            if pacer is not None:
                await pacer.wait()
            result = await providers.chat_complete(
                client, model, [{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=1600)
        try:
            parsed = extract_json_object(result["text"])
            independent = _parse_independent_answers(parsed, bundle)
        except (ValueError, json.JSONDecodeError, TypeError):
            independent = []
        cached.write_text(json.dumps(
            {"cache_key": key, "state_id": bundle["state_id"], "provider": "teacher-auditor",
             "requested_model": model, "returned_model": result["returned_model"],
             "raw_response": result["raw"], "raw_text": result["text"],
             "independent_answers": independent},
            ensure_ascii=False, indent=2), encoding="utf-8")
        auditor_meta.update({"provider": "teacher-auditor", "requested_model": model,
                             "returned_model": result["returned_model"], "cache_key": key})
    out = dict(bundle)
    out["teacher_audit"] = compare_with_teacher(
        bundle, independent, audit_cfg, auditor_meta)
    return out


async def audit_bundles_async(cfg, bundles: list[dict], raw_dir: Path) -> list[dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.teacher_audit.enabled:
        out = []
        for bundle in bundles:
            copied = dict(bundle)
            copied["teacher_audit"] = {
                "enabled": False, "pass": True, "complete": True,
                "confident_disagreements": 0, "questions": []}
            out.append(copied)
        return out
    template = load_teacher_audit_prompt(cfg.teacher_audit.prompt_version)
    provider = cfg.teacher_audit_provider()
    client = None
    if provider.name != "mock":
        client = providers.make_client(provider, providers.require_api_key(provider))
    try:
        sem = asyncio.Semaphore(max(1, cfg.generation.concurrency))
        pacer = RequestPacer(cfg.teacher_audit.requests_per_minute) if client is not None else None
        return list(await asyncio.gather(*[
            _audit_one(sem, client, cfg.teacher_audit.model, cfg.teacher_audit.temperature,
                       template, bundle, raw_dir, cfg.teacher_audit, pacer,
                       f"{provider.name}@{provider.base_url.rstrip('/')}")
            for bundle in bundles
        ]))
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass


def audit_bundles(cfg, bundles: list[dict], raw_dir: Path) -> list[dict]:
    return asyncio.run(audit_bundles_async(cfg, bundles, raw_dir))
