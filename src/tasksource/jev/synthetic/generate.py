"""LLM realization: StateSpec -> state bundle (state + 1..N questions).

One async request per spec (no batch API coupling). Responses are cached
by request_hash = sha256(config_hash + prompt_hash + canonical_json(spec))
so reruns reuse existing responses instead of regenerating.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from . import providers, specs as spec_module
from .schemas import make_question_id

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(version: str) -> str:
    path = PROMPTS_DIR / f"{version}.txt"
    return path.read_text(encoding="utf-8")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def render_prompt(prompt_template: str, spec: dict) -> str:
    return prompt_template.replace("{{SPEC_JSON}}", json.dumps(spec, indent=2, ensure_ascii=False))


def mock_realization(spec: dict) -> dict:
    """Deterministic offline realization (no network) for pilot/tests."""
    import random
    rng = random.Random(int(hashlib.sha256(spec["state_id"].encode()).hexdigest()[:8], 16))
    topic = f"{spec['domain']} {spec['scenario_type']}"
    filler = ["Ticket opened after monitoring fired.",
              "Reporter notes conflicting signals in the logs.",
              "Unrelated background chatter included for realism."][: 1 + spec.get("distractors", 0)]
    state = (f"[{spec['style']}] {topic} ({spec['state_id']}): "
             + " ".join(filler))
    questions = []
    for i, qspec in enumerate(spec["questions"]):
        qid = make_question_id(spec["state_id"], i)
        if qspec["format"] == "choice":
            n = qspec.get("n_options", 4)
            options = [f"Option {chr(65 + j)} for {qspec['skill']} ({spec['state_id']})" for j in range(n)]
            questions.append({"question_id": qid, "format": "choice",
                              "question": f"What is the best {qspec['skill']} decision for this {topic}?",
                              "options": options, "skill": qspec["skill"]})
        elif qspec["format"] == "noul":
            questions.append({"question_id": qid, "format": "noul",
                              "question": f"Does this {topic} need {qspec['skill']}?",
                              "skill": qspec["skill"]})
        else:
            questions.append({"question_id": qid, "format": "score",
                              "question": f"Rate {qspec['skill']} for this {topic}.",
                              "min": qspec.get("min", 0), "max": qspec.get("max", 5),
                              "skill": qspec["skill"]})
    bundle = dict(spec)
    bundle["state"] = state
    bundle["questions"] = questions
    return bundle


def extract_json_object(text: str) -> dict:
    """Extract the first complete JSON object (tolerates fences/prose)."""
    cleaned = text.strip()
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3]
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("no JSON object found in model response")
    depth, in_string, escaped = 0, False, False
    for pos in range(start, len(cleaned)):
        char = cleaned[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start: pos + 1])
    raise ValueError("truncated JSON object in model response")


def parse_bundle(text: str, spec: dict) -> dict:
    """Parse model JSON; expected keys: state, questions[{question, options?, min?, max?}]."""
    data = extract_json_object(text)
    questions = []
    for i, (qspec, q) in enumerate(zip(spec["questions"], data.get("questions", []))):
        entry: dict = {"question_id": make_question_id(spec["state_id"], i),
                       "format": qspec["format"],
                       "question": q.get("question", ""),
                       "skill": qspec.get("skill", "")}
        if qspec["format"] == "choice":
            entry["options"] = list(q.get("options", []))
        if qspec["format"] == "score":
            entry["min"] = q.get("min", qspec.get("min", 0))
            entry["max"] = q.get("max", qspec.get("max", 5))
        questions.append(entry)
    bundle = dict(spec)
    bundle["state"] = data.get("state", "")
    bundle["questions"] = questions
    return bundle


async def _generate_one(sem: asyncio.Semaphore, client, cfg, prompt_template: str,
                        p_hash: str, c_hash: str, spec: dict, raw_dir: Path) -> dict:
    req_hash = providers.request_hash(c_hash, p_hash, spec)
    cached = raw_dir / f"{req_hash}.json"
    if cached.exists():
        record = json.loads(cached.read_text(encoding="utf-8"))
        return {"spec": spec, "bundle": record["bundle"], "request_hash": req_hash, "cached": True,
                "record": record}
    prompt = render_prompt(prompt_template, spec)
    if cfg.provider.name == "mock":
        bundle = mock_realization(spec)
        record = {"request_hash": req_hash, "spec": spec, "prompt": prompt,
                  "provider": "mock", "requested_model": cfg.provider.model,
                  "returned_model": cfg.provider.model,
                  "temperature": cfg.generation.temperature,
                  "prompt_version": cfg.generation.prompt_version, "bundle": bundle}
        cached.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"spec": spec, "bundle": bundle, "request_hash": req_hash, "cached": False, "record": record}
    async with sem:
        last_error: Exception | None = None
        for _ in range(3):  # retry malformed generations, not just transport errors
            result = await providers.chat_complete(
                client, cfg.provider.model,
                [{"role": "user", "content": prompt}],
                temperature=cfg.generation.temperature,
                max_tokens=cfg.generation.max_output_tokens)
            try:
                bundle = parse_bundle(result["text"], spec)
                break
            except (ValueError, KeyError) as exc:
                last_error = exc
        else:
            error_path = raw_dir / f"{req_hash}.error.json"
            error_path.write_text(json.dumps(
                {"request_hash": req_hash, "state_id": spec["state_id"],
                 "error": str(last_error), "raw_text": result["text"][-2000:]},
                ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(f"unparseable generation for {spec['state_id']}: {last_error}")
    record = {"request_hash": req_hash, "spec": spec, "prompt": prompt,
              "raw_response": result["raw"], "raw_text": result["text"],
              "provider": cfg.provider.name, "requested_model": cfg.provider.model,
              "returned_model": result["returned_model"],
              "temperature": cfg.generation.temperature,
              "prompt_version": cfg.generation.prompt_version, "bundle": bundle}
    cached.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"spec": spec, "bundle": bundle, "request_hash": req_hash, "cached": False, "record": record}


async def _safe_generate_one(sem, client, cfg, prompt_template, p_hash, c_hash,
                             spec, raw_dir) -> dict:
    """Fault isolation: one bad spec must not kill a 1k run."""
    try:
        return await _generate_one(sem, client, cfg, prompt_template, p_hash,
                                   c_hash, spec, raw_dir)
    except Exception as exc:  # noqa: BLE001
        return {"spec": spec, "bundle": None, "request_hash": None,
                "cached": False, "record": None, "error": str(exc)}


async def generate_bundles_async(cfg, specs: list[dict], raw_dir: Path) -> list[dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_template = load_prompt(cfg.generation.prompt_version)
    p_hash = prompt_hash(prompt_template)
    c_hash = hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:16]
    client = None
    if cfg.provider.name != "mock":
        api_key = providers.require_api_key(cfg.provider)
        client = providers.make_client(cfg.provider, api_key)
    sem = asyncio.Semaphore(max(1, cfg.generation.concurrency))
    tasks = [_safe_generate_one(sem, client, cfg, prompt_template, p_hash, c_hash, spec, raw_dir)
             for spec in specs]
    results = []
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
    errors = [r for r in results if r.get("bundle") is None]
    if errors:
        (raw_dir / "errors.jsonl").write_text(
            "\n".join(json.dumps({"state_id": r["spec"].get("state_id"),
                                          "error": r.get("error")}) for r in errors),
            encoding="utf-8")
        print(f"[jev-synthetic] generate: {len(errors)} failed, {len(results) - len(errors)} ok "
              f"(see raw/generation/errors.jsonl)", flush=True)
    results = [r for r in results if r.get("bundle") is not None]
    # Restore deterministic spec order.
    order = {spec["state_id"]: i for i, spec in enumerate(specs)}
    results.sort(key=lambda r: order[r["spec"]["state_id"]])
    return results


def generate_bundles(cfg, specs: list[dict], raw_dir: Path) -> list[dict]:
    return asyncio.run(generate_bundles_async(cfg, specs, raw_dir))
