"""Optional LLM critic: cross-question coherence checks.

Deterministic validation cannot tell whether all questions genuinely
refer to the same state or test distinct skills; the critic can.
The critic uses its OWN provider config (see CriticConfig), so e.g.
Luna generation + Albert/DeepSeek critic works correctly.

Critic calls are cached like generation calls:
    hash(critic model + temperature + prompt hash + canonical bundle)
and raw API responses are preserved.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

from . import providers
from .generate import PROMPTS_DIR, extract_json_object, prompt_hash


def load_critic_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.txt").read_text(encoding="utf-8")


def critic_cache_key(model: str, temperature: float, prompt: str, bundle: dict) -> str:
    canonical = json.dumps(
        {"state_id": bundle.get("state_id"), "state": bundle.get("state"),
         "questions": bundle.get("questions")},
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(
        f"{model}\n{temperature}\n{prompt_hash(prompt)}\n{canonical}".encode("utf-8")
    ).hexdigest()


def mock_critique(bundle: dict) -> dict:
    texts = [q.get("question", "") for q in bundle.get("questions", [])]
    issues = []
    if len(set(texts)) != len(texts):
        issues.append("questions are paraphrases of each other")
    return {"pass": not issues, "issues": issues, "score": 1.0 if not issues else 0.0}


class RequestPacer:
    """Space critic calls so a batch stays below the provider's minute limit."""

    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / max(1, requests_per_minute)
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            if now < self.next_start:
                await asyncio.sleep(self.next_start - now)
            self.next_start = time.monotonic() + self.interval


async def _critique_one(sem, client, model: str, temperature: float,
                        template: str, bundle: dict, raw_dir: Path,
                        pacer: RequestPacer | None = None) -> dict:
    key = critic_cache_key(model, temperature, template, bundle)
    cached = raw_dir / f"{key}.json"
    if cached.exists():
        record = json.loads(cached.read_text(encoding="utf-8"))
        return {"state_id": bundle["state_id"], **record["verdict"]}
    if client is None:
        verdict = mock_critique(bundle)
        cached.write_text(json.dumps(
            {"cache_key": key, "state_id": bundle["state_id"],
             "provider": "mock", "model": model, "verdict": verdict},
            ensure_ascii=False, indent=2), encoding="utf-8")
        return {"state_id": bundle["state_id"], **verdict}
    payload = {"state_id": bundle["state_id"], "state": bundle["state"],
               "questions": bundle["questions"]}
    prompt = template.replace("{{BUNDLE_JSON}}", json.dumps(payload, ensure_ascii=False, indent=2))
    async with sem:
        if pacer is not None:
            await pacer.wait()
        result = await providers.chat_complete(
            client, model, [{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=1000)
    try:
        verdict = extract_json_object(result["text"])
        verdict = {"pass": bool(verdict.get("pass", False)),
                   "issues": list(verdict.get("issues", [])),
                   "score": float(verdict.get("score", 0.0))}
    except (ValueError, json.JSONDecodeError, TypeError):
        verdict = {"pass": False, "issues": ["unparseable critic response"], "score": 0.0}
    cached.write_text(json.dumps(
        {"cache_key": key, "state_id": bundle["state_id"],
         "provider": "critic", "requested_model": model,
         "returned_model": result["returned_model"],
         "raw_response": result["raw"], "raw_text": result["text"],
         "verdict": verdict},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return {"state_id": bundle["state_id"], **verdict}


async def critique_bundles_async(cfg, bundles: list[dict], raw_dir: Path) -> list[dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.critic.enabled:
        return [{"state_id": b["state_id"], "pass": True, "issues": [], "score": 1.0} for b in bundles]
    template = load_critic_prompt(cfg.critic.prompt_version)
    provider = cfg.critic_provider()
    client = None
    if provider.name != "mock":
        client = providers.make_client(provider, providers.require_api_key(provider))
    try:
        sem = asyncio.Semaphore(max(1, cfg.generation.concurrency))
        pacer = RequestPacer(cfg.critic.requests_per_minute) if client is not None else None
        out = await asyncio.gather(*[_critique_one(sem, client, cfg.critic.model,
                                                   cfg.critic.temperature, template, b, raw_dir,
                                                   pacer)
                                     for b in bundles])
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass
    return list(out)


def critique_bundles(cfg, bundles: list[dict], raw_dir: Path) -> list[dict]:
    return asyncio.run(critique_bundles_async(cfg, bundles, raw_dir))
