"""Optional LLM critic: cross-question coherence checks.

Deterministic validation cannot tell whether all questions genuinely
refer to the same state or test distinct skills; the critic can.
Kept separate from generation so later you can mix e.g. DeepSeek
generation + Luna critic.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from . import providers
from .generate import PROMPTS_DIR, prompt_hash, render_prompt


def load_critic_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.txt").read_text(encoding="utf-8")


def mock_critique(bundle: dict) -> dict:
    texts = [q.get("question", "") for q in bundle.get("questions", [])]
    issues = []
    if len(set(texts)) != len(texts):
        issues.append("questions are paraphrases of each other")
    return {"pass": not issues, "issues": issues, "score": 1.0 if not issues else 0.0}


async def _critique_one(sem, client, model: str, temperature: float,
                        template: str, bundle: dict) -> dict:
    if client is None:
        return mock_critique(bundle)
    payload = {"state_id": bundle["state_id"], "state": bundle["state"],
               "questions": bundle["questions"]}
    prompt = template.replace("{{BUNDLE_JSON}}", json.dumps(payload, ensure_ascii=False, indent=2))
    async with sem:
        result = await providers.chat_complete(
            client, model, [{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=1000)
    try:
        return json.loads(result["text"])
    except json.JSONDecodeError:
        return {"pass": False, "issues": ["unparseable critic response"], "score": 0.0}


async def critique_bundles_async(cfg, bundles: list[dict], raw_dir: Path) -> list[dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.critic.enabled:
        return [{"state_id": b["state_id"], "pass": True, "issues": [], "score": 1.0} for b in bundles]
    template = load_critic_prompt(cfg.critic.prompt_version)
    client = None
    model = cfg.critic.model
    temperature = cfg.critic.temperature
    if cfg.provider.name != "mock":
        from .config import ProviderConfig
        critic_provider = ProviderConfig(name=cfg.provider.name,
                                         api_key_env=cfg.provider.api_key_env,
                                         base_url=cfg.provider.base_url, model=model)
        client = providers.make_client(critic_provider, providers.require_api_key(critic_provider))
    else:
        client = None
    sem = asyncio.Semaphore(max(1, cfg.generation.concurrency))
    out = await asyncio.gather(*[_critique_one(sem, client, model, temperature, template, b)
                                 for b in bundles])
    results = []
    for bundle, verdict in zip(bundles, out):
        verdict = {"state_id": bundle["state_id"], **verdict}
        (raw_dir / f"{bundle['state_id']}.json").write_text(
            json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(verdict)
    return results


def critique_bundles(cfg, bundles: list[dict], raw_dir: Path) -> list[dict]:
    return asyncio.run(critique_bundles_async(cfg, bundles, raw_dir))
