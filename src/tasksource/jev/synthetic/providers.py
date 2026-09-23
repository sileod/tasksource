"""Generic OpenAI-compatible provider + preflight.

Albert is OpenAI-compatible, so ``AsyncOpenAI`` is used throughout;
switching provider/model is configuration only. ``name == "mock"``
selects a deterministic offline generator (pilot/tests, no network).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from .config import ProviderConfig


@dataclass
class PreflightResult:
    provider: str
    model: str
    returned_model: str
    base_url: str


def require_api_key(provider: ProviderConfig) -> str:
    if provider.name == "mock":
        return "mock-key"
    api_key = os.getenv(provider.api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"{provider.api_key_env} is required for the {provider.name} generator."
        )
    return api_key


def make_client(provider: ProviderConfig, api_key: str):
    """Build a generic OpenAI-compatible async client."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The 'openai' package is required for generation") from exc
    kwargs: dict = {"api_key": api_key}
    if provider.base_url:
        kwargs["base_url"] = provider.base_url
    return AsyncOpenAI(**kwargs)


async def preflight(provider: ProviderConfig) -> PreflightResult:
    """Check API key, verify the configured model exists, return metadata.

    Must run before sampling specs or writing artifacts (except the run dir).
    """
    api_key = require_api_key(provider)
    if provider.name == "mock":
        return PreflightResult(provider="mock", model=provider.model,
                               returned_model=provider.model, base_url="mock://")
    client = make_client(provider, api_key)
    try:
        models = await client.models.list()
    finally:
        try:
            await client.close()
        except Exception:
            pass
    available = [m.id for m in models.data]
    if provider.model not in available:
        raise RuntimeError(
            f"Model {provider.model!r} not listed by {provider.name} "
            f"(base_url={provider.base_url}). Available: {available[:20]}"
        )
    returned = provider.model
    return PreflightResult(provider=provider.name, model=provider.model,
                           returned_model=returned, base_url=provider.base_url)


async def chat_complete(client, model: str, messages: list[dict],
                        temperature: float, max_tokens: int,
                        max_retries: int = 8) -> dict:
    """One chat completion with exponential backoff on 429/5xx.

    Returns raw text + returned model name.
    """
    import asyncio as _asyncio

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = response.choices[0]
            text = choice.message.content or ""
            returned_model = getattr(response, "model", model)
            return {"text": text, "returned_model": returned_model,
                    "raw": response.to_dict() if hasattr(response, "to_dict") else {}}
        except Exception as exc:  # noqa: BLE001 — retry on transient API errors
            status = getattr(exc, "status_code", None)
            retryable = status is None or status == 429 or (status is not None and 500 <= status < 600)
            last_error = exc
            if not retryable or attempt == max_retries - 1:
                raise
            await _asyncio.sleep(min(2.0 ** attempt, 60.0))
    raise last_error  # pragma: no cover


def request_hash(config_hash: str, prompt_hash: str, spec: dict) -> str:
    canonical = json.dumps(spec, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(f"{config_hash}\n{prompt_hash}\n{canonical}".encode("utf-8")).hexdigest()
