"""Typed configuration for the synthetic Jev pipeline (config.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import yaml


@dataclass
class ProviderConfig:
    name: str = "albert"
    api_key_env: str = "ALBERT_API_KEY"
    base_url: str = "https://albert.api.etalab.gouv.fr/v1"
    model: str = "DeepSeek-V4-Flash"


@dataclass
class GenerationConfig:
    temperature: float = 0.8
    concurrency: int = 20
    max_output_tokens: int = 4000
    seed: int = 42
    prompt_version: str = "generate_v1"
    n_states: int = 1000


@dataclass
class SamplerConfig:
    seed: int = 42
    n_states: int = 1000
    questions_per_state: dict = field(default_factory=lambda: {"1": 0.60, "2": 0.25, "3": 0.10, "4": 0.05})
    question_formats: dict = field(default_factory=lambda: {"choice": 0.40, "noul": 0.30, "score": 0.30})
    probability_mixed_formats: float = 0.85
    probability_all_formats_if_n_ge_3: float = 0.70


@dataclass
class CriticConfig:
    enabled: bool = True
    # Full provider config, independent of the generator provider, so
    # e.g. Luna generation + Albert/DeepSeek critic works correctly.
    # When `provider` is absent, it inherits the generator provider.
    provider: ProviderConfig | None = None
    model: str = "deepseek-v4-flash-0731"
    prompt_version: str = "critic_v1"
    temperature: float = 0.0
    requests_per_minute: int = 40


@dataclass
class AnnotatorConfig:
    # Explicit: "mock" = deterministic heuristic placeholder (offline
    # pilot/tests only; labeled as such in outputs), "jev" = real Jev
    # endpoint. "jev" FAILS LOUDLY when the client is unavailable —
    # heuristic output must never masquerade as Jev judgments.
    name: str = "mock"
    version: str = "mock-0.1"
    base_url: str = ""
    # Path appended to base_url for the decisions call, e.g.
    # OpenRouter: base_url=https://openrouter.ai/api + api_path=/api/alpha/decisions
    api_path: str = "/api/alpha/decisions"
    api_key_env: str = "JEV_API_KEY"
    model: str = "jev-mock"


@dataclass
class SelectionConfig:
    # Buckets over observed Jev ambiguity (max_prob based, see select.py).
    buckets: dict = field(default_factory=lambda: {
        "very_confident": 0.20,
        "confident": 0.30,
        "moderately_ambiguous": 0.30,
        "high_ambiguity": 0.15,
        "near_uniform": 0.05,
    })
    max_per_family: int = 0  # 0 = no cap


@dataclass
class SplitConfig:
    train: float = 0.85
    validation: float = 0.075
    test: float = 0.075
    ood_fraction: float = 0.05
    seed_salt: str = "jev-synthetic-split-v1"

    def __post_init__(self):
        if not math.isclose(self.train + self.validation + self.test, 1, abs_tol=1e-6):
            raise ValueError(f"split fractions must sum to 1: {self.train}/{self.validation}/{self.test}")


@dataclass
class AppConfig:
    run_name: str = "deepseek_v4_flash_v1"
    output_dir: str = ".synthetic_runs"
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    annotator: AnnotatorConfig = field(default_factory=AnnotatorConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    def critic_provider(self) -> ProviderConfig:
        """Resolved critic provider (explicit, else the generator provider)."""
        return self.critic.provider if self.critic.provider is not None else self.provider

    def to_dict(self) -> dict:
        critic = dict(self.critic.__dict__)
        critic["provider"] = (self.critic.provider.__dict__
                              if self.critic.provider is not None else None)
        return {
            "run_name": self.run_name,
            "output_dir": self.output_dir,
            "provider": self.provider.__dict__,
            "generation": self.generation.__dict__,
            "sampler": self.sampler.__dict__,
            "critic": critic,
            "annotator": self.annotator.__dict__,
            "selection": self.selection.__dict__,
            "split": self.split.__dict__,
        }


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str) -> AppConfig:
    """Load a config.yaml into a typed AppConfig."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    defaults = AppConfig().to_dict()
    merged = _merge(defaults, raw)
    provider = ProviderConfig(**merged["provider"])
    critic_raw = merged["critic"]
    critic_provider_raw = critic_raw.pop("provider", None)
    if isinstance(critic_provider_raw, dict):
        critic_provider = ProviderConfig(**critic_provider_raw)
    elif isinstance(critic_provider_raw, str):
        # Legacy: bare name inherits the generator's connection details.
        critic_provider = ProviderConfig(
            name=critic_provider_raw, api_key_env=provider.api_key_env,
            base_url=provider.base_url, model=critic_raw.get("model", provider.model))
    else:
        critic_provider = None  # inherit the generator provider at use site
    return AppConfig(
        run_name=merged["run_name"],
        output_dir=merged["output_dir"],
        provider=provider,
        generation=GenerationConfig(**merged["generation"]),
        sampler=SamplerConfig(**merged["sampler"]),
        critic=CriticConfig(provider=critic_provider, **critic_raw),
        annotator=AnnotatorConfig(**merged["annotator"]),
        selection=SelectionConfig(**merged["selection"]),
        split=SplitConfig(**merged["split"]),
    )
