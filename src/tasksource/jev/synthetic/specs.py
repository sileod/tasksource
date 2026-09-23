"""Deterministic StateSpec sampler.

The sampler creates StateSpec dicts (not prose); the RNG controls all
composition. The LLM only instantiates a spec as a plausible scenario,
so dataset composition stays reproducible despite nondeterministic
realization.
"""

from __future__ import annotations

import random

DOMAINS = ["cybersecurity", "customer_support", "healthcare_ops", "finance",
           "devops", "legal_compliance", "logistics", "education"]
SCENARIO_TYPES = ["incident", "ticket", "request", "alert", "review", "escalation"]
STYLES = ["internal_ticket", "chat_transcript", "monitoring_alert",
          "email_thread", "audit_note", "meeting_notes"]
DIFFICULTIES = [1, 2, 3, 4, 5]
AMBIGUITIES = ["low", "moderate", "high"]
STATE_LENGTHS = ["short", "medium", "long"]
EVIDENCE = ["clear", "conflicting", "sparse"]
SKILLS = ["incident_routing", "needs_escalation", "severity", "triage_priority",
          "policy_violation", "sentiment", "factual_grounding", "action_selection"]

SCORE_RANGES = [(0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (0, 10)]
NOUL_CERTAINTY = ["obvious_yes", "likely_yes", "ambiguous", "likely_no", "obvious_no"]


def _weighted_choice(rng: random.Random, weights: dict) -> str:
    roll = rng.random()
    cumulative = 0.0
    items = list(weights.items())
    for key, weight in items:
        cumulative += float(weight)
        if roll < cumulative:
            return key
    return items[-1][0]


def sample_n_questions(rng: random.Random, dist: dict) -> int:
    return int(_weighted_choice(rng, {k: float(v) for k, v in dist.items()}))


def sample_formats(rng: random.Random, n: int, format_weights: dict,
                   p_mixed: float = 0.85, p_all_if_ge3: float = 0.70) -> list[str]:
    """Sample n formats honoring coverage + mixed-format preferences."""
    formats = list(format_weights.keys())
    weights = [float(format_weights[f]) for f in formats]

    def draw() -> str:
        return rng.choices(formats, weights=weights, k=1)[0]

    if n == 1:
        return [draw()]
    if rng.random() >= p_mixed:
        return [draw() for _ in range(n)]
    # Mixed: at least 2 distinct formats (n>=2).
    sampled = [draw() for _ in range(n)]
    if len(set(sampled)) < 2:
        others = [f for f in formats if f != sampled[0]] or formats
        sampled[rng.randrange(n)] = rng.choice(others)
    if n >= 3 and rng.random() < p_all_if_ge3:
        # Strongly prefer all three formats present.
        for missing in ["choice", "noul", "score"]:
            if missing not in sampled:
                sampled[rng.randrange(n)] = missing
    rng.shuffle(sampled)
    return sampled


def sample_question_spec(rng: random.Random, fmt: str) -> dict:
    if fmt == "choice":
        n_options = rng.choices([2, 3, 4, 5, 6],
                                weights=[0.10, 0.25, 0.30, 0.25, 0.10], k=1)[0]
        return {"format": "choice", "skill": rng.choice(SKILLS), "n_options": n_options}
    if fmt == "noul":
        return {"format": "noul", "skill": rng.choice(SKILLS),
                "certainty": rng.choice(NOUL_CERTAINTY)}
    lo, hi = rng.choice(SCORE_RANGES)
    return {"format": "score", "skill": rng.choice(SKILLS), "min": lo, "max": hi}


def sample_spec(index: int, rng: random.Random, sampler_cfg) -> dict:
    n = sample_n_questions(rng, sampler_cfg.questions_per_state)
    formats = sample_formats(rng, n, sampler_cfg.question_formats,
                             sampler_cfg.probability_mixed_formats,
                             sampler_cfg.probability_all_formats_if_n_ge_3)
    return {
        "state_id": f"state_{index:06d}",
        "domain": rng.choice(DOMAINS),
        "scenario_type": rng.choice(SCENARIO_TYPES),
        "style": rng.choice(STYLES),
        "difficulty": rng.choice(DIFFICULTIES),
        "ambiguity": rng.choice(AMBIGUITIES),
        "state_length": rng.choice(STATE_LENGTHS),
        "evidence": rng.choice(EVIDENCE),
        "distractors": rng.choice([0, 1, 2, 3]),
        "certainty_hint": rng.choice(NOUL_CERTAINTY),
        "questions": [sample_question_spec(rng, fmt) for fmt in formats],
    }


def sample_specs(sampler_cfg, n_states: int | None = None) -> list[dict]:
    """Deterministic sampler: same seed + config -> same specs."""
    rng = random.Random(sampler_cfg.seed)
    total = n_states if n_states is not None else sampler_cfg.n_states
    return [sample_spec(i, rng, sampler_cfg) for i in range(total)]
