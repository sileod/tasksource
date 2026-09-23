"""Deterministic StateSpec sampler.

The sampler creates StateSpec dicts (not prose); the RNG controls all
composition. The LLM only instantiates a spec as a plausible scenario,
so dataset composition stays reproducible despite nondeterministic
realization.
"""

from __future__ import annotations

import random

DOMAINS = [
    "cybersecurity", "customer_support", "healthcare_ops", "finance", "devops",
    "legal_compliance", "logistics", "education", "retail", "hospitality",
    "manufacturing", "energy", "telecom", "insurance", "hr_recruiting",
    "procurement", "real_estate", "transportation", "media", "gaming",
    "agriculture", "pharma", "aviation", "maritime", "construction",
    "nonprofit", "government", "research", "marketing", "sales",
    "data_engineering", "ml_platform", "iot", "ecommerce", "food_safety",
    "travel",
]

SCENARIO_TYPES = ["incident", "ticket", "request", "alert", "review",
                  "escalation", "assessment", "appeal"]

STYLES = [
    "internal_ticket", "chat_transcript", "monitoring_alert", "email_thread",
    "audit_note", "meeting_notes", "slack_thread", "incident_postmortem",
    "change_request", "vendor_ticket", "call_transcript", "dashboard_snapshot",
]

DIFFICULTIES = [1, 2, 3, 4, 5]
AMBIGUITY_LEVELS = ["minimal", "low", "moderate", "high", "extreme"]
EVIDENCE_STRUCTURES = [
    "clear", "conflicting", "sparse", "stale", "partial_observation",
    "multi_source", "noisy", "single_witness", "delayed", "redundant",
]
NOUL_CERTAINTY = ["obvious_yes", "likely_yes", "ambiguous", "likely_no", "obvious_no"]

SKILLS = [
    "incident_routing", "needs_escalation", "severity", "triage_priority",
    "policy_violation", "sentiment", "factual_grounding", "action_selection",
    "root_cause", "sla_breach", "fraud_likelihood", "churn_risk",
    "toxicity", "groundedness", "completeness", "urgency",
    "owner_assignment", "refund_approval", "access_justification",
    "data_sensitivity", "compliance_risk", "customer_effort",
    "resolution_confidence", "contradiction",
]

# Skill groups keep the latent task space broad; each domain draws from
# two or three groups (compatibility without a Cartesian product).
SKILL_GROUPS = {
    "ops": ["incident_routing", "needs_escalation", "triage_priority",
            "urgency", "owner_assignment", "action_selection", "root_cause"],
    "trust": ["policy_violation", "factual_grounding", "groundedness",
              "fraud_likelihood", "toxicity", "contradiction",
              "access_justification", "data_sensitivity", "compliance_risk"],
    "service": ["severity", "sentiment", "churn_risk", "customer_effort",
                "resolution_confidence", "completeness", "refund_approval",
                "sla_breach"],
}

DOMAIN_SKILL_GROUPS = {
    "cybersecurity": ["ops", "trust"],
    "customer_support": ["service", "ops"],
    "healthcare_ops": ["ops", "trust"],
    "finance": ["trust", "ops"],
    "devops": ["ops", "service"],
    "legal_compliance": ["trust"],
    "logistics": ["ops", "service"],
    "education": ["service", "trust"],
    "retail": ["service", "trust"],
    "hospitality": ["service", "ops"],
    "manufacturing": ["ops", "trust"],
    "energy": ["ops", "trust"],
    "telecom": ["service", "ops"],
    "insurance": ["trust", "service"],
    "hr_recruiting": ["service", "trust"],
    "procurement": ["trust", "ops"],
    "real_estate": ["service", "trust"],
    "transportation": ["ops", "service"],
    "media": ["service", "trust"],
    "gaming": ["service", "trust"],
    "agriculture": ["ops", "service"],
    "pharma": ["trust", "ops"],
    "aviation": ["ops", "trust"],
    "maritime": ["ops", "trust"],
    "construction": ["ops", "trust"],
    "nonprofit": ["service", "trust"],
    "government": ["trust", "ops"],
    "research": ["trust", "service"],
    "marketing": ["service", "trust"],
    "sales": ["service", "ops"],
    "data_engineering": ["ops", "trust"],
    "ml_platform": ["ops", "trust"],
    "iot": ["ops", "trust"],
    "ecommerce": ["service", "trust"],
    "food_safety": ["trust", "ops"],
    "travel": ["service", "ops"],
}

# Most styles suit every domain; a few are domain-flavored subsets.
DOMAIN_STYLES = {
    "healthcare_ops": ["internal_ticket", "meeting_notes", "call_transcript",
                       "audit_note", "chat_transcript", "email_thread"],
    "aviation": ["monitoring_alert", "incident_postmortem", "internal_ticket",
                 "audit_note", "meeting_notes"],
    "manufacturing": ["monitoring_alert", "dashboard_snapshot", "internal_ticket",
                      "incident_postmortem", "change_request"],
    "devops": ["monitoring_alert", "slack_thread", "incident_postmortem",
               "change_request", "dashboard_snapshot"],
    "ml_platform": ["monitoring_alert", "slack_thread", "dashboard_snapshot",
                    "incident_postmortem", "change_request"],
    "data_engineering": ["monitoring_alert", "slack_thread", "dashboard_snapshot",
                         "internal_ticket", "change_request"],
    "government": ["internal_ticket", "audit_note", "email_thread",
                   "meeting_notes", "vendor_ticket"],
    "legal_compliance": ["audit_note", "email_thread", "internal_ticket",
                         "meeting_notes", "vendor_ticket"],
}

# Numeric scales (the Decisions API accepts at most 10 score levels).
SCORE_RANGES = [(0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (1, 10)]

# Named ordered rubrics (3-7 levels); numeric scales cover the rest.
SEMANTIC_RUBRICS = [
    ["negligible", "low", "moderate", "high", "severe"],
    ["very negative", "negative", "neutral", "positive", "very positive"],
    ["none", "partial", "mostly", "fully"],
    ["strongly oppose", "oppose", "neutral", "support", "strongly support"],
    ["trivial", "minor", "major", "critical"],
    ["very unlikely", "unlikely", "possible", "likely", "very likely"],
    ["absent", "weak", "moderate", "strong"],
    ["unacceptable", "poor", "adequate", "good", "excellent"],
]


def domain_skills(domain: str) -> list[str]:
    """Skill pool compatible with a domain (falls back to all skills)."""
    groups = DOMAIN_SKILL_GROUPS.get(domain, [])
    pool = [skill for group in groups for skill in SKILL_GROUPS.get(group, [])]
    return pool or list(SKILLS)


def domain_styles(domain: str) -> list[str]:
    return DOMAIN_STYLES.get(domain, STYLES)


def numeric_criteria(lo: int, hi: int) -> list[str]:
    return [str(i) for i in range(lo, hi + 1)]


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
    if n >= 3 and rng.random() < p_all_if_ge3:
        # Construct all-three coverage directly: replacing a random
        # position could evict the only instance of another format.
        sampled = ["choice", "noul", "score"] + [draw() for _ in range(n - 3)]
        rng.shuffle(sampled)
        return sampled
    if rng.random() >= p_mixed:
        return [draw() for _ in range(n)]
    # Mixed: at least 2 distinct formats (n>=2).
    sampled = [draw() for _ in range(n)]
    if len(set(sampled)) < 2:
        others = [f for f in formats if f != sampled[0]] or formats
        sampled[rng.randrange(n)] = rng.choice(others)
    rng.shuffle(sampled)
    return sampled


def sample_question_spec(rng: random.Random, fmt: str, skill: str) -> dict:
    if fmt == "choice":
        n_options = rng.choices(
            [2, 3, 4, 5, 6, 7, 8],
            weights=[0.08, 0.22, 0.28, 0.22, 0.12, 0.05, 0.03], k=1)[0]
        return {"format": "choice", "skill": skill, "n_options": n_options}
    if fmt == "noul":
        return {"format": "noul", "skill": skill,
                "certainty": rng.choice(NOUL_CERTAINTY)}
    # Score: Tasksource represents these as ORDERED CRITERIA with the
    # target aligned to them — a healthy mix of numeric scales and
    # named rubrics.
    if rng.random() < 0.5:
        lo, hi = rng.choice(SCORE_RANGES)
        return {"format": "score", "skill": skill, "min": lo, "max": hi,
                "criteria": numeric_criteria(lo, hi)}
    rubric = list(rng.choice(SEMANTIC_RUBRICS))
    return {"format": "score", "skill": skill, "min": 0, "max": len(rubric) - 1,
            "criteria": rubric}


def sample_spec(index: int, rng: random.Random, sampler_cfg) -> dict:
    domain = rng.choice(DOMAINS)
    n = sample_n_questions(rng, sampler_cfg.questions_per_state)
    formats = sample_formats(rng, n, sampler_cfg.question_formats,
                             sampler_cfg.probability_mixed_formats,
                             sampler_cfg.probability_all_formats_if_n_ge_3)
    # Distinct skills per state so questions test distinct aspects.
    pool = domain_skills(domain)
    skills = (rng.sample(pool, n) if n <= len(pool)
              else [rng.choice(pool) for _ in range(n)])
    return {
        "state_id": f"state_{index:06d}",
        "domain": domain,
        "scenario_type": rng.choice(SCENARIO_TYPES),
        "style": rng.choice(domain_styles(domain)),
        "difficulty": rng.choice(DIFFICULTIES),
        "ambiguity": rng.choice(AMBIGUITY_LEVELS),
        "state_length": rng.choice(["short", "medium", "long"]),
        "evidence": rng.choice(EVIDENCE_STRUCTURES),
        "distractors": rng.choice([0, 1, 2, 3]),
        "certainty_hint": rng.choice(NOUL_CERTAINTY),
        "questions": [sample_question_spec(rng, fmt, skill)
                      for fmt, skill in zip(formats, skills)],
    }


def sample_specs(sampler_cfg, n_states: int | None = None) -> list[dict]:
    """Deterministic sampler: same seed + config -> same specs."""
    rng = random.Random(sampler_cfg.seed)
    total = n_states if n_states is not None else sampler_cfg.n_states
    return [sample_spec(i, rng, sampler_cfg) for i in range(total)]
