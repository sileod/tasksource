"""Adjudicate intent, urgency, and workflow impact from cross-view operational signals under explicit precedence rules plus distracting records."""

from ._common import Problem, choice_answer, noul_answer, score_answer, sround

INTENTS = {
    "billing": "Payments, invoices, refunds, or charges.",
    "access": "Authentication, permissions, or account access.",
    "technical": "Product failures, bugs, or integrations.",
    "other": "None of the other options clearly fits.",
}
IMPACT = [
    "Minor inconvenience; normal work can continue.",
    "Important workflow is degraded but a workaround exists.",
    "Core work is blocked and no workaround is available.",
]
BILLING_FLAGS = ["duplicate_charge", "invoice_mismatch", "refund_missing"]


def generate(rng, level=0):
    n_events = sround(4 + 1.2 * level, rng)
    n_distractors = sround(2 + 0.8 * level, rng)
    intent = rng.choice(list(INTENTS))
    urgent = rng.random() < 0.5
    impact = rng.randrange(len(IMPACT))

    billing = {flag: False for flag in BILLING_FLAGS}
    account = {
        "active": rng.choice([True, False]),
        "auth_failures": rng.choice([0, 1]),
        "permission_mismatch": False,
        "tier": rng.choice(["free", "team", "enterprise"]),
    }
    telemetry = {
        "integration_failures": rng.choice([0, 1]),
        "error_rate_percent": rng.choice([0, 5, 10]),
    }

    if intent == "billing":
        billing[rng.choice(BILLING_FLAGS)] = True
        if rng.random() < 0.5:
            account.update(active=True, auth_failures=rng.choice([2, 3]))
        if rng.random() < 0.5:
            telemetry["integration_failures"] = rng.choice([2, 3])
    elif intent == "access":
        account["active"] = True
        if rng.random() < 0.5:
            account["auth_failures"] = rng.choice([2, 3, 4])
        else:
            account["permission_mismatch"] = True
        if rng.random() < 0.5:
            telemetry["error_rate_percent"] = rng.choice([20, 35, 60])
    elif intent == "technical":
        account["permission_mismatch"] = False
        account["auth_failures"] = rng.choice([0, 1])
        if rng.random() < 0.5:
            telemetry["integration_failures"] = rng.choice([2, 3, 4])
        else:
            telemetry["error_rate_percent"] = rng.choice([20, 35, 60])

    if urgent:
        deadline_hours = rng.choice([2, 8, 24, None])
        executive_escalation = deadline_hours is None or rng.random() < 0.35
    else:
        deadline_hours = rng.choice([None, 48, 72, 168])
        executive_escalation = False

    if impact == 0:
        workflow = {
            "core_blocked": False,
            "degraded": False,
            "workaround_available": rng.choice([True, False]),
        }
    elif impact == 1:
        if rng.random() < 0.5:
            workflow = {
                "core_blocked": False,
                "degraded": True,
                "workaround_available": rng.choice([True, False]),
            }
        else:
            workflow = {
                "core_blocked": True,
                "degraded": True,
                "workaround_available": True,
            }
    else:
        workflow = {
            "core_blocked": True,
            "degraded": True,
            "workaround_available": False,
        }

    events = [
        {
            "ts": i + 1,
            "kind": rng.choice(["login", "note", "sync", "payment"]),
            "ok": rng.choice([True, False]),
        }
        for i in range(max(2, n_events))
    ]
    distractors = [
        {
            "id": f"D{i+1}",
            "kind": rng.choice(["campaign", "survey", "feature_flag"]),
            "active": rng.choice([True, False]),
        }
        for i in range(max(0, n_distractors))
    ]
    state = {
        "ticket": {
            "message": "The user reports an operational issue requiring adjudication from the joined records.",
            "channel": rng.choice(["email", "chat", "api"]),
        },
        "billing": billing,
        "account": account,
        "telemetry": telemetry,
        "timeline": {
            "deadline_hours": deadline_hours,
            "executive_escalation": executive_escalation,
        },
        "workflow": workflow,
        "recent_events": events,
        "unrelated_records": distractors,
        "adjudication_rules": {
            "intent": (
                "Choose billing if any billing flag is true. Otherwise choose access when account.active is true and "
                "either account.auth_failures >= 2 or account.permission_mismatch is true. Otherwise choose technical "
                "when telemetry.integration_failures >= 2 or telemetry.error_rate_percent >= 20. Otherwise choose other."
            ),
            "urgency": (
                "Urgent iff timeline.executive_escalation is true or timeline.deadline_hours is not null and <= 24."
            ),
            "workflow_impact": (
                "Use level 2 when workflow.core_blocked is true and no workaround is available; level 1 when "
                "workflow.degraded is true or core work is blocked but a workaround is available; otherwise level 0."
            ),
            "scope": "recent_events and unrelated_records are distractors and do not override the joined current views.",
        },
    }
    questions = {
        "intent": {
            "type": "choice",
            "instructions": "Under state.adjudication_rules.intent, what is the primary operational issue?",
            "criteria": INTENTS,
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Under state.adjudication_rules.urgency, is this request urgent?",
        },
        "workflow_impact": {
            "type": "score",
            "instructions": "Under state.adjudication_rules.workflow_impact, how much does the issue block the user's work?",
            "criteria": IMPACT,
        },
    }
    answers = {
        "intent": choice_answer(intent, list(INTENTS)),
        "is_urgent": noul_answer(urgent),
        "workflow_impact": score_answer(impact, IMPACT),
    }
    return Problem(state, questions, answers)
