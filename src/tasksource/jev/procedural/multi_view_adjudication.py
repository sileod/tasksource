"""Adjudicate intent, urgency, and workflow impact from cross-view operational signals under explicit precedence rules plus distracting records.

From level 2 signals sit next to the rule thresholds (one auth failure short, an
error rate of 19 vs 20, a 25-hour deadline); levels also add affected features.
From level 3 auth failures must be counted from the login events, and at level 4
the deadline is a due time to compare with the current time.
"""

from ._common import Problem, choice_answer, noul_answer, score_answer, sround

INTENTS = {
    "billing": "Payments, invoices, refunds, or charges.",
    "access": "Authentication, permissions, or account access.",
    "technical": "Product failures, bugs, or integrations.",
    "other": "None of the other options clearly fits.",
}
IMPACT = [  # the options restate the workflow impact rule, so they cannot contradict it
    "Level 0: no affected feature is down or slow.",
    "Level 1: some affected feature is down or slow, but no core feature is down without a documented workaround.",
    "Level 2: a core feature is down and has no documented workaround.",
]
BILLING_FLAGS = ["duplicate_charge", "invoice_mismatch", "refund_missing"]


MAX_AFFECTED = [2, 3, 4, 5, 6]


def _values(level):
    """(below, at-or-above) threshold values for auth failures and error rate, and urgent/non-urgent deadlines."""
    if level < 2:
        return ([0, 1], [2, 3, 4]), ([0, 5, 10], [20, 35, 60]), ([2, 8, 24, None], [None, 48, 72, 168])
    return ([1], [2]), ([15, 18, 19], [20, 21, 22]), ([23, 24, None], [25, 26, 30, None])


FEATURES = ["checkout", "login", "search", "reports", "exports", "notifications", "sync", "billing_page"]


def _impact(workflow):
    """Level 2 when a core feature is down without a documented workaround; level 1 when any
    feature is down or slow; otherwise 0 (restored features no longer count)."""
    status = {entry["feature"]: entry["status"] for entry in workflow["affected_features"]}
    if any(status[f] == "down" and f in workflow["core_features"] and f not in workflow["workarounds"]
           for f in status):
        return 2
    return int(any(s in ("down", "slow") for s in status.values()))


def generate(rng, level=0):
    n_events = sround(4 + 1.2 * level, rng)
    n_distractors = sround(2 + 0.8 * level, rng)
    intent = rng.choice(list(INTENTS))
    urgent = rng.random() < 0.5
    impact = rng.randrange(len(IMPACT))

    (auth_low, auth_high), (rate_low, rate_high), (urgent_deadlines, calm_deadlines) = _values(level)
    billing = {flag: False for flag in BILLING_FLAGS}
    account = {
        "active": rng.choice([True, False]),
        "auth_failures": rng.choice(auth_low),
        "permission_mismatch": False,
        "tier": rng.choice(["free", "team", "enterprise"]),
    }
    telemetry = {
        "integration_failures": rng.choice(auth_low),
        "error_rate_percent": rng.choice(rate_low),
    }

    if intent == "billing":
        billing[rng.choice(BILLING_FLAGS)] = True
        if rng.random() < 0.5:
            account.update(active=True, auth_failures=rng.choice(auth_high))
        if rng.random() < 0.5:
            telemetry["integration_failures"] = rng.choice(auth_high)
    elif intent == "access":
        account["active"] = True
        if rng.random() < 0.5:
            account["auth_failures"] = rng.choice(auth_high)
        else:
            account["permission_mismatch"] = True
        if rng.random() < 0.5:
            telemetry["error_rate_percent"] = rng.choice(rate_high)
    elif intent == "technical":
        account["permission_mismatch"] = False
        account["auth_failures"] = rng.choice(auth_low)
        if rng.random() < 0.5:
            telemetry["integration_failures"] = rng.choice(auth_high)
        else:
            telemetry["error_rate_percent"] = rng.choice(rate_high)

    if urgent:
        deadline_hours = rng.choice(urgent_deadlines)
        executive_escalation = deadline_hours is None or rng.random() < 0.35
    else:
        deadline_hours = rng.choice(calm_deadlines)
        executive_escalation = False

    # the level joins three lists: which features are affected and how, which are core, and
    # which have a workaround; drawn freely, then kept when they give the chosen level
    while True:
        affected = rng.sample(FEATURES, rng.randint(1, MAX_AFFECTED[level]))
        workflow = {
            "affected_features": [{"feature": f, "status": rng.choice(["down", "slow", "restored"])} for f in affected],
            "core_features": sorted(rng.sample(FEATURES, 3)),
            "workarounds": sorted(rng.sample(FEATURES, rng.randint(0, 3))) if level >= 2 else [],
        }
        if _impact(workflow) == impact:
            break

    events = [
        {
            "ts": i + 1,
            "kind": rng.choice(["login", "note", "sync", "payment"]),
            "ok": rng.choice([True, False]),
        }
        for i in range(max(2, n_events))
    ]
    derived = level >= 3
    if derived:  # the failed logins in recent_events are the auth failures
        failures = account.pop("auth_failures")
        events = [e for e in events if e["kind"] != "login"]
        events += [{"ts": 0, "kind": "login", "ok": False} for _ in range(failures)]
        events += [{"ts": 0, "kind": "login", "ok": True} for _ in range(rng.randint(0, 3))]
        rng.shuffle(events)
        for i, e in enumerate(events):
            e["ts"] = i + 1
    timeline = {"deadline_hours": deadline_hours, "executive_escalation": executive_escalation}
    if level >= 4:  # hours left must be computed from two clock times
        now = rng.randint(0, 6 * 24)
        timeline = {"now_hour": now, "due_hour": None if deadline_hours is None else now + deadline_hours,
                    "executive_escalation": executive_escalation}
    distractors = [
        {
            "id": f"D{i+1}",
            "kind": rng.choice(["campaign", "survey", "feature_flag"]),
            "active": rng.choice([True, False]),
        }
        for i in range(max(0, n_distractors))
    ]
    state = {
        "ticket": {"channel": rng.choice(["email", "chat", "api"])},
        "billing": billing,
        "account": account,
        "telemetry": telemetry,
        "timeline": timeline,
        "workflow": workflow,
        "recent_events": events,
        "unrelated_records": distractors,
        "adjudication_rules": {
            "intent": (
                "Choose billing if any billing flag is true. Otherwise choose access when account.active is true and "
                f"either {'the number of failed login events in recent_events' if derived else 'account.auth_failures'} "
                ">= 2 or account.permission_mismatch is true. Otherwise choose technical "
                "when telemetry.integration_failures >= 2 or telemetry.error_rate_percent >= 20. Otherwise choose other."
            ),
            "urgency": (
                "Urgent iff timeline.executive_escalation is true or "
                + ("timeline.due_hour is not null and at most 24 hours after timeline.now_hour." if level >= 4
                   else "timeline.deadline_hours is not null and <= 24.")
            ),
            "workflow_impact": (
                "Level 2 when a core feature is down and has no documented workaround; otherwise level 1 when any "
                "affected feature is down or slow; otherwise level 0. Restored features are no longer affected."
            ),
            "scope": ("unrelated_records are distractors; recent_events matter only through the failed logins the intent "
                      "rule counts." if derived else
                      "recent_events and unrelated_records are distractors and do not override the joined current views."),
        },
    }
    questions = {
        "intent": {
            "type": "choice",
            "instructions": "Following the intent rule, what is the primary operational issue?",
            "criteria": INTENTS,
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Following the urgency rule, is this request urgent?",
        },
        "workflow_impact": {
            "type": "score",
            "instructions": "Following the workflow impact rule, how much does the issue block the user's work?",
            "criteria": IMPACT,
        },
    }
    answers = {
        "intent": choice_answer(intent, list(INTENTS)),
        "is_urgent": noul_answer(urgent),
        "workflow_impact": score_answer(impact, IMPACT),
    }
    return Problem(state, questions, answers)
