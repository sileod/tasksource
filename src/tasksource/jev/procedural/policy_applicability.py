"""Apply prioritized structured policies to an access request."""

from ._common import Problem, choice_answer, noul_answer, score_answer, sround

ROLES = ["analyst", "engineer", "manager"]
TEAMS = ["alpha", "beta", "gamma"]
ACTIONS = ["read", "write", "approve"]
SENSITIVITY = ["public", "internal", "restricted"]
RISK = ["Routine policy outcome.", "Sensitive or exceptional outcome requiring review.", "Denied high-risk request or policy conflict."]


def _matches(policy, request):
    return (
        request["subject"]["role"] in policy["roles"]
        and request["subject"]["team"] in policy["teams"]
        and request["subject"]["clearance"] >= policy["min_clearance"]
        and request["action"] in policy["actions"]
        and SENSITIVITY.index(request["resource"]["sensitivity"]) <= policy["max_sensitivity"]
    )


def generate(rng, level=0):
    n = max(2, sround(4 + 0.8 * level, rng))
    request = {
        "subject": {"role": rng.choice(ROLES), "team": rng.choice(TEAMS), "clearance": rng.randrange(3)},
        "resource": {"team": rng.choice(TEAMS), "sensitivity": rng.choice(SENSITIVITY)},
        "action": rng.choice(ACTIONS),
    }
    policies = []
    for i in range(n):
        policies.append({
            "id": f"P{i+1}",
            "priority": n - i,
            "effect": rng.choice(["allow", "deny"]),
            "roles": rng.sample(ROLES, rng.randint(1, len(ROLES))),
            "teams": rng.sample(TEAMS, rng.randint(1, len(TEAMS))),
            "min_clearance": rng.randrange(3),
            "actions": rng.sample(ACTIONS, rng.randint(1, len(ACTIONS))),
            "max_sensitivity": rng.randrange(len(SENSITIVITY)),
        })
    policies[-1].update(
        roles=[request["subject"]["role"]],
        teams=[request["subject"]["team"]],
        min_clearance=request["subject"]["clearance"],
        actions=[request["action"]],
        max_sensitivity=SENSITIVITY.index(request["resource"]["sensitivity"]),
    )
    matching = [p for p in policies if _matches(p, request)]
    governing = max(matching, key=lambda p: p["priority"])
    allowed = governing["effect"] == "allow"
    conflict = len({p["effect"] for p in matching}) > 1
    restricted = request["resource"]["sensitivity"] == "restricted"
    risk_index = 2 if (not allowed and restricted) or conflict else 1 if restricted else 0
    options = [p["id"] for p in policies]
    state = {
        "request": request,
        "policies": policies,
        "semantics": "A policy matches when all listed constraints match. Apply the highest-priority matching policy; default deny only if none match.",
        "review_rule": "Level 2 when the governing policy denies a restricted resource or the matching policies disagree on effect; otherwise level 1 when the resource is restricted; otherwise level 0.",
    }
    questions = {
        "access_allowed": {"type": "noul", "instructions": "Does the governing policy allow the requested action?"},
        "governing_policy": {"type": "choice", "instructions": "Which matching policy has the highest priority?", "criteria": {x: x for x in options}},
        "review_risk": {"type": "score", "instructions": "Following the review rule, how risky is this policy outcome for automated execution?", "criteria": RISK},
    }
    answers = {
        "access_allowed": noul_answer(allowed),
        "governing_policy": choice_answer(governing["id"], options),
        "review_risk": score_answer(risk_index, RISK),
    }
    return Problem(state, questions, answers)
