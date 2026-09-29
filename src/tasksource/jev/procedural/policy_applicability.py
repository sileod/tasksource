"""Apply prioritized structured policies to an access request.

Level 0 has two policies with one or two constraints each; level 4 has about a
dozen policies with up to five constraints, many of them higher-priority near
misses that fail on a single constraint (often clearance or sensitivity by one step).
"""

from ._common import Problem, choice_answer, noul_answer, score_answer, sround

ROLES = ["analyst", "engineer", "manager"]
TEAMS = ["alpha", "beta", "gamma"]
ACTIONS = ["read", "write", "approve"]
SENSITIVITY = ["public", "internal", "restricted"]
RISK = [  # the options restate the review rule
    "Level 0: the resource is not restricted and the matching policies agree on effect.",
    "Level 1: the resource is restricted, but the governing policy allows it and the matching policies agree on effect.",
    "Level 2: the governing policy denies a restricted resource, or the matching policies disagree on effect.",
]
FIELDS = ["roles", "teams", "min_clearance", "actions", "max_sensitivity"]
N_POLICIES = [2, 3, 5, 8, 12]
N_CONSTRAINTS = [(1, 1), (1, 2), (2, 4), (3, 5), (4, 5)]
NEAR_MISS = [0.0, 0.3, 0.5, 0.7, 0.8]
SEMANTICS = ("A policy matches when every constraint it lists holds; a constraint it does not list matches anything. "
             "roles, teams, and actions list the allowed values; min_clearance is the lowest clearance allowed; "
             "max_sensitivity is the most sensitive resource allowed, in the order public < internal < restricted. "
             "Apply the matching policy with the highest priority, where a larger priority number wins; deny if none match.")


def _holds(field, value, request):
    subject = request["subject"]
    if field == "roles":
        return subject["role"] in value
    if field == "teams":
        return subject["team"] in value
    if field == "actions":
        return request["action"] in value
    if field == "min_clearance":
        return subject["clearance"] >= value
    return SENSITIVITY.index(request["resource"]["sensitivity"]) <= SENSITIVITY.index(value)


def _matches(policy, request):
    return all(_holds(field, policy[field], request) for field in FIELDS if field in policy)


def _value(rng, field, request, satisfied):
    """A constraint value that holds (or fails) for ``request``; failures stay one step away when they can."""
    subject = request["subject"]
    if field in ("roles", "teams", "actions"):
        pool = {"roles": ROLES, "teams": TEAMS, "actions": ACTIONS}[field]
        own = {"roles": subject["role"], "teams": subject["team"], "actions": request["action"]}[field]
        others = [x for x in pool if x != own]
        chosen = rng.sample(others, rng.randint(0 if satisfied else 1, len(others)))
        return sorted(chosen + [own] if satisfied else chosen, key=pool.index)
    if field == "min_clearance":
        clearance = subject["clearance"]
        return rng.randint(0, clearance) if satisfied else clearance + 1
    rank = SENSITIVITY.index(request["resource"]["sensitivity"])
    return SENSITIVITY[rng.randint(rank, 2) if satisfied else rank - 1]


def _can_fail(field, request):
    if field == "min_clearance":
        return request["subject"]["clearance"] < 2
    if field == "max_sensitivity":
        return request["resource"]["sensitivity"] != "public"
    return True


def _policy(rng, request, fields, fail=None):
    """Constraints on ``fields``, all holding for ``request`` except ``fail`` (then the policy misses it)."""
    return {field: _value(rng, field, request, field != fail) for field in fields}


def generate(rng, level=0):
    n = max(2, sround(N_POLICIES[level] * rng.uniform(0.8, 1.2), rng))
    request = {
        "subject": {"role": rng.choice(ROLES), "team": rng.choice(TEAMS), "clearance": rng.randrange(3)},
        "resource": {"team": rng.choice(TEAMS), "sensitivity": rng.choice(SENSITIVITY)},
        "action": rng.choice(ACTIONS),
    }
    low, high = N_CONSTRAINTS[level]
    policies = []
    for i in range(n):
        fields = sorted(rng.sample(FIELDS, rng.randint(low, high)), key=FIELDS.index)
        failable = [f for f in fields if _can_fail(f, request)]
        if i == n - 1:  # the lowest priority always matches, so some policy governs
            body = _policy(rng, request, fields)
        elif failable and rng.random() < NEAR_MISS[level]:
            body = _policy(rng, request, fields, fail=rng.choice(failable))
        else:
            body = {field: rng.choice([
                _value(rng, field, request, True),
                _value(rng, field, request, False) if _can_fail(field, request) else _value(rng, field, request, True),
            ]) for field in fields}
        policies.append({"id": f"P{i+1}", "priority": n - i, "effect": rng.choice(["allow", "deny"]), **body})
    if level >= 2:  # priority must then be read, not inferred from the listing order
        rng.shuffle(policies)
    matching = [p for p in policies if _matches(p, request)]
    governing = max(matching, key=lambda p: p["priority"])
    allowed = governing["effect"] == "allow"
    conflict = len({p["effect"] for p in matching}) > 1
    restricted = request["resource"]["sensitivity"] == "restricted"
    risk_index = 2 if (not allowed and restricted) or conflict else 1 if restricted else 0
    options = sorted((p["id"] for p in policies), key=lambda x: int(x[1:]))
    state = {
        "request": request,
        "policies": policies,
        "semantics": SEMANTICS,
        "review_rule": "Level 2 when the governing policy denies a restricted resource or the matching policies disagree on effect; otherwise level 1 when the resource is restricted; otherwise level 0.",
    }
    questions = {
        "access_allowed": {"type": "noul", "instructions": "Does the governing policy allow the requested action?"},
        "governing_policy": {"type": "choice", "instructions": "Which policy governs the request, i.e. has the highest priority among the policies whose constraints all hold?", "criteria": {x: x for x in options}},
        "review_risk": {"type": "score", "instructions": "Following the review rule, how risky is this policy outcome for automated execution?", "criteria": RISK},
    }
    answers = {
        "access_allowed": noul_answer(allowed),
        "governing_policy": choice_answer(governing["id"], options),
        "review_risk": score_answer(risk_index, RISK),
    }
    return Problem(state, questions, answers)
