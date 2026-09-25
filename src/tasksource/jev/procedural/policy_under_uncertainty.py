"""Apply prioritized policies to a request whose requester role is uncertain.

The role is known only through past request counts (the prior) and reports of
stated reliability; answers are the exact posterior, pushed through the same
policy rules as policy_applicability: the probability that the request is
allowed and the distribution over the governing policy.
"""

from fractions import Fraction

from ._common import Problem, sround
from .policy_applicability import ACTIONS, ROLES, SENSITIVITY, TEAMS, _matches

RELIABILITY = [Fraction(2, 3), Fraction(3, 4), Fraction(4, 5), Fraction(9, 10)]
SOURCES = ["intake form", "directory lookup", "manager's note", "badge log"]
NONE = "none (default deny)"
PLURAL = {"analyst": "analysts", "engineer": "engineers", "manager": "managers"}


def _times(fraction):
    return f"right {fraction.numerator} times in {fraction.denominator}"


def _posterior(prior, reports, roles):
    """P(role | reports): a report names the true role with its reliability, else another possible role uniformly."""
    weights = {}
    for role in roles:
        weight = prior[role]
        for said, reliability in reports:
            weight *= reliability if said == role else (1 - reliability) / (len(roles) - 1)
        weights[role] = weight
    total = sum(weights.values())
    return {role: weight / total for role, weight in weights.items()}


def _governing(policies, request):
    matching = [p for p in policies if _matches(p, request)]
    return max(matching, key=lambda p: p["priority"]) if matching else None


def _probabilities(distribution, options):
    """Floats for exact fractions, summing to one."""
    values = {option: float(distribution.get(option, 0)) for option in options}
    largest = max(values, key=values.get)
    values[largest] += 1.0 - sum(values.values())
    return values


def _choice(distribution, options):
    probabilities = _probabilities(distribution, options)
    return {"type": "choice", "choice": max(probabilities, key=probabilities.get),
            "probabilities": probabilities, "confidence": max(probabilities.values())}


def _policy(rng, i, n_policies):
    return {
        "id": f"P{i+1}",
        "priority": n_policies - i,
        "effect": rng.choice(["allow", "deny"]),
        "roles": sorted(rng.sample(ROLES, rng.randint(1, len(ROLES))), key=ROLES.index),
        "teams": sorted(rng.sample(TEAMS, rng.randint(1, len(TEAMS))), key=TEAMS.index),
        "min_clearance": rng.randrange(3),
        "actions": sorted(rng.sample(ACTIONS, rng.randint(1, len(ACTIONS))), key=ACTIONS.index),
        "max_sensitivity": rng.randrange(len(SENSITIVITY)),
    }


def _effect(policies, request, subject, role):
    policy = _governing(policies, {**request, "subject": {**subject, "role": role}})
    return policy["effect"] if policy else "deny"


def generate(rng, level=0):
    n_policies = max(2, sround(3 + 0.6 * level, rng))
    n_reports = max(1, sround(1 + 0.5 * level, rng))
    roles = sorted(rng.sample(ROLES, 2 if rng.random() < 0.6 - 0.1 * level else 3), key=ROLES.index)
    history = {role: rng.randint(2, 30) for role in roles}
    prior = {role: Fraction(count, sum(history.values())) for role, count in history.items()}
    true_role = rng.choices(roles, weights=[history[r] for r in roles])[0]
    reports = []
    for source in rng.sample(SOURCES, min(n_reports, len(SOURCES))):
        reliability = rng.choice(RELIABILITY)
        wrong = [r for r in roles if r != true_role]
        said = true_role if rng.random() < reliability else rng.choice(wrong)
        reports.append((source, said, reliability))

    subject = {"role": None, "team": rng.choice(TEAMS), "clearance": rng.randrange(3)}
    request = {"subject": subject, "resource": {"team": rng.choice(TEAMS), "sensitivity": rng.choice(SENSITIVITY)},
               "action": rng.choice(ACTIONS)}
    # the role decides the outcome in most problems; in the rest the evidence about it is irrelevant
    role_matters = rng.random() < 0.7
    while True:
        policies = [_policy(rng, i, n_policies) for i in range(n_policies)]
        effects = {_effect(policies, request, subject, role) for role in roles}
        if (len(effects) > 1) == role_matters:
            break

    posterior = _posterior(prior, [(said, reliability) for _, said, reliability in reports], roles)
    allowed, governing = Fraction(0), {}
    for role, p in posterior.items():
        policy = _governing(policies, {**request, "subject": {**subject, "role": role}})
        key = policy["id"] if policy else NONE
        governing[key] = governing.get(key, 0) + p
        allowed += p if policy and policy["effect"] == "allow" else 0

    counts = ", ".join(f"{history[r]} by {PLURAL[r]}" for r in roles)
    state = {
        "request": {**request, "subject": {**subject, "role": "unknown"}},
        "role_evidence": {
            "history": f"Of the last {sum(history.values())} requests from this account, {counts}. "
                       "No other role has used it.",
            "reports": [{"source": source, "says": said, "reliability": _times(reliability)}
                        for source, said, reliability in reports],
        },
        "policies": policies,
        "semantics": (
            "A policy matches when all listed constraints match. Apply the highest-priority matching policy; "
            "deny if none match. The requester's role is one of the roles in the history, in proportion to "
            "the history counts before the reports are considered. Each report is independent: it names the "
            "true role as often as its reliability says, and otherwise names one of the other possible roles, "
            "each equally likely."
        ),
    }
    options = [p["id"] for p in policies] + [NONE]
    questions = {
        "access_allowed": {"type": "noul", "instructions":
                           "Given the uncertainty about the requester's role, how likely is the request to be allowed?"},
        "governing_policy": {"type": "choice", "instructions": "Which policy governs the request?",
                             "criteria": {x: x for x in options}},
        "requester_role": {"type": "choice", "instructions": "What is the requester's role?",
                           "criteria": {x: x for x in roles}},
    }
    answers = {
        "access_allowed": {"type": "noul", "noul": round(float(allowed), 6)},
        "governing_policy": _choice(governing, options),
        "requester_role": _choice(posterior, roles),
    }
    data = {"prior": prior, "reports": reports, "posterior": posterior, "request": request, "true_role": true_role}
    return Problem(state, questions, answers, data)
