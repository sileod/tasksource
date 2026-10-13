"""Adjudicate natural-language records against compositional requirements.

Difficulty grows records and independent criteria, and adds nested disjunctions,
conditionals, cross-field budgets and conflicting or missing information.
Published states contain only prose, not oracle facts or expression trees.
"""

from ._common import Problem, choice_answer, noul_answer, score_answer

N_RECORDS = [4, 7, 11, 17, 25]
N_CONSTRAINTS = [1, 2, 3, 4, 6]
LABELS = ["eligible", "ineligible", "undetermined"]
COUNTS = [str(i) for i in range(9)] + ["9 or more"]
DOMAINS = ["service proposals", "workshop offers", "supplier listings", "venue packages", "project bids"]
REGIONS = ["north", "south", "east", "west"]
SEMANTICS = (
    "Assess each numbered requirement separately using only the descriptions provided. "
    "Explicitly absent information and conflicting claims count as unknown, not false. "
    "A comparison with an unknown operand is unknown. AND is false if any argument is false, "
    "true if all are true, otherwise unknown. OR is true if any argument is true, "
    "false if all are false, otherwise unknown. NOT reverses true and false and preserves unknown. "
    "'If A, then B' means (NOT A) OR B. A record is eligible if every requirement is true, "
    "ineligible if at least one is false, and undetermined otherwise. "
    "For a decisive requirement, take the lowest numbered false requirement, or if none is false, "
    "the lowest numbered unknown requirement; use 'none' if every requirement is true. "
    "Removing a requirement leaves every other requirement in force. Day numbers and limits are inclusive."
)


def _atom(op, field, value):
    return {"op": op, "field": field, "value": value}


def _requirement(rng, kind):
    """One predicate and its corresponding user-facing verbalization."""
    if kind == "fee":
        bound = rng.choice([120, 180, 250, 330, 420])
        return {"kind": kind, "limit": bound, "expr": _atom("le", "fee", bound),
                "text": rng.choice([f"The quoted base price must not exceed €{bound}.",
                                    f"Only base prices of €{bound} or less are acceptable."])}
    if kind == "total":
        bound = rng.choice([180, 250, 360, 480])
        return {"kind": kind, "limit": bound,
                "expr": {"op": "sum_le", "fields": ["fee", "extras"], "value": bound},
                "text": f"The combined base price and additional charges must be at most €{bound}."}
    if kind == "lead":
        bound = rng.choice([2, 4, 7, 10])
        return {"kind": kind, "limit": bound, "expr": _atom("le", "lead_days", bound),
                "text": f"Availability must begin within {bound} days."}
    if kind == "capacity":
        bound = rng.choice([10, 20, 35, 50])
        return {"kind": kind, "limit": bound, "expr": _atom("ge", "capacity", bound),
                "text": f"Capacity must accommodate at least {bound} people."}
    if kind == "region":
        allowed = sorted(rng.sample(REGIONS, rng.randint(1, 2)), key=REGIONS.index)
        return {"kind": kind, "allowed": allowed, "expr": {"op": "in", "field": "region", "values": allowed},
                "text": "The location must be in " + " or ".join(f"the {r} region" for r in allowed) + "."}
    if kind == "membership":
        return {"kind": kind, "expr": {"op": "not", "arg": _atom("eq", "membership", True)},
                "text": "Exclude offers that require annual membership."}
    if kind == "certified":
        return {"kind": kind, "expr": _atom("eq", "certified", True),
                "text": "The provider must hold a current certification."}
    if kind == "deposit":
        return {"kind": kind, "expr": _atom("eq", "deposit", False),
                "text": "No mandatory deposit is permitted."}
    if kind == "rating":
        bound = rng.choice([3, 4])
        return {"kind": kind, "limit": bound, "expr": _atom("ge", "rating", bound),
                "text": f"The published rating must be at least {bound} out of 5."}
    if kind == "start":
        bound = rng.choice([8, 12, 17, 23])
        return {"kind": kind, "limit": bound, "expr": _atom("le", "start_day", bound),
                "text": f"The start date must be on or before day {bound} of the planning period."}
    if kind == "mode":
        onsite = _atom("eq", "mode", "in person")
        remote = {"op": "and", "args": [_atom("eq", "mode", "remote"), _atom("eq", "onboarding", True)]}
        return {"kind": kind, "expr": {"op": "or", "args": [onsite, remote]},
                "text": "Either in-person delivery, or remote delivery with onboarding included, is required."}
    if kind == "conditional":
        bound = rng.choice([4, 8, 12, 24])
        return {"kind": kind, "limit": bound,
                "expr": {"op": "implies", "if": _atom("eq", "expedited", True),
                         "then": _atom("le", "response_hours", bound)},
                "text": f"If service is expedited, the response time must be no more than {bound} hours."}
    if kind == "support":
        desk = _atom("eq", "desk_support", True)
        chat = {"op": "and", "args": [_atom("eq", "chat_support", True),
                                    _atom("eq", "extended_hours", True)]}
        return {"kind": kind, "expr": {"op": "or", "args": [desk, chat]},
                "text": "Support must include either a dedicated help desk, or live chat with extended hours."}
    raise ValueError(kind)


def _facts(rng, req, outcome, level):
    """Give one clause a known true, known false, or indeterminate witness."""
    kind = req["kind"]
    limit = req.get("limit")
    if kind in ("fee", "lead", "start"):
        value = max(0, limit - rng.choice([0, 1, 2])) if outcome == "pass" else limit + rng.choice([1, 2])
        return {{"fee": "fee", "lead": "lead_days", "start": "start_day"}[kind]:
                None if outcome == "unknown" else value}
    if kind in ("capacity", "rating"):
        value = limit + rng.choice([0, 1]) if outcome == "pass" else limit - 1
        return {{"capacity": "capacity", "rating": "rating"}[kind]:
                None if outcome == "unknown" else value}
    if kind == "total":
        extra = rng.choice([10, 20, 30, 40])
        difference = -rng.choice([0, 1, 2]) if outcome == "pass" else rng.choice([1, 2])
        return {"fee": limit - extra + difference, "extras": None if outcome == "unknown" else extra}
    if kind == "region":
        options = req["allowed"] if outcome == "pass" else [x for x in REGIONS if x not in req["allowed"]]
        return {"region": None if outcome == "unknown" else rng.choice(options)}
    if kind in ("membership", "deposit"):
        return {kind: None if outcome == "unknown" else outcome == "fail"}
    if kind == "certified":
        return {"certified": None if outcome == "unknown" else outcome == "pass"}
    if kind == "mode":
        if outcome == "pass":
            if rng.random() < 0.5:
                return {"mode": "in person", "onboarding": rng.choice([True, False, None])}
            return {"mode": "remote", "onboarding": True}
        return {"mode": "remote", "onboarding": False if outcome == "fail" else None}
    if kind == "conditional":
        if outcome == "pass" and rng.random() < 0.3:
            return {"expedited": False, "response_hours": limit + 1}
        if outcome == "pass" and level >= 4 and rng.random() < 0.2:
            return {"expedited": None, "response_hours": limit}
        return {"expedited": True, "response_hours":
                None if outcome == "unknown" else (limit if outcome == "pass" else limit + 1)}
    if kind == "support":
        if outcome == "pass":
            return ({"desk_support": True, "chat_support": False, "extended_hours": False}
                    if rng.random() < 0.5 else
                    {"desk_support": False, "chat_support": True, "extended_hours": True})
        return {"desk_support": False, "chat_support": True,
                "extended_hours": False if outcome == "fail" else None}
    raise ValueError(kind)


DESCRIPTIONS = {
    "fee": ("The base price is €{v}.", "A base quote of €{v} is stated."),
    "extras": ("Additional charges are €{v}.", "There is an extra fee of €{v}."),
    "lead_days": ("Availability begins in {v} {unit}.", "The waiting time is {v} {unit}."),
    "capacity": ("Capacity is {v} people.", "It accommodates {v} participants."),
    "region": ("The location is in the {v} region.", "It operates in the {v} region."),
    "membership": ("Annual membership is required.", "An annual subscription is mandatory."),
    "certified": ("The provider holds a current certification.", "Certification is currently valid."),
    "deposit": ("A deposit is mandatory.", "A deposit must be paid."),
    "rating": ("The published rating is {v}/5.", "Reviews give it {v} out of 5."),
    "start_day": ("The start date is day {v} of the planning period.", "It starts on day {v} of the planning period."),
    "mode": ("Delivery is {v}.", "The offering uses {v} delivery."),
    "onboarding": ("Onboarding is included.", "Setup assistance is part of the offer."),
    "expedited": ("The service is expedited.", "Priority handling is enabled."),
    "response_hours": ("Response time is {v} hours.", "A response takes {v} hours."),
    "desk_support": ("A dedicated help desk is available.", "Dedicated desk support is included."),
    "chat_support": ("Live chat support is available.", "The provider offers live chat."),
    "extended_hours": ("Support operates extended hours.", "Extended support hours are included."),
}
NEGATIVE = {
    "membership": "Annual membership is not required.",
    "certified": "The provider does not hold a current certification.",
    "deposit": "No deposit is required.",
    "onboarding": "Onboarding is not included.",
    "expedited": "Service uses standard rather than expedited handling.",
    "desk_support": "There is no dedicated help desk.",
    "chat_support": "Live chat support is not offered.",
    "extended_hours": "Support does not operate extended hours.",
}
NAMES = {
    "fee": "the base price", "extras": "additional charges", "lead_days": "the waiting time",
    "capacity": "capacity", "region": "the location", "membership": "membership requirements",
    "certified": "certification", "deposit": "the deposit policy", "rating": "the rating",
    "start_day": "the starting day", "mode": "delivery mode", "onboarding": "onboarding",
    "expedited": "priority handling", "response_hours": "response time",
    "desk_support": "desk support", "chat_support": "live chat support",
    "extended_hours": "support hours",
}


def _describe(rng, field, value, level):
    if value is None:
        if level >= 4 and rng.random() < 0.35:
            if field in NEGATIVE:
                positive = DESCRIPTIONS[field][0].format(v=True)
                negative = NEGATIVE[field]
            elif field == "region":
                a, b = rng.sample(REGIONS, 2)
                positive, negative = f"{a} region", f"{b} region"
            elif field == "mode":
                positive, negative = "remote", "in person"
            elif field in ("fee", "extras"):
                positive, negative = "€180", "€310"
            elif field in ("lead_days", "start_day"):
                positive, negative = "day 2", "day 9"
            elif field == "capacity":
                positive, negative = "10 people", "30 people"
            elif field == "rating":
                positive, negative = "3/5", "5/5"
            else:
                positive, negative = "4 hours", "20 hours"
            return f"The sources conflict about {NAMES[field]}: one reports {positive}; another reports {negative}."
        return rng.choice([f"The listing does not specify {NAMES[field]}.",
                           f"No information about {NAMES[field]} is provided."])
    if value is False and field in NEGATIVE:
        return NEGATIVE[field]
    unit = "day" if value == 1 and field == "lead_days" else "days"
    return rng.choice(DESCRIPTIONS[field]).format(v=value, unit=unit)


def _tri_not(value):
    return None if value is None else not value


def _tri_and(values):
    return False if False in values else None if None in values else True


def _tri_or(values):
    return True if True in values else None if None in values else False


def _evaluate(expr, facts):
    op = expr["op"]
    if op in ("le", "ge", "eq", "in"):
        value = facts.get(expr["field"])
        if value is None:
            return None
        if op == "le":
            return value <= expr["value"]
        if op == "ge":
            return value >= expr["value"]
        if op == "in":
            return value in expr["values"]
        return value == expr["value"]
    if op == "sum_le":
        numbers = [facts.get(field) for field in expr["fields"]]
        return None if None in numbers else sum(numbers) <= expr["value"]
    if op == "not":
        return _tri_not(_evaluate(expr["arg"], facts))
    if op == "and":
        return _tri_and([_evaluate(x, facts) for x in expr["args"]])
    if op == "or":
        return _tri_or([_evaluate(x, facts) for x in expr["args"]])
    if op == "implies":
        return _tri_or([_tri_not(_evaluate(expr["if"], facts)), _evaluate(expr["then"], facts)])
    raise ValueError(op)


def _status(results):
    return "ineligible" if False in results else "undetermined" if None in results else "eligible"


def generate(rng, level=0):
    """Return one multi-question Jev problem, without exposing hidden facts."""
    n = N_RECORDS[level] + rng.randint(0, 2)
    simple = ["fee", "lead", "capacity", "region", "membership", "certified", "deposit", "rating", "start"]
    advanced = ["mode", "support", "total", "conditional"]
    forced = []
    if level == 2:
        forced = [rng.choice(["mode", "support", "total"])]
    elif level == 3:
        forced = ["conditional", rng.choice(["mode", "support", "total"])]
    elif level >= 4:
        forced = ["conditional", rng.choice(["mode", "support"]), "total"]
    kinds = list(forced)
    pool = simple + (advanced[:3] if level == 2 else advanced if level >= 3 else [])
    rng.shuffle(pool)
    for kind in pool:
        if len(kinds) == N_CONSTRAINTS[level]:
            break
        if kind in kinds or (kind == "fee" and "total" in kinds) or (kind == "total" and "fee" in kinds):
            continue
        kinds.append(kind)
    rng.shuffle(kinds)
    requirements = [_requirement(rng, kind) for kind in kinds]
    domain = rng.choice(DOMAINS)
    records, source = [], []
    # Vary the eligible count independently of level, including no-match cases.
    if rng.random() < 0.1:
        weights = [0, 0.6, 0.4]
    else:
        eligible_share = rng.choice([0.15, 0.3, 0.45, 0.6])
        unknown_share = rng.choice([0.2, 0.35, 0.5])
        weights = [eligible_share, (1 - eligible_share) * (1 - unknown_share),
                   (1 - eligible_share) * unknown_share]
    outcomes = rng.choices(LABELS, weights=weights, k=n)
    for i, desired in enumerate(outcomes):
        chosen = rng.randrange(len(requirements))
        facts = {}
        for j, req in enumerate(requirements):
            variant = "pass" if j != chosen or desired == "eligible" else (
                "fail" if desired == "ineligible" else "unknown")
            case = _facts(rng, req, variant, level)
            assert not (set(facts) & set(case)), "Requirement groups must use disjoint fields"
            facts.update(case)
        if level >= 3 and desired == "ineligible" and len(requirements) > 2 and rng.random() < 0.2:
            another = rng.choice([j for j in range(len(requirements)) if j != chosen])
            facts.update(_facts(rng, requirements[another], "unknown", level))
        results = [_evaluate(req["expr"], facts) for req in requirements]
        assert _status(results) == desired
        snippets = [_describe(rng, field, value, level) for field, value in facts.items()]
        extra = rng.choice([
            "The listing was refreshed after its last review.",
            "A reference code is available on request.",
            "It was entered into the catalogue this quarter.",
            "The offer carries an internal tracking number.",
        ])
        if level >= 1:
            snippets.append(extra)
        if level >= 2:
            rng.shuffle(snippets)
        record_id = f"R{i+1}"
        records.append({"id": record_id, "description": " ".join(snippets)})
        source.append({"id": record_id, "facts": facts, "results": results, "status": desired})
    rng.shuffle(records)
    probe = rng.choice(source)
    if rng.random() < 0.7:
        failing = [i for i, value in enumerate(probe["results"]) if value is not True]
        requirement_index = rng.choice(failing) if failing else rng.randrange(len(requirements))
    else:
        requirement_index = rng.randrange(len(requirements))
    p_id = probe["id"]
    c_id = f"C{requirement_index+1}"
    eligibility = {r["id"]: r["status"] for r in source}
    first = next((r["id"] for r in records if eligibility[r["id"]] == "eligible"), "none")
    result = probe["results"]
    bad = [j for j, x in enumerate(result) if x is False]
    uncertain = [j for j, x in enumerate(result) if x is None]
    decisive = f"C{(bad or uncertain)[0]+1}" if bad or uncertain else "none"
    relaxed = _status([v for j, v in enumerate(result) if j != requirement_index])
    state = {
        "user_request": (f"Select eligible {domain} under all of the following conditions:\n"
                         + "\n".join(f"C{i+1}. {req['text']}" for i, req in enumerate(requirements))),
        "records": records,
        "semantics": SEMANTICS,
    }
    constraint_options = [f"C{i+1}" for i in range(len(requirements))] + ["none"]
    record_options = [r["id"] for r in records] + ["none"]
    questions = {
        "probe_eligibility": {"type": "choice", "instructions": f"What is the eligibility status of {p_id}?",
                              "criteria": {x: x for x in LABELS}},
        "requirement_status": {"type": "choice", "instructions": f"For {p_id}, is {c_id} met, violated or unknown?",
                               "criteria": {x: x for x in ["met", "violated", "unknown"]}},
        "eligible_count": {"type": "score", "instructions": "How many listed records are eligible?",
                           "criteria": COUNTS},
        "first_eligible": {"type": "choice", "instructions": "Which record is the first eligible one in the displayed order? Choose none if there is no eligible record.",
                            "criteria": {x: x for x in record_options}},
        "decisive_constraint": {"type": "choice", "instructions": f"For {p_id}, which numbered requirement is decisive under the stated ordering rule?",
                                "criteria": {x: x for x in constraint_options}},
        "eligible_if_relaxed": {"type": "noul", "instructions": f"If requirement {c_id} is removed, is {p_id} definitely eligible under every remaining requirement?"},
        "probe_fully_eligible": {"type": "noul", "instructions": f"Is {p_id} definitely eligible, rather than ineligible or undetermined?"},
    }
    answers = {
        "probe_eligibility": choice_answer(probe["status"], LABELS),
        "requirement_status": choice_answer({True: "met", False: "violated", None: "unknown"}[result[requirement_index]],
                                            ["met", "violated", "unknown"]),
        "eligible_count": score_answer(min(sum(v == "eligible" for v in eligibility.values()), 9), COUNTS),
        "first_eligible": choice_answer(first, record_options),
        "decisive_constraint": choice_answer(decisive, constraint_options),
        "eligible_if_relaxed": noul_answer(relaxed == "eligible"),
        "probe_fully_eligible": noul_answer(probe["status"] == "eligible"),
    }
    return Problem(state, questions, answers, {"requirements": requirements, "source": source,
                                              "probe": p_id, "constraint_index": requirement_index})
