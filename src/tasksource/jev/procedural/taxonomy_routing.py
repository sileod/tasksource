"""Route a ticket to the one category, among 4 to 60, whose rule it satisfies.

Level 0 has a handful of one-condition rules; level 4 has dozens of rules of two
to four conditions and ticket amounts and ages on or next to the thresholds.

Each state is a fresh routing guide: category names come from a bounded pool,
but every row draws new rules (conjunctions of one to three conditions), many of
them sharing a condition with the right category. The ticket meets every
condition of exactly one category and fails at least one of every other, so the
answer is exact while nearly every option is a plausible near-miss.
"""

from ._common import Problem, choice_answer, noul_answer, phrase, render_records, score_answer, sround

AREAS = ["billing", "shipping", "returns", "accounts", "security", "hardware", "software", "partners",
         "compliance", "onboarding"]
DESKS = ["intake", "escalations", "priority", "review", "specialists", "backlog", "desk-a", "desk-b"]
CATEGORIES = [f"{area}/{desk}" for area in AREAS for desk in DESKS]  # 80 names, recurring across splits
PRODUCTS = ["router", "laptop", "phone", "printer", "camera", "tablet"]
ISSUES = ["refund", "damage", "login", "delay", "invoice", "defect"]
TIERS = ["free", "plus", "business"]
REGIONS = ["north", "south", "east", "west"]
CHANNELS = ["email", "chat", "phone"]
SIZES = [5, 10, 18, 30, 45]
CONDITIONS = [(1, 1), (1, 2), (1, 3), (2, 3), (2, 4)]
AMOUNTS = [50, 100, 200, 500]
AGES = [7, 14, 30, 60]
MET_COUNTS = ["0", "1", "2", "3", "4"]  # rules have at most four conditions

FIELDS = {"product": PRODUCTS, "issue": ISSUES, "tier": TIERS, "region": REGIONS, "channel": CHANNELS}


def _condition(rng):
    kind = rng.choice(["field", "field", "field", "amount", "age"])
    if kind == "field":
        field = rng.choice(list(FIELDS))
        return ("eq", field, rng.choice(FIELDS[field]))
    if kind == "amount":
        return (rng.choice(["gt", "le"]), "amount", rng.choice(AMOUNTS))
    return (rng.choice(["gt", "le"]), "age_days", rng.choice(AGES))


def _holds(condition, ticket):
    op, field, value = condition
    if op == "eq":
        return ticket[field] == value
    return ticket[field] > value if op == "gt" else ticket[field] <= value


def _text(condition):
    op, field, value = condition
    name = field.replace("_", " ")
    if op == "eq":
        return f"{name} is {value}"
    return f"{name} {'above' if op == 'gt' else 'at most'} {value}"


def _ticket(rng, level):
    ticket = {"product": rng.choice(PRODUCTS), "issue": rng.choice(ISSUES), "tier": rng.choice(TIERS),
              "region": rng.choice(REGIONS), "channel": rng.choice(CHANNELS),
              "amount": rng.randint(10, 900), "age_days": rng.randint(1, 90)}
    if level >= 3:  # on or next to a threshold, where "above" and "at most" are easy to confuse
        ticket["amount"] = rng.choice(AMOUNTS) + rng.choice([-5, 0, 0, 5])
        ticket["age_days"] = rng.choice(AGES) + rng.choice([-1, 0, 0, 1])
    return ticket


def _rule(rng, ticket, satisfied, sizes):
    """``sizes`` (low, high) conditions on distinct fields, all true of ``ticket`` iff ``satisfied``."""
    while True:
        conditions, fields = [], set()
        for _ in range(rng.randint(*sizes)):
            condition = _condition(rng)
            if condition[1] not in fields:
                fields.add(condition[1])
                conditions.append(condition)
        if all(_holds(c, ticket) for c in conditions) == satisfied:
            return conditions


def _near_rule(rng, ticket, gold, sizes):
    """A failing rule that shares one of the gold conditions."""
    for _ in range(50):
        rule = _rule(rng, ticket, False, (max(1, sizes[0] - 1), max(1, sizes[1] - 1)))
        shared = rng.choice(gold)
        failing = [c for c in rule if not _holds(c, ticket)]
        if shared[1] not in {c[1] for c in rule}:
            return [shared, failing[0], *[c for c in rule if c is not failing[0]]][:max(2, sizes[1])]
    return _rule(rng, ticket, False, sizes)


def generate(rng, level=0):
    n = min(len(CATEGORIES), max(4, sround(SIZES[level] * rng.uniform(0.8, 1.3), rng)))
    names = rng.sample(CATEGORIES, n)
    sizes = CONDITIONS[level]
    ticket = _ticket(rng, level)
    gold = rng.choice(names)
    rules = {gold: _rule(rng, ticket, True, sizes)}
    for name in names:
        if name != gold:
            near = sizes[1] > 1 and rng.random() < 0.6
            rules[name] = _near_rule(rng, ticket, rules[gold], sizes) if near else _rule(rng, ticket, False, sizes)

    probe = gold if rng.random() < 0.4 else rng.choice([name for name in names if name != gold])
    met = sum(_holds(c, ticket) for c in rules[probe])

    guide = [{"category": name, "rule": " and ".join(map(_text, rules[name]))} for name in names]
    style = rng.choice(["lines", "table", "json"])
    ticket_text = "; ".join(f"{field.replace('_', ' ')}: {value}" for field, value in ticket.items())
    state = (f"Routing guide (a ticket goes to the category whose conditions it all meets; exactly one does):\n"
             f"{render_records(guide, style)}\n\nTicket: {ticket_text}")
    questions = {
        "route": {"type": "choice", "criteria": {name: name for name in names}, "instructions": phrase(rng, [
            "Which category should this ticket be routed to?", "Where does the routing guide send this ticket?",
            "Which category's conditions does the ticket meet?"])},
        "belongs_to": {"type": "noul", "instructions": phrase(rng, [
            "Does this ticket belong in {c}?", "Should the ticket be routed to {c}?"], c=probe)},
        "conditions_met": {"type": "score", "criteria": MET_COUNTS, "instructions": phrase(rng, [
            "How many of the conditions listed for {c} does the ticket meet?",
            "Count the conditions of {c} that hold for this ticket."], c=probe)},
    }
    answers = {
        "route": choice_answer(gold, names),
        "belongs_to": noul_answer(probe == gold),
        "conditions_met": score_answer(met, MET_COUNTS),
    }
    data = {"ticket": ticket, "rules": {name: rules[name] for name in names}, "gold": gold, "probe": probe}
    return Problem(state, questions, answers, data)
