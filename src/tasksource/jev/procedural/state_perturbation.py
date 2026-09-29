"""Compare minimally edited operational records and identify semantic change, changed dimension, and risk direction under an explicit additive risk rule.

Levels add records and simultaneous changes (whose risk effects can offset),
amounts near the threshold, and look-alike fields that change without being material.
"""

from ._common import Problem, choice_answer, noul_answer, score_answer, sround

DIMENSIONS = ["authorization", "status", "financial", "ownership", "none"]
RISK = ["Lower operational risk than before.", "No material risk change.", "Higher operational risk than before."]
RISK_RULE = {
    "unauthorized_points": 3,
    "blocked_status_points": 2,
    "amount_threshold": 500,
    "amount_threshold_points": 1,
    "unassigned_owner_points": 1,
    "otherwise_points": 0,
    "direction": "Add the applicable points over all records. Compare after with before: lower total means lower risk, equal means no material risk change, and higher total means higher risk.",
}
MATERIAL = "Material fields are authorized, status, amount, and owner; every other field is non-material."
STATUSES = ["open", "blocked", "resolved"]
OWNERS = ["alice", "bob", "unassigned"]
N_RECORDS = [1, 2, 4, 6, 8]
N_CHANGED = [1, 1, 2, 3, 4]


def _risk(record):
    return (
        (0 if record["authorized"] else RISK_RULE["unauthorized_points"])
        + (RISK_RULE["blocked_status_points"] if record["status"] == "blocked" else RISK_RULE["otherwise_points"])
        + (RISK_RULE["amount_threshold_points"] if record["amount"] >= RISK_RULE["amount_threshold"] else RISK_RULE["otherwise_points"])
        + (RISK_RULE["unassigned_owner_points"] if record["owner"] == "unassigned" else RISK_RULE["otherwise_points"])
    )


def _amount(rng, level):
    return rng.choice([100, 250, 500, 900]) if level < 2 else rng.choice([100, 900, *range(450, 560, 10)])


def _record(rng, level, i):
    record = {"id": f"R{i+1}", "authorized": rng.choice([True, False]), "status": rng.choice(STATUSES),
              "amount": _amount(rng, level), "owner": rng.choice(OWNERS), "note": rng.choice(["routine", "reviewed", "imported"])}
    if level >= 2:  # look-alike, non-material fields
        record.update(status_note=rng.choice(STATUSES), amount_quoted=_amount(rng, level), previous_owner=rng.choice(OWNERS))
    return record


def _change(rng, level, record, dimension):
    after = dict(record)
    if dimension == "authorization":
        after["authorized"] = not record["authorized"]
    elif dimension == "status":
        after["status"] = rng.choice([x for x in STATUSES if x != record["status"]])
    elif dimension == "financial":
        after["amount"] = rng.choice([x for x in range(0, 1000) if x != record["amount"] and _amount_ok(x, level)])
    elif dimension == "ownership":
        after["owner"] = rng.choice([x for x in OWNERS if x != record["owner"]])
    decoys = [f for f in ("note", "status_note", "amount_quoted", "previous_owner") if f in record]
    for field in rng.sample(decoys, rng.randint(1 if dimension == "none" else 0, min(len(decoys), level + 1))):
        after[field] = {"note": lambda: rng.choice([x for x in ["routine", "reviewed", "imported"] if x != record["note"]]),
                        "status_note": lambda: rng.choice([x for x in STATUSES if x != record["status_note"]]),
                        "amount_quoted": lambda: rng.choice([x for x in range(450, 560, 10) if x != record["amount_quoted"]]),
                        "previous_owner": lambda: rng.choice([x for x in OWNERS if x != record["previous_owner"]])}[field]()
    return after


def _amount_ok(x, level):
    return x in (100, 250, 500, 900) if level < 2 else x % 10 == 0 and (x in (100, 900) or 450 <= x < 560)


def generate(rng, level=0):
    n = N_RECORDS[level]
    before = [_record(rng, level, i) for i in range(n)]
    changed = set(rng.sample(range(n), min(n, N_CHANGED[level])))
    if n == 1 and rng.random() < 0.2:
        changed = set()
    dimensions = [rng.choice(DIMENSIONS[:-1]) if i in changed else "none" for i in range(n)]
    after = [_change(rng, level, record, dimension) for record, dimension in zip(before, dimensions)]
    probe = rng.choice(sorted(changed)) if changed and (n == len(changed) or rng.random() < 0.5) else \
        rng.choice([i for i in range(n) if i not in changed] or [0])
    delta = sum(map(_risk, after)) - sum(map(_risk, before))
    risk_index = 0 if delta < 0 else 2 if delta > 0 else 1
    record_id = before[probe]["id"]
    state = {"before": before, "after": after, "materiality": MATERIAL, "risk_rule": RISK_RULE}
    questions = {
        "material_change": {
            "type": "noul",
            "instructions": f"Did any material field of record {record_id} change?",
        },
        "changed_dimension": {
            "type": "choice",
            "instructions": f"Which material dimension of record {record_id} changed? Choose none when only non-material fields changed.",
            "criteria": {x: x for x in DIMENSIONS},
        },
        "risk_direction": {
            "type": "score",
            "instructions": "Following the risk rule, how did total operational risk change from before to after?",
            "criteria": RISK,
        },
    }
    answers = {
        "material_change": noul_answer(dimensions[probe] != "none"),
        "changed_dimension": choice_answer(dimensions[probe], DIMENSIONS),
        "risk_direction": score_answer(risk_index, RISK),
    }
    return Problem(state, questions, answers, {"probe": record_id, "dimensions": dimensions})
