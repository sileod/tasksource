"""Compare minimally edited operational states and identify semantic change, changed dimension, and risk direction under an explicit additive risk rule."""

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
    "direction": "Add the applicable points for each record. Compare after with before: lower total means lower risk, equal means no material risk change, and higher total means higher risk.",
}


def _risk(record):
    return (
        (0 if record["authorized"] else RISK_RULE["unauthorized_points"])
        + (RISK_RULE["blocked_status_points"] if record["status"] == "blocked" else RISK_RULE["otherwise_points"])
        + (RISK_RULE["amount_threshold_points"] if record["amount"] >= RISK_RULE["amount_threshold"] else RISK_RULE["otherwise_points"])
        + (RISK_RULE["unassigned_owner_points"] if record["owner"] == "unassigned" else RISK_RULE["otherwise_points"])
    )


def generate(rng, level=0):
    n_noise_fields = sround(2 + 1.0 * level, rng)
    before = {
        "authorized": rng.choice([True, False]),
        "status": rng.choice(["open", "blocked", "resolved"]),
        "amount": rng.choice([100, 250, 500, 900]),
        "owner": rng.choice(["alice", "bob", "unassigned"]),
        "note": rng.choice(["routine", "reviewed", "imported"]),
    }
    after = dict(before)
    dimension = rng.choice(DIMENSIONS)
    if dimension == "authorization":
        after["authorized"] = not before["authorized"]
    elif dimension == "status":
        after["status"] = rng.choice([x for x in ["open", "blocked", "resolved"] if x != before["status"]])
    elif dimension == "financial":
        after["amount"] = rng.choice([x for x in [100, 250, 500, 900] if x != before["amount"]])
    elif dimension == "ownership":
        after["owner"] = rng.choice([x for x in ["alice", "bob", "unassigned"] if x != before["owner"]])
    else:
        after["note"] = rng.choice([x for x in ["routine", "reviewed", "imported"] if x != before["note"]])
    noise = {f"aux_{i+1}": rng.randint(0, 9) for i in range(max(0, n_noise_fields))}
    before.update(noise)
    after.update(noise)
    delta = _risk(after) - _risk(before)
    risk_index = 0 if delta < 0 else 2 if delta > 0 else 1
    material = dimension != "none"
    state = {"before": before, "after": after, "risk_rule": RISK_RULE}
    questions = {
        "material_change": {
            "type": "noul",
            "instructions": "Did authorization, status, amount, or owner change? Ignore note and aux_* fields.",
        },
        "changed_dimension": {
            "type": "choice",
            "instructions": "Which material dimension changed? Choose none when only non-material fields changed.",
            "criteria": {x: x for x in DIMENSIONS},
        },
        "risk_direction": {
            "type": "score",
            "instructions": "Following the risk rule, how did operational risk change from before to after?",
            "criteria": RISK,
        },
    }
    answers = {
        "material_change": noul_answer(material),
        "changed_dimension": choice_answer(dimension, DIMENSIONS),
        "risk_direction": score_answer(risk_index, RISK),
    }
    return Problem(state, questions, answers)
