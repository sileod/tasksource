"""Judge support, conflict, and decisive provenance from duplicated, invalid, and contradictory evidence."""

from ._common import Problem, choice_answer, noul_answer, sround


def generate(rng, level=0):
    n_evidence = sround(6 + 1.4 * level, rng)
    n_origins = max(3, sround(4 + 0.6 * level, rng))
    origins = [f"S{i+1}" for i in range(n_origins)]
    evidence = []
    roles = ["support", "support", "contradict", "irrelevant"]
    for i in range(max(4, n_evidence)):
        evidence.append({
            "id": f"E{i+1}",
            "origin": rng.choice(origins),
            "role": rng.choice(roles),
            "valid": rng.random() < 0.8,
            "reliability": rng.choice([1, 2, 3]),
        })
    if rng.random() < 0.85:  # usually guarantee one valid support, without fixing its origin
        evidence[0].update(origin=rng.choice(origins), role="support", valid=True)
    valid_support = {}
    valid_contradict = set()
    for item in evidence:
        if not item["valid"]:
            continue
        if item["role"] == "support":
            valid_support[item["origin"]] = max(valid_support.get(item["origin"], 0), item["reliability"])
        elif item["role"] == "contradict":
            valid_contradict.add(item["origin"])
    supported = len(valid_support) >= 2 and not valid_contradict
    conflict = bool(valid_support and valid_contradict)
    decisive = "none"
    if valid_support:
        decisive = min(valid_support, key=lambda origin: (-valid_support[origin], origin))
    options = origins + ["none"]
    state = {
        "claim": "The deployment is sufficiently supported as the cause of the incident.",
        "rule": "Support is sufficient only with at least two independent valid supporting origins and no valid contradictory origin. Duplicates from one origin count once.",
        "evidence": evidence,
    }
    questions = {
        "claim_supported": {"type": "noul", "instructions": "Under state.rule, is the claim sufficiently supported?"},
        "has_conflict": {"type": "noul", "instructions": "Is there at least one valid supporting origin and at least one valid contradictory origin?"},
        "strongest_support_origin": {
            "type": "choice",
            "instructions": "Which valid supporting origin has the highest reliability? Break ties lexicographically; choose none if there is no valid support.",
            "criteria": {option: option for option in options},
        },
    }
    answers = {
        "claim_supported": noul_answer(supported),
        "has_conflict": noul_answer(conflict),
        "strongest_support_origin": choice_answer(decisive, options),
    }
    return Problem(state, questions, answers)
