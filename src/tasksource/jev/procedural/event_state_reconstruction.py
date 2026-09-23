"""Reconstruct owner, open status, and severity from shuffled events plus a stale snapshot."""

from ._common import Problem, choice_answer, noul_answer, score_answer, sround

OWNERS = ["alice", "bob", "carol", "unassigned"]
SEVERITY = ["Routine.", "Degraded service requiring attention.", "Critical user-blocking incident."]


def generate(rng, level=0):
    n_events = sround(6 + 1.5 * level, rng)
    owner = rng.choice(OWNERS[:-1])
    is_open = True
    severity = rng.randrange(3)
    initial = {"owner": owner, "open": is_open, "severity": severity}
    history = [dict(initial)]
    events = []
    ts = 100
    for _ in range(max(3, n_events)):
        ts += rng.randint(1, 8)
        kind = rng.choice(["assign", "severity", "resolve", "reopen", "note"])
        event = {"ts": ts, "kind": kind}
        if kind == "assign":
            owner = rng.choice(OWNERS)
            event["owner"] = owner
        elif kind == "severity":
            severity = rng.randrange(3)
            event["severity"] = severity
        elif kind == "resolve":
            is_open = False
        elif kind == "reopen":
            is_open = True
        else:
            event["text"] = rng.choice(["ack", "triage", "customer update"])
        events.append(event)
        history.append({"owner": owner, "open": is_open, "severity": severity})
    snapshot_cut = rng.randrange(len(events))
    snapshot = history[snapshot_cut + 1]
    presented = list(events)
    rng.shuffle(presented)
    state = {
        "initial_state": initial,
        "stale_snapshot": {"as_of": events[snapshot_cut]["ts"], **snapshot},
        "events": presented,
        "rule": "Start from initial_state and apply events in ascending timestamp order. stale_snapshot is an older redundant view and must not override later events.",
    }
    questions = {
        "current_owner": {"type": "choice", "instructions": "Who owns the incident after replaying the event log?", "criteria": {x: x for x in OWNERS}},
        "is_open": {"type": "noul", "instructions": "Is the incident open after replaying the event log?"},
        "current_severity": {"type": "score", "instructions": "What is the incident's current severity after replaying the event log?", "criteria": SEVERITY},
    }
    answers = {
        "current_owner": choice_answer(owner, OWNERS),
        "is_open": noul_answer(is_open),
        "current_severity": score_answer(severity, SEVERITY),
    }
    return Problem(state, questions, answers)
