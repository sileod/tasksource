"""Reconstruct owner, open status, and severity from an event log plus a stale snapshot.

Level 0 has three events in order; levels add events, shuffle the log (level 2),
and void earlier entries as recorded in error (level 3).
"""

from ._common import Problem, choice_answer, noul_answer, score_answer, sround

OWNERS = ["alice", "bob", "carol", "unassigned"]
SEVERITY = ["Routine.", "Degraded service requiring attention.", "Critical user-blocking incident."]
SEVERITY_NAMES = ["routine", "degraded", "critical"]  # as written in the state; SEVERITY describes them
N_EVENTS = [3, 5, 8, 12, 16]


def _replay(initial, events):
    """Owner, open, and severity after applying the non-voided events in timestamp order."""
    voided = {e["voids_ts"] for e in events if e["kind"] == "void"}
    owner, is_open, severity = initial["owner"], initial["open"], initial["severity"]
    for e in sorted(events, key=lambda e: e["ts"]):
        if e["ts"] in voided:
            continue
        if e["kind"] == "assign":
            owner = e["owner"]
        elif e["kind"] == "severity":
            severity = e["severity"]
        elif e["kind"] == "resolve":
            is_open = False
        elif e["kind"] == "reopen":
            is_open = True
    return owner, is_open, severity


def generate(rng, level=0):
    n_events = sround(N_EVENTS[level] * rng.uniform(0.85, 1.15), rng)
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
        if level >= 3 and events and rng.random() < 0.2:
            # void an earlier state-changing entry; state is then replayed without it
            voidable = [e for e in events if e["kind"] in ("assign", "severity", "resolve", "reopen") and not e.get("voided")]
            if voidable:
                target = rng.choice(voidable)
                target["voided"] = True
                event = {"ts": ts, "kind": "void", "voids_ts": target["ts"]}
                events.append(event)
                owner, is_open, severity = _replay(initial, events)
                history.append({"owner": owner, "open": is_open, "severity": severity})
                continue
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
    owner, is_open, severity = _replay(initial, events)
    presented = [{k: v for k, v in e.items() if k != "voided"} for e in events]
    if level >= 2:
        rng.shuffle(presented)
    named = lambda record: {**record, "severity": SEVERITY_NAMES[record["severity"]]} if "severity" in record else record
    presented = [named(e) for e in presented]
    state = {"initial_state": named(initial), "events": presented,
             "rule": "Start from initial_state and apply events in ascending timestamp order."}
    if level >= 1:
        snapshot_cut = rng.randrange(len(events))
        state["stale_snapshot"] = named({"as_of": events[snapshot_cut]["ts"], **history[snapshot_cut + 1]})
        state["rule"] += " stale_snapshot is an older redundant view and must not override later events."
    if level >= 3:
        state["rule"] += " A void event cancels the entry with timestamp voids_ts, as if it never happened."
    questions = {
        "current_owner": {"type": "choice", "instructions": "Who owns the incident after replaying the event log?", "criteria": {x: x for x in OWNERS}},
        "is_open": {"type": "noul", "instructions": "Is the incident open after replaying the event log?"},
        "current_severity": {"type": "score", "instructions": "What is the incident's current severity after replaying the event log?", "criteria": [f"{name}: {text}" for name, text in zip(SEVERITY_NAMES, SEVERITY)]},
    }
    answers = {
        "current_owner": choice_answer(owner, OWNERS),
        "is_open": noul_answer(is_open),
        "current_severity": score_answer(severity, [f"{name}: {text}" for name, text in zip(SEVERITY_NAMES, SEVERITY)]),
    }
    return Problem(state, questions, answers)
