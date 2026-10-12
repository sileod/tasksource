"""Simulate persistent event-action rules over a timestamped event stream.

Levels add rule contention and priorities, guarded rules, cooldowns, toggles,
and out-of-order presentation. Exactly one eligible rule fires per signal.
"""

from ._common import Problem, choice_answer, noul_answer, score_answer

N_RULES = [1, 2, 3, 5, 8]
N_OBJECTS = [2, 3, 4, 6, 8]
N_EVENTS = [3, 5, 8, 12, 18]
KINDS = ["arrive", "check", "expire"]
COUNTS = [str(i) for i in range(9)] + ["9 or more"]
SEMANTICS = ("Process events in increasing timestamp order, not list order. Control events enable or disable "
             "the named rule immediately. Each signal can fire at most one enabled rule. A rule matches when "
             "its event_type and target match the signal (target='any' matches all), its min_value is not "
             "greater than the signal value, and its required_state is null or equals the target's current "
             "state. A rule can fire at timestamp t only if it has never fired or t - last_fired >= cooldown. "
             "Choose the eligible rule with greatest priority, breaking ties by smaller numeric rule ID. "
             "Firing sets the target's boolean state to the rule's set_state value; it also updates last_fired "
             "even when the value was already satisfied. A control event never fires a rule.")


def _replay(state):
    values = {x["id"]: x["state"] for x in state["objects"]}
    enabled = {r["id"]: r["enabled"] for r in state["rules"]}
    last = {}
    fired = {}
    timeline = []
    for event in sorted(state["events"], key=lambda e: e["ts"]):
        if event["kind"] == "control":
            enabled[event["rule"]] = event["enabled"]
            continue
        matches = [r for r in state["rules"] if enabled[r["id"]] and
                   r["event_type"] == event["kind"] and r["target"] in ("any", event["target"]) and
                   event["value"] >= r["min_value"] and
                   (r["required_state"] is None or r["required_state"] == values[event["target"]]) and
                   (r["id"] not in last or event["ts"] - last[r["id"]] >= r["cooldown"])]
        selected = min(matches, key=lambda r: (-r["priority"], int(r["id"][1:]))) if matches else None
        rid = selected["id"] if selected else "none"
        if selected:
            last[rid] = event["ts"]
            values[event["target"]] = selected["set_state"]
        fired[event["ts"]] = rid
        timeline.append({"ts": event["ts"], "rule": rid, "target": event["target"]})
    return values, enabled, fired, timeline


def generate(rng, level=0):
    objects = [{"id": f"O{i+1}", "state": bool(rng.getrandbits(1))} for i in range(N_OBJECTS[level])]
    rules = []
    for i in range(N_RULES[level]):
        rules.append({
            "id": f"R{i+1}", "event_type": rng.choice(KINDS),
            "target": rng.choice(["any", rng.choice(objects)["id"]]) if level >= 1 else "any",
            "min_value": rng.randint(0, 5) if level >= 2 else 0,
            "required_state": rng.choice([None, False, True]) if level >= 4 else None,
            "set_state": bool(rng.getrandbits(1)),
            "priority": rng.randint(1, 4) if level >= 2 else 1,
            "cooldown": rng.randint(2, 6) if level >= 3 else 0,
            "enabled": True,
        })
    events = []
    t = 0
    for i in range(N_EVENTS[level]):
        t += rng.randint(1, 3)
        if level >= 3 and i > 1 and rng.random() < 0.2:
            rule = rng.choice(rules)
            events.append({"ts": t, "kind": "control", "rule": rule["id"], "enabled": bool(rng.getrandbits(1))})
        else:
            from_rule = rng.choice(rules)
            events.append({"ts": t,
                           "kind": from_rule["event_type"] if rng.random() < 0.65 else rng.choice(KINDS),
                           "target": from_rule["target"] if from_rule["target"] != "any" and rng.random() < 0.8 else rng.choice(objects)["id"],
                           "value": rng.randint(0, 9)})
    signals = [e for e in events if e["kind"] != "control"]
    if level >= 4:
        rng.shuffle(events)
    state = {"objects": objects, "rules": rules, "events": events, "semantics": SEMANTICS}
    values, enabled, fired, timeline = _replay(state)
    probe_event = rng.choice(signals)
    probe_obj = rng.choice(objects)["id"]
    if rng.random() < 0.5:
        probe_obj = probe_event["target"]
    probe_rule = rng.choice(rules)["id"]
    n_fired = sum(rid != "none" for rid in fired.values())
    options = [r["id"] for r in rules] + ["none"]
    questions = {
        "rule_on_signal": {"type": "choice", "instructions": f"Which rule fires for the signal at timestamp {probe_event['ts']}?",
                           "criteria": {x: x for x in options}},
        "object_final_state": {"type": "noul", "instructions": f"After processing all events, is {probe_obj}.state true?"},
        "rule_enabled_at_end": {"type": "noul", "instructions": f"Is {probe_rule} enabled after processing all events?"},
        "total_firings": {"type": "score", "instructions": "How many rule firings occur in total (including no-op writes)?",
                          "criteria": COUNTS},
        "object_ever_targeted": {"type": "noul", "instructions": f"Does any rule fire on {probe_obj} during the event sequence?"},
    }
    answers = {
        "rule_on_signal": choice_answer(fired[probe_event["ts"]], options),
        "object_final_state": noul_answer(values[probe_obj]),
        "rule_enabled_at_end": noul_answer(enabled[probe_rule]),
        "total_firings": score_answer(min(n_fired, 9), COUNTS),
        "object_ever_targeted": noul_answer(any(e["target"] == probe_obj and e["rule"] != "none" for e in timeline)),
    }
    return Problem(state, questions, answers, {"timeline": timeline, "values": values,
                                                     "probe_event": probe_event["ts"], "probe_object": probe_obj,
                                                     "probe_rule": probe_rule})
