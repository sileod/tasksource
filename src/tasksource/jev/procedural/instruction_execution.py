"""Ground instructions in entity sets and apply guarded state changes.

Levels add groups, compound filters and exceptions, unavailable entities,
transaction policies, deferred operations, and ambiguous references.
"""

from ._common import Problem, choice_answer, noul_answer, score_answer

DOMAINS = [
    ("job", "running", "start", "stop"),
    ("document", "approved", "approve", "unapprove"),
    ("account", "enabled", "enable", "disable"),
    ("ticket", "escalated", "escalate", "de-escalate"),
    ("shipment", "held", "hold", "release"),
]
SITES = ["north", "south", "central", "west"]
TIERS = ["standard", "priority", "critical"]
COUNTS = [str(i) for i in range(9)] + ["9 or more"]
RESOLUTIONS = ["execute", "partial", "blocked", "no_change", "clarify", "schedule", "rule", "query"]
N_ENTITIES = [3, 5, 8, 14, 22]
SEMANTICS = ("A request selects the named entities after any exclusions. Immediate updates set the named boolean "
             "property to the requested value. Unavailable entities cannot be changed. Atomic requests change "
             "nothing if any selected entity is unavailable; best-effort requests skip unavailable entities. "
             "Ordered requests perform steps in order, including repeated updates, and skip unavailable entities "
             "like best-effort requests. Clarify an ambiguous or "
             "missing name without changing anything. A query only reads state. A schedule or recurring rule "
             "is registered now and does not immediately modify entity properties.")


def _count(n):
    return min(n, 9)


def _execute(state, spec):
    entities = {e["id"]: dict(e) for e in state["entities"]}
    ids = spec["ids"]
    kind = spec["kind"]
    if kind == "clarify":
        status = "clarify"
    elif kind in ("query", "schedule", "rule"):
        status = kind
    elif not ids:
        status = "no_change"
    else:
        unavailable = [i for i in ids if not entities[i]["available"]]
        policy = state["execution_policy"]
        if unavailable and policy == "atomic":
            status = "blocked"
        else:
            available = [i for i in ids if entities[i]["available"]]
            if not available:
                status = "blocked"
            else:
                status = "partial" if unavailable else "execute"
                for value in spec["values"]:
                    for i in available:
                        entities[i][state["property"]] = value
                if status == "execute" and len(spec["values"]) == 1 and all(
                    entities[i][state["property"]] == next(e for e in state["entities"] if e["id"] == i)[state["property"]]
                    for i in ids
                ):
                    status = "no_change"
    changed = sum(e[state["property"]] != entities[e["id"]][state["property"]] for e in state["entities"])
    return status, entities, changed


def generate(rng, level=0):
    noun, prop, positive, negative = rng.choice(DOMAINS)
    n = N_ENTITIES[level] + rng.randint(0, max(1, level))
    entities = [
        {"id": f"E{i+1}", "name": f"{noun} {i+1}", "site": rng.choice(SITES),
         "tier": rng.choice(TIERS), "available": True, prop: bool(rng.getrandbits(1))}
        for i in range(n)
    ]
    groups = {"batch A": [e["id"] for e in entities if int(e["id"][1:]) % 2],
              "batch B": [e["id"] for e in entities if not int(e["id"][1:]) % 2]}
    wanted = bool(rng.getrandbits(1))
    verb = positive if wanted else negative
    selector_kind = rng.choice(["id", "group"] if level == 1 else
                               ["id", "group", "filter"] if level >= 2 else ["id"])
    if selector_kind == "id":
        chosen = rng.choice(entities)
        ids = [chosen["id"]]
        selector = chosen["name"]
    elif selector_kind == "group":
        group = rng.choice(list(groups))
        ids = list(groups[group])
        selector = f"all {noun}s in {group}"
    else:
        exemplar = rng.choice(entities)
        site = exemplar["site"]
        tier = exemplar["tier"] if level >= 3 else None
        ids = [e["id"] for e in entities if e["site"] == site and (tier is None or e["tier"] == tier)]
        selector = f"all {noun}s at site {site}" + (f" with tier {tier}" if tier else "")
    if level >= 2 and len(ids) > 1 and rng.random() < 0.5:
        exclude = rng.choice(ids)
        ids.remove(exclude)
        selector += f", except {next(e['name'] for e in entities if e['id'] == exclude)}"
    kind = "immediate"
    if level >= 2:
        kind = rng.choices(["immediate", "query", "schedule", "rule"], [6, 1, 1, 1])[0]
    if level >= 3 and rng.random() < 0.12:
        kind = "clarify"
        if rng.random() < 0.75:
            a, b = rng.sample(entities, 2)
            b["name"] = a["name"]
            selector = a["name"]
        else:
            selector = f"{noun} not listed"
        ids = []
    if level >= 2 and kind == "immediate":
        for i in ids:
            if rng.random() < (0.15 if level == 2 else 0.35):
                next(e for e in entities if e["id"] == i)["available"] = False
    policy = "atomic" if level < 3 else rng.choice(["atomic", "best_effort", "ordered"])
    values = [wanted]
    if level >= 4 and kind == "immediate" and rng.random() < 0.4:
        policy = "ordered"
        values = [not wanted, wanted]
    if kind == "query":
        request = f"Are {selector} currently {prop}?"
    elif kind == "schedule":
        request = f"At 18:30, {verb} {selector}."
    elif kind == "rule":
        request = f"Whenever a review event occurs, {verb} {selector}."
    elif kind == "clarify":
        request = f"Please {verb} {selector}."
    elif len(values) > 1:
        reverse = negative if wanted else positive
        request = f"First {reverse} {selector}, then {verb} them."
    else:
        request = f"Please {verb} {selector}."
    state = {"entity_type": noun, "property": prop, "entities": entities, "groups": groups,
             "execution_policy": policy, "semantics": SEMANTICS, "request": request}
    spec = {"kind": kind, "ids": ids, "values": values}
    status, final, changed = _execute(state, spec)
    probe = rng.choice(entities)
    if ids and rng.random() < 0.6:
        target_id = rng.choice(ids)
        probe = next(e for e in entities if e["id"] == target_id)
    questions = {
        "resolution": {"type": "choice", "instructions": "How should the request be handled under the stated execution policy?",
                       "criteria": {x: x for x in RESOLUTIONS}},
        "target_count": {"type": "score", "instructions": "How many entities are unambiguously selected by the request, before checking availability?",
                         "criteria": COUNTS},
        "probe_selected": {"type": "noul", "instructions": f"Is {probe['id']} in the unambiguous target set?"},
        "probe_final_state": {"type": "noul", "instructions": f"Immediately after handling the request, is {probe['id']}.{prop} true?"},
        "changed_count": {"type": "score", "instructions": "How many entity properties change immediately (net changes)?",
                          "criteria": COUNTS},
        "deferred_registration": {"type": "choice", "instructions": "What is registered for later, if anything?",
                                  "criteria": {x: x for x in ["nothing", "one-time schedule", "recurring rule"]}},
    }
    answers = {
        "resolution": choice_answer(status, RESOLUTIONS),
        "target_count": score_answer(_count(len(ids)), COUNTS),
        "probe_selected": noul_answer(probe["id"] in ids),
        "probe_final_state": noul_answer(final[probe["id"]][prop]),
        "changed_count": score_answer(_count(changed), COUNTS),
        "deferred_registration": choice_answer({"schedule": "one-time schedule", "rule": "recurring rule"}.get(status, "nothing"),
                                               ["nothing", "one-time schedule", "recurring rule"]),
    }
    return Problem(state, questions, answers, {"spec": spec, "final": final, "probe": probe["id"]})
