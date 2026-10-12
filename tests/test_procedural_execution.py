"""Independent oracles, determinism, label schemas, and scaling for new generators."""

import json
import random
import unittest
from functools import lru_cache
from itertools import product

from tasksource.jev.procedural import (
    event_triggered_execution, goal_directed_planning, graph_navigation, instruction_execution,
)

MODULES = [instruction_execution, goal_directed_planning, event_triggered_execution, graph_navigation]


def samples(module, level, n=100):
    return [module.generate(random.Random(f"{module.__name__}:{level}:{i}"), level) for i in range(n)]


def plan_oracle(state, forbidden=None, starting=None):
    steps = state["steps"]
    initial = frozenset(s["id"] for s in steps if s["completed"]) if starting is None else starting
    goal = set(state["goal"])

    @lru_cache(None)
    def solve(done):
        if goal <= set(done):
            return 0
        choices = []
        for s in steps:
            if s["id"] == forbidden or s["id"] in done or not s["available"]:
                continue
            if set(s["requires_all"]) <= set(done) and (not s["requires_any"] or set(s["requires_any"]) & set(done)):
                answer = solve(frozenset((*done, s["id"])))
                if answer is not None:
                    choices.append(answer + 1)
        return min(choices) if choices else None

    legal = [s["id"] for s in steps if s["id"] not in initial and s["available"]
             and set(s["requires_all"]) <= set(initial)
             and (not s["requires_any"] or set(s["requires_any"]) & set(initial))]
    return solve(initial), legal


def event_oracle(state):
    values = {o["id"]: o["state"] for o in state["objects"]}
    enabled = {r["id"]: r["enabled"] for r in state["rules"]}
    last, events, targets = {}, {}, []
    for e in sorted(state["events"], key=lambda x: x["ts"]):
        if e["kind"] == "control":
            enabled[e["rule"]] = e["enabled"]
            continue
        chosen = None
        for r in sorted(state["rules"], key=lambda x: (-x["priority"], int(x["id"][1:]))):
            if not enabled[r["id"]] or r["event_type"] != e["kind"]:
                continue
            if r["target"] not in ("any", e["target"]) or e["value"] < r["min_value"]:
                continue
            if r["required_state"] is not None and r["required_state"] != values[e["target"]]:
                continue
            if r["id"] in last and e["ts"] - last[r["id"]] < r["cooldown"]:
                continue
            chosen = r
            break
        events[e["ts"]] = chosen["id"] if chosen else "none"
        if chosen:
            values[e["target"]] = chosen["set_state"]
            last[chosen["id"]] = e["ts"]
            targets.append(e["target"])
    return values, enabled, events, targets


def graph_oracle(state, removed=None):
    nodes = state["nodes"]
    inf = float("inf")
    dist = {(a, b): 0 if a == b else inf for a, b in product(nodes, repeat=2)}
    for e in state["edges"]:
        if e["open"] and e["id"] != removed:
            dist[e["from"], e["to"]] = min(dist[e["from"], e["to"]], e["cost"])
    for k in nodes:
        for a in nodes:
            for b in nodes:
                dist[a, b] = min(dist[a, b], dist[a, k] + dist[k, b])
    via, start, end = state["via"], state["start"], state["finish"]
    middle = via or end
    cost = dist[start, middle] + (dist[via, end] if via else 0)
    if cost == inf:
        return None, "none"
    hops = [e["to"] for e in state["edges"] if e["open"] and e["id"] != removed
            and e["from"] == start and e["cost"] + dist[e["to"], middle] == dist[start, middle]]
    return cost, min(hops) if hops else "none"


class ProceduralExecutionTest(unittest.TestCase):
    def test_schema_and_determinism(self):
        for module in MODULES:
            for level in range(5):
                for p in samples(module, level):
                    self.assertEqual(set(p.questions), set(p.answers))
                    json.dumps([p.state, p.questions, p.answers], allow_nan=False)
                    for name, q in p.questions.items():
                        a = p.answers[name]
                        self.assertEqual(q["type"], a["type"])
                        if a["type"] == "noul":
                            self.assertTrue(0 <= a["noul"] <= 1)
                        else:
                            options = list(q["criteria"])
                            self.assertEqual(len(set(options)), len(options))
                            if a["type"] == "score":
                                self.assertLessEqual(len(options), 10)
                                self.assertIn(a["score"], range(len(options)))
                            else:
                                self.assertIn(a["choice"], options)
                                self.assertEqual(set(a["probabilities"]), set(options))
                                self.assertEqual(sum(a["probabilities"].values()), 1)
                self.assertEqual(module.generate(random.Random(31), level),
                                 module.generate(random.Random(31), level))

    def test_instruction_effects(self):
        for level in range(5):
            for p in samples(instruction_execution, level, 200):
                state, spec = p.state, p.data["spec"]
                before = {e["id"]: e for e in state["entities"]}
                final = {eid: dict(e) for eid, e in before.items()}
                ids, prop = spec["ids"], state["property"]
                unavailable = [i for i in ids if not before[i]["available"]]
                kind = spec["kind"]
                if kind in ("clarify", "query", "schedule", "rule"):
                    status = kind
                elif not ids:
                    status = "no_change"
                elif unavailable and state["execution_policy"] == "atomic":
                    status = "blocked"
                else:
                    allowed = [i for i in ids if before[i]["available"]]
                    status = "partial" if unavailable and allowed else "execute" if allowed else "blocked"
                    for value in spec["values"] if allowed else []:
                        for i in allowed:
                            final[i][prop] = value
                    if status == "execute" and len(spec["values"]) == 1 and all(
                        final[i][prop] == before[i][prop] for i in ids
                    ):
                        status = "no_change"
                changed = sum(e[prop] != final[eid][prop] for eid, e in before.items())
                a = p.answers
                self.assertEqual(a["resolution"]["choice"], status)
                self.assertEqual(a["target_count"]["score"], min(len(ids), 9))
                self.assertEqual(a["changed_count"]["score"], min(changed, 9))
                self.assertEqual(a["probe_selected"]["noul"], float(p.data["probe"] in ids))
                self.assertEqual(a["probe_final_state"]["noul"], float(final[p.data["probe"]][prop]))
                self.assertEqual(a["deferred_registration"]["choice"],
                                 {"schedule": "one-time schedule", "rule": "recurring rule"}.get(status, "nothing"))
                self.assertEqual(final, p.data["final"])

    def test_planning_optimality(self):
        for level in range(5):
            for p in samples(goal_directed_planning, level, 140):
                state, probe = p.state, p.data["probe"]
                distance, legal = plan_oracle(state)
                name = state["steps"][probe]["id"]
                a = p.answers
                self.assertEqual(a["goal_reachable"]["noul"], float(distance is not None))
                self.assertEqual(a["minimal_actions"]["score"], 9 if distance is None else min(distance, 8))
                self.assertEqual(a["probe_applicable"]["noul"], float(name in legal))
                self.assertEqual(a["first_legal_action"]["choice"],
                                 min(legal, key=lambda x: int(x[1:])) if legal else "none")
                start = frozenset([name, *[s["id"] for s in state["steps"] if s["completed"]]])
                rest = plan_oracle(state, starting=start)[0] if name in legal else None
                self.assertEqual(a["probe_optimal_first"]["noul"],
                                 float(distance is not None and name in legal and rest == distance - 1))
                without = plan_oracle(state, forbidden=name)[0]
                self.assertEqual(a["probe_required"]["noul"],
                                 float(distance is not None and (without is None or without > distance)))

    def test_event_replay(self):
        for level in range(5):
            for p in samples(event_triggered_execution, level, 200):
                d, a = p.data, p.answers
                values, enabled, events, targets = event_oracle(p.state)
                self.assertEqual(a["rule_on_signal"]["choice"], events[d["probe_event"]])
                self.assertEqual(a["object_final_state"]["noul"], float(values[d["probe_object"]]))
                self.assertEqual(a["rule_enabled_at_end"]["noul"], float(enabled[d["probe_rule"]]))
                self.assertEqual(a["total_firings"]["score"], min(sum(x != "none" for x in events.values()), 9))
                self.assertEqual(a["object_ever_targeted"]["noul"], float(d["probe_object"] in targets))

    def test_graph_shortest_paths(self):
        for level in range(5):
            for p in samples(graph_navigation, level, 200):
                cost, hop = graph_oracle(p.state)
                a, d = p.answers, p.data
                self.assertEqual(a["reachable"]["noul"], float(cost is not None))
                self.assertEqual(a["shortest_cost"]["choice"], "unreachable" if cost is None else str(cost))
                self.assertEqual(a["first_hop"]["choice"], hop)
                self.assertEqual(a["within_budget"]["noul"], float(cost is not None and cost <= d["budget"]))
                alternative, _ = graph_oracle(p.state, d["probe"])
                self.assertEqual(a["edge_essential"]["noul"],
                                 float(cost is not None and (alternative is None or alternative > cost)))

    def test_structural_scaling_and_diversity(self):
        for module, key in ((instruction_execution, "entities"), (goal_directed_planning, "steps"),
                            (event_triggered_execution, "rules"), (graph_navigation, "nodes")):
            levels = [samples(module, level, 120) for level in range(5)]
            sizes = [sum(len(p.state[key]) for p in ps) / len(ps) for ps in levels]
            self.assertEqual(sizes, sorted(sizes))
            self.assertEqual(len(set(sizes)), 5)
            for ps in levels:
                self.assertEqual(len({json.dumps(p.state, sort_keys=True) for p in ps}), len(ps))
        self.assertTrue(any(p.data["spec"]["kind"] in ("schedule", "rule", "clarify")
                            for p in samples(instruction_execution, 4)))
        self.assertTrue(any(s["requires_any"] for p in samples(goal_directed_planning, 4)
                            for s in p.state["steps"]))
        self.assertTrue(any(e["kind"] == "control" for p in samples(event_triggered_execution, 4)
                            for e in p.state["events"]))
        self.assertTrue(any(p.state["via"] for p in samples(graph_navigation, 4)))

    def test_split_conversion(self):
        try:
            from scripts.build_procedural_jev import build_task
            from tasksource.jev.procedural import jev_rows
        except ImportError:
            self.skipTest("dataset building dependencies not installed")
        for module in MODULES:
            task = module.__name__.rsplit(".", 1)[-1]
            ds = build_task(task, {"train": 30, "validation": 5, "test": 5}, list(range(5)))
            self.assertFalse(set(ds["train"]["state"]) & set(ds["test"]["state"]))
            for example in ds["train"].select(range(3)):
                self.assertEqual(set(json.loads(example["questions"])), set(json.loads(example["answers"])))
                rows = jev_rows(example, f"procedural-typed-decisions/{task}", "train", 0)
                self.assertTrue(all(len(row["options"]) == len(row["target"]) or
                                    row["kind"] == "noul" and len(row["target"]) == 1 for row in rows))


if __name__ == "__main__":
    unittest.main()
