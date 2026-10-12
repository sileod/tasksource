"""Compute directed shortest paths with closures, weights, and waypoints.

Levels grow the graph and introduce unequal costs, closed edges, optional
mandatory waypoints, and near-optimal alternatives.
"""

from heapq import heappop, heappush

from ._common import Problem, choice_answer, noul_answer

SIZES = [4, 6, 9, 13, 18]
SEMANTICS = ("Edges are directed and can be traversed only when open=true. Each traversal adds its positive "
             "integer cost. A route must go from start to finish; if via is not null, the route must visit via "
             "at least once before finishing. Repeated nodes are allowed. Minimize total cost. When multiple "
             "cheapest routes exist, choose the lexicographically smallest first hop. An edge is essential "
             "when closing that edge would make the cheapest valid route more expensive or unreachable.")


def _shortest(state, removed=None, origin=None, visited_via=False):
    start = state["start"] if origin is None else origin
    via = state["via"]
    goal = state["finish"]
    graph = {}
    for e in state["edges"]:
        if e["open"] and e["id"] != removed:
            graph.setdefault(e["from"], []).append((e["to"], e["cost"]))
    initial_seen = visited_via or via is None or start == via
    queue = [(0, start, initial_seen)]
    best = {(start, initial_seen): 0}
    while queue:
        cost, node, seen = heappop(queue)
        if cost != best[node, seen]:
            continue
        if node == goal and seen:
            return cost
        for nxt, weight in graph.get(node, []):
            key = (nxt, seen or nxt == via)
            candidate = cost + weight
            if candidate < best.get(key, float("inf")):
                best[key] = candidate
                heappush(queue, (candidate, *key))
    return None


def _first_hop(state, optimum):
    if optimum is None or state["start"] == state["finish"] and state["via"] is None:
        return "none"
    options = []
    for edge in state["edges"]:
        if edge["open"] and edge["from"] == state["start"]:
            remaining = _shortest(state, origin=edge["to"],
                                  visited_via=(state["via"] is None or state["start"] == state["via"]))
            if remaining is not None and edge["cost"] + remaining == optimum:
                options.append(edge["to"])
    return min(options) if options else "none"


def generate(rng, level=0):
    n = SIZES[level] + (rng.randrange(2) if level else 0)
    nodes = sorted(rng.sample([f"N{i+1:02d}" for i in range(70)], n))
    edges = {}

    def add(a, b):
        if a != b and (a, b) not in edges:
            edges[a, b] = {"id": f"E{len(edges)+1}", "from": a, "to": b,
                           "cost": rng.randint(1, 9) if level >= 2 else 1, "open": True}

    backbone = rng.sample(nodes[1:-1], min(len(nodes) - 2, [1, 2, 3, 4, 5][level]))
    path = [nodes[0], *backbone, nodes[-1]]
    for a, b in zip(path, path[1:]):
        add(a, b)
    n_edges = [5, 10, 19, 32, 55][level]
    while len(edges) < n_edges:
        a, b = rng.sample(nodes, 2)
        add(a, b)
    for edge in edges.values():
        if level >= 2 and rng.random() < [0, 0, 0.16, 0.22, 0.28][level]:
            edge["open"] = False
    if level >= 2 and rng.random() < 0.16:
        for e in edges.values():
            if e["to"] == nodes[-1]:
                e["open"] = False
    via = rng.choice(nodes[1:-1]) if level >= 3 and rng.random() < 0.65 else None
    state = {"nodes": nodes, "edges": list(edges.values()), "start": nodes[0], "finish": nodes[-1],
             "via": via, "semantics": SEMANTICS}
    rng.shuffle(state["edges"])
    cost = _shortest(state)
    first_hop = _first_hop(state, cost)
    along = [e for e in state["edges"] if e["open"] and e["from"] == state["start"] and e["to"] == first_hop]
    probe = rng.choice(along) if along and rng.random() < 0.65 else rng.choice(state["edges"])
    without = _shortest(state, removed=probe["id"])
    essential = cost is not None and (without is None or without > cost)
    budget = max(0, (cost if cost is not None else rng.randint(2, 15)) + rng.choice([-3, -2, -1, 0, 1, 2, 3]))
    if cost is None:
        possible_costs = ["unreachable", *map(str, range(1, 7))]
        answer = "unreachable"
    else:
        answer = str(cost)
        near = [max(0, cost + d) for d in (-5, -3, -2, -1, 1, 2, 3, 5, 8)]
        possible_costs = [str(cost), *[str(x) for x in near if x != cost]]
        possible_costs += ["unreachable"]
    possible_costs = list(dict.fromkeys(possible_costs))
    rng.shuffle(possible_costs)
    possible_costs = possible_costs[:min(8, len(possible_costs))]
    if answer not in possible_costs:
        possible_costs[-1] = answer
    hop_options = nodes[1:] + ["none"]
    questions = {
        "reachable": {"type": "noul", "instructions": "Does any valid route from start to finish exist?"},
        "shortest_cost": {"type": "choice", "instructions": "What is the minimum total route cost? Choose unreachable if no valid route exists.",
                          "criteria": {c: c for c in possible_costs}},
        "first_hop": {"type": "choice", "instructions": "What is the lexicographically smallest first hop among minimum-cost routes? Choose none if unreachable.",
                      "criteria": {c: c for c in hop_options}},
        "within_budget": {"type": "noul", "instructions": f"Does a valid route exist with total cost at most {budget}?"},
        "edge_essential": {"type": "noul", "instructions": f"Is edge {probe['id']} essential to achieving the minimum route cost?"},
    }
    answers = {
        "reachable": noul_answer(cost is not None),
        "shortest_cost": choice_answer(answer, possible_costs),
        "first_hop": choice_answer(first_hop, hop_options),
        "within_budget": noul_answer(cost is not None and cost <= budget),
        "edge_essential": noul_answer(essential),
    }
    return Problem(state, questions, answers, {"cost": cost, "probe": probe["id"], "budget": budget})
