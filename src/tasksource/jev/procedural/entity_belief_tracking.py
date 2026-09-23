"""Track world state and agent-specific beliefs across partially witnessed object moves."""

from ._common import Problem, choice_answer, noul_answer, sround

AGENTS = ["alice", "bob", "carol"]
LOCATIONS = ["desk", "locker", "archive", "lab"]


def generate(rng, level=0):
    n_events = sround(5 + 1.4 * level, rng)
    n_objects = sround(2 + 0.35 * level, rng)
    objects = [f"item_{i+1}" for i in range(max(1, n_objects))]
    world = {obj: rng.choice(LOCATIONS) for obj in objects}
    beliefs = {agent: dict(world) for agent in AGENTS}
    initial = dict(world)
    events = []
    for i in range(max(2, n_events)):
        obj = rng.choice(objects)
        dest = rng.choice([x for x in LOCATIONS if x != world[obj]])
        witnesses = rng.sample(AGENTS, rng.randint(0, len(AGENTS)))
        world[obj] = dest
        for agent in witnesses:
            beliefs[agent][obj] = dest
        events.append({"step": i + 1, "object": obj, "destination": dest, "witnesses": witnesses})
    target_obj = rng.choice(objects)
    target_agent = rng.choice(AGENTS)
    state = {
        "initial_locations": initial,
        "events": events,
        "belief_rule": "A move always changes the true location. An agent updates that object's believed location only when listed as a witness; otherwise the agent keeps its previous belief.",
        "query_agent": target_agent,
        "query_object": target_obj,
    }
    questions = {
        "world_location": {"type": "choice", "instructions": "Where is query_object actually located after all events?", "criteria": {x: x for x in LOCATIONS}},
        "agent_belief_location": {"type": "choice", "instructions": "Where does query_agent believe query_object is after all events?", "criteria": {x: x for x in LOCATIONS}},
        "belief_matches_world": {"type": "noul", "instructions": "Does query_agent's final belief about query_object match the true final location?"},
    }
    actual = world[target_obj]
    believed = beliefs[target_agent][target_obj]
    answers = {
        "world_location": choice_answer(actual, LOCATIONS),
        "agent_belief_location": choice_answer(believed, LOCATIONS),
        "belief_matches_world": noul_answer(actual == believed),
    }
    return Problem(state, questions, answers)
