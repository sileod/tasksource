"""Track world state and agent-specific beliefs across partially witnessed object moves."""

from ._common import Problem, choice_answer, noul_answer, sround

AGENTS = ["alice", "bob", "carol"]
LOCATIONS = ["desk", "locker", "archive", "lab"]
MORE_LOCATIONS = LOCATIONS + ["shelf", "drawer", "cabinet", "garage", "attic", "basement", "kitchen", "office",
                              "mailroom", "vault", "studio", "workshop"]
LONG_LIST_SHARE = 0.3


def generate(rng, level=0):
    n_events = sround(5 + 1.4 * level, rng)
    n_objects = sround(2 + 0.35 * level, rng)
    objects = [f"item_{i+1}" for i in range(max(1, n_objects))]
    locations = rng.sample(MORE_LOCATIONS, rng.randint(8, len(MORE_LOCATIONS))) if rng.random() < LONG_LIST_SHARE else LOCATIONS
    world = {obj: rng.choice(locations) for obj in objects}
    beliefs = {agent: dict(world) for agent in AGENTS}
    initial = dict(world)
    events = []
    for i in range(max(2, n_events)):
        obj = rng.choice(objects)
        dest = rng.choice([x for x in locations if x != world[obj]])
        witnesses = rng.sample(AGENTS, rng.randint(0, len(AGENTS)))
        world[obj] = dest
        for agent in witnesses:
            beliefs[agent][obj] = dest
        events.append({"step": i + 1, "object": obj, "destination": dest, "witnesses": witnesses})
    target_obj = rng.choice(objects)
    target_agent = rng.choice(AGENTS)
    state = {
        "locations": locations,
        "initial_locations": initial,
        "events": events,
        "belief_rule": "A move always changes the true location. An agent updates that object's believed location only when listed as a witness; otherwise the agent keeps its previous belief.",
    }
    questions = {
        "world_location": {"type": "choice", "instructions": f"Where is {target_obj} actually located after all events?", "criteria": {x: x for x in locations}},
        "agent_belief_location": {"type": "choice", "instructions": f"Where does {target_agent} believe {target_obj} is after all events?", "criteria": {x: x for x in locations}},
        "belief_matches_world": {"type": "noul", "instructions": f"After all events, does {target_agent} believe {target_obj} is where it actually is?"},
    }
    actual = world[target_obj]
    believed = beliefs[target_agent][target_obj]
    answers = {
        "world_location": choice_answer(actual, locations),
        "agent_belief_location": choice_answer(believed, locations),
        "belief_matches_world": noul_answer(actual == believed),
    }
    return Problem(state, questions, answers)
