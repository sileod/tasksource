"""Track world state and agent-specific beliefs across partially witnessed object moves.

Levels add events, objects, and agents; from level 2 a second-order question asks
where one agent thinks another agent believes an object is.
"""

from ._common import Problem, choice_answer, noul_answer, sround

AGENTS = ["alice", "bob", "carol", "dave"]
LOCATIONS = ["desk", "locker", "archive", "lab"]
MORE_LOCATIONS = LOCATIONS + ["shelf", "drawer", "cabinet", "garage", "attic", "basement", "kitchen", "office",
                              "mailroom", "vault", "studio", "workshop"]
N_EVENTS = [3, 5, 8, 11, 15]
N_OBJECTS = [1, 2, 2, 3, 3]
N_AGENTS = [2, 3, 3, 4, 4]
RULE = ("Everyone starts knowing the initial locations. A move always changes the true location. The witnesses of "
        "a move see it and see who else witnessed it. An agent's belief about an object changes only when they "
        "witness a move of it. What an agent thinks another agent believes changes only when the first agent "
        "witnesses a move: it becomes the destination if the other agent witnessed that move too, and stays as it "
        "was otherwise.")


def generate(rng, level=0):
    agents = AGENTS[:N_AGENTS[level]]
    locations = rng.sample(MORE_LOCATIONS, rng.randint(len(LOCATIONS), len(MORE_LOCATIONS)))
    objects = [f"item_{i+1}" for i in range(N_OBJECTS[level])]
    world = {obj: rng.choice(locations) for obj in objects}
    initial = dict(world)
    beliefs = {agent: dict(world) for agent in agents}
    nested = {(a, b): dict(world) for a in agents for b in agents if a != b}  # what a thinks b believes
    events = []
    for i in range(max(2, sround(N_EVENTS[level] * rng.uniform(0.85, 1.15), rng))):
        obj = rng.choice(objects)
        dest = rng.choice([x for x in locations if x != world[obj]])
        witnesses = sorted(rng.sample(agents, rng.randint(0, len(agents))), key=agents.index)
        world[obj] = dest
        for a in witnesses:
            beliefs[a][obj] = dest
            for b in witnesses:
                if b != a:
                    nested[a, b][obj] = dest
        events.append({"step": i + 1, "object": obj, "destination": dest, "witnesses": witnesses})
    target_obj = rng.choice(objects)
    target_agent = rng.choice(agents)
    other = rng.choice([a for a in agents if a != target_agent])
    state = {"locations": locations, "initial_locations": initial, "events": events, "belief_rule": RULE}
    criteria = {x: x for x in locations}
    questions = {
        "world_location": {"type": "choice", "instructions": f"Where is {target_obj} actually located after all events?", "criteria": criteria},
        "agent_belief_location": {"type": "choice", "instructions": f"Where does {target_agent} believe {target_obj} is after all events?", "criteria": criteria},
        "belief_matches_world": {"type": "noul", "instructions": f"After all events, does {target_agent} believe {target_obj} is where it actually is?"},
    }
    actual, believed = world[target_obj], beliefs[target_agent][target_obj]
    answers = {
        "world_location": choice_answer(actual, locations),
        "agent_belief_location": choice_answer(believed, locations),
        "belief_matches_world": noul_answer(actual == believed),
    }
    if level >= 2:
        questions["nested_belief_location"] = {"type": "choice", "criteria": criteria, "instructions":
            f"After all events, where does {target_agent} think {other} believes {target_obj} is?"}
        answers["nested_belief_location"] = choice_answer(nested[target_agent, other][target_obj], locations)
    data = {"initial": initial, "events": events, "object": target_obj, "agent": target_agent, "other": other}
    return Problem(state, questions, answers, data)
