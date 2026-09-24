"""Count, compare, and sum over an inventory; every answer is unique by construction."""

from ._common import COLORS, OBJECTS, STYLES, Problem, choice_answer, noul_answer, phrase, render_records, score_answer, sround

SIZES = [5, 10, 18, 30, 45]
CATEGORIES = ["tools", "kitchen", "garden", "office", "outdoor"]
COUNTS = [str(i) for i in range(21)]


def generate(rng, level=0):
    n = max(3, sround(SIZES[level] * rng.uniform(0.8, 1.2), rng))
    names = rng.sample([f"{c} {o}" for c in COLORS for o in OBJECTS], n)
    items = [{"item": name, "category": rng.choice(CATEGORIES), "quantity": rng.randint(1, 20),
              "in_stock": rng.random() < 0.75} for name in names]

    sizes = {c: sum(i["category"] == c for i in items) for c in CATEGORIES}
    nonempty = [c for c in CATEGORIES if 1 <= sizes[c] <= 20]
    category = rng.choice(nonempty if nonempty and rng.random() < 0.9 else
                          [c for c in CATEGORIES if sizes[c] <= 20])
    members = [i for i in items if i["category"] == category]
    count = len(members)
    if members:  # decide the stock answer first so it is balanced at every size
        out_of_stock = rng.random() < 0.5
        for i in members:
            i["in_stock"] = True
        if out_of_stock:
            for i in rng.sample(members, rng.randint(1, max(1, len(members) // 2))):
                i["in_stock"] = False

    # Largest quantity within the category when it has several items, else overall;
    # the leader is bumped until strictly largest so the answer is unique.
    pool = members if len(members) >= 2 else items
    scope = f"{category} items" if pool is members else "items"
    leader = max(pool, key=lambda i: i["quantity"])
    runner_up = max((i["quantity"] for i in pool if i is not leader), default=0)
    leader["quantity"] = max(leader["quantity"], runner_up + 1)
    rivals = sorted((i["item"] for i in pool if i is not leader), key=lambda _: rng.random())
    extra = [i["item"] for i in items if i["item"] not in rivals and i is not leader]
    options = sorted([leader["item"], *(rivals + extra)[:5]], key=lambda _: rng.random())

    total = sum(i["quantity"] for i in members)
    threshold = max(0, total + rng.choice([-3, -2, -1, 0, 1, 2]))

    questions = {
        "count_in_category": {"type": "score", "criteria": COUNTS, "instructions": phrase(rng, [
            "How many items are in the {c} category?", "Count the {c} items.",
            "How many listed items belong to {c}?"], c=category)},
        "largest_quantity": {"type": "choice", "criteria": {o: o for o in options}, "instructions": phrase(rng, [
            "Among the {s}, which has the largest quantity?", "Which of the {s} has the highest quantity?"],
            s=scope)},
        "any_out_of_stock": {"type": "noul", "instructions": phrase(rng, [
            "Is any {c} item out of stock?", "Is at least one {c} item not in stock?"], c=category)},
        "total_above": {"type": "noul", "instructions": phrase(rng, [
            "Is the total quantity of {c} items greater than {t}?",
            "Do the {c} items add up to more than {t} units?"], c=category, t=threshold)},
    }
    answers = {
        "count_in_category": score_answer(count, COUNTS),
        "largest_quantity": choice_answer(leader["item"], options),
        "any_out_of_stock": noul_answer(any(not i["in_stock"] for i in members)),
        "total_above": noul_answer(total > threshold),
    }
    data = {"items": items, "category": category, "scope_category": pool is members, "threshold": threshold}
    return Problem(render_records(items, rng.choice(STYLES)), questions, answers, data)
