"""Find one record among many whose ids differ from the target by one or two digits."""

from ._common import CITIES, STYLES, Problem, choice_answer, noul_answer, phrase, render_records, sround

SIZES = [8, 20, 50, 120, 250]
DOMAINS = [("locker", "city"), ("shipment", "destination"), ("badge", "office"), ("account", "branch")]


def _near(key, rng):
    """A different id sharing most digits with ``key``: a swap or a one-digit change."""
    digits = list(key)
    i, j = rng.sample(range(4), 2)
    if digits[i] != digits[j] and rng.random() < 0.5:
        digits[i], digits[j] = digits[j], digits[i]
    else:
        digits[i] = rng.choice([d for d in "0123456789" if d != digits[i]])
    return "".join(digits)


def generate(rng, level=0):
    noun, field = rng.choice(DOMAINS)
    n = max(4, sround(SIZES[level] * rng.uniform(0.7, 1.3), rng))
    target = f"{rng.randrange(10_000):04d}"
    keys = {target}
    while len(keys) < min(n, 4):  # guaranteed near misses of the target
        keys.add(_near(target, rng))
    while len(keys) < n:
        keys.add(_near(target, rng) if rng.random() < 0.2 else f"{rng.randrange(10_000):04d}")
    keys = sorted(keys, key=lambda _: rng.random())
    records = [{"id": f"{noun[0].upper()}-{key}", field: rng.choice(CITIES)} for key in keys]
    index = keys.index(target)
    gold = records[index]
    near = [r for r in records if r is not gold and sum(a != b for a, b in zip(r["id"], gold["id"])) <= 2]

    others = list(dict.fromkeys(r[field] for r in near if r[field] != gold[field]))
    others += [c for c in rng.sample(CITIES, len(CITIES)) if c != gold[field] and c not in others]
    options = sorted([gold[field], *others[:5]], key=lambda _: rng.random())

    proposed = gold[field] if rng.random() < 0.5 else others[0]
    listed = {r["id"] for r in records}
    if rng.random() < 0.5:
        probe = rng.choice(near)["id"]
    else:
        probe = gold["id"]
        while probe in listed:
            probe = f"{noun[0].upper()}-{_near(target, rng)}"

    style = rng.choice(STYLES + ("prose",))
    if style == "prose":
        state = " ".join(f"The {noun} {r['id']} has {field} {r[field]}." for r in records)
    else:
        state = render_records(records, style)
    key = gold["id"]
    questions = {
        "value_of_id": {"type": "choice", "criteria": {c: c for c in options}, "instructions": phrase(rng, [
            "Which {field} is listed for {noun} {key}?", "What is the {field} of {noun} {key}?",
            "Look up {noun} {key}. Which {field} does it have?"], field=field, noun=noun, key=key)},
        "id_has_value": {"type": "noul", "instructions": phrase(rng, [
            "Is {noun} {key} listed with {field} {value}?", "Does {noun} {key} have {field} {value}?"],
            field=field, noun=noun, key=key, value=proposed)},
        "id_listed": {"type": "noul", "instructions": phrase(rng, [
            "Is there a {noun} with id {probe}?", "Does the list include {noun} {probe}?"],
            noun=noun, probe=probe)},
    }
    answers = {
        "value_of_id": choice_answer(gold[field], options),
        "id_has_value": noul_answer(proposed == gold[field]),
        "id_listed": noul_answer(probe in listed),
    }
    data = {"records": records, "field": field, "key": key, "proposed": proposed, "probe": probe}
    return Problem(state, questions, answers, data)
