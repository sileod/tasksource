"""Find one record among many whose ids differ from the target by one or two digits.

From level 2 the question names an original id that was reissued one to three
times; the chain must be followed to the listed id, and near-miss ids have
reissue chains of their own.
"""

from ._common import CITIES, STYLES, Problem, choice_answer, noul_answer, phrase, render_records, sround

SIZES = [8, 20, 50, 120, 250]
HOPS = [0, 0, 1, 2, 3]
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
    keys = sorted(sorted(keys), key=lambda _: rng.random())  # sort first: set order depends on the hash seed
    records = [{"id": f"{noun[0].upper()}-{key}", field: rng.choice(CITIES)} for key in keys]
    index = keys.index(target)
    gold = records[index]
    near = [r for r in records if r is not gold and sum(a != b for a, b in zip(r["id"], gold["id"])) <= 2]

    others = list(dict.fromkeys(r[field] for r in near if r[field] != gold[field]))
    others += [c for c in rng.sample(CITIES, len(CITIES)) if c != gold[field] and c not in others]
    n_options = rng.randint(6, len(CITIES))  # varied option counts, 6 to 20
    options = sorted([gold[field], *others[:n_options - 1]], key=lambda _: rng.random())

    proposed = gold[field] if rng.random() < 0.5 else others[0]
    listed = {r["id"] for r in records}
    if rng.random() < 0.5:
        probe = rng.choice(near)["id"]
    else:
        probe = gold["id"]
        while probe in listed:
            probe = f"{noun[0].upper()}-{_near(target, rng)}"

    prefix = f"{noun[0].upper()}-"
    used = set(keys) | {probe[len(prefix):]}  # the id_listed probe must not turn out to be a reissued id

    def fresh(near_to):
        while True:
            candidate = _near(near_to, rng) if rng.random() < 0.7 else f"{rng.randrange(10_000):04d}"
            if candidate not in used:
                used.add(candidate)
                return candidate

    reissued = {}  # old id -> new id
    chain = [target]
    for _ in range(HOPS[level]):
        chain.insert(0, fresh(chain[0]))
        reissued[prefix + chain[0]] = prefix + chain[1]
    for _ in range(2 * HOPS[level]):  # near-miss chains that end at other records
        end = rng.choice([k for k in keys if k != target])
        start = fresh(chain[0])
        if rng.random() < 0.5:
            middle = fresh(start)
            reissued[prefix + start], reissued[prefix + middle] = prefix + middle, prefix + end
        else:
            reissued[prefix + start] = prefix + end

    style = rng.choice(STYLES + ("prose",))
    if style == "prose":
        state = " ".join(f"The {noun} {r['id']} has {field} {r[field]}." for r in records)
    else:
        state = render_records(records, style)
    if reissued:
        moves = sorted(reissued.items(), key=lambda _: rng.random())
        state += (f"\n\nReissued {noun} ids (a reissued id is no longer listed; look up its new id):\n"
                  + "\n".join(f"{old} -> {new}" for old, new in moves))
    key = prefix + chain[0]
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
    data = {"records": records, "field": field, "key": key, "reissued": reissued, "proposed": proposed, "probe": probe}
    return Problem(state, questions, answers, data)
