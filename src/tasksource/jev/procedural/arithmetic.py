"""Exact arithmetic over an order, an account ledger, or a schedule.

Numeric answers are choices among the gold value and typical slips (a skipped
line, a sign error, an hour carry, a rule applied the wrong way). Each state
asks a random subset of its questions, in random order, including graded
probabilities (k out of n) and descriptive ordinal scores.
"""

import json

from ._common import OBJECTS, Problem, choice_answer, noul_answer, phrase, render_records, score_answer, sround

COUNTS = [str(i) for i in range(11)]
CURRENCIES = ["$", "€", "£"]
TASK_NAMES = ["email triage", "code review", "standup", "design sync", "report writing",
              "client call", "inventory check", "backup check", "planning", "interviews"]
SPEND = ["At most half of the budget.", "More than half of the budget, but within it.", "Over the budget."]
DRIFT = ["Fell by more than 50.", "Changed by 50 or less.", "Rose by more than 50."]


def _numeric_choice(rng, gold, slips, fmt, offsets=(1, 2, 5, 10, 20)):
    """Gold plus four distinct wrong values, typical slips first, then near misses.

    Gold's rank among the five is uniform, so sorted options do not reveal it.
    """
    near = [gold + sign * o for o in offsets for sign in (-1, 1)]
    below, above = [], []
    for v in rng.sample(slips, len(slips)) + rng.sample(near, len(near)):
        if v >= 0 and v != gold and v not in below + above:
            (below if v < gold else above).append(v)
    rank = rng.choice([r for r in range(5) if r <= len(below) and 4 - r <= len(above)])
    values = sorted(below[:rank] + [gold] + above[:4 - rank])
    if rng.random() < 0.5:
        values = rng.sample(values, len(values))
    options = [fmt(v) for v in values]
    return {o: o for o in options}, fmt(gold)


def _probability(k, n):
    return {"type": "noul", "noul": round(k / n, 6)}


def _near(rng, value, step):
    """A threshold just above or below ``value``, each half the time."""
    return value + rng.choice([-1, 1]) * step * rng.randint(1, 3)


def _order(rng, level):
    n = min(len(OBJECTS), sround(2 + level * 1.2 + rng.random(), rng))
    lines = [{"item": item, "unit_price": rng.randint(2, 20 + 15 * level), "quantity": rng.randint(1, 3 + level)}
             for item in rng.sample(OBJECTS, n)]
    cost = lambda line: line["unit_price"] * line["quantity"]
    top = max(lines, key=cost)
    while sum(cost(line) == cost(top) for line in lines) > 1:
        top["quantity"] += 1
    total = sum(map(cost, lines))
    threshold = max(20, 5 * round(_near(rng, total, 10) / 5))
    discount, shipping = 5 * rng.randint(1, min(4, threshold // 20)), rng.choice([4, 6, 8])
    due = total - discount if total >= threshold else total + shipping
    wrong_rule = total + shipping if total >= threshold else total - discount
    slips = [total, wrong_rule, due - cost(lines[-1]), due - cost(top) + top["unit_price"], due + 10, due - 10]

    sym = rng.choice(CURRENCIES)
    money = lambda v: f"{sym}{v}"
    rule = f"Orders of {money(threshold)} or more get {money(discount)} off; smaller orders pay {money(shipping)} shipping."
    style = rng.choice(["json", "table", "csv", "lines", "prose"])
    if style == "prose":
        body = " ".join(phrase(rng, ["{q} x {i} at {p} each.", "{i}: {q} units at {p} per unit."],
                               q=l["quantity"], i=l["item"], p=money(l["unit_price"])) for l in lines)
        state = f"Order: {body}\nRule: {rule}"
    elif style == "json":
        state = json.dumps({"currency": sym, "lines": lines, "rule": rule}, ensure_ascii=False)
    else:
        state = f"Order lines (prices in {sym}):\n{render_records(lines, style)}\n\nRule: {rule}"

    criteria, gold = _numeric_choice(rng, due, slips, money)
    budget = max(1, _near(rng, due, 5))
    line_cut = rng.choice(sorted({cost(l) for l in lines}))
    over = sum(cost(l) > line_cut for l in lines)
    quantities = sorted({l["quantity"] for l in lines})
    units = rng.choice(quantities[1:] or quantities)  # the minimum would make it certain
    bulk = sum(l["quantity"] >= units for l in lines)
    pool = {
        "amount_due": ({"type": "choice", "criteria": criteria, "instructions": phrase(rng, [
            "How much is due for this order after the rule?", "What is the final amount to pay?",
            "After applying the rule, what does the order cost?"])}, choice_answer(gold, list(criteria))),
        "largest_line": ({"type": "choice", "criteria": {l["item"]: l["item"] for l in lines},
                          "instructions": phrase(rng, ["Which line costs the most in total?",
                                                       "Which item accounts for the largest share of the bill?"])},
                         choice_answer(top["item"], [l["item"] for l in lines])),
        "lines_above": ({"type": "score", "criteria": COUNTS, "instructions": phrase(rng, [
            "How many lines cost more than {c} in total?", "Count the lines whose total exceeds {c}."],
            c=money(line_cut))}, score_answer(over, COUNTS)),
        "random_line_bulk": ({"type": "noul", "instructions": phrase(rng, [
            "If one order line is picked uniformly at random, what is the probability that its quantity is at least {u}?",
            "What is the chance that a randomly chosen line orders {u} or more units?"], u=units)},
            _probability(bulk, n)),
    }
    if rng.random() < 0.5:
        pool["within_budget"] = ({"type": "noul", "instructions": phrase(rng, [
            "Is the amount due at most {b}?", "Does the order fit within a budget of {b}?"], b=money(budget))},
            noul_answer(due <= budget))
    else:
        budget = rng.choice([due * 2 + rng.randint(0, 5), due + rng.randint(1, 20), max(1, due - rng.randint(1, 20))])
        level_index = 2 if due > budget else 0 if 2 * due <= budget else 1
        pool["budget_use"] = ({"type": "score", "criteria": SPEND, "instructions": phrase(rng, [
            "How does the amount due compare with a budget of {b}?",
            "Against a {b} budget, how much does this order use?"], b=money(budget))},
            score_answer(level_index, SPEND))
    data = {"scenario": "order", "lines": lines, "threshold": threshold, "discount": discount,
            "shipping": shipping, "due": due, "budget": budget, "line_cut": line_cut, "units": units}
    return state, pool, data


def _ledger(rng, level):
    n = min(10, sround(3 + 1.6 * level, rng))
    transactions = [{"day": day, "type": rng.choice(["deposit", "withdrawal"]),
                     "amount": 5 * rng.randint(1, 10 + 10 * level)}
                    for day in sorted(rng.sample(range(1, 29), n))]
    signed = [t["amount"] if t["type"] == "deposit" else -t["amount"] for t in transactions]
    running = []
    for delta in signed:
        running.append((running[-1] if running else 0) + delta)
    lowest = min(running)
    start = max(0, 5 * round((-lowest + rng.choice([-1, 1]) * 5 * rng.randint(1, 6)) / 5))
    balances = [start + r for r in running]
    final = balances[-1]
    flipped = rng.randrange(n)
    slips = [final - 2 * signed[flipped], final - signed[flipped], final - start, final + 10, final - 10, start - sum(signed)]

    sym = rng.choice(CURRENCIES)
    money = lambda v: f"{sym}{v}"
    style = rng.choice(["json", "table", "csv", "lines", "prose"])
    if style == "prose":
        body = " ".join(phrase(rng, ["On day {d}, a {t} of {a}.", "Day {d}: {t} of {a}."],
                               d=t["day"], t=t["type"], a=money(t["amount"])) for t in transactions)
        state = f"The account opens the month at {money(start)}. {body}"
    elif style == "json":
        state = json.dumps({"opening_balance": start, "currency": sym, "transactions": transactions},
                           ensure_ascii=False)
    else:
        state = f"Opening balance: {money(start)}\n{render_records(transactions, style)}"

    fmt = lambda v: f"-{sym}{-v}" if v < 0 else money(v)
    criteria, gold = _numeric_choice(rng, final, slips, fmt) if final >= 0 else (None, None)
    days = [f"day {t['day']}" for t in transactions]
    low_days = [d for d, b in zip(days, balances) if b == min(balances)]
    deposits = sum(t["type"] == "deposit" for t in transactions)
    change = final - start
    pool = {
        "went_negative": ({"type": "noul", "instructions": phrase(rng, [
            "Does the balance ever drop below zero?", "Is the account overdrawn at any point?"])},
            noul_answer(min(balances) < 0)),
        "withdrawal_count": ({"type": "score", "criteria": COUNTS, "instructions": phrase(rng, [
            "How many withdrawals are there?", "Count the withdrawals."])}, score_answer(n - deposits, COUNTS)),
        "random_is_deposit": ({"type": "noul", "instructions": phrase(rng, [
            "If one transaction is picked uniformly at random, what is the probability that it is a deposit?",
            "What is the chance that a randomly chosen transaction is a deposit?"])}, _probability(deposits, n)),
        "net_change": ({"type": "score", "criteria": DRIFT, "instructions": phrase(rng, [
            "From opening to closing, how did the balance change?",
            "Compare the final balance with the opening balance."])},
            score_answer(0 if change < -50 else 2 if change > 50 else 1, DRIFT)),
    }
    if criteria:
        pool["final_balance"] = ({"type": "choice", "criteria": criteria, "instructions": phrase(rng, [
            "What is the balance after the last transaction?", "What is the closing balance?"])},
            choice_answer(gold, list(criteria)))
    if len(low_days) == 1:
        pool["lowest_day"] = ({"type": "choice", "criteria": {d: d for d in days}, "instructions": phrase(rng, [
            "After which day's transaction is the balance lowest?", "On which day does the balance reach its minimum?"])},
            choice_answer(low_days[0], days))
    data = {"scenario": "ledger", "start": start, "transactions": transactions, "balances": balances}
    return state, pool, data


def _clock(minutes):
    return f"{minutes // 60 % 24:02d}:{minutes % 60:02d}"


def _duration(minutes, rng):
    if minutes >= 60 and rng.random() < 0.5:
        hours, rest = divmod(minutes, 60)
        return f"{hours} h {rest} min" if rest else f"{hours} h"
    return f"{minutes} min"


def _schedule(rng, level):
    n = sround(2 + level + rng.random(), rng)
    tasks = [{"task": name, "minutes": 5 * rng.randint(2, 9 + 3 * level)} for name in rng.sample(TASK_NAMES, n)]
    top = max(tasks, key=lambda t: t["minutes"])
    while sum(t["minutes"] == top["minutes"] for t in tasks) > 1:
        top["minutes"] += 5
    start = 5 * rng.randint(7 * 12, 13 * 12)
    gap = rng.choice([0, 0, 5, 10, 15]) if level >= 1 else 0
    starts, clock = [], start
    for t in tasks:
        starts.append(clock)
        clock += t["minutes"] + gap
    finish = clock - gap
    slips = [finish + gap, finish - (n - 1) * gap, finish - tasks[-1]["minutes"], finish + 60, finish - 60, finish + 10]

    rule = f"Tasks run back to back in this order{f', with a {gap}-minute break between tasks' if gap else ''}."
    rows = [{"task": t["task"], "duration": _duration(t["minutes"], rng)} for t in tasks]
    style = rng.choice(["table", "csv", "lines", "prose"])
    if style == "prose":
        body = ", then ".join(f"{r['task']} ({r['duration']})" for r in rows)
        state = f"Starting at {_clock(start)}: {body}. {rule}"
    else:
        state = f"Start: {_clock(start)}\n{render_records(rows, style)}\n\n{rule}"

    criteria, gold = _numeric_choice(rng, finish, slips, _clock, offsets=(5, 10, 15, 30))
    deadline = _near(rng, finish, 5)
    noon = sum(s < 12 * 60 for s in starts)
    long = rng.choice([20, 30, 45])
    pool = {
        "finish_time": ({"type": "choice", "criteria": criteria, "instructions": phrase(rng, [
            "At what time does the last task end?", "When is everything finished?"])},
            choice_answer(gold, list(criteria))),
        "done_by_deadline": ({"type": "noul", "instructions": phrase(rng, [
            "Is everything done by {t}?", "Does the last task end at or before {t}?"], t=_clock(deadline))},
            noul_answer(finish <= deadline)),
        "longest_task": ({"type": "choice", "criteria": {t["task"]: t["task"] for t in tasks},
                          "instructions": phrase(rng, ["Which task takes the longest?", "Which task is the longest?"])},
                         choice_answer(top["task"], [t["task"] for t in tasks])),
        "starts_before_noon": ({"type": "score", "criteria": COUNTS, "instructions": phrase(rng, [
            "How many tasks start before 12:00?", "Count the tasks that begin before noon."])},
            score_answer(noon, COUNTS)),
        "random_is_long": ({"type": "noul", "instructions": phrase(rng, [
            "If one task is picked uniformly at random, what is the probability that it lasts more than {m} minutes?",
            "What is the chance that a randomly chosen task takes longer than {m} minutes?"], m=long)},
            _probability(sum(t["minutes"] > long for t in tasks), n)),
    }
    data = {"scenario": "schedule", "start": start, "tasks": tasks, "gap": gap, "finish": finish,
            "deadline": deadline, "long": long}
    return state, pool, data


def generate(rng, level=0):
    state, pool, data = rng.choice([_order, _ledger, _schedule])(rng, level)
    k = min(len(pool), rng.choice([2, 3, 3, 4, 4, 5]))
    chosen = rng.sample(sorted(pool), k)
    questions = {qid: pool[qid][0] for qid in chosen}
    answers = {qid: pool[qid][1] for qid in chosen}
    return Problem(state, questions, answers, data)
