"""Filter one table by two conditions, join it with a second, and compare years."""

import json

from ._common import CITIES, PEOPLE, Problem, choice_answer, noul_answer, phrase, render_records, score_answer, sround

SIZES = [6, 10, 16, 24, 36]
TEAMS = ["billing", "search", "mobile", "security", "data", "support"]
COUNTS = [str(i) for i in range(13)]


def generate(rng, level=0):
    n = min(len(PEOPLE) - len(TEAMS), max(4, sround(SIZES[level] * rng.uniform(0.8, 1.2), rng)))
    cities = rng.sample(CITIES, 4)
    people = [{"name": name, "team": rng.choice(TEAMS), "city": rng.choice(cities),
               "start_year": rng.randint(2008, 2024)} for name in rng.sample(PEOPLE, n)]
    managers = rng.sample([p for p in PEOPLE if p not in {q["name"] for q in people}], len(TEAMS))
    teams = [{"team": team, "manager": manager} for team, manager in zip(TEAMS, managers)]

    person = rng.choice(people)
    for other in people:  # make the (team, city) pair unique to the chosen person
        if other is not person and (other["team"], other["city"]) == (person["team"], person["city"]):
            other["city"] = rng.choice([c for c in cities if c != person["city"]])
    near = [p["name"] for p in people if p is not person and (p["team"] == person["team"] or p["city"] == person["city"])]
    rest = [p["name"] for p in people if p is not person and p["name"] not in near]
    options = sorted([person["name"], *(rng.sample(near, len(near)) + rest)[:5]], key=lambda _: rng.random())

    subject = rng.choice(people)
    manager = {t["team"]: t["manager"] for t in teams}[subject["team"]]
    year = subject["start_year"] + rng.choice([-2, -1, 1, 2])
    while True:  # exact counts only: redraw rather than clip to the scale
        city, since = rng.choice(cities), rng.randint(2010, 2022)
        matches = sum(p["city"] == city and p["start_year"] >= since for p in people)
        if matches < len(COUNTS):
            break
    listed = rng.sample(managers, len(managers))

    style = rng.choice(["json", "table", "csv"])
    if style == "json":
        state = json.dumps({"people": people, "teams": teams}, ensure_ascii=False)
    else:
        state = f"people:\n{render_records(people, style)}\n\nteams:\n{render_records(teams, style)}"
    questions = {
        "find_person": {"type": "choice", "criteria": {o: o for o in options}, "instructions": phrase(rng, [
            "Who is on the {t} team and based in {c}?", "Which person works in {c} on the {t} team?"],
            t=person["team"], c=person["city"])},
        "manager_of": {"type": "choice", "criteria": {m: m for m in listed}, "instructions": phrase(rng, [
            "Who manages the team that {p} belongs to?", "Who is the manager of {p}'s team?"], p=subject["name"])},
        "started_before": {"type": "noul", "instructions": phrase(rng, [
            "Did {p} start before {y}?", "Is {p}'s start year earlier than {y}?"], p=subject["name"], y=year)},
        "count_matching": {"type": "score", "criteria": COUNTS, "instructions": phrase(rng, [
            "How many people based in {c} started in {y} or later?",
            "Count the people in {c} whose start year is {y} or later."], c=city, y=since)},
    }
    answers = {
        "find_person": choice_answer(person["name"], options),
        "manager_of": choice_answer(manager, listed),
        "started_before": noul_answer(subject["start_year"] < year),
        "count_matching": score_answer(matches, COUNTS),
    }
    data = {"people": people, "teams": teams, "person": person["name"], "subject": subject["name"],
            "year": year, "city": city, "since": since}
    return Problem(state, questions, answers, data)
