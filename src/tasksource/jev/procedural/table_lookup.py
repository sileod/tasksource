"""Filter one table by two conditions, join it with a second, and compare years.

From level 2 the person is identified through their manager (a join), and from
level 3 also by start year, with same-team, same-city colleagues who started later.
"""

import json

from ._common import CITIES, PEOPLE, Problem, choice_answer, noul_answer, phrase, render_records, score_answer, sround

SIZES = [6, 10, 16, 24, 36]
TEAMS = ["billing", "search", "mobile", "security", "data", "support"]
COUNTS = [str(i) for i in range(10)]  # Jev scores have at most 10 levels


def generate(rng, level=0):
    n = min(len(PEOPLE) - len(TEAMS), max(4, sround(SIZES[level] * rng.uniform(0.8, 1.2), rng)))
    cities = rng.sample(CITIES, 4)
    people = [{"name": name, "team": rng.choice(TEAMS), "city": rng.choice(cities),
               "start_year": rng.randint(2008, 2024)} for name in rng.sample(PEOPLE, n)]
    managers = rng.sample([p for p in PEOPLE if p not in {q["name"] for q in people}], len(TEAMS))
    teams = [{"team": team, "manager": manager} for team, manager in zip(TEAMS, managers)]

    person = rng.choice(people)
    by_year = level >= 3
    cut = min(person["start_year"] + rng.randint(1, 3), 2025) if by_year else None
    peers = [p for p in people if p is not person and (p["team"], p["city"]) == (person["team"], person["city"])]
    if by_year:  # colleagues sharing team and city started too late to match
        for other in rng.sample([p for p in people if p is not person], min(2, len(people) - 1)):
            other.update(team=person["team"], city=person["city"])
        for other in people:
            if other is not person and (other["team"], other["city"]) == (person["team"], person["city"]):
                other["start_year"] = rng.randint(cut, cut + 4)
    else:  # make the (team, city) pair unique to the chosen person
        for other in peers:
            other["city"] = rng.choice([c for c in cities if c != person["city"]])
    near = [p["name"] for p in people if p is not person and (p["team"] == person["team"] or p["city"] == person["city"])]
    rest = [p["name"] for p in people if p is not person and p["name"] not in near]
    n_options = rng.randint(6, 40)  # varied option counts, capped by the table
    options = sorted([person["name"], *(rng.sample(near, len(near)) + rest)[:n_options - 1]], key=lambda _: rng.random())

    subject = rng.choice(people)
    manager = {t["team"]: t["manager"] for t in teams}[subject["team"]]
    year = subject["start_year"] + rng.choice([-2, -1, 1, 2])
    while True:  # exact counts only: redraw rather than clip to the scale
        city, since = rng.choice(cities), rng.randint(2010, 2022)
        matches = sum(p["city"] == city and p["start_year"] >= since for p in people)
        if matches < len(COUNTS):
            break
    listed = rng.sample(managers, len(managers))
    team_manager = {t["team"]: t["manager"] for t in teams}[person["team"]]

    style = rng.choice(["json", "table", "csv"])
    if style == "json":
        state = json.dumps({"people": people, "teams": teams}, ensure_ascii=False)
    else:
        state = f"people:\n{render_records(people, style)}\n\nteams:\n{render_records(teams, style)}"
    questions = {
        "find_person": {"type": "choice", "criteria": {o: o for o in options}, "instructions": phrase(rng, [
            "Who is on {t} and based in {c}{y}?", "Which person works in {c} on {t}{y}?"],
            t=f"the team managed by {team_manager}" if level >= 2 else f"the {person['team']} team",
            c=person["city"], y=f" and started before {cut}" if by_year else "")},
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
            "year": year, "city": city, "since": since, "cut": cut}
    return Problem(state, questions, answers, data)
