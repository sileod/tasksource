"""Measure how hard each procedural level is for Jev: chance-adjusted accuracy per task and level.

Generates ``--per-level`` fresh states per level (seeded by ``calibrate:task:level:i``,
never a published row), asks Jev through scripts/audit_jev_tasks.py, and prints
kappa = (accuracy - chance) / (1 - chance) per task x level and per question.
Level 0 should be easy-ish for Jev and level 4 tough.

    PYTHONPATH=.:src python scripts/calibrate_procedural_levels.py --tasks needle_retrieval --budget-usd 0.2
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tasksource.jev.procedural import SOURCE_PREFIX, TASKS, jev_rows


def write_shards(tasks, per_level, levels, out):
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    level_of, graded = {}, {}
    for task in tasks:
        rows = []
        for level in levels:
            for i in range(per_level):
                problem = TASKS[task].generate(random.Random(f"calibrate:{task}:{level}:{i}"), level)
                state = problem.state if isinstance(problem.state, str) else json.dumps(problem.state, ensure_ascii=False)
                example = {"state": state, "questions": json.dumps(problem.questions),
                           "answers": json.dumps(problem.answers)}
                for row in jev_rows(example, SOURCE_PREFIX + task, "validation", f"{level}-{i}"):
                    level_of[row["id"]] = level
                    if row["kind"] == "noul" and row["target"][0] not in (0.0, 1.0):
                        graded[row["id"]] = row["target"][0]
                    rows.append(row)
        pd.DataFrame(rows).to_parquet(shards / f"validation-{task}.parquet")
    return shards, level_of, graded


def kappa(group):
    accuracy, chance = group.correct.mean(), group.chance.mean()
    return round((accuracy - chance) / (1 - chance), 2)


def report(out, level_of, graded):
    decisions = pd.read_json(out / "decisions.jsonl", lines=True)
    decisions = decisions[decisions.id.isin(level_of)]
    decisions["level"] = decisions.id.map(level_of)
    decisions["task"] = decisions.source.str.split("/").str[1]
    decisions["question"] = decisions.id.str.split(":").str[3]
    decisions["chance"] = decisions.jev_probabilities.map(lambda p: 0.5 if len(p) == 1 else 1 / len(p))
    decisions["correct"] = decisions.gold == decisions.jev
    by_level = decisions.groupby(["task", "level"])[["correct", "chance"]].apply(kappa).unstack()
    by_level["all"] = decisions.groupby("task")[["correct", "chance"]].apply(kappa)
    by_question = decisions.groupby(["task", "question", "level"])[["correct", "chance"]].apply(kappa).unstack()
    print(by_level.to_string(), "\n", by_question.to_string(), sep="\n")
    # probability answers: kappa of the rounded answer says little, so also the mean absolute error
    soft = decisions[decisions.id.isin(graded)].copy()
    if len(soft):
        soft["error"] = (soft.jev_probabilities.str[0] - soft.id.map(graded)).abs()
        print("\nmean absolute error on graded probabilities (0 is exact):")
        print(soft.groupby(["task", "question", "level"]).error.mean().round(3).unstack().to_string())
    return by_level


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="*", default=sorted(TASKS))
    parser.add_argument("--levels", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--per-level", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("build/procedural-level-calibration"))
    parser.add_argument("--budget-usd", type=float, default=0.5, help="cumulative over --out's cache")
    args = parser.parse_args()
    shards, level_of, graded = write_shards(args.tasks, args.per_level, args.levels, args.out)
    subprocess.run([sys.executable, str(Path(__file__).with_name("audit_jev_tasks.py")), "--shards", str(shards),
                    "--out", str(args.out), "--per-task", "100000", "--skip-adjudication",
                    "--budget-usd", str(args.budget_usd), "--tasks", *[SOURCE_PREFIX + t for t in args.tasks]],
                   check=True)
    report(args.out, level_of, graded)


if __name__ == "__main__":
    main()
