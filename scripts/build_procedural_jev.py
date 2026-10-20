#!/usr/bin/env python3
"""Generate and optionally publish tasksource/procedural-typed-decisions, one config per task.

The generators are reasoning-core's Jev tasks (see ``tasksource.jev.procedural``); the reasoning-core commit and
task versions are recorded in ``provenance.json``, so publish from a clean reasoning-core checkout.

Every row is one generated state with several typed Jev questions over it
(`questions`, `answers` as JSON), plus one flat label column per question so
that each question is also an ordinary Tasksource task. Generation is
deterministic: row ``i`` of a split is seeded by ``task:split:i`` alone.
States are unique within a split, and evaluation states never occur in train.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value, concatenate_datasets, load_from_disk
from huggingface_hub import HfApi

from tasksource.jev.procedural import REPO_ID, TASKS, reasoning_core_provenance

MAX_CLASS_NAMES = 100
# No label of a yes/no or choice question goes above this share of its answers: a state
# drops the question when its label would; states left without questions are skipped.
MAX_LABEL_SHARE = 0.6
BALANCE_WARMUP = 20
# Yes/no questions are cheap to ask and dominate decision data: at most this share of a task's questions.
MAX_YES_NO_SHARE = 1 / 3


def balance_label(answer):
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5
    return answer.get("choice")  # graded scores keep their natural distribution


def yes_no(answer):
    return answer["type"] == "noul" and answer["noul"] in (0.0, 1.0)


def balanced_questions(problem, counts, max_share=MAX_LABEL_SHARE, warmup=BALANCE_WARMUP,
                       max_yes_no=MAX_YES_NO_SHARE):
    """The problem's questions whose label is not already over-represented, and yes/no questions while
    under their share of the task's questions; updates ``counts``."""
    kept = {}
    asked = counts.setdefault(None, Counter())  # the task's questions: yes/no or not
    # least-asked questions first, so the yes/no budget is shared among a task's yes/no questions
    order = sorted(problem.questions, key=lambda qid: sum(counts.get(qid, Counter()).values()))
    for qid in order:
        spec, answer = problem.questions[qid], problem.answers[qid]
        label = balance_label(answer)
        seen = counts.setdefault(qid, Counter())
        total = sum(seen.values())
        if label is not None and total >= warmup and seen[label] + 1 > max_share * (total + 1):
            continue
        binary, everything = asked[True], sum(asked.values())
        if yes_no(answer) and everything >= warmup and binary + 1 > max_yes_no * (everything + 1):
            continue
        seen[label] += 1
        asked[yes_no(answer)] += 1
        kept[qid] = spec
    return {qid: spec for qid, spec in problem.questions.items() if qid in kept}


def generate_split(task, split, rows, levels, exclude, max_attempts=50):
    module, seen, out, counts = TASKS[task], set(exclude), [], {}
    for index in range(rows * max_attempts):
        if len(out) == rows:
            break
        rng = random.Random(f"{task}:{split}:{index}")
        level = rng.choice(levels)
        problem = module.generate(rng, level)
        text = isinstance(problem.state, str)
        state = problem.state if text else json.dumps(problem.state, ensure_ascii=False)
        key = state if text else json.dumps(problem.state, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        questions = balanced_questions(problem, counts)
        if not questions:
            continue
        answers = {qid: problem.answers[qid] for qid in questions}
        out.append({
            "id": f"{task}:{split}:{len(out)}",
            "level": level,
            "state": state,
            "questions": json.dumps(questions, ensure_ascii=False),
            "answers": json.dumps(answers, ensure_ascii=False),
        })
    return out, seen


def label_features(rows):
    """One flat column per question: a ClassLabel, a float for graded Noul answers,
    or the chosen option's text when a choice has no small fixed vocabulary."""
    specs, values = {}, {}
    for row in rows:
        answers = json.loads(row["answers"])
        for qid, spec in json.loads(row["questions"]).items():
            specs.setdefault(qid, []).append(spec)
            values.setdefault(qid, []).append(answers[qid])
    features, readers = {}, {}
    for qid, qspecs in specs.items():
        kind = qspecs[0]["type"]
        if kind == "noul":
            if {a["noul"] for a in values[qid]} <= {0.0, 1.0}:
                features[qid] = ClassLabel(names=["false", "true"])
                readers[qid] = lambda a: int(a["noul"])
            else:
                features[qid] = Value("float32")
                readers[qid] = lambda a: a["noul"]
        elif kind == "score":
            features[qid] = ClassLabel(names=list(qspecs[0]["criteria"]))
            readers[qid] = lambda a: int(a["score"])
        else:
            widest = max((list(s["criteria"]) for s in qspecs), key=len)
            names = list(dict.fromkeys(widest + [c for s in qspecs for c in s["criteria"]]))
            if len(names) > MAX_CLASS_NAMES:  # open answers such as amounts or times
                features[qid] = Value("string")
                readers[qid] = lambda a: a["choice"]
            else:
                features[qid] = ClassLabel(names=names)
                readers[qid] = lambda a, names=names: names.index(a["choice"])
    return features, readers


def pretty_order(rows, first_rows=1_000, seed=0, key=lambda row: row["level"]):
    """Buckets (levels by default) round-robin in a display prefix, easiest first, then the remaining rows shuffled."""
    buckets = {}
    for row in rows:
        buckets.setdefault(key(row), []).append(row)
    prefix, queues = [], [list(bucket) for _, bucket in sorted(buckets.items())]
    while len(prefix) < min(first_rows, len(rows)):
        for queue in queues:
            if queue and len(prefix) < first_rows:
                prefix.append(queue.pop(0))
    rest = [row for queue in queues for row in queue]
    random.Random(seed).shuffle(rest)
    return prefix + rest


def build_task(task, sizes, levels, pretty_rows=1_000):
    splits, seen = {}, set()
    for split, rows in sizes.items():
        splits[split], seen = generate_split(task, split, rows, levels, seen)
    splits["train"] = pretty_order(splits["train"], pretty_rows)
    labels, readers = label_features(splits["train"])
    features = Features({
        "id": Value("string"), "level": Value("int32"), "state": Value("string"),
        "questions": Value("string"), "answers": Value("string"), **labels,
    })
    for rows in splits.values():
        for row in rows:
            answers = json.loads(row["answers"])
            # States may ask a subset of their task's questions; the rest are null.
            row.update({qid: read(answers[qid]) if qid in answers else None
                        for qid, read in readers.items()})
    return DatasetDict({
        split: Dataset.from_list(rows, features=features) for split, rows in splits.items()
    })


COMMON = ["id", "task", "level", "state", "questions", "answers"]


def build_all(output, tasks, pretty_rows=1_000, seed=0):
    """Every task in one config (the default): shared columns only, the train prefix cycling
    through levels and tasks, everything else shuffled."""
    parts = {task: load_from_disk(str(output / task)) for task in tasks}
    splits = {}
    for split in parts[tasks[0]]:
        rows = [{"task": task, **{c: r[c] for c in COMMON if c != "task"}}
                for task, dataset in parts.items() for r in dataset[split]]
        if split == "train":
            rows = pretty_order(rows, pretty_rows, seed, key=lambda r: (r["level"], r["task"]))
        else:
            random.Random(seed).shuffle(rows)
        splits[split] = Dataset.from_list([{c: r[c] for c in COMMON} for r in rows])
    return DatasetDict(splits)


def summarize(task, dataset):
    summary = {"task": task, "rows": {split: len(rows) for split, rows in dataset.items()}}
    train = dataset["train"]
    for column, feature in train.features.items():
        if isinstance(feature, ClassLabel):
            counts = Counter(k for k in train[column] if k is not None)
            asked = sum(counts.values())
            summary[column] = {feature.int2str(k)[:24]: round(v / asked, 3)
                               for k, v in sorted(counts.items())}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "build" / "procedural-typed-decisions")
    parser.add_argument("--card", type=Path, default=root / "dataset_cards" / "procedural-typed-decisions.md")
    parser.add_argument("--tasks", nargs="*", default=sorted(TASKS))
    parser.add_argument("--train", type=int, default=20_000)
    parser.add_argument("--eval", type=int, default=1_000, help="Rows in each of validation and test.")
    parser.add_argument("--levels", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--pretty-rows", type=int, default=1_000,
                        help="Train rows shown first, cycling through levels; the rest is shuffled.")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default=REPO_ID)
    return parser.parse_args()


def main(args):
    sizes = {"train": args.train, "validation": args.eval, "test": args.eval}
    provenance = {**reasoning_core_provenance(), "levels": args.levels, "sizes": sizes,
                  "seeding": "row i of a split: random.Random(f'{task}:{split}:{i}') draws the level, then the state seed",
                  "script": "https://github.com/sileod/tasksource/blob/main/scripts/build_procedural_jev.py"}
    built = {}
    for task in args.tasks:
        built[task] = build_task(task, sizes, args.levels, args.pretty_rows)
        built[task].save_to_disk(args.output / task)
    summaries = [summarize(task, dataset) for task, dataset in built.items()]
    if summaries:  # --tasks with no names rebuilds only the combined config from the saved tasks
        (args.output / "build-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    built["all"] = build_all(args.output, sorted(TASKS), args.pretty_rows)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    if args.upload:
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        for task, dataset in built.items():
            dataset.push_to_hub(args.repo_id, config_name=task,
                                commit_message=f"Publish {task}")
        api.upload_file(path_or_fileobj=str(args.output / "provenance.json"), path_in_repo="provenance.json",
                        repo_id=args.repo_id, repo_type="dataset", commit_message="Record generator provenance")
        # The card explicitly maps each config to its Parquet files. Upload it
        # last so named configs remain loadable after push_to_hub updates metadata.
        api.upload_file(path_or_fileobj=str(args.card), path_in_repo="README.md",
                        repo_id=args.repo_id, repo_type="dataset",
                        commit_message="Document procedural-typed-decisions configurations")


if __name__ == "__main__":
    main(parse_args())
