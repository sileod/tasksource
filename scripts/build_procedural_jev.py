#!/usr/bin/env python3
"""Generate and optionally publish tasksource/procedural-jev, one config per task.

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

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi

from tasksource.jev.procedural import REPO_ID, TASKS


def generate_split(task, split, rows, levels, exclude, max_attempts=50):
    module, seen, out = TASKS[task], set(exclude), []
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
        out.append({
            "id": f"{task}:{split}:{len(out)}",
            "level": level,
            "state": state,
            "questions": json.dumps(problem.questions, ensure_ascii=False),
            "answers": json.dumps(problem.answers, ensure_ascii=False),
        })
    return out, seen


def label_features(rows):
    """One flat column per question: a ClassLabel, or a float for graded Noul answers."""
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
            features[qid] = ClassLabel(names=names)
            readers[qid] = lambda a, names=names: names.index(a["choice"])
    return features, readers


def build_task(task, sizes, levels):
    splits, seen = {}, set()
    for split, rows in sizes.items():
        splits[split], seen = generate_split(task, split, rows, levels, seen)
    labels, readers = label_features(splits["train"])
    features = Features({
        "id": Value("string"), "level": Value("int32"), "state": Value("string"),
        "questions": Value("string"), "answers": Value("string"), **labels,
    })
    for rows in splits.values():
        for row in rows:
            answers = json.loads(row["answers"])
            row.update({qid: read(answers[qid]) for qid, read in readers.items()})
    return DatasetDict({
        split: Dataset.from_list(rows, features=features) for split, rows in splits.items()
    })


def summarize(task, dataset):
    summary = {"task": task, "rows": {split: len(rows) for split, rows in dataset.items()}}
    train = dataset["train"]
    for column, feature in train.features.items():
        if isinstance(feature, ClassLabel):
            counts = Counter(train[column])
            summary[column] = {feature.int2str(k)[:24]: round(v / len(train), 3)
                               for k, v in sorted(counts.items())}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "build" / "procedural-jev")
    parser.add_argument("--card", type=Path, default=root / "dataset_cards" / "procedural-jev.md")
    parser.add_argument("--tasks", nargs="*", default=sorted(TASKS))
    parser.add_argument("--train", type=int, default=20_000)
    parser.add_argument("--eval", type=int, default=1_000, help="Rows in each of validation and test.")
    parser.add_argument("--levels", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default=REPO_ID)
    return parser.parse_args()


def main(args):
    sizes = {"train": args.train, "validation": args.eval, "test": args.eval}
    built = {}
    for task in args.tasks:
        built[task] = build_task(task, sizes, args.levels)
        built[task].save_to_disk(args.output / task)
    summaries = [summarize(task, dataset) for task, dataset in built.items()]
    (args.output / "build-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    if args.upload:
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        for task, dataset in built.items():
            dataset.push_to_hub(args.repo_id, config_name=task,
                                commit_message=f"Publish {task}")
        # The card explicitly maps each config to its Parquet files. Upload it
        # last so named configs remain loadable after push_to_hub updates metadata.
        api.upload_file(path_or_fileobj=str(args.card), path_in_repo="README.md",
                        repo_id=args.repo_id, repo_type="dataset",
                        commit_message="Document procedural-jev configurations")


if __name__ == "__main__":
    main(parse_args())
