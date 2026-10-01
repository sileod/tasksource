"""Build tasksource/synthetic-typed-decisions from workflow runs (src/tasksource/jev/synthetic/workflows.py).

One row per state, in the procedural dataset's format: `questions` is a Jev request and `answers` holds
Jev's soft labels. Splits are by workflow, so validation and test use decision schemas unseen in train.

    python scripts/build_synthetic_jev.py .synthetic_runs/workflows_v2 [--upload]
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

REPO = "tasksource/synthetic-typed-decisions"


def jev_question(q):
    if q["type"] == "noul":
        return {"type": "noul", "instructions": q["question"]}
    if q["type"] == "score":
        return {"type": "score", "instructions": q["question"], "criteria": list(q["options"])}
    return {"type": "choice", "instructions": q["question"], "criteria": {o: o for o in q["options"]}}


def jev_answer(q):
    p = [float(x) for x in q["jev"]]
    if q["type"] == "noul":
        return {"type": "noul", "noul": p[0]}
    top = max(range(len(p)), key=p.__getitem__)
    if q["type"] == "score":
        return {"type": "score", "score": float(top), "legend": dict(enumerate(q["options"])),
                "probabilities": {str(i): x for i, x in enumerate(p)}, "confidence": p[top]}
    return {"type": "choice", "choice": q["options"][top], "probabilities": dict(zip(q["options"], p)),
            "confidence": p[top]}


def split_of(workflow_id, salt="synthetic-typed-decisions-v1"):
    position = int(hashlib.sha256(f"{salt}:{workflow_id}".encode()).hexdigest()[:8], 16) / 16 ** 8
    return "train" if position < 0.8 else "validation" if position < 0.9 else "test"


def rows(run):
    workflows = {w["workflow_id"]: w for w in map(json.loads, (run / "workflows.jsonl").open())}
    for item in map(json.loads, (run / "items.jsonl").open()):
        kept = [q for q in item["questions"] if q["kept"] and q.get("jev")]
        if not kept:
            continue
        workflow = workflows[item["workflow_id"]]
        yield split_of(f"{run.name}/{item['workflow_id']}"), {
            "id": f"{run.name}/{item['state_id']}",
            "workflow": f"{run.name}/{item['workflow_id']}",
            "application": workflow["application"],
            "domain": workflow["domain"],
            "source": workflow["source"],
            "state": item["state"],
            "questions": json.dumps({q["id"]: jev_question(q) for q in kept}, ensure_ascii=False),
            "answers": json.dumps({q["id"]: jev_answer(q) for q in kept}, ensure_ascii=False),
            "skills": json.dumps({q["id"]: q["skill"] for q in kept}),
            "checker": json.dumps({q["id"]: q["check"]["probabilities"] for q in kept}),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    splits = {"train": [], "validation": [], "test": []}
    for run in args.runs:
        for split, row in rows(run):
            splits[split].append(row)
    for split, items in splits.items():
        random.Random(0).shuffle(items)
        n = sum(len(json.loads(r["questions"])) for r in items)
        print(f"{split}: {len(items)} states, {n} decisions, {len({r['workflow'] for r in items})} workflows")
    dataset = DatasetDict({split: Dataset.from_list(items) for split, items in splits.items()})
    if args.upload:
        dataset.push_to_hub(REPO)


if __name__ == "__main__":
    main()
