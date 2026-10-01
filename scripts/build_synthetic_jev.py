"""Build tasksource/synthetic-typed-decisions from workflow runs (src/tasksource/jev/synthetic/workflows.py).

One row per state, in the procedural dataset's format: `questions` is a Jev request and `answers` holds
Jev's soft labels. Splits are by workflow, so validation and test use decision schemas unseen in train.

    python scripts/build_synthetic_jev.py .synthetic_runs/workflows_v2 [more runs] [--upload]
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

REPO = "tasksource/synthetic-typed-decisions"
CARD = Path(__file__).resolve().parent.parent / "dataset_cards" / "synthetic-typed-decisions.md"


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
        HfApi().upload_file(path_or_fileobj=card(splits).encode(), path_in_repo="README.md",
                            repo_id=REPO, repo_type="dataset", commit_message="Dataset card")


def card(splits):
    """The card with its size section filled from the data."""
    answers = [a for items in splits.values() for r in items for a in json.loads(r["answers"]).values()]
    kinds = Counter(a["type"] for a in answers)
    conf = [max(a["noul"], 1 - a["noul"]) if a["type"] == "noul" else a["confidence"] for a in answers]
    rows = "\n".join(f"| {split} | {len(items):,} | {sum(len(json.loads(r['questions'])) for r in items):,} | "
                     f"{len({r['workflow'] for r in items})} |" for split, items in splits.items())
    size = (f"| split | items | decisions | workflows |\n|---|---|---|---|\n{rows}\n\n"
            f"Decisions: {kinds['choice']:,} choice, {kinds['noul']:,} noul, {kinds['score']:,} score. Jev is at "
            f"least 0.95 confident on {sum(c >= .95 for c in conf) / len(conf):.0%} of them and below 0.7 on "
            f"{sum(c < .7 for c in conf) / len(conf):.0%}.")
    text = CARD.read_text(encoding="utf-8")
    start, end = text.index("<!-- size -->"), text.index("<!-- /size -->")
    return text[:start] + "<!-- size -->\n" + size + "\n" + text[end:]


if __name__ == "__main__":
    main()
