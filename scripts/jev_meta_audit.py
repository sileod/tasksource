"""Ask Jev yes/no questions about sampled decisions of a Jev build: is the gold answer right, is the
question ambiguous, does it need information the text lacks, is it trivial, is it malformed.

Each decision is shown with its gold answer; the mean probabilities per source point to sources worth
reading (the probabilities are a screen, not a verdict).

    PYTHONPATH=.:src python scripts/jev_meta_audit.py --per-task 20
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from scripts.audit_jev_tasks import gold_answer, option_labels, sample_rows
from tasksource.jev.synthetic.annotate import annotate_bundle_jev
from tasksource.jev.synthetic.config import AnnotatorConfig

CHECKS = {
    "wrong": "Is the marked answer wrong, so that another option is clearly the better answer?",
    "ambiguous": "Is the question ambiguous for this text, so that a careful reader could reasonably defend "
                 "a different option?",
    "missing": "Does answering depend on information the text does not give, such as missing context or "
               "options whose meaning is not defined?",
    "trivial": "Is the answer obvious from surface cues alone, without understanding the text?",
    "malformed": "Is the example malformed: garbled text, options that do not fit the question, or a "
                 "question that makes no sense?",
}


def shown(row, max_chars):
    options = ["no", "yes"] if row["kind"] == "noul" else list(row["options"])
    gold = gold_answer(row)
    listing = "\n".join(f"{label}. {option}" for label, option in zip(option_labels(len(options)), options))
    return (f"Text:\n{row['state'][:max_chars]}\n\nQuestion: {row['question']}\n\nOptions:\n{listing}\n\n"
            f"Marked answer: {option_labels(len(options))[gold]}. {options[gold]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=Path, default=Path("build/tasksource-jev-typed-decisions-v11/data"))
    parser.add_argument("--out", type=Path, default=Path("build/jev-meta-audit"))
    parser.add_argument("--per-task", type=int, default=20)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--jev-model", default="~typesafe/jev-latest")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=6000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in sample_rows(args.shards, args.per_task, 1, args.tasks) if gold_answer(r) is not None]
    print(f"{len(rows)} decisions from {len({r['source'] for r in rows})} sources", flush=True)
    config = AnnotatorConfig(name="jev", version=args.jev_model, base_url="https://openrouter.ai",
                             api_key_env="JEV_OPENROUTER_API_KEY", model=args.jev_model)
    key = os.environ[config.api_key_env]

    def check(row):
        bundle = {"state_id": row["id"], "state": shown(row, args.max_chars),
                  "questions": [{"question_id": name, "format": "noul", "question": question, "options": []}
                                for name, question in CHECKS.items()]}
        try:
            labelled = annotate_bundle_jev(bundle, config, key, args.out / "cache")
        except Exception as error:
            print(f"failed {row['id']}: {str(error)[:120]}", flush=True)
            return None
        return {"source": row["source"], "id": row["id"],
                **{name: a["probabilities"][0] for name, a in zip(CHECKS, labelled["annotations"])}}

    with ThreadPoolExecutor(args.concurrency) as pool:
        records = [r for r in pool.map(check, rows) if r]
    frame = pd.DataFrame(records)
    frame.to_csv(args.out / "rows.csv", index=False)
    per_source = frame.groupby("source")[list(CHECKS)].mean().round(3)
    per_source.insert(0, "n", frame.groupby("source").size())
    per_source.sort_values("wrong", ascending=False).to_csv(args.out / "per-source.csv")
    print(per_source.sort_values("wrong", ascending=False).head(40).to_string())
    print(json.dumps(frame[list(CHECKS)].mean().round(3).to_dict()))


if __name__ == "__main__":
    main()
