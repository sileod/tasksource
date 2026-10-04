"""Second detectors for jev_error_pass.py: decision models' gold probability on published source examples.

- Jev on a sample (``--per-source`` examples per source): with the DeepSeek screening, two independent
  detectors give a capture-recapture estimate of the wrong labels both miss, per source.
- Mercury Decide (free) on every example, as extra recall. Its failures never block: a failed request
  is skipped and retried on the next run, and the run ends after ``--max-failures``.

Rows the detectors doubt (gold probability below ``--doubt``) are written with their content, so
``jev_error_pass.py --confirm`` can confirm them like DeepSeek's flags.

    set -a; . ~/.jev_synth.env; set +a
    PYTHONPATH=.:src python scripts/jev_second_detector.py --model jev
    PYTHONPATH=.:src python scripts/jev_second_detector.py --model mercury
"""

import argparse
import argparse as _argparse
import json
from pathlib import Path

from datasets import load_dataset

from scripts.audit_jev_tasks import annotate, bundles_of, gold_probability
from scripts.jev_error_pass import REPO, ROW_FIELDS, checkable
from tasksource.jev.augmentations import stable_fraction
from tasksource.jev.synthetic.config import AnnotatorConfig

MODELS = {"jev": "~typesafe/jev-latest", "mercury": "inception/mercury-decide:free"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--out", type=Path, default=Path("build/jev-error-pass"))
    parser.add_argument("--per-source", type=int, help="sample size per source (default: 30 for jev, all for mercury)")
    parser.add_argument("--doubt", type=float, default=0.2, help="gold probability under which a row is kept for confirmation")
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--max-failures", type=int, default=2000)
    args = parser.parse_args()
    per_source = args.per_source if args.per_source is not None else 30 if args.model == "jev" else 0

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"second-{args.model}.jsonl"
    done = {json.loads(line)["example_id"] for line in path.open()} if path.exists() else set()
    rows, seen = [], set(done)
    for split in ("train", "validation", "test"):
        full = load_dataset(REPO, "full", split=split)
        full = full.filter(lambda v: v == "direct", input_columns="variant", desc=f"direct {split}")
        for row in full.select_columns(list(ROW_FIELDS) + ["variant"]):
            if row["example_id"] not in seen and checkable(row):
                seen.add(row["example_id"])
                rows.append({k: row[k] for k in ROW_FIELDS})
    if per_source:  # a stable random sample: the same examples on every run
        rows.sort(key=lambda row: stable_fraction(row["example_id"], "second-detector"))
        counts, sample = {}, []
        for row in rows:
            counts[row["source"]] = counts.get(row["source"], 0) + 1
            if counts[row["source"]] <= per_source:
                sample.append(row)
        rows = [row for row in sample if row["example_id"] not in done]
    print(f"{args.model}: {len(done)} done, {len(rows)} to score", flush=True)

    model = MODELS[args.model]
    config = AnnotatorConfig(name=args.model, version=model, base_url="https://openrouter.ai",
                             api_path="/api/alpha/decisions", api_key_env="JEV_OPENROUTER_API_KEY", model=model)
    for start in range(0, len(rows), args.chunk * 10):
        bundles = bundles_of(rows[start:start + args.chunk * 10])
        try:
            results, spent = annotate(bundles, config, _argparse.Namespace(
                out=args.out / f"cache-{args.model}", chunk=args.chunk, concurrency=args.concurrency,
                budget_usd=args.budget_usd))
        except Exception as error:  # never block the error pass on a detector
            print(f"{args.model} stopped: {str(error)[:300]}", flush=True)
            break
        written = 0
        with path.open("a") as handle:
            for bundle in bundles:
                result = results.get(id(bundle))
                for row, annotation in zip(bundle["rows"], (result or {}).get("annotations", [])):
                    probability = gold_probability(row, annotation["probabilities"])
                    record = {"example_id": row["example_id"], "source": row["source"], "model": args.model,
                              "gold_probability": round(probability, 4)}
                    if probability < args.doubt:
                        record["row"] = row
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
        failed = sum(1 for bundle in bundles if id(bundle) not in results)
        print(f"{args.model}: {start + len(bundles)} bundles, {written} scored, {failed} failed, ${spent:.3f}", flush=True)
        if args.model == "jev" and spent >= args.budget_usd:
            break
        if failed >= args.max_failures or (written == 0 and failed):
            print(f"{args.model}: too many failures, stopping (rerun to resume)", flush=True)
            break


if __name__ == "__main__":
    main()
