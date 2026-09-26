"""Per-task Jev accuracy on tasksource, with a stronger model adjudicating disagreements.

For each source of a Jev build, up to ``--per-task`` direct decisions are drawn
(test split first, then validation, then train). Questions over one state go to
Jev in one request. Where Jev's top answer differs from the gold label, an
adjudicator (``--adjudicator``, via litlm) answers the question blind, without
seeing either label. Its answer tells label noise or ambiguity (it sides with
Jev) apart from Jev errors (it sides with gold).

    PYTHONPATH=.:src python scripts/audit_jev_tasks.py --shards build/tasksource-jev-typed-decisions-v11/data

Jev responses are cached under ``<out>/jev-cache`` and adjudications by litlm, so
reruns only pay for what is missing. ``--budget-usd`` stops Jev spending, counted
from the cached usage.
"""

import argparse
import glob
import json
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from tasksource.jev.synthetic.annotate import annotate_bundle_jev, jev_cache_key
from tasksource.jev.synthetic.config import AnnotatorConfig

SPLIT_PREFERENCE = ("test", "validation", "train")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def sample_rows(shards, per_task, seed, tasks=None):
    """Up to ``per_task`` direct decisions per source, from its most held-out split."""
    files = defaultdict(dict)
    for path in glob.glob(str(Path(shards) / "*.parquet")):
        split, stem = Path(path).stem.split("-", 1)
        files[stem][split] = path
    rows = []
    for stem, splits in sorted(files.items()):
        picked = []
        for split in SPLIT_PREFERENCE:
            if split not in splits or len(picked) >= per_task:
                continue
            table = pq.read_table(splits[split]).to_pandas()
            table = table[table.variant == "direct"]
            if tasks and not table.source.isin(tasks).any():
                break
            order = random.Random(f"{seed}/{stem}/{split}").sample(range(len(table)), len(table))
            picked.extend(table.iloc[order[: per_task - len(picked)]].to_dict("records"))
        rows.extend(picked)
    return rows


def bundles_of(rows):
    """One Jev request per (source, state); question ids are unique within it."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["state"])].append(row)
    bundles = []
    for (source, state), members in grouped.items():
        questions = [{"question_id": f"q{i}", "format": m["kind"], "question": m["question"],
                      "options": list(m["options"])} for i, m in enumerate(members)]
        bundles.append({"state_id": members[0]["id"], "state": state, "questions": questions,
                        "rows": members})
    return bundles


def gold_answer(row):
    """Index of the gold option (0/1 for noul: no/yes), or None when the target is a tie."""
    target = list(row["target"])
    if row["kind"] == "noul":
        return None if target[0] == 0.5 else int(target[0] > 0.5)
    best = max(target)
    return None if target.count(best) > 1 else target.index(best)


def predicted_answer(kind, probabilities):
    if kind == "noul":
        return int(probabilities[0] > 0.5)
    return max(range(len(probabilities)), key=probabilities.__getitem__)


def gold_probability(row, probabilities):
    """Probability Jev puts on the gold answer."""
    gold = gold_answer(row)
    if row["kind"] == "noul":
        return probabilities[0] if gold else 1 - probabilities[0]
    return probabilities[gold]


def cached_cost(cache_dir):
    return sum(json.loads(path.read_text()).get("usage", {}).get("cost", 0) or 0
               for path in cache_dir.glob("*.json"))


def annotate(bundles, config, args):
    cache_dir = args.out / "jev-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ[config.api_key_env]
    for path in cache_dir.glob("*.json"):  # entries cut short by an interrupted older run
        try:
            json.loads(path.read_text())
        except ValueError:
            path.unlink()
    spent = cached_cost(cache_dir)
    todo = [b for b in bundles if not (cache_dir / f"{jev_cache_key(b, config)}.json").exists()]
    print(f"Jev: {len(bundles) - len(todo)} cached, {len(todo)} to request, ${spent:.4f} spent", flush=True)
    results, failures = {}, 0

    def one(bundle):
        request = {k: bundle[k] for k in ("state_id", "state", "questions")}
        return annotate_bundle_jev(request, config, api_key, cache_dir)

    # cached bundles first, then new ones in chunks so the budget check stays close
    pending = {id(bundle) for bundle in todo}
    for bundle in bundles:
        if id(bundle) not in pending:
            results[id(bundle)] = one(bundle)
    for start in range(0, len(todo), args.chunk):
        if spent >= args.budget_usd:
            print(f"Budget reached (${spent:.4f}); {len(todo) - start} bundles left", flush=True)
            break
        chunk = todo[start:start + args.chunk]
        with ThreadPoolExecutor(args.concurrency) as pool:
            futures = {pool.submit(one, bundle): bundle for bundle in chunk}
            for future in as_completed(futures):
                try:
                    results[id(futures[future])] = future.result()
                except Exception as error:  # a failed request is retried on the next run
                    failures += 1
                    if failures <= 5:
                        print(f"Jev failure: {str(error)[:200]}", flush=True)
        spent += sum(json.loads(path.read_text()).get("usage", {}).get("cost", 0) or 0
                     for path in (cache_dir / f"{jev_cache_key(b, config)}.json" for b in chunk) if path.exists())
        print(f"Jev: {start + len(chunk)}/{len(todo)} requested, ${spent:.4f} spent, {failures} failures", flush=True)
    return results, spent


def adjudication_prompt(row):
    kind, question, options = row["kind"], row["question"], list(row["options"])
    if kind == "noul":
        options = ["no", "yes"]
    labels = option_labels(len(options))
    listing = "\n".join(f"{label}. {option}" for label, option in zip(labels, options))
    scale = " The options are ordered levels, lowest first." if kind == "score" else ""
    return (f"Read the text and answer the question.{scale}\n\n<text>\n{row['state']}\n</text>\n\n"
            f"Question: {question}\n\nOptions:\n{listing}\n\n"
            "Reason step by step about the text, the question and each option, checking facts and implicit\n"
            "assumptions. Then give the label of the best option inside <answer></answer> tags.")


def option_labels(n_options):
    """Letters, or numbers when there are more options than letters."""
    return list(LETTERS[:n_options]) if n_options <= len(LETTERS) else [str(i + 1) for i in range(n_options)]


def parse_letter(reply, n_options):
    match = re.search(r"<answer>\s*\(?([A-Z]|\d+)\b", str(reply))
    labels = option_labels(n_options)
    return labels.index(match.group(1)) if match and match.group(1) in labels else None


def adjudicate(records, args):
    from litlm import complete

    per_task = defaultdict(list)
    for record in records:
        if record["jev"] != record["gold"]:
            per_task[record["row"]["source"]].append(record)
    disputed = [r for items in per_task.values() for r in items[: args.adjudicate_per_task]]
    print(f"Adjudicating {len(disputed)} disagreements with {args.adjudicator}", flush=True)
    if not disputed:
        return
    replies = complete([adjudication_prompt(r["row"]) for r in disputed], model=args.adjudicator,
                       caching=True, max_tokens=args.adjudicator_max_tokens, temperature=0.0,
                       max_concurrency=args.adjudicator_concurrency, rpm=args.adjudicator_rpm, timeout=180)
    for _ in range(3):  # rate limits: retry only the failed positions, more slowly
        if not replies.failures:
            break
        time.sleep(60)
        replies.resume(max_concurrency=2, rpm=args.adjudicator_rpm / 2, timeout=180, num_retries=5)
    for record, reply in zip(disputed, replies):
        n_options = 2 if record["row"]["kind"] == "noul" else len(record["row"]["options"])
        record["adjudicated"] = parse_letter(reply, n_options)


def report(records, args):
    per_task = defaultdict(list)
    for record in records:
        per_task[record["row"]["source"]].append(record)
    lines = []
    for source, items in sorted(per_task.items()):
        disputed = [r for r in items if r["jev"] != r["gold"]]
        judged = [r for r in disputed if r.get("adjudicated") is not None]
        lines.append({
            "source": source,
            "n": len(items),
            "kinds": ",".join(sorted({r["row"]["kind"] for r in items})),
            "split": ",".join(sorted({r["row"]["split"] for r in items})),
            "jev_accuracy": round(sum(r["jev"] == r["gold"] for r in items) / len(items), 4),
            "jev_gold_probability": round(sum(r["gold_probability"] for r in items) / len(items), 4),
            "disagreements": len(disputed),
            "adjudicated": len(judged),
            "adjudicator_sides_gold": sum(r["adjudicated"] == r["gold"] for r in judged),
            "adjudicator_sides_jev": sum(r["adjudicated"] == r["jev"] for r in judged),
            "adjudicator_other": sum(r["adjudicated"] not in (r["gold"], r["jev"]) for r in judged),
        })
    frame = pd.DataFrame(lines)
    frame["label_doubt"] = (frame.adjudicator_sides_jev / frame.adjudicated.where(frame.adjudicated > 0)).round(3)
    # accuracy if the disagreements where the adjudicator sides with Jev were label errors
    frame["jev_accuracy_adjudicated"] = ((frame.jev_accuracy * frame.n + frame.adjudicator_sides_jev)
                                         / frame.n).round(4)
    frame.to_csv(args.out / "per-task.csv", index=False)
    with (args.out / "decisions.jsonl").open("w") as handle:
        for r in records:
            handle.write(json.dumps({"id": r["row"]["id"], "source": r["row"]["source"], "kind": r["row"]["kind"],
                                     "gold": r["gold"], "jev": r["jev"], "jev_probabilities": r["probabilities"],
                                     "adjudicated": r.get("adjudicated")}) + "\n")
    total = {"tasks": len(frame), "decisions": int(frame.n.sum()),
             "jev_accuracy": round(float((frame.jev_accuracy * frame.n).sum() / frame.n.sum()), 4),
             "macro_jev_accuracy": round(float(frame.jev_accuracy.mean()), 4),
             "disagreements": int(frame.disagreements.sum()),
             "adjudicator_sides_gold": int(frame.adjudicator_sides_gold.sum()),
             "adjudicator_sides_jev": int(frame.adjudicator_sides_jev.sum()),
             "adjudicator_other": int(frame.adjudicator_other.sum())}
    # procedural gold is computed exactly: siding with Jev there is an adjudicator error
    procedural = frame[frame.source.str.startswith("procedural-typed-decisions/")]
    if procedural.adjudicated.sum():
        total["adjudicator_error_rate_on_procedural"] = round(
            float(procedural.adjudicator_sides_jev.sum() + procedural.adjudicator_other.sum())
            / float(procedural.adjudicated.sum()), 4)
        total["procedural_adjudicated"] = int(procedural.adjudicated.sum())
    (args.out / "summary.json").write_text(json.dumps(total, indent=2) + "\n")
    print(json.dumps(total), flush=True)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=Path, default=Path("build/tasksource-jev-typed-decisions-v11/data"))
    parser.add_argument("--out", type=Path, default=Path("build/jev-task-audit"))
    parser.add_argument("--per-task", type=int, default=200)
    parser.add_argument("--tasks", nargs="*", help="restrict to these sources")
    parser.add_argument("--limit-tasks", type=int, help="first N sources only (a pilot)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jev-model", default="~typesafe/jev-latest")
    parser.add_argument("--jev-key-env", default="JEV_OPENROUTER_API_KEY")
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--adjudicator", default="albert/deepseek-v4-flash-0731")
    parser.add_argument("--adjudicator-max-tokens", type=int, default=4000)
    parser.add_argument("--adjudicator-concurrency", type=int, default=4)
    parser.add_argument("--adjudicator-rpm", type=float, default=20)  # Albert allows 50, shared with other uses of the key
    parser.add_argument("--adjudicate-per-task", type=int, default=40)
    parser.add_argument("--skip-adjudication", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = sample_rows(args.shards, args.per_task, args.seed, args.tasks)
    if args.tasks:
        rows = [r for r in rows if r["source"] in args.tasks]
    if args.limit_tasks:
        keep = sorted({r["source"] for r in rows})[: args.limit_tasks]
        rows = [r for r in rows if r["source"] in keep]
    bundles = bundles_of(rows)
    print(f"{len(rows)} decisions from {len({r['source'] for r in rows})} sources in {len(bundles)} requests",
          flush=True)
    config = AnnotatorConfig(name="jev", version=args.jev_model, base_url="https://openrouter.ai",
                             api_path="/api/alpha/decisions", api_key_env=args.jev_key_env, model=args.jev_model)
    results, spent = annotate(bundles, config, args)

    records = []
    for bundle in bundles:
        result = results.get(id(bundle))
        if result is None:
            continue
        for row, annotation in zip(bundle["rows"], result["annotations"]):
            gold = gold_answer(row)
            if gold is None:
                continue
            probabilities = annotation["probabilities"]
            records.append({"row": row, "gold": gold, "probabilities": probabilities,
                            "jev": predicted_answer(row["kind"], probabilities),
                            "gold_probability": gold_probability(row, probabilities)})
    if not args.skip_adjudication:
        adjudicate(records, args)
    report(records, args)
    print(f"Jev spend: ${spent:.4f}", flush=True)


if __name__ == "__main__":
    main()
