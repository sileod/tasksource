"""Error detection over the published tasksource-jev-typed-decisions: DeepSeek V4 Flash (Albert) checks
the gold of every source example, several examples of one source per prompt.

One direct row per `example_id` is checked, examples of the `default` config first. Verdicts go to
build/jev-error-pass/verdicts.jsonl (resumable), flagged rows to flagged.jsonl.

Batched screening over-flags hard examples (adversarial NLI), so `--confirm` re-checks each flagged
example alone: DeepSeek reasoning step by step, and Jev. An example is bad when both reject the gold
(DeepSeek picks another option and Jev gives the gold less than 0.3). Bad examples go to
src/tasksource/metadata/jev_bad_examples.csv, which the build drops.

    set -a; . ~/.jev_synth.env; set +a
    PYTHONPATH=.:src python scripts/jev_error_pass.py --limit 2000   # probe
    PYTHONPATH=.:src python scripts/jev_error_pass.py                # everything
    PYTHONPATH=.:src python scripts/jev_error_pass.py --confirm
"""

import argparse
import json
import os
import re
import threading
import time
from pathlib import Path

from datasets import load_dataset

from scripts.audit_jev_tasks import (adjudication_prompt, annotate, bundles_of, gold_answer, gold_probability,
                                     option_labels, parse_letter)
from tasksource.jev.synthetic.config import AnnotatorConfig

REPO = "tasksource/tasksource-jev-typed-decisions"
VERDICTS = ("ok", "wrong", "ambiguous", "malformed")

INSTRUCTIONS = """You check the gold answers of a dataset ({source}). For each example below, decide:
- ok: the gold answer is correct, or a reasonable reading of the question (hard but right is ok);
- wrong: another option is clearly correct;
- ambiguous: several options are equally defensible, or the text does not allow an answer;
- malformed: the text, question or options are broken, truncated or garbled so the task makes no sense.
Labels follow the dataset's own conventions (e.g. NLI "neutral" when the text does not settle it);
judge against those conventions, not your preferences. Be conservative: flag only clear problems.

{examples}

Reply with one JSON object per example, one per line, nothing else:
{{"i": <number>, "verdict": "ok|wrong|ambiguous|malformed", "better": "<label of the correct option or null>", "reason": "<at most 12 words>"}}"""


def render(index, row, max_chars):
    options = ["no", "yes"] if row["kind"] == "noul" else list(row["options"])
    target = list(row["target"])
    gold = int(target[0] > 0.5) if row["kind"] == "noul" else target.index(max(target))
    labels = option_labels(len(options))
    state = row["state"] if len(row["state"]) <= max_chars else row["state"][:max_chars] + " [...]"
    listing = "\n".join(f"{label}. {option}" for label, option in zip(labels, options))
    scale = " (ordered levels, lowest first)" if row["kind"] == "score" else ""
    return (f"### Example {index}\n<text>\n{state}\n</text>\nQuestion: {row['question']}\n"
            f"Options{scale}:\n{listing}\nGold: {labels[gold]}")


def checkable(row):
    """Hard labels only: soft crowd distributions and uncertain yes/no targets are not errors."""
    target = list(row["target"])
    if row["kind"] == "noul":
        return abs(target[0] - 0.5) >= 0.3
    return max(target) >= 0.99


def batches(rows, max_items, max_chars):
    """Requests of up to ``max_items`` examples of one source and ``max_chars`` of text."""
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    for source, items in by_source.items():
        batch, size = [], 0
        for row in items:
            length = min(len(row["state"]), 4 * max_chars)
            if batch and (len(batch) == max_items or size + length > max_chars):
                yield source, batch
                batch, size = [], 0
            batch.append(row)
            size += length
        if batch:
            yield source, batch


ROW_FIELDS = ("id", "example_id", "source", "kind", "state", "question", "options", "target")
BAD = Path(__file__).resolve().parent.parent / "src" / "tasksource" / "metadata" / "jev_bad_examples.csv"


def parse(reply, batch):
    out = {}
    for line in str(reply).splitlines():
        match = re.search(r"\{.*\}", line)
        if not match:
            continue
        try:
            item = json.loads(match.group(0))
            row = batch[int(item["i"]) - 1]
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        if item.get("verdict") in VERDICTS:
            out[row["example_id"]] = {"example_id": row["example_id"], "source": row["source"], "id": row["id"],
                                      "verdict": item["verdict"], "better": item.get("better"),
                                      "reason": str(item.get("reason", ""))[:200],
                                      "row": {k: row[k] for k in ROW_FIELDS}}
    return out


def confirm(args):
    """Re-check flagged examples one by one (DeepSeek reasoning, Jev) and write the bad examples."""
    import pandas as pd
    from litlm import complete
    verdicts = {}
    for line in (args.out / "verdicts.jsonl").open():
        verdict = json.loads(line)
        verdicts[verdict["example_id"]] = verdict
    flagged = [v for v in verdicts.values() if v["verdict"] != "ok"]
    path = args.out / "confirmed.jsonl"
    confirmed = {json.loads(line)["example_id"]: json.loads(line) for line in path.open()} if path.exists() else {}
    todo = [v for v in flagged if v["example_id"] not in confirmed]
    print(f"{len(verdicts)} checked, {len(flagged)} flagged, {len(todo)} to confirm", flush=True)
    rows = [v["row"] for v in todo]
    jev_args = argparse.Namespace(out=args.out, chunk=500, concurrency=32, budget_usd=args.jev_budget)
    config = AnnotatorConfig(name="jev", version="~typesafe/jev-latest", base_url="https://openrouter.ai",
                             api_path="/api/alpha/decisions", api_key_env="JEV_OPENROUTER_API_KEY",
                             model="~typesafe/jev-latest")
    bundles = bundles_of(rows)
    results, _ = annotate(bundles, config, jev_args)
    jev = {}
    for bundle in bundles:
        result = results.get(id(bundle))
        for row, annotation in zip(bundle["rows"], (result or {}).get("annotations", [])):
            jev[row["example_id"]] = gold_probability(row, annotation["probabilities"])
    keys = [key for key in args.keys if os.environ.get(key)]
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        replies = complete([adjudication_prompt(v["row"]) for v in chunk], model=args.model,
                           api_key=os.environ[keys[(start // args.chunk) % len(keys)]], temperature=0.0,
                           max_tokens=4000, max_concurrency=args.concurrency, rpm=args.rpm, timeout=300,
                           num_retries=5, show_progress=False)
        with path.open("a") as handle:
            for verdict, reply in zip(chunk, replies):
                row = verdict["row"]
                n_options = 2 if row["kind"] == "noul" else len(row["options"])
                answer = parse_letter(reply, n_options)
                if answer is None or verdict["example_id"] not in jev:
                    continue  # retried on the next run
                record = {k: verdict[k] for k in ("example_id", "source", "id", "verdict", "reason")}
                record.update(deepseek=answer, gold=gold_answer(row), jev_gold_probability=round(jev[verdict["example_id"]], 3))
                record["bad"] = answer != record["gold"] and record["jev_gold_probability"] < 0.3
                confirmed[record["example_id"]] = record
                handle.write(json.dumps(record) + "\n")
        print(f"confirmed {start + len(chunk)}/{len(todo)} {time.strftime('%H:%M')}", flush=True)
    bad = pd.DataFrame([r for r in confirmed.values() if r["bad"]],
                       columns=["example_id", "source", "id", "verdict", "reason", "deepseek", "gold", "jev_gold_probability"])
    bad.sort_values(["source", "example_id"]).to_csv(BAD, index=False)
    checked = pd.Series([v["source"] for v in verdicts.values()]).value_counts()
    rates = (bad.source.value_counts().reindex(checked.index, fill_value=0) / checked).rename("bad_rate")
    pd.concat([checked.rename("checked"), rates], axis=1).sort_values("bad_rate", ascending=False).to_csv(
        args.out / "per-source.csv", index_label="source")
    print(f"{len(bad)} bad examples of {len(verdicts)} checked -> {BAD}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("build/jev-error-pass"))
    parser.add_argument("--model", default="albert/deepseek-v4-flash-0731")
    parser.add_argument("--keys", nargs="+", default=["ALBERT_API_KEY", "ALBERT_API_KEY_2"])
    parser.add_argument("--rpm", type=float, default=45, help="per key; Albert allows 50")
    parser.add_argument("--concurrency", type=int, default=12, help="per key")
    parser.add_argument("--max-items", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=16_000, help="text per request; longer states go alone")
    parser.add_argument("--chunk", type=int, default=400, help="requests per key between saves")
    parser.add_argument("--limit", type=int, help="examples to check (a probe)")
    parser.add_argument("--confirm", action="store_true", help="re-check flagged examples, write the bad list")
    parser.add_argument("--jev-budget", type=float, default=5.0)
    args = parser.parse_args()
    from litlm import Failure, complete
    if args.confirm:
        return confirm(args)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "verdicts.jsonl"
    done = {json.loads(line)["example_id"] for line in path.open()} if path.exists() else set()
    default_ids = set(load_dataset(REPO, "default", split="train").select_columns(["example_id"])[:]["example_id"])
    rows, seen = [], set(done)
    for split in ("train", "validation", "test"):
        full = load_dataset(REPO, "full", split=split)
        full = full.filter(lambda v: v == "direct", input_columns="variant", desc=f"direct {split}")
        for row in full:
            if row["example_id"] not in seen and checkable(row):
                seen.add(row["example_id"])
                rows.append(row)
    rows.sort(key=lambda row: row["example_id"] not in default_ids)  # stable: default first, then the rest
    if args.limit:
        rows = rows[:args.limit]
    requests = list(batches(rows, args.max_items, args.max_chars))
    print(f"{len(done)} done; {len(rows)} examples to check in {len(requests)} requests", flush=True)

    lock = threading.Lock()

    def worker(key, mine):
        for start in range(0, len(mine), args.chunk):
            chunk = mine[start:start + args.chunk]
            prompts = [INSTRUCTIONS.format(source=source, examples="\n\n".join(
                render(i, row, 4 * args.max_chars) for i, row in enumerate(batch, 1))) for source, batch in chunk]
            replies = complete(prompts, model=args.model, api_key=os.environ[key], temperature=0.0,
                               max_tokens=8000, max_concurrency=args.concurrency, rpm=args.rpm, timeout=300,
                               num_retries=5, show_progress=False)
            verdicts = {}
            for (_, batch), reply in zip(chunk, replies):
                if reply is not None and not isinstance(reply, Failure):  # failures retry on the next run
                    verdicts.update(parse(reply, batch))
            with lock, path.open("a") as handle:
                for verdict in verdicts.values():
                    if verdict["verdict"] == "ok":
                        verdict = {k: v for k, v in verdict.items() if k != "row"}  # keep rows of flags only
                    handle.write(json.dumps(verdict, ensure_ascii=False) + "\n")
            print(f"[{key}] {start + len(chunk)}/{len(mine)} requests, {len(verdicts)} verdicts "
                  f"({sum(v['verdict'] != 'ok' for v in verdicts.values())} flagged) {time.strftime('%H:%M')}",
                  flush=True)

    keys = [key for key in args.keys if os.environ.get(key)]
    threads = [threading.Thread(target=worker, args=(key, requests[i::len(keys)])) for i, key in enumerate(keys)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
