"""Error detection over the published tasksource-jev-typed-decisions: DeepSeek V4 Flash (Albert) checks
the gold of every source example, several examples of one source per prompt.

One direct row per `example_id` is checked, examples of the `default` config first. Verdicts go to
build/jev-error-pass/verdicts.jsonl (resumable), flagged rows to flagged.jsonl.

Batched screening over-flags hard examples (adversarial NLI), so `--confirm` re-checks each flagged
example alone: DeepSeek reasoning step by step after examples of the same source, and Jev. An example is
bad when both reject the gold (DeepSeek picks another option and Jev gives the gold less than 0.3).
Models cannot tell hard from wrong, so constructed sources are not checked, and a source with more than
MAX_BAD_RATE bad is listed for review instead of losing its examples. The removable examples go to
build/jev-error-pass/bad-examples.csv; once reviewed, copying them to
src/tasksource/metadata/jev_bad_examples.csv makes the build drop them.

    set -a; . ~/.jev_synth.env; set +a
    PYTHONPATH=.:src python scripts/jev_error_pass.py --limit 2000   # probe
    PYTHONPATH=.:src python scripts/jev_error_pass.py                # everything
    PYTHONPATH=.:src python scripts/jev_error_pass.py --confirm
"""

import argparse
import json
import os
import re
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

Return every displayed example exactly once, using these indices: {indices}.
Reply with one JSON object, nothing else:
{{"items":[{{"i": <number>, "verdict": "ok|wrong|ambiguous|malformed", "better": "<label of the correct option or null>", "reason": "<at most 12 words>"}}]}}"""


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
    """Hard labels of curated sources only: soft crowd distributions and uncertain yes/no targets are not
    errors, and constructed labels are right by design."""
    if re.search(CONSTRUCTED, row["source"]):
        return False
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
# labels from a generator, solver or simulator: a model rejecting them means a hard example, not an error
CONSTRUCTED = (r"^procedural-typed-decisions/|logical-entailment|FOL-nli|LogicNLI|FLD|proofwriter|ruletaker|PARARULE|"
               r"robustLR|clutrr|stepgame|babi_nli|SpaRTUN|spartqa|tomi-nli|mindgames|corr2cause|cladder|nlgraph|"
               r"conceptrules|regset|satisfiability|monotonicity|missing-item|synthetic-retrieval-NLI")
# bad apples are rare: above this rate a source disagrees systematically (a convention, or hard data) and is
# reviewed by reading it instead of losing its flagged examples
MAX_BAD_RATE = 0.08
REFERENCES = 3  # examples judged ok, shown when confirming flags of the same source
BAD = Path(__file__).resolve().parent.parent / "src" / "tasksource" / "metadata" / "jev_bad_examples.csv"


def parse(reply, batch):
    out = {}
    decoder = json.JSONDecoder()
    text = str(reply)
    for match in re.finditer(r'\{', text):
        try:
            item, _ = decoder.raw_decode(text[match.start():])
            if not (type(item.get('i')) is int or
                    isinstance(item.get('i'), str) and item['i'].isdigit()):
                continue
            index = int(item["i"])
            if not 1 <= index <= len(batch):
                continue
            row = batch[index - 1]
            if row is None:
                continue
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        if item.get("verdict") in VERDICTS:
            out[row["example_id"]] = {"example_id": row["example_id"], "source": row["source"], "id": row["id"],
                                      "verdict": item["verdict"], "better": item.get("better"),
                                      "reason": str(item.get("reason", ""))[:200],
                                      "row": {k: row[k] for k in ROW_FIELDS}}
    return out


def confirmation_prompt(row, references):
    """The adjudication prompt, after examples of the same source with their gold: conventions such as
    SNLI's "contradiction" for unrelated captions are the dataset's, not errors."""
    if not references:
        return adjudication_prompt(row)
    shown = "\n\n".join(render(i, reference, 4000) for i, reference in enumerate(references, 1))
    return (f"Examples from the same dataset ({row['source']}) with their gold answers, showing its labeling "
            f"conventions:\n\n{shown}\n\nNow the example to answer, following those conventions.\n\n"
            + adjudication_prompt(row))


def capture_recapture(verdicts, second, out):
    """Per source, on Jev's sample: DeepSeek's flags and Jev's doubts are two catches of the wrong labels;
    their overlap estimates how many both miss (Chapman's estimator)."""
    import pandas as pd
    rows = []
    for example_id, models in second.items():
        if "jev" in models and example_id in verdicts:
            verdict = verdicts[example_id]
            rows.append({"source": verdict["source"], "deepseek": verdict.get("screen", verdict["verdict"]) != "ok",
                         "jev": "row" in models["jev"]})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    stats = frame.groupby("source").agg(sample=("jev", "size"), deepseek=("deepseek", "sum"), jev=("jev", "sum"),
                                        both=("jev", lambda j: (j & frame.loc[j.index, "deepseek"]).sum()))
    stats["estimated_wrong"] = (stats.deepseek + 1) * (stats.jev + 1) / (stats.both + 1) - 1
    stats["estimated_missed"] = (stats.estimated_wrong - (stats.deepseek + stats.jev - stats.both)).clip(lower=0)
    stats["missed_rate"] = stats.estimated_missed / stats["sample"]
    stats.sort_values("missed_rate", ascending=False).round(3).to_csv(out / "capture-recapture.csv")
    print(f"capture-recapture: {stats.estimated_missed.sum():.0f} wrong labels missed by both in "
          f"{stats['sample'].sum()} sampled examples", flush=True)


def confirm(args):
    """Re-check flagged examples one by one (DeepSeek reasoning, Jev) and write the bad examples."""
    import pandas as pd
    from litlm import complete
    verdicts = {}
    for line in (args.out / "verdicts.jsonl").open():
        verdict = json.loads(line)
        verdicts[verdict["example_id"]] = verdict
    second = {}  # gold probabilities of decision models (jev_second_detector.py); their doubts get confirmed too
    for detector in sorted(args.out.glob("second-*.jsonl")):
        for line in detector.open():
            record = json.loads(line)
            second.setdefault(record["example_id"], {})[record["model"]] = record
            if "row" in record and record["example_id"] in verdicts and verdicts[record["example_id"]]["verdict"] == "ok":
                verdicts[record["example_id"]] = {**verdicts[record["example_id"]], "row": record["row"],
                                                  "verdict": "wrong", "reason": f"{record['model']} doubt",
                                                  "screen": "ok"}
    capture_recapture(verdicts, second, args.out)
    flagged = [v for v in verdicts.values() if v["verdict"] != "ok" and checkable(v['row'])]
    references = {}
    for verdict in verdicts.values():
        if verdict["verdict"] == "ok" and "row" in verdict:
            references.setdefault(verdict["source"], []).append(verdict["row"])
    path = args.out / "confirmed.jsonl"
    confirmed = {json.loads(line)["example_id"]: json.loads(line) for line in path.open()} if path.exists() else {}
    todo = [v for v in flagged if v["example_id"] not in confirmed]
    if args.confirm_limit:
        import random
        random.Random(43).shuffle(todo)
        todo = todo[:args.confirm_limit]
    print(f"{len(verdicts)} checked, {len(flagged)} flagged, {len(todo)} to confirm", flush=True)
    rows = [v["row"] for v in todo]
    jev_args = argparse.Namespace(out=args.out, chunk=500, concurrency=32, budget_usd=args.jev_budget)
    config = AnnotatorConfig(name="jev", version="~typesafe/jev-latest", base_url="https://openrouter.ai",
                             api_path="/api/alpha/decisions", api_key_env="JEV_OPENROUTER_API_KEY",
                             model="~typesafe/jev-latest")
    jev = {v['example_id']: second[v['example_id']]['jev']['gold_probability'] for v in todo
           if 'jev' in second.get(v['example_id'], {})}
    bundles = bundles_of([row for row in rows if row['example_id'] not in jev])
    results, _ = annotate(bundles, config, jev_args)
    for bundle in bundles:
        result = results.get(id(bundle))
        for row, annotation in zip(bundle["rows"], (result or {}).get("annotations", [])):
            jev[row["example_id"]] = gold_probability(row, annotation["probabilities"])
    keys = [key for key in args.keys if os.environ.get(key)]
    if not keys:
        raise ValueError('No configured provider keys are available')
    # Only request confirmations with an independent detector result available.
    # A reached Jev budget must not trigger thousands of unusable DeepSeek calls.
    todo = [v for v in todo if v['example_id'] in jev]
    print(f'{len(todo)} flags have independent scores available for confirmation', flush=True)
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        replies = complete([confirmation_prompt(v["row"], references.get(v["source"], [])[:REFERENCES]) for v in chunk],
                           model=args.model,
                           api_key_envs=keys, per_key_rpm=args.rpm, temperature=0.0,
                           max_tokens=8192, max_concurrency=args.concurrency, timeout=300,
                           extra_body={'chat_template_kwargs': {'thinking': True}},
                           response_format={'type': 'text'},
                           num_retries=0, caching=True, show_progress=False, progress_interval=30)
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
    checked = pd.Series([v["source"] for v in verdicts.values()]).value_counts()
    sources = pd.concat([checked.rename("checked"), bad.source.value_counts().reindex(checked.index, fill_value=0)
                         .rename("bad")], axis=1)
    sources["bad_rate"] = sources.bad / sources.checked
    sources["action"] = ["review" if rate > MAX_BAD_RATE else "remove" if n else "" for rate, n in
                         zip(sources.bad_rate, sources.bad)]
    sources.sort_values("bad_rate", ascending=False).to_csv(args.out / "per-source.csv", index_label="source")
    removed = bad[bad.source.isin(sources.index[sources.action == "remove"])]
    removed.sort_values(["source", "example_id"]).to_csv(args.bad, index=False)
    print(f"{len(bad)} bad examples of {len(verdicts)} checked; {len(removed)} removed -> {args.bad}; "
          f"{(sources.action == 'review').sum()} sources to review -> {args.out / 'per-source.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("build/jev-error-pass"))
    parser.add_argument("--model", default="albert/deepseek-v4-flash-0731")
    parser.add_argument("--keys", nargs="+", default=sorted(k for k in os.environ if re.fullmatch(r"ALBERT_API_KEY(_\d+)?", k)))
    parser.add_argument("--rpm", type=float, default=45, help="per key; Albert allows 50")
    parser.add_argument("--concurrency", type=int, default=12, help="per key")
    parser.add_argument("--max-items", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=16_000, help="text per request; longer states go alone")
    parser.add_argument("--chunk", type=int, default=400, help="requests per key between saves")
    parser.add_argument("--limit", type=int, help="examples to check (a probe)")
    parser.add_argument("--confirm", action="store_true", help="re-check flagged examples, write the bad list")
    parser.add_argument("--confirm-limit", type=int, help="limit pending confirmations for a bounded pilot")
    parser.add_argument("--jev-budget", type=float, default=5.0)
    parser.add_argument("--bad", type=Path, help="where --confirm writes the removable examples (default: in --out; "
                        f"copy them to {BAD.name} once reviewed, and the build drops them)")
    args = parser.parse_args()
    from litlm import Failure, complete
    if args.confirm:
        args.bad = args.bad or args.out / "bad-examples.csv"
        return confirm(args)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "verdicts.jsonl"
    done = {json.loads(line)["example_id"] for line in path.open()} if path.exists() else set()
    default_ids = set(load_dataset(REPO, "default", split="train").select_columns(["example_id"])[:]["example_id"])
    rows, seen = [], set(done)
    for split in ("train", "validation", "test"):
        full = load_dataset(REPO, "full", split=split)
        full = full.filter(lambda identifiers, variants: [variant == 'direct' and identifier not in done
                            for identifier, variant in zip(identifiers, variants)], batched=True,
                           input_columns=['example_id', 'variant'], desc=f'pending direct {split}')
        for row in full:
            if row["example_id"] not in seen and checkable(row):
                seen.add(row["example_id"])
                rows.append(row)
    raw_path = args.out / 'screen-retry-raw.jsonl'
    if raw_path.exists():
        by_id = {row['example_id']: row for row in rows}
        recovered = {}
        for line in raw_path.open():
            record = json.loads(line)
            if not record.get('failed'):
                recovered.update(parse(record['text'], [by_id.get(key) for key in record['example_ids']]))
        with path.open('a') as handle:
            for record in recovered.values():
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        rows = [row for row in rows if row['example_id'] not in recovered]
        done.update(recovered)
        print(f'{len(recovered)} pending verdicts recovered from raw responses', flush=True)
    rows.sort(key=lambda row: row["example_id"] not in default_ids)  # stable: default first, then the rest
    if args.limit:
        rows = rows[:args.limit]
    requests = list(batches(rows, args.max_items, args.max_chars))
    print(f"{len(done)} done; {len(rows)} examples to check in {len(requests)} requests", flush=True)

    references = {}
    if path.exists():
        for line in path.open():
            verdict = json.loads(line)
            if verdict["verdict"] == "ok" and "row" in verdict:
                references[verdict["source"]] = references.get(verdict["source"], 0) + 1

    keys = [key for key in args.keys if os.environ.get(key)]
    if not keys:
        raise ValueError('No configured provider keys are available')
    prompts = [INSTRUCTIONS.format(source=source, indices=', '.join(str(i) for i in range(1, len(batch) + 1)), examples='\n\n'.join(
        render(i, row, 4 * args.max_chars) for i, row in enumerate(batch, 1))) for source, batch in requests]
    settled = 0
    def save(index, reply):
        nonlocal settled
        settled += 1
        source, batch = requests[index]
        verdicts = parse(reply, batch) if reply is not None and not isinstance(reply, Failure) else {}
        with (args.out / 'screen-retry-raw.jsonl').open('a') as handle:
            handle.write(json.dumps({'source': source, 'example_ids': [row['example_id'] for row in batch],
                                     'text': str(reply), 'failed': isinstance(reply, Failure),
                                     'error': str(reply.error) if isinstance(reply, Failure) else None,
                                     'received_verdicts': len(verdicts)}, ensure_ascii=False) + '\n')
        with path.open('a') as handle:
            for verdict in verdicts.values():
                if verdict['verdict'] == 'ok':
                    references[source] = references.get(source, 0) + 1
                    if references[source] > REFERENCES:
                        verdict = {k: v for k, v in verdict.items() if k != 'row'}
                handle.write(json.dumps(verdict, ensure_ascii=False) + '\n')
        if settled % args.chunk == 0 or settled == len(requests):
            print(f'{settled}/{len(requests)} requests checkpointed {time.strftime("%H:%M")}', flush=True)
    replies = complete(prompts, model=args.model, api_key_envs=keys, per_key_rpm=args.rpm,
                       temperature=0.0, max_tokens=8000, max_concurrency=args.concurrency * len(keys), timeout=300,
                       num_retries=0, caching=True, json=True, show_progress=False, progress_interval=30, on_result=save)
    print(replies.summary(), flush=True)


if __name__ == "__main__":
    main()
