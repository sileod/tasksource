"""Per-source label audit of the Jev ``extraction`` config with an LLM.

For each source, up to ``--per-source`` rows are drawn (test split first, then validation, then train).
A model fills every field blind, as an extraction model would; where its answer differs from the gold
one, a judge sees the text, the field, and both answers in random order, and says which is correct.
Siding with the model points to label noise or ambiguity, siding with gold to model errors.

    PYTHONPATH=.:src python scripts/audit_jev_extraction.py --data build/jev-extraction/dataset \\
        --model albert/deepseek-v4-flash-0731 --api-key-envs ALBERT_API_KEY ALBERT_API_KEY_2

``--model`` defaults to ``OPENAI_MODEL`` on ``OPENAI_BASE_URL`` (see ``tasksource.llm``); litlm routes
(``openrouter/...``) pass through. Calls are cached by litlm.
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset, load_from_disk

from tasksource import llm

SPLIT_PREFERENCE = ("test", "validation", "train")


def sample(dataset, per_source, seed):
    picked, seen = [], defaultdict(int)
    for split in SPLIT_PREFERENCE:
        if split not in dataset:
            continue
        rows = dataset[split].shuffle(seed=seed)
        for row in rows:
            if seen[row["source"]] < per_source:
                seen[row["source"]] += 1
                picked.append(row)
    return picked


def extraction_prompt(row):
    fields = [{"name": f["name"], "type": f["type"], "instruction": f["instruction"]} for f in row["fields"]]
    return ("Extract the fields from the text. Copy values verbatim from the text. A `text` field has one value, "
            "a `list` field every matching value. Use null when the text does not contain the answer.\n\n"
            f"Text:\n{row['state']}\n\nFields:\n{json.dumps(fields, ensure_ascii=False, indent=1)}\n\n"
            'Answer with one JSON object: {"<name>": null | ["<value>", ...], ...}')


def norm(text):
    return re.sub(r"\W+", " ", text.casefold()).strip()


def agree(field, gold, predicted):
    """Gold and model answers match: both null, a shared alternative (``text``) or the same set (``list``)."""
    gold = {norm(v["text"]) for v in gold["values"]} if not gold["null"] else set()
    predicted = {norm(v) for v in predicted or [] if isinstance(v, str) and norm(v)}
    if not gold or not predicted:
        return gold == predicted
    return bool(gold & predicted) if field["type"] == "text" else gold == predicted


def show(answer):
    return "null (not in the text)" if not answer else json.dumps(answer, ensure_ascii=False)


def judge_prompt(item, flip):
    row, field = item["row"], item["field"]
    a, b = (item["predicted"], item["gold"]) if flip else (item["gold"], item["predicted"])
    kind = "one value (any correct span suffices)" if field["type"] == "text" else "every matching value"
    return (f"Text:\n{row['state']}\n\nField `{field['name']}` ({kind}): {field['instruction']}\n\n"
            f"Answer A: {show(a)}\nAnswer B: {show(b)}\n\n"
            "Which answer is correct for this text? Check each value against the text and the field's "
            "definition. If both name the same things and differ only in span extent (an added unit, title or "
            "gloss), reply BOTH; "
            "a null answer and a non-null one cannot both be correct. Reply with one word: A, B, BOTH or NEITHER.")


def run(prompts, args, json_mode=False, reasoning_effort=None):
    from litlm import complete
    if not prompts:
        return []
    replies = complete(prompts, model=args.model, json=json_mode, caching=True, max_tokens=args.max_tokens,
                       temperature=0.0, timeout=300, num_retries=0, api_key_envs=args.api_key_envs or None,
                       per_key_rpm=args.per_key_rpm if args.api_key_envs else None,
                       rpm=None if args.api_key_envs else args.rpm, reasoning_effort=reasoning_effort)
    for _ in range(3):  # rate limits: retry only the failed positions
        if not replies.failures:
            break
        time.sleep(30)
        replies.resume(num_retries=3, timeout=300)
    return replies


def parse_json(reply):
    if isinstance(reply, dict):  # litlm parses JSON replies
        return reply
    match = re.search(r"\{.*\}", str(reply or ""), re.S)
    try:
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="save_to_disk directory (default: the Hub config)")
    parser.add_argument("--repo-id", default="tasksource/tasksource-jev-typed-decisions")
    parser.add_argument("--output", type=Path, default=Path("build/audit-jev-extraction"))
    parser.add_argument("--per-source", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--api-key-envs", nargs="*", help="key pool, round-robin (litlm)")
    parser.add_argument("--per-key-rpm", type=float, default=40)
    parser.add_argument("--rpm", type=float, default=60)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort", help="for the blind extraction (default: off)")
    parser.add_argument("--judge-reasoning-effort", default="low")
    args = parser.parse_args()
    args.model = llm.litlm_model(args.model)
    dataset = load_from_disk(args.data) if args.data else load_dataset(args.repo_id, "extraction")
    rows = sample(dataset, args.per_source, args.seed)
    print(f"extracting {len(rows)} rows with {args.model}", flush=True)
    replies = run([extraction_prompt(r) for r in rows], args, json_mode=True, reasoning_effort=args.reasoning_effort)

    items = []
    for row, reply in zip(rows, replies):
        parsed = parse_json(reply)
        for field, gold in zip(row["fields"], row["answers"]):
            predicted = parsed.get(field["name"]) if isinstance(parsed, dict) else None
            predicted = [predicted] if isinstance(predicted, str) else predicted
            items.append({"row": row, "field": field, "gold": [v["text"] for v in gold["values"]] or None,
                          "predicted": predicted or None, "parsed": parsed is not None,
                          "agree": parsed is not None and agree(field, gold, predicted)})
    print(f"parsed {sum(i['parsed'] for i in items)}/{len(items)} fields", flush=True)
    # A model answer that is not verbatim in the text cannot beat the gold span: no judge call.
    for item in items:
        if item["parsed"] and not item["agree"] and not all(
                isinstance(v, str) and v in item["row"]["state"] for v in item["predicted"] or []):
            item["verdict"] = "gold"
    disputed = [i for i in items if i["parsed"] and not i["agree"] and "verdict" not in i]
    rng = random.Random(args.seed)
    flips = [rng.random() < 0.5 for _ in disputed]
    print(f"judging {len(disputed)} disagreements", flush=True)
    verdicts = run([judge_prompt(i, f) for i, f in zip(disputed, flips)], args, reasoning_effort=args.judge_reasoning_effort)
    for item, flip, reply in zip(disputed, flips, verdicts):
        words = re.findall(r"\b(A|B|BOTH|NEITHER)\b", str(reply or "").upper())
        word = words[-1] if words else None
        item["verdict"] = {"A": "model" if flip else "gold", "B": "gold" if flip else "model",
                           "BOTH": "both", "NEITHER": "neither"}.get(word)

    per_source = defaultdict(list)
    for item in items:
        per_source[item["row"]["source"]].append(item)
    args.output.mkdir(parents=True, exist_ok=True)
    lines = ["| source | fields | parsed | agree | disputed | gold | model | both | neither | est. label error |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for source, group in sorted(per_source.items()):
        parsed = [i for i in group if i["parsed"]]
        counts = {k: sum(i.get("verdict") == k for i in parsed if not i["agree"]) for k in ("gold", "model", "both", "neither")}
        error = (counts["model"] + counts["neither"]) / max(1, len(parsed))
        summary[source] = dict(fields=len(group), parsed=len(parsed), agree=sum(i["agree"] for i in parsed),
                               disputed=len(parsed) - sum(i["agree"] for i in parsed), **counts, label_error=round(error, 3))
        s = summary[source]
        lines.append(f"| {source} | {s['fields']} | {s['parsed']} | {s['agree']} | {s['disputed']} | {s['gold']} | "
                     f"{s['model']} | {s['both']} | {s['neither']} | {error:.1%} |")
    report = (f"# Extraction label audit\n\nModel `{args.model}`, {args.per_source} rows per source, blind extraction "
              "then a judge on disagreements; a model answer not verbatim in the text counts for gold. Estimated label error: fields where the judge prefers the model "
              "answer or neither, over parsed fields.\n\n" + "\n".join(lines) + "\n")
    (args.output / "report.md").write_text(report)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    with open(args.output / "items.jsonl", "w") as out:
        for item in items:
            row = item.pop("row")
            out.write(json.dumps(dict(item, id=row["id"], source=row["source"], split=row["split"]), ensure_ascii=False) + "\n")
    print(report)


if __name__ == "__main__":
    main()
