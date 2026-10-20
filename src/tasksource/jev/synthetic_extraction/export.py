"""Export a generation run to tasksource/synthetic-extraction rows and optionally publish them.

    PYTHONPATH=src python -m tasksource.jev.synthetic_extraction.export --run DIR [--upload]

Rows follow the ``extraction`` config of tasksource-jev-typed-decisions; only verified fields are kept, and a
document needs at least ``MIN_FIELDS`` of them. Splits are by schema (an evaluation schema is never seen in
train), so evaluation measures transfer to a new application's fields.
"""
import argparse
import hashlib
import json
import subprocess
import zlib
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

from ...citation import card_section
from ..extraction import FEATURES, check
from . import __main__ as generation

REPO_ID = "tasksource/synthetic-extraction"
MIN_FIELDS = 2
SPLIT_BUCKETS = (("validation", 1), ("test", 1))  # out of 10 buckets of crc32(schema_id); the rest is train


def split_of(schema_id: str) -> str:
    bucket = zlib.crc32(schema_id.encode()) % 10
    for split, size in SPLIT_BUCKETS:
        if bucket < size:
            return split
        bucket -= size
    return "train"


def to_row(item: dict, schema: dict) -> dict | None:
    kept = [f for f in item["fields"] if f["kept"]]
    if len(kept) < MIN_FIELDS:
        return None
    split = split_of(item["schema_id"])
    row = {"id": item["document_id"], "source": "synthetic-extraction", "split": split, "state": item["document"],
           "fields": [{k: f[k] for k in ("name", "type", "instruction")} for f in kept],
           "answers": [{"null": not f["spans"], "values": f["spans"]} for f in kept],
           "metadata": json.dumps({
               "schema_id": item["schema_id"], "domain": item["domain"], "document_source": schema["source"],
               "document_kind": schema["document_kind"], "application": schema["application"],
               "plans": {f["name"]: item["plans"][f["name"]] for f in kept},
               "judged": [f["name"] for f in kept if f["judge"]],
               "gold_from_readers": [f["name"] for f in kept if f.get("gold_from") == "readers"],
               "dropped": {f["name"]: f["drop"] for f in item["fields"] if not f["kept"]}}, ensure_ascii=False)}
    return check(row)


def load_rows(run: Path) -> dict[str, list[dict]]:
    schemas = {s["schema_id"]: s for s in map(json.loads, (run / "schemas.jsonl").open())}
    items = {}
    for line in (run / "items.jsonl").open():  # a resumed run may repeat a document: the last one wins
        item = json.loads(line)
        items[item["document_id"]] = item
    splits = {"train": [], "validation": [], "test": []}
    for document_id in sorted(items):
        row = to_row(items[document_id], schemas[items[document_id]["schema_id"]])
        if row:
            splits[row["split"]].append(row)
    return splits


def field_stats(run: Path) -> dict:
    fields = [f for line in (run / "items.jsonl").open() for f in json.loads(line)["fields"]]
    return {"fields": len(fields), "kept": sum(f["kept"] for f in fields),
            "dropped": dict(Counter(f["drop"] for f in fields if not f["kept"]))}


def provenance(run: Path, args_used: dict) -> dict:
    root = Path(__file__).resolve().parents[4]
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    prompts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(generation.PROMPTS.glob("*.txt"))}
    return {"generator": "https://github.com/sileod/tasksource/tree/main/src/tasksource/jev/synthetic_extraction",
            "commit": commit, "prompts_sha256": prompts, "run": args_used, "field_verification": field_stats(run)}


def card(summary: dict, prov: dict) -> str:
    configs = "".join(f"  - split: {s}\n    path: data/{s}-*.parquet\n" for s in summary)
    run, stats = prov["run"], prov["field_verification"]
    return f"""---
pretty_name: synthetic-extraction
language:
- en
license: apache-2.0
task_categories:
- token-classification
- question-answering
tags:
- tasksource
- jev
- extraction
- clairic
- synthetic
configs:
- config_name: default
  data_files:
{configs}---

# synthetic-extraction

LLM-written extraction requests: one natural document and the fields an application would extract from it, typed
`text` (one span) or `list` (every matching span), answered with character spans or null, as extraction models
(e.g. Cloudflare's Clairic) are asked. It covers fields that need reading rather than a solver (the corrected date in
a thread, who is assigned an action, the item a complaint is about); computable fields are in
[procedural-extraction](https://huggingface.co/datasets/tasksource/procedural-extraction). It is a small booster for
the `extraction` config of
[tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions).

## How it is made

1. **Design**: per domain and document source, `{run['writer']}` drafts an application's schema: a document kind
   and typed fields with precise instructions.
2. **Write**: it writes documents with an intended answer per field, each planned as present, present beside a near
   miss (a similar value that does not qualify), or absent beside a near miss, so nulls are earned.
3. **Verify**: answers must be verbatim spans. Two blind readers of other model families
   ({', '.join(f'`{m}`' for m in run['readers'])}) fill every field. A field is kept when both agree with the
   writer (up to case, edge punctuation and a leading article); when one disagrees, a judge sees both answers in a
   random order and the field is kept only for an unambiguous verdict for the writer. When both disagree it is
   dropped, unless both copy the same span and the judge rules for it over the writer's answer (the writer missed a
   value it wrote). Documents keep their verified fields (at least {MIN_FIELDS}).

{stats['kept']:,} of {stats['fields']:,} written fields passed verification. Splits are by schema: validation and
test schemas never occur in train. Prompts, models and the generation commit are in
[provenance.json](provenance.json).

| split | rows |
|---|---|
""" + "".join(f"| {s} | {n:,} |\n" for s, n in summary.items()) + f"""
## Schema

`state` (the document), `fields` (`[{{name, type, instruction}}]`), `answers` (one per field: `{{null, values:
[{{text, start, end}}]}}`, `state[start:end] == text`), `metadata` (JSON: schema, domain, document source, field
plans, judged and dropped fields).

{card_section()}"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="output directory of a generation run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--writer", default="deepseek-v4-flash-0731")
    parser.add_argument("--readers", nargs="+", default=["gpt-oss-120b", "gemma-4-31b-it"])
    parser.add_argument("--judge")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args(argv)
    output = args.output or args.run / "export"
    splits = {s: r for s, r in load_rows(args.run).items() if r}
    dataset = DatasetDict({s: Dataset.from_list(r, features=FEATURES) for s, r in splits.items()})
    dataset.save_to_disk(str(output / "data"))
    summary = {s: len(r) for s, r in splits.items()}
    prov = provenance(args.run, {"writer": args.writer, "readers": args.readers, "judge": args.judge or args.writer,
                                 "seed": args.seed, "schemas": sum(1 for _ in (args.run / "schemas.jsonl").open())})
    (output / "provenance.json").write_text(json.dumps({**prov, "rows": summary}, indent=1) + "\n")
    (output / "README.md").write_text(card(summary, prov))
    print(summary, prov["field_verification"])
    if args.upload:
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        dataset.push_to_hub(args.repo_id, commit_message="Publish synthetic-extraction")
        for path in ("provenance.json", "README.md"):
            api.upload_file(path_or_fileobj=str(output / path), path_in_repo=path, repo_id=args.repo_id,
                            repo_type="dataset", commit_message=f"Add {path}")
        print(api.dataset_info(args.repo_id).sha)


if __name__ == "__main__":
    main()
