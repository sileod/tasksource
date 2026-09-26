"""Build tasksource-instruct and tasksource_dpo_pairs from the task catalog.

Every catalog task with hard labels is loaded with pinned Hub revisions and
recast as an instruction (``tasksource.recast_instruct``): a prompt that offers
the answers, and the gold answer as target. The same rows give preference
pairs: the gold answer is ``chosen`` and another offered answer ``rejected``.
Token classification has no single alternative answer and stays instruct-only.

The build is resumable: each task is written to ``data/`` and recorded in
``build-report.jsonl`` (rows, source revisions, code commit); ``--finalize``
interleaves the tasks, adds license columns, and writes ``sources.yaml``.

    PYTHONPATH=.:src python scripts/build_instruct_dataset.py --finalize --upload
"""

import argparse
import json
import random
import time
import traceback
from pathlib import Path

import yaml
from datasets import DatasetDict, Features, Sequence, Value, load_dataset
from huggingface_hub import HfApi

from scripts.build_jev_dataset import (
    JEV_EXCLUDED_SOURCES, PUBLISH_EXCLUDED_PREFIXES, code_state, latest_records, resolve_revisions, slug,
    source_licenses, source_provenance,
)
from scripts.push_dataset_card import push_card
from tasksource import list_tasks, load_task
from tasksource.recast import recast_instruct

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
MAX_BYTES = 131_072  # a complete prompt and target, as in the Jev build
FEATURES = Features({"inputs": Value("string"), "targets": Value("string"), "options": Sequence(Value("string")),
                     "task": Value("string")})


def select_tasks(args):
    frame = list_tasks(instruct=True)
    frame = frame[~frame.id.str.startswith(PUBLISH_EXCLUDED_PREFIXES) & ~frame.id.isin(JEV_EXCLUDED_SOURCES)]
    if args.tasks:
        frame = frame[frame.id.isin(args.tasks)]
    return frame.head(args.limit) if args.limit else frame


def build_task(row, args):
    provenance = source_provenance(row.id)
    pins = resolve_revisions(provenance)
    revision = pins.get(provenance.get("dataset"))
    file_pins = {repo: pins.get(repo) for repo in provenance.get("data_files_from", []) if pins.get(repo)}
    dataset = load_task(row.id, max_rows=args.max_rows, max_rows_eval=args.max_rows_eval,
                        data_file_pins=file_pins, **({"revision": revision} if revision else {}))
    dataset = recast_instruct(dataset, question=dataset.question, seed=0, options=True)
    rows = {}
    for split, split_rows in dataset.items():
        if split not in SPLITS or not len(split_rows):
            continue
        split_rows = split_rows.filter(
            lambda x: len(x["inputs"].encode()) + len(x["targets"].encode()) <= MAX_BYTES)
        split_rows = split_rows.add_column("task", [row.id] * len(split_rows)).cast(FEATURES)
        split_rows.to_parquet(args.output / "data" / f"{split}-{slug(row.id)}.parquet")
        rows[split] = len(split_rows)
    return rows, {repo: sha for repo, sha in pins.items() if sha}


def build(args):
    (args.output / "data").mkdir(parents=True, exist_ok=True)
    report = args.output / "build-report.jsonl"
    done = {task for task, record in latest_records(report).items() if record["status"] == "ok"}
    tasks = select_tasks(args)
    print(f"Selected {len(tasks)} tasks; {len(done & set(tasks.id))} already built", flush=True)
    for position, row in enumerate(tasks.itertuples(index=False), start=1):
        if row.id in done:
            continue
        started = time.time()
        print(f"[{position}/{len(tasks)}] {row.id}", flush=True)
        try:
            rows, revisions = build_task(row, args)
            record = {"task": row.id, "task_type": row.task_type, "status": "ok", "rows": rows,
                      "revisions": revisions, "code": code_state()}
        except Exception as error:
            record = {"task": row.id, "status": "error", "error": f"{type(error).__name__}: {error}",
                      "traceback": traceback.format_exc()[-2000:]}
        record["seconds"] = round(time.time() - started, 1)
        with report.open("a") as handle:
            handle.write(json.dumps(record) + "\n")


def interleave(rows):
    """Round-robin tasks, so every stretch of the split mixes them."""
    tasks = rows.data.column("task").to_pylist()
    queues = {}
    for index, task in enumerate(tasks):
        queues.setdefault(task, []).append(index)
    order, depth = [], 0
    while queues:
        for task in list(queues):
            if depth < len(queues[task]):
                order.append(queues[task][depth])
            else:
                del queues[task]
        depth += 1
    return rows.select(order).flatten_indices()  # later column reads stay fast


def preference_pairs(rows):
    """Gold answer as ``chosen``, another offered answer (fixed per row) as ``rejected``."""
    def pair(x, index):
        others = [option for option in x["options"] if option != x["targets"]]
        rejected = random.Random(f"{x['task']}/{index}").choice(others) if others and x["targets"] in x["options"] else None
        return {"prompt": x["inputs"], "chosen": x["targets"], "rejected": rejected}
    pairs = rows.map(pair, with_indices=True, remove_columns=["inputs", "targets", "options"])
    return pairs.filter(lambda rejected: rejected is not None, input_columns="rejected")


def finalize(args):
    records = {task: record for task, record in latest_records(args.output / "build-report.jsonl").items()
               if record["status"] == "ok"}
    selected = set(select_tasks(args).id)
    files = {split: [args.output / "data" / f"{split}-{slug(task)}.parquet" for task in sorted(records)
                     if task in selected and split in records[task]["rows"]] for split in SPLITS}
    instruct = DatasetDict({split: interleave(load_dataset("parquet", data_files=[str(f) for f in paths], split="train"))
                            for split, paths in files.items() if paths})
    tasks = {split: rows.data.column("task").to_pylist() for split, rows in instruct.items()}
    licenses = source_licenses(sorted(set().union(*tasks.values())))
    def add_licenses(rows):
        tasks = rows.data.column("task").to_pylist()
        rows = rows.add_column("license", [licenses[t]["license"] for t in tasks])
        return rows.add_column("license_use", [licenses[t]["license_use"] for t in tasks])
    instruct = DatasetDict({split: add_licenses(rows) for split, rows in instruct.items()})
    pairs = DatasetDict({split: preference_pairs(rows) for split, rows in instruct.items()})
    instruct = DatasetDict({split: rows.remove_columns("options") for split, rows in instruct.items()})

    sources = {}
    for task in sorted(set().union(*tasks.values())):
        info = {"rows": {split: records[task]["rows"][split] for split in SPLITS if split in records[task]["rows"]}}
        info.update(source_provenance(task))
        info["revisions"] = records[task].get("revisions", {})
        info.update(licenses[task])
        sources[task] = info
    document = {"tasksource_commit": code_state()["git_commit"],
                "datasets": sorted({r for i in sources.values() for r in [i.get("dataset"), *i.get("originals", [])] if r}),
                "sources": sources}
    (args.output / "sources.yaml").write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=120))
    summary = {name: {split: len(rows) for split, rows in data.items()} for name, data in
               (("instruct", instruct), ("dpo_pairs", pairs))}
    (args.output / "build-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return instruct, pairs


def upload(args, instruct, pairs):
    api = HfApi()
    for repo_id, data, card in ((args.repo_id, instruct, "tasksource-instruct.md"),
                                (args.dpo_repo_id, pairs, "tasksource-dpo-pairs.md")):
        data.push_to_hub(repo_id, max_shard_size="512MB", commit_message="Rebuild from the tasksource catalog")
        push_card(ROOT / "dataset_cards" / card, repo_id, "Dataset card")
        for name in ("sources.yaml", "build-report.jsonl", "build-summary.json"):
            api.upload_file(path_or_fileobj=str(args.output / name), path_in_repo=name, repo_id=repo_id,
                            repo_type="dataset", commit_message=f"Add {name}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "tasksource-instruct")
    parser.add_argument("--max-rows", type=int, default=30_000)
    parser.add_argument("--max-rows-eval", type=int, default=500)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default="tasksource/tasksource-instruct")
    parser.add_argument("--dpo-repo-id", default="tasksource/tasksource_dpo_pairs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.finalize_only:
        build(args)
    if args.finalize or args.finalize_only:
        instruct, pairs = finalize(args)
        if args.upload:
            upload(args, instruct, pairs)
