"""Build zero-shot-label-nli from the catalog's classification tasks (``tasksource.recast.recast_label_nli``).

``default`` has premise, hypothesis ("This example is <label>.") and entailment/contradiction labels;
``triplet`` pairs each premise with the gold and a wrong hypothesis (anchor, positive, negative).
Tasks whose labels are already NLI labels, bare numbers, or describe the annotation are left out.
Training rows are balanced per task (``balance``).

The build is resumable and parallel: each task is written to ``data/`` and recorded in
``build-report.jsonl``; ``--finalize`` interleaves the tasks and adds license columns.

    PYTHONPATH=.:src python scripts/build_label_nli.py --workers 16 --finalize --upload
"""

import argparse
from collections import Counter
import json
import random
import re
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import yaml
from datasets import ClassLabel, DatasetDict, Features, Value, concatenate_datasets, load_dataset
from huggingface_hub import DatasetCard, HfApi

from scripts.build_instruct_dataset import interleave
from scripts.build_jev_dataset import (
    JEV_EXCLUDED_SOURCES, MAX_MAJORITY_RATIO, PUBLISH_EXCLUDED_PREFIXES, code_state, latest_records,
    resolve_revisions, slug, source_licenses, source_provenance,
)
from tasksource import list_tasks, load_task
from tasksource.jev.augmentations import stable_fraction
from tasksource.recast import recast_label_nli

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
MAX_BYTES = 131_072
NLI = ClassLabel(names=["entailment", "neutral", "contradiction"])
FEATURES = Features({"premise": Value("string"), "hypothesis": Value("string"), "positive": Value("string"),
                     "negative": Value("string"), "labels": NLI, "task": Value("string")})  # recast="label_nli"
GENERIC_LABELS = {"true", "false", "yes", "no"}  # say little about the text without the task's question
GENERIC_CAP = 1_000
SCRIPT_URL = "https://github.com/sileod/tasksource/blob/main/scripts/build_label_nli.py"


def select_tasks(args):
    frame = list_tasks(multilingual=args.multilingual)  # without improper label sets
    frame = frame[(frame.task_type == "Classification") & (frame.soft_labels != True)]
    frame = frame[~frame.id.str.startswith(PUBLISH_EXCLUDED_PREFIXES) & ~frame.id.isin(JEV_EXCLUDED_SOURCES)]
    if args.tasks:
        frame = frame[frame.id.isin(args.tasks)]
    return frame.head(args.limit) if args.limit else frame


def eligible(names, question=None):
    """Labels that read as classes of the text: named, not NLI labels already, no bare numbers,
    and not bare answers to the task's question ("This example is yes." drops the question)."""
    if not names or len(names) < 2:
        return "labels have no names"
    if question and {name.strip().lower() for name in names} <= GENERIC_LABELS:
        return "labels answer the task question"
    if "entail" in " ".join(names).lower():
        return "already NLI"
    if any(re.fullmatch(r"-?[\d.]+", name.strip()) for name in names):  # "This example is 2." says nothing
        return "numeric labels"
    if any("annotators" in name.lower() for name in names):
        return "labels describe the annotation, not the text"


def build_task(task, args):
    provenance = source_provenance(task)
    pins = resolve_revisions(provenance)
    revision = pins.get(provenance.get("dataset"))
    file_pins = {repo: pins.get(repo) for repo in provenance.get("data_files_from", []) if pins.get(repo)}
    kwargs = dict(max_rows=args.max_rows, max_rows_eval=args.max_rows_eval, multilingual=args.multilingual,
                  data_file_pins=file_pins, **({"revision": revision} if revision else {}))
    dataset = load_task(task, **kwargs)
    reason = eligible(getattr(dataset["train"].features["labels"], "names", None), dataset.question)
    if reason:
        return "skipped", reason, {}
    rows = {}
    for split, split_rows in recast_label_nli(dataset, task).items():
        split_rows = split_rows.filter(lambda premise: len(premise.encode()) <= MAX_BYTES, input_columns="premise")
        if split in SPLITS and len(split_rows):
            split_rows.cast(FEATURES).to_parquet(args.output / "data" / f"{split}-{slug(task)}.parquet")
            rows[split] = len(split_rows)
    return "ok", rows, {repo: sha for repo, sha in pins.items() if sha}


def balance(rows, task):
    """Training rows of one task: the majority gold label at most MAX_MAJORITY_RATIO times the
    other labels, and at most GENERIC_CAP rows when labels are only true/false/yes/no. Kept rows
    are a stable random sample."""
    gold = rows.data.column("positive").to_pylist()
    order = sorted(range(len(gold)), key=lambda i: stable_fraction(f"{task}/{i}", "label-nli-balance"))
    counts = {}
    for label in gold:
        counts[label] = counts.get(label, 0) + 1
    majority = max(counts, key=counts.get)
    limit = MAX_MAJORITY_RATIO * (len(gold) - counts[majority])
    seen, keep = 0, []
    for i in order:
        if gold[i] == majority:
            if seen >= limit:
                continue
            seen += 1
        keep.append(i)
    if {label[len("This example is "):-1].lower() for label in counts} <= GENERIC_LABELS:
        keep = keep[:GENERIC_CAP]
    return rows.select(sorted(keep)).flatten_indices()


def run_task(job):
    task, args = job
    started = time.time()
    try:
        status, rows, revisions = build_task(task, args)
        record = {"task": task, "status": status, "code": code_state()}
        record.update({"rows": rows, "revisions": revisions} if status == "ok" else {"reason": rows})
    except Exception as error:
        record = {"task": task, "status": "error", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc()[-2000:]}
    record["seconds"] = round(time.time() - started, 1)
    return record


def build(args):
    (args.output / "data").mkdir(parents=True, exist_ok=True)
    report = args.output / "build-report.jsonl"
    done = {task for task, record in latest_records(report).items() if record["status"] in ("ok", "skipped")}
    todo = [task for task in select_tasks(args).id if task not in done]
    print(f"{len(done)} tasks done, {len(todo)} to build", flush=True)
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for position, record in enumerate(pool.imap_unordered(run_task, [(task, args) for task in todo]), start=1):
            with report.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"[{position}/{len(todo)}] {record['task']}: {record['status']} {record.get('rows') or record.get('reason') or record.get('error', '')[:200]}", flush=True)


def finalize(args):
    records = {task: record for task, record in latest_records(args.output / "build-report.jsonl").items()
               if record["status"] == "ok"}
    selected = set(select_tasks(args).id)
    files = {split: [str(args.output / "data" / f"{split}-{slug(task)}.parquet") for task in sorted(records)
                     if task in selected and split in records[task]["rows"]] for split in SPLITS}
    def load(split, path, task):
        rows = load_dataset("parquet", data_files=path, split="train")
        return balance(rows, task) if split == "train" else rows
    data = DatasetDict({split: interleave(concatenate_datasets([
        load(split, path, task) for task in sorted(records) if task in selected and split in records[task]["rows"]
        for path in [str(args.output / "data" / f"{split}-{slug(task)}.parquet")]]))
        for split in SPLITS if files[split]})
    tasks = sorted(set().union(*(rows.unique("task") for rows in data.values())))
    licenses = source_licenses(tasks)

    def add_licenses(rows):
        names = rows.data.column("task").to_pylist()
        rows = rows.add_column("license", [licenses[t]["license"] for t in names])
        return rows.add_column("license_use", [licenses[t]["license_use"] for t in names])
    data = DatasetDict({split: add_licenses(rows) for split, rows in data.items()})
    nli = DatasetDict({split: rows.select_columns(["premise", "hypothesis", "labels", "task", "license", "license_use"])
                       for split, rows in data.items()})
    triplet = DatasetDict({split: rows.rename_columns({"premise": "anchor"}).select_columns(
        ["anchor", "positive", "negative", "task", "license", "license_use"]) for split, rows in data.items()})

    published = {split: Counter(rows.data.column("task").to_pylist()) for split, rows in data.items()}
    sources = {}
    for task in tasks:
        info = {"rows": {split: published[split][task] for split in published if published[split][task]}, **source_provenance(task), "revisions": records[task].get("revisions", {})}
        sources[task] = {**info, **licenses[task]}
    document = {"tasksource_commit": code_state()["git_commit"], "sources": sources,
                "datasets": sorted({r for i in sources.values() for r in [i.get("dataset"), *i.get("originals", [])] if r})}
    (args.output / "sources.yaml").write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=120))
    summary = {"tasks": len(tasks), "rows": {split: len(rows) for split, rows in nli.items()}}
    (args.output / "build-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return nli, triplet


def upload(args, nli, triplet):
    api = HfApi()
    nli.push_to_hub(args.repo_id, max_shard_size="512MB", commit_message="Rebuild from the tasksource catalog")
    triplet.push_to_hub(args.repo_id, config_name="triplet", max_shard_size="512MB",
                        commit_message="Rebuild triplet config from the tasksource catalog")
    card = DatasetCard.load(args.repo_id, repo_type="dataset")
    if SCRIPT_URL not in card.text:
        card.text = card.text.rstrip() + (f"\n\nBuilt with [build_label_nli.py]({SCRIPT_URL}); [sources.yaml](sources.yaml) "
                                          "lists every task with its revision, license and row counts.\n")
        card.push_to_hub(args.repo_id, repo_type="dataset", commit_message="Card: link the build script")
    for name in ("sources.yaml", "build-report.jsonl", "build-summary.json"):
        api.upload_file(path_or_fileobj=str(args.output / name), path_in_repo=name, repo_id=args.repo_id,
                        repo_type="dataset", commit_message=f"Add {name}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--multilingual", action="store_true")
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--max-rows-eval", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id")
    args = parser.parse_args()
    name = ("multilingual-" if args.multilingual else "") + "zero-shot-label-nli"
    args.output = args.output or ROOT / "build" / name
    args.repo_id = args.repo_id or f"tasksource/{name}"
    return args


if __name__ == "__main__":
    args = parse_args()
    if not args.finalize_only:
        build(args)
    if args.finalize or args.finalize_only:
        nli, triplet = finalize(args)
        if args.upload:
            upload(args, nli, triplet)
