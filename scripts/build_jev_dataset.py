#!/usr/bin/env python3
"""Build and optionally upload the tasksource-jev dataset.

Each task/split is written as its own Parquet shard, making interrupted builds
resumable. The JSONL report is append-only and records successes and failures.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
import numpy as np
import pandas as pd
from tasksource import list_tasks, load_task


SUPPORTED_TYPES = {"Classification", "MultipleChoice"}


def slug(task_id):
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:10]
    readable = "".join(c if c.isalnum() else "-" for c in task_id).strip("-")[:70]
    return f"{readable}-{digest}"


def read_completed(report_path):
    completed = set()
    if not report_path.exists():
        return completed
    for line in report_path.read_text().splitlines():
        record = json.loads(line)
        if record["status"] == "ok":
            completed.add(record["task"])
    return completed


def latest_records(report_path):
    records = {}
    if report_path.exists():
        for line in report_path.read_text().splitlines():
            record = json.loads(line)
            records[record["task"]] = record
    return records


def append_report(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def to_training_row(example, index, task_id, split):
    """Convert the lossless internal recast to the common Jev training schema."""
    options = list(example["criteria"])
    label = int(example["label"])
    target = [0.0] * len(options)
    target[label] = 1.0
    return {
        "id": f"{slug(task_id)}:{split}:{index}",
        "kind": "choice",
        "options": options,
        "target": target,
        "state": example["state"],
        "question": example["instructions"],
        "source": task_id,
    }


def pretty_order(dataset, first_rows=1_000):
    """Round-robin sources in a display prefix without shuffling the remainder."""
    first_rows = min(first_rows, len(dataset))
    if first_rows < 2 or "source" not in dataset.column_names:
        return dataset
    sources = dataset["source"]
    names = sorted(set(sources))
    if len(names) < 2:
        return dataset
    per_source = (first_rows + len(names) - 1) // len(names)
    buckets = {name: [] for name in names}
    for index, source in enumerate(sources):
        bucket = buckets[source]
        if len(bucket) < per_source:
            bucket.append(index)
    prefix = []
    for offset in range(per_source):
        for name in names:
            if offset < len(buckets[name]):
                prefix.append(buckets[name][offset])
                if len(prefix) == first_rows:
                    break
        if len(prefix) == first_rows:
            break
    selected = np.zeros(len(dataset), dtype=bool)
    selected[prefix] = True
    order = np.concatenate((np.asarray(prefix), np.flatnonzero(~selected)))
    return dataset.select(order)


def publish_dataset(output, repo_id, pretty_rows=1_000):
    data_files = {}
    for split in ("train", "validation", "test"):
        files = sorted((output / "data").glob(f"{split}-*.parquet"))
        if files:
            data_files[split] = [str(path) for path in files]
    dataset = load_dataset("parquet", data_files=data_files)
    if "train" in dataset:
        dataset["train"] = pretty_order(dataset["train"], pretty_rows)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(output / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add tasksource-jev dataset card",
    )
    dataset.push_to_hub(
        repo_id,
        max_shard_size="256MB",
        commit_message="Publish tasksource-jev Parquet dataset",
    )
    for name in ("build-report.jsonl", "build-summary.json", "failed-tasks.json", "outdated-datasets.json"):
        path = output / name
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add {name}",
            )


def migrate_legacy_shards(data_dir):
    """Upgrade resumable shards created before the public schema was finalized."""
    for path in sorted(data_dir.glob("*.parquet")):
        shard = load_dataset("parquet", data_files=str(path), split="train")
        if "id" in shard.column_names:
            continue
        split = path.name.split("-", 1)[0]
        task_id = shard[0]["task"]
        upgraded = shard.map(
            to_training_row,
            with_indices=True,
            fn_kwargs={"task_id": task_id, "split": split},
            remove_columns=shard.column_names,
        )
        temporary = path.with_suffix(".parquet.tmp")
        upgraded.to_parquet(temporary)
        temporary.replace(path)


def select_tasks(args):
    frame = list_tasks(instruct=True)
    frame["multilingual"] = False
    frame["source_id"] = frame.id
    if not args.english_only:
        multilingual = list_tasks(multilingual=True, instruct=True)
        multilingual["multilingual"] = True
        multilingual["source_id"] = "multilingual/" + multilingual.id
        frame = pd.concat([frame, multilingual], ignore_index=True)
    frame = frame[frame.task_type.isin(SUPPORTED_TYPES)]
    if args.tasks:
        wanted = set(args.tasks)
        frame = frame[frame.source_id.isin(wanted)]
        missing = wanted - set(frame.source_id)
        if missing:
            raise ValueError(f"Unknown or unsupported tasks: {sorted(missing)}")
    if args.limit:
        frame = frame.head(args.limit)
    return frame


def build(args):
    output = args.output.resolve()
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    migrate_legacy_shards(data_dir)
    report_path = output / "build-report.jsonl"
    completed = read_completed(report_path)
    tasks = select_tasks(args)
    print(f"Selected {len(tasks)} tasks; {len(completed)} already complete", flush=True)

    for position, row in enumerate(tasks.itertuples(index=False), start=1):
        task_id = row.source_id
        if task_id in completed:
            print(f"[{position}/{len(tasks)}] skip {task_id}", flush=True)
            continue
        started = time.time()
        print(f"[{position}/{len(tasks)}] build {task_id}", flush=True)
        try:
            dataset = load_task(
                row.id,
                recast="jev",
                multilingual=row.multilingual,
                max_rows=args.max_rows,
                max_rows_eval=args.max_rows_eval,
            )
            split_rows = {}
            for split, split_dataset in dataset.items():
                split_dataset = split_dataset.map(
                    to_training_row,
                    with_indices=True,
                    fn_kwargs={"task_id": task_id, "split": split},
                    remove_columns=split_dataset.column_names,
                )
                path = data_dir / f"{split}-{slug(task_id)}.parquet"
                split_dataset.to_parquet(path)
                split_rows[split] = len(split_dataset)
            record = {
                "task": task_id,
                "task_type": row.task_type,
                "status": "ok",
                "rows": split_rows,
                "seconds": round(time.time() - started, 3),
            }
            completed.add(task_id)
        except Exception as error:
            record = {
                "task": task_id,
                "task_type": row.task_type,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
                "seconds": round(time.time() - started, 3),
            }
        append_report(report_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    if args.finalize:
        shutil.copyfile(args.card, output / "README.md")
        selected = set(tasks.source_id)
        latest = latest_records(report_path)
        failures = [
            latest[task] for task in sorted(selected & set(latest))
            if latest[task]["status"] == "error"
        ]
        outdated = [
            record for record in failures
            if "Dataset scripts are no longer supported" in record.get("error", "")
        ]
        (output / "failed-tasks.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (output / "outdated-datasets.json").write_text(
            json.dumps(outdated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        summary = {
            "selected_tasks": len(tasks),
            "completed_tasks": len(completed & selected),
            "failed_tasks": len(failures),
            "outdated_datasets": len(outdated),
            "parquet_files": len(list(data_dir.glob("*.parquet"))),
        }
        (output / "build-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary), flush=True)

    if args.upload:
        publish_dataset(output, args.repo_id, args.pretty_rows)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "build" / "tasksource-jev")
    parser.add_argument("--card", type=Path, default=root / "dataset_cards" / "tasksource-jev.md")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--english-only", action="store_true",
        help="Exclude the multilingual Tasksource catalog.",
    )
    parser.add_argument("--max-rows", type=int, default=30_000)
    parser.add_argument("--max-rows-eval", type=int, default=3_000)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default="tasksource/tasksource-jev")
    parser.add_argument(
        "--pretty-rows", type=int, default=1_000,
        help="Deterministically interleave sources in this many leading train rows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
