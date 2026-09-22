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


def select_tasks(args):
    frame = list_tasks(instruct=True)
    frame = frame[frame.task_type.isin(SUPPORTED_TYPES)]
    if args.tasks:
        wanted = set(args.tasks)
        frame = frame[frame.id.isin(wanted)]
        missing = wanted - set(frame.id)
        if missing:
            raise ValueError(f"Unknown or unsupported tasks: {sorted(missing)}")
    if args.limit:
        frame = frame.head(args.limit)
    return frame


def build(args):
    output = args.output.resolve()
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_path = output / "build-report.jsonl"
    completed = read_completed(report_path)
    tasks = select_tasks(args)
    print(f"Selected {len(tasks)} tasks; {len(completed)} already complete", flush=True)

    for position, row in enumerate(tasks.itertuples(index=False), start=1):
        task_id = row.id
        if task_id in completed:
            print(f"[{position}/{len(tasks)}] skip {task_id}", flush=True)
            continue
        started = time.time()
        print(f"[{position}/{len(tasks)}] build {task_id}", flush=True)
        try:
            dataset = load_task(
                task_id,
                recast="jev",
                max_rows=args.max_rows,
                max_rows_eval=args.max_rows_eval,
            )
            split_rows = {}
            for split, split_dataset in dataset.items():
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
        selected = set(tasks.id)
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
        subprocess.run(
            ["hf", "repo", "create", args.repo_id, "--repo-type", "dataset", "--exist-ok"],
            check=True,
        )
        subprocess.run(
            ["hf", "upload-large-folder", args.repo_id, str(output), "--repo-type", "dataset"],
            check=True,
        )


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "build" / "tasksource-jev")
    parser.add_argument("--card", type=Path, default=root / "dataset_cards" / "tasksource-jev.md")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-rows", type=int, default=30_000)
    parser.add_argument("--max-rows-eval", type=int, default=3_000)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default="tasksource/tasksource-jev")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
