#!/usr/bin/env python3
"""Generate and optionally publish tasksource/procedural-extraction from reasoning-core's extraction tasks.

One config per task, in the schema of the ``extraction`` config of tasksource-jev-typed-decisions (``state``,
``fields``, ``answers`` with character offsets, ``metadata``). Generation is deterministic: row ``i`` of a split
draws its level and ``state_seed`` from ``task:split:i`` alone, so the data is fixed by the reasoning-core commit
recorded in ``provenance.json``. States are unique within a split, and evaluation states never occur in train.

    PYTHONPATH=.:src:../rc-jev-extraction python scripts/build_procedural_extraction.py --upload
"""
import argparse
import json
import random
import subprocess
from pathlib import Path

from datasets import Dataset, DatasetDict, Value
from huggingface_hub import HfApi

from tasksource.jev.extraction import FEATURES, check

REPO_ID = "tasksource/procedural-extraction"
TASKS = ["record_field_extraction", "state_update_extraction", "conditional_list_extraction",
         "log_query_extraction"]
LEVELS = [0, 1, 2, 3, 4]
# Albert deepseek-v4-flash, mean per-field score over 30 states per cell (reasoning-core 38b6189); GLiNER-large
# fills each field with its best span for the instruction as a label.
CALIBRATION = {
    "record_field_extraction": ([1.00, 0.91, 0.84, 0.67, 0.58], [0.76, 0.64, 0.46, 0.41, 0.38]),
    "state_update_extraction": ([0.84, 0.76, 0.77, 0.68, 0.51], [0.58, 0.46, 0.37, 0.36, 0.29]),
    "conditional_list_extraction": ([0.88, 0.79, 0.70, 0.64, 0.52], [0.29, 0.17, 0.12, 0.05, 0.10]),
    "log_query_extraction": ([0.86, 0.76, 0.68, 0.63, 0.59], [0.31, 0.16, 0.09, 0.24, 0.15]),
}


def generate_split(task, split, rows, exclude, max_attempts=5):
    from reasoning_core.decision import extraction_row
    seen, out = set(exclude), []
    for index in range(rows * max_attempts):
        if len(out) == rows:
            break
        rng = random.Random(f"{task}:{split}:{index}")
        level, state_seed = rng.choice(LEVELS), rng.randrange(2 ** 32)
        entry = task.generate_example(level=level, state_seed=state_seed)
        row = extraction_row(entry, split)
        if row["state"] in seen:
            continue
        seen.add(row["state"])
        row["id"] = f"{task.task_name}:{split}:{len(out)}"
        out.append(check(row))
    return out, seen


def pretty_order(rows, first_rows=1_000, seed=0):
    """Levels round-robin in a display prefix, easiest first, then the rest shuffled."""
    queues = {}
    for row in rows:
        queues.setdefault(json.loads(row["metadata"])["level"], []).append(row)
    prefix, queues = [], [queues[k] for k in sorted(queues)]
    while len(prefix) < min(first_rows, len(rows)):
        for queue in queues:
            if queue and len(prefix) < first_rows:
                prefix.append(queue.pop(0))
    rest = [row for queue in queues for row in queue]
    random.Random(seed).shuffle(rest)
    return prefix + rest


def build_task(name, sizes):
    from reasoning_core import get_task
    task, splits, seen = get_task(name), {}, set()
    for split, rows in sizes.items():
        splits[split], seen = generate_split(task, split, rows, seen)
    splits["train"] = pretty_order(splits["train"])
    features = FEATURES.copy()
    features["level"] = Value("int32")
    for rows in splits.values():
        for row in rows:
            row["level"] = json.loads(row["metadata"])["level"]
    return DatasetDict({s: Dataset.from_list(r, features=features) for s, r in splits.items()})


def provenance():
    import reasoning_core
    from reasoning_core import get_task
    root = Path(reasoning_core.__file__).resolve().parents[1]
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "reasoning_core"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"reasoning-core checkout {root} has uncommitted task changes; commit them first")
    return {"generator": "https://github.com/sileod/reasoning-core", "commit": commit,
            "tasks": {name: {"task_version": get_task(name).task_version,
                             "module": f"reasoning_core/tasks/jev/{name}.py"} for name in TASKS},
            "levels": LEVELS, "seeding": "row i of a split: random.Random(f'{task}:{split}:{i}') draws level and state_seed",
            "script": "https://github.com/sileod/tasksource/blob/main/scripts/build_procedural_extraction.py"}


def card(summary, prov):
    configs = "".join(
        f"- config_name: {name}\n  data_files:\n" + "".join(
            f"  - split: {split}\n    path: {name}/{split}-*.parquet\n" for split in ("train", "validation", "test"))
        for name in TASKS)
    rows = "\n".join(f"| `{name}` | {summary[name]['train']:,} | {summary[name]['validation']:,} | "
                     f"{summary[name]['test']:,} | " + " / ".join(f"{v:.2f}" for v in CALIBRATION[name][0]) + " | "
                     + " / ".join(f"{v:.2f}" for v in CALIBRATION[name][1]) + " |" for name in TASKS)
    return f"""---
pretty_name: procedural-extraction
language:
- en
license: mit
task_categories:
- token-classification
- question-answering
tags:
- tasksource
- jev
- extraction
- clairic
- procedural
- synthetic
configs:
{configs}---

# procedural-extraction

Procedurally generated extraction requests: one text and several typed fields to fill from it in one pass, as
extraction models (e.g. Cloudflare's Clairic) are asked. Each field is `text` (one span) or `list` (every
matching span) and is answered with character spans of the text or null. Answers are computed by construction,
so spans are exact and labels noise-free; nulls are earned (the text mentions a negated, proposed, former or
other entity's value, not nothing). The tasks are generated by
[reasoning-core](https://github.com/sileod/reasoning-core) (commit `{prov['commit'][:7]}`, see
[provenance.json](provenance.json)) and are a procedural component of the `extraction` config of
[tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions).

| config | what is extracted |
|---|---|
| `record_field_extraction` | A person's city, employer, job title, phone, email and birth date, from prose about several people with near-miss names; negated, former, hoped-for and relatives' values as confusers; "same city/company as X" chains |
| `state_update_extraction` | The value in force (or before the last change) of plan slots, from a timestamped chat of statements, proposals approved/rejected/pending, corrections and undos, out of order from level 2 |
| `conditional_list_extraction` | Every order id or product meeting conjunctive, negated or numeric conditions after later updates, or the heaviest such order; prose, bullets or a table |
| `log_query_extraction` | Every request id or user of log lines matching filters on level, exact service (auth vs auth-proxy), status, latency and user, or the latest/slowest match; plain, logfmt or JSON lines |

Levels 0 to 4 are drawn uniformly; `level` is a column. Mean per-field score by level (0 / 1 / 2 / 3 / 4):

| config | train | validation | test | DeepSeek-v4-flash | GLiNER-large |
|---|---|---|---|---|---|
{rows}

## Schema

`state` (the text), `fields` (`[{{name, type, instruction}}]`), `answers` (one per field: `{{null, values:
[{{text, start, end}}]}}`, `state[start:end] == text`), `metadata` (JSON: level, state seed), `level`.

## Citation

```bibtex
@inproceedings{{sileo-2024-tasksource,
    title = "tasksource: A Large Collection of {{NLP}} tasks with a Structured Dataset Preprocessing Framework",
    author = "Sileo, Damien",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    year = "2024",
    url = "https://aclanthology.org/2024.lrec-main.1361",
}}
```
"""


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=root / "build" / "procedural-extraction")
    parser.add_argument("--train", type=int, default=10_000)
    parser.add_argument("--eval", type=int, default=500)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()
    prov = provenance()
    summary = {}
    for name in TASKS:
        dataset = build_task(name, {"train": args.train, "validation": args.eval, "test": args.eval})
        dataset.save_to_disk(str(args.output / name))
        summary[name] = {split: len(rows) for split, rows in dataset.items()}
        print(name, summary[name], flush=True)
    (args.output / "provenance.json").write_text(json.dumps({**prov, "rows": summary}, indent=1) + "\n")
    (args.output / "README.md").write_text(card(summary, prov))
    if args.upload:
        from datasets import load_from_disk
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        for name in TASKS:
            load_from_disk(str(args.output / name)).push_to_hub(args.repo_id, config_name=name,
                                                                 commit_message=f"Publish {name}")
        for path in ("provenance.json", "README.md"):
            api.upload_file(path_or_fileobj=str(args.output / path), path_in_repo=path, repo_id=args.repo_id,
                            repo_type="dataset", commit_message=f"Add {path}")
        print(api.dataset_info(args.repo_id).sha)


if __name__ == "__main__":
    main()
