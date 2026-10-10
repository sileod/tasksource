"""Write the English, multilingual, and visual task catalogs.

    PYTHONPATH=src python scripts/write_task_catalogs.py

A quick glance at every task: its id (linked to the annotation), type, source
dataset, and whether it has a question; then the soft-label annotations.
"""

import re
from pathlib import Path

from tasksource import list_tasks

ROOT = Path(__file__).resolve().parents[1]


def cell(text):
    return " ".join(str(text or "").split()).replace("|", "\\|")


def dataset_link(name):
    if name in ("csv", "json", "parquet", "text") or "/" not in str(name):
        return cell(name)
    return f"[{name}](https://hf.co/datasets/{name})"


def annotation_lines(source):
    """Line number of each top-level annotation assignment."""
    return {m.group(1): source.count("\n", 0, m.start()) + 1
            for m in re.finditer(r"^(\w+)\s*=", source, flags=re.M)}


def write(path, multilingual=False, vision=False):
    tasks = list_tasks(multilingual=multilingual, vision=vision)
    name = 'vision_tasks' if vision else ('multilingual_tasks' if multilingual else 'tasks')
    kind = 'visual' if vision else ('multilingual' if multilingual else 'English')
    flag = ', vision=True' if vision else (', multilingual=True' if multilingual else '')
    module = f"src/tasksource/{name}.py"
    lines_of = annotation_lines((ROOT / module).read_text())
    lines = [
        f"{len(tasks)} {kind} tasks. Load one with `load_task(id{flag})`; "
        f"the annotations are in [{name}.py]({module})"
        + ". Evaluation benchmarks are in [eval_only.py](src/tasksource/eval_only.py); "
        + "other excluded annotations are in [parked.py](src/tasksource/parked.py).",
        "",
        "| # | id | type | dataset | question |",
        "|--:|---|---|---|:-:|",
    ]
    for index, row in enumerate(tasks.itertuples(), 1):
        lines.append("| " + " | ".join([
            str(index), f"[{cell(row.id)}]({module}#L{lines_of[row.preprocessing_name]})", row.task_type,
            dataset_link(row.dataset_name), "✓" if getattr(row.mapping, "question", None) else "",
        ]) + " |")
    soft = list_tasks(multilingual=multilingual, vision=vision, soft=True)
    soft = soft[soft.soft_labels]
    if len(soft):
        lines += [
            "", "## Soft labels", "",
            "Annotations whose label is a distribution (annotator votes, rater shares, survey counts), "
            "loaded with `load_task(id, soft=True)`. Those with a hard view are also listed above, by "
            "their majority label; the others have soft labels only. `votes` are shares of annotators, "
            "`mean` a numeric value interpolated over score anchors (see each task for its meaning); "
            "annotators is the typical count per item (vote shares from fewer than "
            "five are coarse, and the Jev build leaves them out).",
            "",
            "| id | kind | aggregation | annotators | default view | dataset |",
            "|---|---|---|--:|---|---|",
        ]
        for row in soft.itertuples():
            lines.append("| " + " | ".join([
                f"[{cell(row.id)}]({module}#L{lines_of[row.preprocessing_name]})", row.mapping.kind,
                row.mapping.aggregation, str(row.mapping.annotators or ""),
                ("regression" if row.mapping.regression else row.mapping.hard_type) or "",
                dataset_link(row.dataset_name),
            ]) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(f"{path.name}: {len(tasks)} tasks")


if __name__ == "__main__":
    write(ROOT / "catalog_english.md", multilingual=False)
    write(ROOT / "catalog_multilingual.md", multilingual=True)
    write(ROOT / "catalog_vision.md", vision=True)
