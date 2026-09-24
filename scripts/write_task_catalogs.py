"""Write the task catalogs tasks.md (English) and mtasks.md (multilingual).

    PYTHONPATH=src python scripts/write_task_catalogs.py

A quick glance at every task: its id (linked to the annotation), type, source
dataset, and whether it has a question.
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


def write(path, multilingual):
    tasks = list_tasks(multilingual=multilingual)
    module = f"src/tasksource/{'multilingual_tasks' if multilingual else 'tasks'}.py"
    lines_of = annotation_lines((ROOT / module).read_text())
    lines = [
        f"{len(tasks)} {'multilingual' if multilingual else 'English'} tasks. Load one with "
        f"`load_task(id{', multilingual=True' if multilingual else ''})`; the annotations are in "
        f"[{'multilingual_tasks' if multilingual else 'tasks'}.py](src/tasksource/"
        f"{'multilingual_tasks' if multilingual else 'tasks'}.py), and tasks kept out on purpose are in "
        "[parked.py](src/tasksource/parked.py).",
        "",
        "| id | type | dataset | question |",
        "|---|---|---|:-:|",
    ]
    for row in tasks.itertuples():
        lines.append("| " + " | ".join([
            f"[{cell(row.id)}]({module}#L{lines_of[row.preprocessing_name]})", row.task_type,
            dataset_link(row.dataset_name), "✓" if getattr(row.mapping, "question", None) else "",
        ]) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(f"{path.name}: {len(tasks)} tasks")


if __name__ == "__main__":
    write(ROOT / "tasks.md", multilingual=False)
    write(ROOT / "mtasks.md", multilingual=True)
