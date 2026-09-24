"""Write the task catalogs tasks.md (English) and mtasks.md (multilingual).

    PYTHONPATH=src python scripts/write_task_catalogs.py

One compact row per task: its id, type, source dataset, question, and the
annotation fields that differ from the template defaults.
"""

from pathlib import Path

from tasksource import list_tasks

ROOT = Path(__file__).resolve().parents[1]
SHOWN_ELSEWHERE = {"dataset_name", "config_name", "task_id", "question", "load_dataset_kwargs"}
DEFAULTS = {"sentence1": "sentence1", "sentence2": "sentence2", "labels": "labels", "inputs": "input",
            "tokens": "tokens", "splits": ("train", "validation", "test")}


def describe(value):
    if callable(value):
        return "fn"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(describe(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{…}" if value else ""
    return str(value)


def fields(mapping):
    """The annotation fields that differ from the template defaults, on one line."""
    parts = []
    for key, value in mapping.to_dict().items():
        value = getattr(mapping, key)
        if key in SHOWN_ELSEWHERE or value is None or DEFAULTS.get(key) == value:
            continue
        if key in ("pre_process", "post_process"):
            if getattr(value, "__name__", "") != "identity":
                parts.append(key)
            continue
        if key == "splits" and tuple(value) == DEFAULTS["splits"]:
            continue
        text = describe(value)
        if text:
            parts.append(f"{key}={text}")
    return ", ".join(parts)


def cell(text):
    return " ".join(str(text or "").split()).replace("|", "\\|")


def dataset_link(name):
    if name in ("csv", "json", "parquet", "text") or "/" not in str(name):
        return cell(name)
    return f"[{name}](https://hf.co/datasets/{name})"


def write(path, multilingual):
    tasks = list_tasks(multilingual=multilingual)
    lines = [
        f"{len(tasks)} {'multilingual' if multilingual else 'English'} tasks. Load one with "
        f"`load_task(id{', multilingual=True' if multilingual else ''})`; the annotations are in "
        f"[{'multilingual_tasks' if multilingual else 'tasks'}.py](src/tasksource/"
        f"{'multilingual_tasks' if multilingual else 'tasks'}.py), and tasks kept out on purpose are in "
        "[parked.py](src/tasksource/parked.py).",
        "",
        "| id | type | dataset | config | question | fields |",
        "|---|---|---|---|---|---|",
    ]
    for row in tasks.itertuples():
        lines.append("| " + " | ".join([
            cell(row.id), row.task_type, dataset_link(row.dataset_name), cell(row.config_name),
            cell(getattr(row.mapping, "question", None)), cell(fields(row.mapping)),
        ]) + " |")
    path.write_text("\n".join(lines) + "\n")
    print(f"{path.name}: {len(tasks)} tasks")


if __name__ == "__main__":
    write(ROOT / "tasks.md", multilingual=False)
    write(ROOT / "mtasks.md", multilingual=True)
