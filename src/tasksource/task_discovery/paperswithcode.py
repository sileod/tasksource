"""Retrieve decision-oriented candidates from the archived PWC dataset export."""
import gzip
import hashlib
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "pwc-archive/files"
REVISION = "1af3a546770f9b98665523ccba4b22b4f3e18c1f"
FILENAME = "jul-28-datasets.json.gz"
SHA256 = "6ef923aaf24ff62c6ef567518728d4c85dba1b23eecbc8e1bb5ad5d6990fe25f"
DECISION_TASKS = (
    "classification", "multiple choice", "multiple-choice", "sentiment", "stance",
    "entailment", "inference", "word sense", "event detection", "relation extraction",
    "intent", "coreference", "argument mining", "fact verification", "fact checking",
    "semantic parsing", "relevance", "ranking", "grounding", "counting", "preference",
    "sarcasm", "emotion", "hate speech", "offensive", "question answering",
)


def candidates(rows, decision_only=True):
    for row in rows:
        tasks = sorted({t["task"] for t in row.get("tasks", [])})
        decision_tasks = [t for t in tasks if any(k in t.casefold() for k in DECISION_TASKS)]
        if decision_only and not decision_tasks:
            continue
        text = " ".join(str(row.get(k) or "") for k in ("homepage", "description", "data_loaders"))
        paper = row.get("verified_paper") or row.get("paper")
        yield dict(name=row["name"], aliases=[row["full_name"]] if row.get("full_name") else [],
            sources=[dict(source="paperswithcode", url=row["url"])],
            dataset_urls=[row["homepage"]] if row.get("homepage") else [],
            github=sorted(set(re.findall(r"https?://github\.com/[\w.-]+/[\w.-]+", text))),
            huggingface=sorted({url.rstrip(".") for url in re.findall(r"https?://huggingface\.co/datasets/[\w.-]+/[\w.-]+", text)}),
            papers=[paper] if paper else [], tasks=tasks, decision_tasks=decision_tasks,
            languages=row.get("languages") or [], modalities=row.get("modalities") or [],
            description=row.get("short_description") or row.get("description") or "",
            train_status="unchecked")


def retrieve(decision_only=True, snapshot=None):
    path = Path(snapshot or hf_hub_download(REPO, FILENAME, repo_type="dataset", revision=REVISION))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError("PWC snapshot checksum mismatch")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = list(candidates(json.load(stream), decision_only))
    return rows, dict(source="paperswithcode", repository=REPO, revision=REVISION,
        file=FILENAME, sha256=digest, decision_only=decision_only,
        decision_task_terms=list(DECISION_TASKS), candidates=len(rows))
