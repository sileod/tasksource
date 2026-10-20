"""Build (and publish) the ``extraction`` config of tasksource-jev-typed-decisions.

    PYTHONPATH=.:src python scripts/build_jev_extraction.py --output build/jev-extraction
    PYTHONPATH=.:src python scripts/build_jev_extraction.py --output build/jev-extraction --push

Rows follow ``tasksource.jev.extraction``: a text, typed fields and character-span answers. Sources
are catalog tasks: the extractive question answering catalog (``list_tasks(extractive=True)``,
questions grouped by context) and the named-entity token tasks (one list field per entity type).
Native splits are kept; evaluation rows whose text also occurs in training are dropped; each source is
capped (``--max-rows`` training rows, ``--max-rows-eval`` per evaluation split), sampled by seed.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, Value
from huggingface_hub import HfApi

from tasksource import load_task, task_licenses
from tasksource.jev.extraction import FEATURES, ner_rows, qa_rows

REPO = "tasksource/tasksource-jev-typed-decisions"
# NewsQA's redistribution terms (Microsoft's NewsQA license over CNN articles) are unresolved.
EXCLUDED_QA = {"newsqa"}
# Multi-span answers are sets (every span is part of the answer), not alternatives.
LIST_QA = {"quoref"}
NER = {
    "conll2003/ner_tags": None, "wnut_17/wnut_17": None, "ncbi_disease/ncbi_disease": None,
    "jnlpba/jnlpba": None, "ontonotes_english/SpeedOfMagic--ontonotes_english": None,
    "few-nerd/supervised": None, "acronym_identification": None,
    "finer-139": "Every amount reported as {name}, as written.",
    "wiesp2022-ner": None, "broad-twitter-corpus": None, "finer-ord": None, "anatem": None,
    "stackoverflow-ner": None, "legallens-ner": None, "gum-ner": None, "opinion-target/en": None,
}
SPLITS = ("train", "validation", "test")


def row_id(row):
    key = json.dumps([row["source"], row["state"], [f["instruction"] for f in row["fields"]]], ensure_ascii=False)
    return f"{row['source']}:{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def source_rows(task, extractive, args):
    """``{split: [rows]}`` of one task, before capping."""
    dataset = load_task(task, extractive=extractive)
    out = {}
    for split in SPLITS:
        if split not in dataset or not len(dataset[split]):
            continue
        if extractive:
            rows = list(qa_rows(dataset[split], task, split, args.max_fields, kind="list" if task in LIST_QA else "text"))
        else:
            names = dataset[split].features["labels"].feature.names
            kwargs = {"instruction": NER[task]} if NER[task] else {}
            rows = list(ner_rows(dataset[split], task, split, names, max_fields=args.max_fields, seed=args.seed, **kwargs))
        out[split] = rows
    return out


def capped(rows, limit, seed, name):
    if limit is None or len(rows) <= limit:
        return rows
    return random.Random(f"{seed}:{name}").sample(rows, limit)


def build(args):
    qa = [t for t in __import__("tasksource").list_tasks(extractive=True).id if t not in EXCLUDED_QA]
    tasks = [(t, True) for t in qa] + [(t, False) for t in NER]
    if args.tasks:
        tasks = [(t, e) for t, e in tasks if t in args.tasks]
    licenses = {}
    for extractive in (True, False):
        ids = [t for t, e in tasks if e == extractive]
        if ids:
            frame = task_licenses(ids, extractive=extractive, fresh=True)
            licenses.update({r.id: (r.license, r.license_use) for r in frame.itertuples()})
    splits, report = {s: [] for s in SPLITS}, {}
    for task, extractive in tasks:
        print(f"loading {task}", flush=True)
        rows = source_rows(task, extractive, args)
        counts = {}
        for split, items in rows.items():  # repeated texts (boilerplate sentences): keep the first
            for r in items:
                r["id"] = row_id(r)
            unique = list({r["id"]: r for r in reversed(items)}.values())[::-1]
            counts[f"{split}_duplicates_dropped"] = len(items) - len(unique)
            rows[split] = unique
        for split, items in list(rows.items()):  # hidden test labels (all O / no answers): not evaluation data
            if not any(not a["null"] for r in items for a in r["answers"]):
                counts[f"{split}_unlabeled_dropped"] = len(rows.pop(split))
        train_states = {r["state"] for r in rows.get("train", [])}
        for split, items in rows.items():
            if split != "train":
                before = len(items)
                items = [r for r in items if r["state"] not in train_states]
                counts[f"{split}_overlap_dropped"] = before - len(items)
            items = capped(items, args.max_rows if split == "train" else args.max_rows_eval, args.seed, f"{task}:{split}")
            license, use = licenses.get(task, ("unspecified", "unspecified"))
            for r in items:
                r["license"], r["license_use"] = license, use
            splits[split] += items
            counts[split] = len(items)
            counts[f"{split}_fields"] = sum(len(r["fields"]) for r in items)
            counts[f"{split}_null_fields"] = sum(a["null"] for r in items for a in r["answers"])
        report[task] = counts
        print(f"  {counts}", flush=True)
    features = FEATURES.copy()
    features["license"], features["license_use"] = Value("string"), Value("string")
    dataset = DatasetDict({s: Dataset.from_list(sorted(r, key=lambda x: x["id"]), features=features)
                           for s, r in splits.items() if r})
    for split, rows in dataset.items():
        duplicated = [i for i, n in Counter(rows["id"]).items() if n > 1]
        if duplicated:
            raise ValueError(f"{split}: duplicate ids {duplicated[:3]}")
    dataset = DatasetDict({s: rows.shuffle(seed=args.seed) for s, rows in dataset.items()})
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": report, "excluded": sorted(EXCLUDED_QA), "max_rows": args.max_rows,
                "max_rows_eval": args.max_rows_eval, "max_fields": args.max_fields, "seed": args.seed,
                "rows": {s: len(r) for s, r in dataset.items()},
                "revisions": {t: __import__("tasksource").task_provenance(t, extractive=e).get("revision") for t, e in tasks}}
    (args.output / "build-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    dataset.save_to_disk(str(args.output / "dataset"))
    return dataset, manifest


CARD = """## Extraction config

`load_dataset("{repo}", "extraction")`: one text and several typed fields to fill from it, answered in one
pass, as extraction models (e.g. Cloudflare's Clairic) are asked. {rows}

- `state`: the text. `fields`: `[{{name, type, instruction}}]`; `type` is `text` (one span; the instruction is
  a question) or `list` (every span of a kind, e.g. every person).
- `answers`: one per field, in order: `{{null, values: [{{text, start, end}}]}}`, character offsets into
  `state` (`state[start:end] == text`, checked for every span). `null` means the text does not hold the
  answer (an unanswerable question, an absent entity type). Several values of a `text` field are
  alternative gold spans; of a `list` field, every item.
- Extractive question answering (SQuAD 2.0, AdversarialQA, Natural Questions, HotpotQA, Quoref, MuSiQue,
  DROP single-span, DuoRC verbatim spans, ConcurrentQA, FairytaleQA explicit spans) groups the questions
  asked about one context, up to {max_fields} fields per row.
  Named entities (CoNLL-2003, OntoNotes, WNUT-17, Few-NERD, NCBI-disease, JNLPBA, acronyms, FiNER-139
  and astronomy, biomedical, financial, software, legal, GUM, Twitter and English opinion-target sources)
  ask for every entity type of the dataset, or the types present plus absent ones up to {max_fields}.
  Entity texts are the source tokens joined by spaces.
- Native splits, except those without released labels; evaluation texts that also occur in training are dropped. Each source is capped at
  {max_rows:,} training rows and {max_rows_eval:,} per evaluation split. `license`/`license_use` per row;
  `metadata` keeps source question ids or entity types. See [build-manifest.json](build-manifest.json).

The same question answering sources load in SQuAD format for `AutoModelForQuestionAnswering` with
`tasksource.load_task(id, extractive=True)`.
"""


TAGS = ["extraction", "information-extraction", "extractive-qa", "span-extraction", "named-entity-recognition",
        "structured-extraction", "clairic"]
TASK_IDS = ["extractive-qa", "named-entity-recognition"]
ROW = ("| `extraction` | A text and several typed fields (questions, entity types) answered with character spans "
       "or null, Clairic-style. See [extraction format and sources](extraction/README.md). |")


def root_card(text):
    """The Hub root card with the extraction tags, task ids, config row and load line."""
    import yaml
    from scripts.build_jev_dataset import card_header
    meta, body = card_header(text)
    for key, values in (("tags", TAGS), ("task_ids", TASK_IDS)):
        meta[key] = list(dict.fromkeys((meta.get(key) or []) + values))
    if "`extraction`" not in body:
        lines = body.split("\n")
        at = next(i for i, line in enumerate(lines) if line.startswith("| `vision`"))
        lines.insert(at + 1, ROW)
        body = "\n".join(lines)
    vision = 'vision = load_dataset("tasksource/tasksource-jev-typed-decisions", "vision")\n'
    if "extraction = load_dataset" not in body and vision in body:
        body = body.replace(vision, vision + 'extraction = load_dataset("tasksource/tasksource-jev-typed-decisions", "extraction")\n')
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True) + "---\n" + body


def push(dataset, manifest, args):
    dataset.push_to_hub(args.repo_id, config_name="extraction", data_dir="extraction", max_shard_size="256MB",
                        commit_message="Add the extraction config")
    api = HfApi()
    api.upload_file(path_or_fileobj=str(args.output / "build-manifest.json"), path_in_repo="extraction/build-manifest.json",
                    repo_id=args.repo_id, repo_type="dataset", commit_message="Add extraction build manifest")
    rows = ", ".join(f"{n:,} {s}" for s, n in manifest["rows"].items())
    card = CARD.format(repo=args.repo_id, rows=f"Rows: {rows}.", max_fields=args.max_fields,
                       max_rows=args.max_rows, max_rows_eval=args.max_rows_eval)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="extraction/README.md", repo_id=args.repo_id,
                    repo_type="dataset", commit_message="Document the extraction config")
    from huggingface_hub import hf_hub_download
    from scripts.build_jev_dataset import CHANGELOG, dataset_changelog, tag_release
    root = Path(hf_hub_download(args.repo_id, "README.md", repo_type="dataset", force_download=True)).read_text()
    api.upload_file(path_or_fileobj=root_card(root).encode(), path_in_repo="README.md", repo_id=args.repo_id,
                    repo_type="dataset", commit_message="Card: extraction config, tags")
    api.upload_file(path_or_fileobj=dataset_changelog(CHANGELOG.read_text(encoding="utf-8")).encode(),
                    path_in_repo="CHANGELOG.md", repo_id=args.repo_id, repo_type="dataset",
                    commit_message="Changelog: extraction config")
    tag_release(api, args.repo_id, CHANGELOG)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("build/jev-extraction"))
    parser.add_argument("--tasks", nargs="*", help="only these task ids (smoke runs)")
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--max-rows-eval", type=int, default=500)
    parser.add_argument("--max-fields", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo-id", default=REPO)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    dataset, manifest = build(args)
    print(json.dumps(manifest["rows"]), flush=True)
    if args.push:
        push(dataset, manifest, args)


if __name__ == "__main__":
    main()
