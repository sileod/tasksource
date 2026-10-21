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
# Procedural extraction (reasoning-core generators, scripts/build_procedural_extraction.py): one source per
# generator, each capped like any other source, so equal shares. Pinned Hub snapshot of the published data.
PROCEDURAL_REPO, PROCEDURAL_REVISION = "tasksource/procedural-extraction", "ee40c9664ab772661f96aa71e4977d2abe686816"
PROCEDURAL = ["record_field_extraction", "state_update_extraction", "conditional_list_extraction",
              "log_query_extraction", "typed_mention_extraction", "business_document_extraction",
              "nested_structure_extraction", "html_extraction", "date_range_extraction", "spec_match_extraction"]
# Multi-span answers are sets (every span is part of the answer), not alternatives.
LIST_QA = {"quoref", "cuad/spans", "qasper/spans", "tab/pii", "ud/ewt", "ud/gum", "preco/mentions", "redocred/entities",
           "qasrl/orig", "bioscope/abstracts", "bioscope/papers", "sherlock/scopes", "multispanqa",
           "rams/role", "casie/role", "phee", "gum/segmentation", "abstrct/neoplasm"}
NER = {
    "conll2000": "Every {name} chunk in the text, as written.",
    "vuamc": "Every metaphorically used word or phrase in the text, as written.",
    "fce/errors": "Every erroneous word or phrase in the text, as written.",
    "semeval2014/laptop-aspects": "Every laptop aspect term in the review, as written.",
    "scierc/entities": None,
    "toxic-spans": "Every toxic span in the text, as written.",
    "litbank/entities": None, "linnaeus": None,
    "crossner/politics": None,
    "crossner/science": None,
    "crossner/music": None,
    "crossner/literature": None,
    "crossner/ai": None,
    "skillspan/skill": "Every skill requirement in the text, as written.",
    "skillspan/knowledge": "Every knowledge requirement in the text, as written.",
    "fabner": None, "buster": None, "biored/entities": None,
    "medmentions/st21pv": None, "chia": None,
    "bionlp-cg/entities": None, "bioinfer/entities": None, "bc4chemd": None,
    "uner/en_ewt": None,
    "harveyner": None,
    # Multilingual peppering (MULTILINGUAL, SOURCE_ROWS).
    "uner/pt_bosque": None, "uner/zh_gsd": None,
    "masakhaner2/swa": None, "massive-slots/hi": "Every {name} in the utterance, as written.",
    "inspec/keyphrases": "Every keyphrase in the text, as written.",

    "bc5cdr": None, "tweetner7": None, "mit-movie-trivia": None, "mit-restaurant": None,
    "conll04": None, "bc2gm": None,
    "conll2003/ner_tags": None, "wnut_17/wnut_17": None, "ncbi_disease/ncbi_disease": None,
    "jnlpba/jnlpba": None, "ontonotes_english/SpeedOfMagic--ontonotes_english": None,
    "few-nerd/supervised": None, "acronym_identification": None,
    "finer-139": "Every amount reported as {name}, as written.",
    "wiesp2022-ner": None, "broad-twitter-corpus": None, "finer-ord": None, "anatem": None,
    "stackoverflow-ner": None, "legallens-ner": None, "gum-ner": None, "opinion-target/en": None,

    "maven/triggers": "Every word or phrase that triggers a {name} event, as written.",
    "slurp/slots": None,
    "ebm-nlp/participants": "Every {name} in the trial abstract, as written.",
    "ebm-nlp/interventions": "Every {name} in the trial abstract, as written.",
    "ebm-nlp/outcomes": "Every {name} in the trial abstract, as written.",
}
# NER sources of the multilingual catalog (``list_tasks(multilingual=True)``).
MULTILINGUAL = {"uner/pt_bosque", "uner/zh_gsd", "masakhaner2/swa", "massive-slots/hi"}
SPLITS = ("train", "validation", "test")
# A collection is a task-id prefix (swde/auto, swde/book... -> swde). Its training budget is shared equally by its
# sources, so a suite split into many configs weighs like one source; procedural generators share a larger budget.
COLLECTION_ROWS = {"procedural": 80_000,
    # Their non-English members are capped by SOURCE_ROWS; raised so uner/en_ewt and tydiqa/english keep max_rows.
    "uner": 30_000, "tydiqa": 70_000}
# Valid but low-signal sources (label audit v2026.10.20.6: subjective or crowd-noisy span boundaries), capped
# lower than other sources rather than parked.
SOURCE_ROWS = {"inspec/keyphrases": 1_000, "toxic-spans": 2_000, "ebm-nlp/outcomes": 2_000, "legallens-ner": 1_000,
    # Multilingual peppering: 1,600 training rows per language, shared by its sources (Swahili: QA + NER).
    "uner/pt_bosque": 1_600, "uner/zh_gsd": 1_600, "masakhaner2/swa": 800, "massive-slots/hi": 1_600,
    "tydiqa/arabic": 1_600, "tydiqa/bengali": 1_600, "tydiqa/finnish": 1_600, "tydiqa/korean": 1_600,
    "tydiqa/russian": 1_600, "tydiqa/swahili": 800}


def collection(task):
    return task.split("/", 1)[0]


def train_limits(tasks, max_rows, max_collection_rows):
    """Training cap per task: ``max_rows``, lowered so that each collection stays within its budget."""
    size = Counter(collection(t) for t in tasks)
    return {t: min(max_rows, SOURCE_ROWS.get(t, max_rows),
                   COLLECTION_ROWS.get(collection(t), max_collection_rows) // size[collection(t)]) for t in tasks}


def row_id(row):
    key = json.dumps([row["source"], row["state"], [f["instruction"] for f in row["fields"]]], ensure_ascii=False)
    return f"{row['source']}:{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def procedural_rows(task):
    from datasets import load_dataset
    config = task.split("/", 1)[1]
    dataset = load_dataset(PROCEDURAL_REPO, config, revision=PROCEDURAL_REVISION)
    return {split: [{**{k: v for k, v in r.items() if k != "level"}, "source": task} for r in dataset[split]]
            for split in SPLITS if split in dataset}


def catalog(task, extractive):
    """``load_task``/``task_licenses`` keyword arguments selecting the catalog of ``task``."""
    return {"extractive": extractive, "multilingual": task in MULTILINGUAL}


def source_rows(task, extractive, args):
    """``{split: [rows]}`` of one task, before capping."""
    if extractive is None:
        return procedural_rows(task)
    dataset = load_task(task, **catalog(task, extractive))
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


def without_excess_empty(rows, share, seed, name):
    """Rows whose every field is null, subsampled to at most ``share`` of the result; order kept."""
    empty = [i for i, r in enumerate(rows) if all(a["null"] for a in r["answers"])]
    keep = min(len(empty), int(share / (1 - share) * (len(rows) - len(empty))))
    dropped = set(empty) - set(random.Random(f"{seed}:{name}:empty").sample(empty, keep))
    return [r for i, r in enumerate(rows) if i not in dropped], len(dropped)


def build(args):
    qa = [t for t in __import__("tasksource").list_tasks(extractive=True).id if t not in EXCLUDED_QA]
    tasks = [(t, True) for t in qa] + [(t, False) for t in NER] + [(f"procedural/{t}", None) for t in PROCEDURAL]
    if args.tasks:
        tasks = [(t, e) for t, e in tasks if t in args.tasks]
    licenses = {}
    for extractive, multilingual in ((True, False), (False, False), (False, True)):
        ids = [t for t, e in tasks if e == extractive and (t in MULTILINGUAL) == multilingual]
        if ids:
            frame = task_licenses(ids, extractive=extractive, multilingual=multilingual, fresh=True)
            licenses.update({r.id: (r.license, r.license_use) for r in frame.itertuples()})
    licenses.update({t: ("mit", "commercial") for t, e in tasks if e is None})
    limits = train_limits([t for t, _ in tasks], args.max_rows, args.max_collection_rows)
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
        if extractive is False:  # texts without any entity: at most --max-empty-share of a split
            for split, items in rows.items():
                rows[split], counts[f"{split}_empty_dropped"] = without_excess_empty(
                    items, args.max_empty_share, args.seed, f"{task}:{split}")
        train_states = {r["state"] for r in rows.get("train", [])}
        for split, items in rows.items():
            if split != "train":
                before = len(items)
                items = [r for r in items if r["state"] not in train_states]
                counts[f"{split}_overlap_dropped"] = before - len(items)
            items = capped(items, limits[task] if split == "train" else args.max_rows_eval, args.seed, f"{task}:{split}")
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
                "max_rows_eval": args.max_rows_eval, "max_collection_rows": args.max_collection_rows,
                "collection_rows": COLLECTION_ROWS, "source_rows": SOURCE_ROWS, "max_fields": args.max_fields, "max_empty_share": args.max_empty_share, "seed": args.seed,
                "rows": {s: len(r) for s, r in dataset.items()},
                "revisions": {t: PROCEDURAL_REVISION if e is None else
                              __import__("tasksource").task_provenance(t, **catalog(t, e)).get("revision") for t, e in tasks}}
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
  DROP single-span, DuoRC, ConcurrentQA, FairytaleQA, QuAC with history, windowed CUAD, TyDi English,
  Qasper spans, PolicyQA, SubjQA, COVID-QA, TimeQA, TAT-QA text spans, ROPES, QAConv and MultiDoc2Dial
  grounding, MultiSpanQA, QA-SRL) groups the questions asked about one context, up to {max_fields} fields per row.
  Named entities (CoNLL-2003, OntoNotes, WNUT-17, Few-NERD, NCBI-disease, JNLPBA, acronyms, FiNER-139,
  MAVEN event triggers, SLURP slots, EBM-NLP PICO spans, and astronomy, biomedical, financial, software,
  legal, GUM, Twitter and English opinion-target sources) ask for every entity type of the dataset, or
  the types present plus absent ones up to {max_fields}. Entity texts are the source tokens joined by spaces.
  Other span tagging (skills, keyphrases, aspect terms, toxic spans, metaphors, learner errors, BioScope and
  ConanDoyle-neg negation and speculation scopes, AbstRCT claims and premises, GUM elementary discourse units)
  asks for every span of its kind. Event arguments (PHEE adverse drug events, RAMS and CASIE) ask for every
  argument of each role of a trigger given by its character offsets.
- Structural extraction uses English UD EWT/GUM dependency subtrees (`ud/...`), PreCo coreference
  mention sets (`preco/mentions`), human-revised Re-DocRED relation targets (`redocred/entities`),
  and CoNLL-2000 phrase chunks. UD fields select a unique head and keep full contiguous subtrees;
  a missing dependent is null and a non-projective positive field is omitted.
- Procedural extraction ([procedural-extraction](https://huggingface.co/datasets/tasksource/procedural-extraction),
  sources `procedural/...`): person records with near-miss names and confusers, plan slots in force after a chat
  of proposals and undos, conditional order lists after updates, log queries, typed mentions, business documents
  with amendments, nested configurations with override layers, raw HTML pages, records dated in mixed formats
  within a range and products matching specs in mixed units and number styles, at difficulty levels 0 to 4;
  spans and nulls exact by construction.
- Mostly English, with a small multilingual share: TyDi QA (Arabic, Bengali, Finnish, Korean, Russian, Swahili),
  UNER (Portuguese, Chinese), MasakhaNER 2 (Swahili) and MASSIVE slots (Hindi), at most 1,600 training rows per
  language.
- Native splits, except those without released labels; evaluation texts that also occur in training are dropped. Each source is capped at
  {max_rows:,} training rows (a suite of configs, such as `swde/...`, shares {max_collection_rows:,}) and {max_rows_eval:,} per evaluation split; at most {max_empty:.0%} of an entity source's
  texts mention none of its entity types. `license`/`license_use` per row;
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
                       max_rows=args.max_rows, max_rows_eval=args.max_rows_eval,
                       max_collection_rows=manifest.get("max_collection_rows", args.max_rows),
                       max_empty=manifest["max_empty_share"])
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
    parser.add_argument("--max-collection-rows", type=int, default=20_000,
                        help="training rows shared by the sources of a collection (task-id prefix)")
    parser.add_argument("--max-fields", type=int, default=8)
    parser.add_argument("--max-empty-share", type=float, default=0.2,
                        help="entity sources: share of texts without any entity kept per split")
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
