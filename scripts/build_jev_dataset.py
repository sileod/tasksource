#!/usr/bin/env python3
"""Build and optionally publish the tasksource-jev-typed-decisions dataset.

Each task/split is written as its own Parquet shard, making interrupted builds
resumable. The JSONL report is append-only and records successes and failures.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, List, Value, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
import numpy as np
import pandas as pd
import yaml
from tasksource import list_tasks, load_task, task_provenance
from tasksource.preprocess import sample_dataset
from tasksource.metadata.weights import task_weight
from tasksource.metadata.dpi_licenses import LICENSE_USE as DPI_USE, REPO_LICENSES as DPI_REPOS, TASK_LICENSES as DPI_TASKS
from tasksource.jev.augmentations import augment_jev_internal, stable_fraction
from tasksource.jev.prompt_augmentations import (
    published_pair_style, published_question_style,
)
from tasksource.jev import procedural
from tasksource.jev.derived import VARIANT as PACKED_VARIANT, add_packed_classification, packed_items
from tasksource.jev.length import LengthBudget, render_request
from tasksource.jev.options import gold_position_violations


SUPPORTED_TYPES = {"Classification", "MultipleChoice", "TokenClassification", "SoftLabeling"}
PUBLISH_EXCLUDED_PREFIXES = ("bigbench/", "mmlu/", "blimp/")
JEV_TOKEN_TASKS = {
    "conll2003/ner_tags", "wnut_17/wnut_17",
}


# Vote shares from fewer annotators than this are coarse (one of three is 0.33): such
# soft annotations enter Jev by their hard majority view, if they have one.
MIN_ANNOTATORS = 5


def load_jev_task(row, max_rows, max_rows_eval, revision=None, data_file_pins=None):
    """``revision`` and ``data_file_pins`` pin the task's Hub dataset and ``hf://``
    data files to the commits recorded in the build report."""
    return load_task(
        row.id, recast="jev", multilingual=row.multilingual,
        soft=row.task_type == "SoftLabeling", min_annotators=MIN_ANNOTATORS,
        max_rows=max_rows, max_rows_eval=max_rows_eval, data_file_pins=data_file_pins,
        **({"revision": revision} if revision else {}),
    )


TRAINING_FEATURES = Features({
    "id": Value("string"), "kind": Value("string"), "options": List(Value("string")),
    "target": List(Value("float64")), "state": Value("string"), "question": Value("string"),
    "source": Value("string"), "variant": Value("string"), "split": Value("string"),
})


NATIVE_SOURCES = (
    [procedural.SOURCE_PREFIX + name for name in sorted(procedural.TASKS)]
)


def load_native_task(source_id, max_rows, max_rows_eval, revision=None, data_file_pins=None):
    """Procedural sources, authored as typed Jev questions grouped by state (no recast)."""
    dataset = load_dataset(procedural.REPO_ID, source_id[len(procedural.SOURCE_PREFIX):], revision=revision)
    dataset = sample_dataset(dataset, max_rows, max_rows_eval)
    rows = {split: [row for index, example in enumerate(examples)
                    for row in procedural.jev_rows(example, source_id, normalized_split(split), index)]
            for split, examples in dataset.items()}
    return DatasetDict({
        split: Dataset.from_list(split_rows, features=TRAINING_FEATURES)
        for split, split_rows in rows.items() if split_rows  # e.g. hidden test labels
    })


def normalized_split(split):
    return "dev" if split == "validation" else split


def filter_request_lengths(rows, max_bytes=131_072, budget=None):
    """Drop rows whose complete rendered request exceeds byte or exact-token budgets."""
    if not max_bytes and budget is None:
        return rows, 0

    def fits(row):
        question = {
            "question_id": row.get("question_id", "decision"),
            "kind": row["kind"],
            "question": row["question"],
            "options": row["options"],
        }
        questions = [question]
        if max_bytes and len(render_request(row["state"], questions).encode("utf-8")) > max_bytes:
            return False
        return budget is None or budget.fits(row["state"], questions)

    before = len(rows)
    rows = rows.filter(fits)
    return rows, before - len(rows)


def slug(task_id):
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:10]
    readable = "".join(c if c.isalnum() else "-" for c in task_id).strip("-")[:70]
    return f"{readable}-{digest}"


# Arguments that change what a task's shards contain: the build fingerprint, so a resumed build
# never mixes shards built with different settings. The code state is recorded per shard
# (report "code") but not enforced: any commit would otherwise invalidate every shard.
SHARD_PARAMETERS = ("max_rows", "max_rows_eval", "noul_rate", "score_rate", "permutation_rate", "prompt_rate",
                    "paired_format_rate", "pack_rate", "pack_max_tokens", "pack_tokenizer", "pack_max_items",
                    "max_request_bytes", "max_request_tokens", "request_tokenizer")


def _git(*arguments):
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=False).stdout


def code_state():
    """Commit plus a hash of uncommitted changes under src/ and scripts/, untracked files included."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(_git("diff", "--binary", "HEAD", "--", "src", "scripts"))
    for name in sorted(_git("ls-files", "--others", "--exclude-standard", "--", "src", "scripts").decode().split()):
        if not name.endswith((".pyc", ".pyo")):
            digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    return {"git_commit": _git("rev-parse", "HEAD").decode().strip(), "uncommitted_sha256": digest.hexdigest()}


def build_fingerprint(args):
    shard_args = {name: getattr(args, name, None) for name in SHARD_PARAMETERS}
    payload = json.dumps(shard_args, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def read_completed(report_path, data_dir=None, fingerprint=None):
    """Tasks whose latest successful shards exist; with ``fingerprint``, return
    ``(completed, stale)``: ``stale`` tasks have shards from another fingerprint."""
    completed, stale = set(), set()
    # only a task's latest record counts: a later failed rebuild voids an earlier success
    for task, record in latest_records(report_path).items():
        if record["status"] != "ok":
            continue
        splits = record.get("rows", {})
        if data_dir is None or (
            splits and all(
                (data_dir / f"{split}-{slug(task)}.parquet").exists()
                for split in splits
            )
        ):
            current = fingerprint is None or record.get("fingerprint") == fingerprint
            (completed if current else stale).add(task)
    return completed if fingerprint is None else (completed, stale)


def resolve_revisions(provenance):
    """Commit sha of each Hub repo a source loads, at the revision it requests."""
    api, pins = HfApi(), {}
    requested = {**({provenance["dataset"]: provenance.get("revision")} if provenance.get("dataset") else {}),
                 **{repo: provenance.get("data_file_revisions", {}).get(repo)
                    for repo in provenance.get("data_files_from", [])}}
    for repo, revision in requested.items():
        try:
            pins[repo] = api.dataset_info(repo, revision=revision).sha
        except Exception:  # gated or offline: loading will tell; the report records no pin
            pins[repo] = None
    return pins


def build_manifest(args, tasks):
    """Record enough provenance to rerun the pipeline from the same inputs."""
    root = Path(__file__).resolve().parents[1]
    def display_path(value):
        try:
            return str(value.resolve().relative_to(root))
        except ValueError:
            return value.name

    def git(*arguments):
        result = subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, check=False
        )
        return result.stdout
    status = git("status", "--porcelain").decode("utf-8", "replace").splitlines()
    diff = git("diff", "--binary", "HEAD", "--", "src", "scripts", "dataset_cards")
    versions = {}
    for package in ("datasets", "huggingface_hub", "pyarrow", "numpy", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "git_commit": git("rev-parse", "HEAD").decode().strip(),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "fingerprint": build_fingerprint(args),
        "working_tree_changes": status,
        "python": sys.version.split()[0],
        "packages": versions,
        "parameters": {
            key: display_path(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "selected_sources": list(tasks.source_id),
    }


def fixed_source_audit(baseline_path, selected, latest):
    """Compare previously script-bound task IDs with this build's results."""
    if baseline_path is None:
        return None
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    old_ids = sorted({record["task"] for record in baseline})
    audit = {"succeeded": [], "failed": [], "not_attempted": [],
             "not_in_current_catalog": []}
    for source in old_ids:
        if source not in selected:
            audit["not_in_current_catalog"].append(source)
        elif source not in latest:
            audit["not_attempted"].append(source)
        elif latest[source]["status"] == "ok":
            audit["succeeded"].append(source)
        else:
            audit["failed"].append({
                "task": source,
                "error": latest[source].get("error", ""),
            })
    return audit


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
    if "target" in example:  # soft labels: the distribution itself
        target = list(example["target"])
    else:
        label = int(example["label"])
        if not 0 <= label < len(options):
            raise ValueError(f"{task_id}:{split}:{index}: label {label} outside {len(options)} options")
        target = [0.0] * len(options)
        target[label] = 1.0
    source_row = example.get("source_row", index)
    question_id = example.get("question_id", "decision")
    # soft annotations of one dataset share a group, so one input's questions stay together
    group_id = f"{slug(example.get('group') or task_id)}:{split}:{source_row}"
    row_id = group_id if question_id == "decision" else f"{group_id}:{question_id}"
    return {
        "id": row_id,
        "kind": example.get("kind") or "choice",  # ordinal tasks: part score
        "options": options,
        "target": target,
        "state": example["state"],
        "question": example["instructions"],
        "source": task_id,
        "variant": "direct",
        "split": normalized_split(split),
    }


def write_pack_audit(output, split, task_id, records):
    """Member IDs of each packed group, kept out of the lightweight schema."""
    if not records:
        return
    path = output / "pack-audit" / f"{split}-{slug(task_id)}.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    ), encoding="utf-8")


def check_gold_positions(rows, split, multiple_choice_sources):
    """Reject multiple-choice sources whose gold answers favor one slot."""
    sources, targets = [], []
    columns = rows.select_columns(["source", "kind", "variant", "target"])
    for batch in columns.iter(batch_size=10_000):
        for source, kind, variant, target in zip(
            batch["source"], batch["kind"], batch["variant"], batch["target"]
        ):
            if source in multiple_choice_sources and kind == "choice" and variant == "direct":
                sources.append(source)
                targets.append(target)
    violations = gold_position_violations(sources, targets)
    if violations:
        raise ValueError(
            f"Gold answers concentrated in one option slot in {split} "
            f"(source: share, slot, rows): {violations}"
        )


# Shorter matches are generic utterances ("okay", "Quick!") that recur
# naturally across splits, not contamination.
MIN_OVERLAP_CHARS = 40


def _normalized(text):
    return re.sub(r"\W+", " ", str(text)).casefold().strip()


def content_keys(rows):
    """Keys for train-overlap checks: (state, options) for direct rows, and
    each item text for packed rows, whose members may be capped away."""
    decisions, texts = set(), set()
    columns = rows.select_columns(["state", "options", "variant"])
    for batch in columns.iter(batch_size=10_000):
        for state, options, variant in zip(batch["state"], batch["options"], batch["variant"]):
            if variant == "direct":
                decisions.add((_normalized(state), tuple(sorted(map(_normalized, options)))))
                texts.add(_normalized(state))
            elif variant == PACKED_VARIANT:
                texts.update(map(_normalized, packed_items(state)))
    return decisions, texts


def drop_train_overlap(rows, train_keys):
    """Drop eval source-row groups and packs whose content is in train."""
    decisions, texts = train_keys
    leaked = set()
    columns = rows.select_columns(["id", "state", "options", "variant"])
    for batch in columns.iter(batch_size=10_000):
        for identifier, state, options, variant in zip(
            batch["id"], batch["state"], batch["options"], batch["variant"]
        ):
            if variant == "direct":
                key = (_normalized(state), tuple(sorted(map(_normalized, options))))
                content = len(key[0]) + max(map(len, key[1]), default=0)
                hit = content >= MIN_OVERLAP_CHARS and key in decisions
            elif variant == PACKED_VARIANT:
                hit = any(
                    len(item) >= MIN_OVERLAP_CHARS and item in texts
                    for item in map(_normalized, packed_items(state))
                )
            else:
                continue
            if hit:
                leaked.add(source_row_group(identifier))
    ids = rows.select_columns(["id"])[:]["id"]
    keep = [i for i, identifier in enumerate(ids) if source_row_group(identifier) not in leaked]
    return rows.select(keep), len(rows) - len(keep)


def pretty_order(dataset, first_rows=1_000, seed=0):
    """Round-robin sources in a display prefix, then shuffle the remainder.

    The prefix shows variety in the Hub viewer. Without the shuffle, capped
    splits keep build order and put all procedural rows last.
    """
    first_rows = min(first_rows, len(dataset))
    shuffle = np.random.default_rng(seed).permutation
    if first_rows < 2 or "source" not in dataset.column_names:
        return dataset.select(shuffle(len(dataset)))
    # ``dataset["source"]`` is a lazy Column in recent datasets releases.
    # Repeated scalar indexing inside the loops below repeatedly rebuilds an
    # Arrow column and makes publication effectively quadratic.  Format the
    # metadata columns once instead.
    display_columns = ["source"]
    if "variant" in dataset.column_names:
        display_columns.append("variant")
    display = dataset.select_columns(display_columns)[:]
    sources = display["source"]
    variants = display.get("variant")
    names = sorted(set(sources))
    if len(names) < 2:
        return dataset.select(shuffle(len(dataset)))
    per_source = (first_rows + len(names) - 1) // len(names)
    buckets = {name: [] for name in names}
    first_by_variant = {name: {} for name in names} if variants else None
    for index, source in enumerate(sources):
        bucket = buckets[source]
        if len(bucket) < per_source:
            bucket.append(index)
        if variants:
            first_by_variant[source].setdefault(variants[index], index)
    if variants:
        primary = ("direct", "instruction_paraphrase", "paired_text_format")
        for name in names:
            offset = min(int(stable_fraction(name, "preview-variant") * 3), 2)
            priority = (
                *primary[offset:], *primary[:offset],
                "criteria_permutation", "label_verification",
            )
            chosen = [
                first_by_variant[name][variant]
                for variant in priority if variant in first_by_variant[name]
            ]
            buckets[name] = list(dict.fromkeys((*chosen, *buckets[name])))[:per_source]
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
    order = np.concatenate((np.asarray(prefix), shuffle(np.flatnonzero(~selected))))
    return dataset.select(order)


def source_family(source):
    """Balance dataset families, not each configuration as an independent task."""
    parts = source.split("/")
    if parts[0] in {"multilingual", procedural.SOURCE_PREFIX.rstrip("/")} and len(parts) > 1:
        return "/".join(parts[:2])
    return parts[0]


def _ranked_family_groups(source_buckets, identifiers):
    """Round-robin source configs while keeping each source-row group intact."""
    active = deque()
    for source in sorted(source_buckets):
        groups = {}
        for index in source_buckets[source]:
            groups.setdefault(source_row_group(identifiers[index]), []).append(index)
        active.append(iter(sorted(
            groups.items(),
            key=lambda item: stable_fraction(item[0], "publish-cap"),
        )))
    while active:
        iterator = active.popleft()
        try:
            group = next(iterator)
        except StopIteration:
            continue
        yield group
        active.append(iterator)


MAX_MULTILINGUAL_SHARE = 0.2  # a ceiling on multilingual families' share of each format


def diverse_cap(dataset, max_rows):
    """Cap by dataset family, sampling configs and preserving question groups."""
    if max_rows is None or len(dataset) <= max_rows:
        return dataset
    metadata_columns = ["source"]
    if "id" in dataset.column_names:
        metadata_columns.append("id")
    metadata = dataset.select_columns(metadata_columns)[:]
    sources = metadata["source"]
    selected = []
    deferred = []
    buckets = {}
    for index, source in enumerate(sources):
        family = source_family(source)
        buckets.setdefault(family, {}).setdefault(source, []).append(index)
    # each family's share of the cap scales with its weight (metadata/weights.py)
    weights = {family: max(task_weight(source) for source in configs) for family, configs in buckets.items()}
    multilingual = sum(weight for family, weight in weights.items() if family.startswith("multilingual/"))
    english = sum(weights.values()) - multilingual
    if english and multilingual > MAX_MULTILINGUAL_SHARE * (english + multilingual):
        scale = MAX_MULTILINGUAL_SHARE * english / ((1 - MAX_MULTILINGUAL_SHARE) * multilingual)
        weights = {family: weight * scale if family.startswith("multilingual/") else weight
                   for family, weight in weights.items()}
    total_weight = sum(weights.values())
    quotas = {family: max(1, int(max_rows * weight / total_weight)) for family, weight in weights.items()}
    family_sizes = {
        family: sum(len(indices) for indices in configs.values())
        for family, configs in buckets.items()
    }
    base_possible = sum(min(quotas[family], size) for family, size in family_sizes.items())
    overflow_needed = max_rows - base_possible
    total_extra = sum(max(0, size - quotas[family]) for family, size in family_sizes.items())
    ids = (
        metadata["id"]
        if "id" in metadata
        else list(map(str, range(len(dataset))))
    )
    for family in sorted(buckets):
        quota = quotas[family]
        extra = max(0, family_sizes[family] - quota)
        overflow_limit = (
            32 + (overflow_needed * extra + total_extra - 1) // total_extra
            if extra and total_extra else 0
        )
        source_rows = 0
        source_deferred = 0
        for group_id, indices in _ranked_family_groups(buckets[family], ids):
            if source_rows + len(indices) <= quota:
                selected.extend(indices)
                source_rows += len(indices)
            elif source_deferred < overflow_limit:
                deferred.append((group_id, indices))
                source_deferred += 1
    if len(selected) < max_rows:
        deferred.sort(
            key=lambda item: stable_fraction(item[0], "publish-overflow")
        )
        for _, indices in deferred:
            if len(selected) + len(indices) <= max_rows:
                selected.extend(indices)
            if len(selected) == max_rows:
                break
    return dataset.select(sorted(selected))


# Target share of each format in a capped split. Graded and procedural shares are
# reserved; the other formats share the rest, and one short of rows passes its
# unused share to them in proportion to theirs.
FORMAT_SHARES = {"Classification": 0.47, "MultipleChoice": 0.30, "TokenClassification": 0.03,
                 "graded": 0.10, "procedural": 0.10}
RESERVED_FORMATS = ("graded", "procedural")


def row_format(source, formats=None):
    if source.startswith(procedural.SOURCE_PREFIX):
        return "procedural"
    task_type = (formats or {}).get(source, "Classification")
    return "graded" if task_type == "SoftLabeling" else task_type  # soft labels: the graded share


def format_budgets(sizes, max_rows, shares):
    """Row budget per format: reserved shares first, then the rest by share."""
    budgets = {name: min(sizes[name], int(max_rows * shares[name])) for name in RESERVED_FORMATS if name in sizes}
    remaining = max_rows - sum(budgets.values())
    active = {name for name in sizes if name not in budgets}
    while active:
        total = sum(shares[name] for name in active)
        short = [name for name in active if sizes[name] <= remaining * shares[name] / total]
        if not short:
            for name in active:
                budgets[name] = int(remaining * shares[name] / total)
            break
        for name in short:
            budgets[name] = sizes[name]
            remaining -= sizes[name]
            active.remove(name)
    return budgets


def share_cap(dataset, max_rows, procedural_share=FORMAT_SHARES["procedural"], formats=None):
    """Cap a split with a fixed share per format, then by family within each format.

    ``formats`` maps a source task to its task type (Classification,
    MultipleChoice, TokenClassification, SoftLabeling: the graded share);
    procedural sources are recognized by prefix. Without fixed shares, classification would take most
    rows, and the seven procedural generators, the only source of several
    decision shapes, would get under 2%.
    """
    if max_rows is None or len(dataset) <= max_rows:
        return diverse_cap(dataset, max_rows)
    sources = dataset.select_columns(["source"])[:]["source"]
    groups = {}
    for index, source in enumerate(sources):
        groups.setdefault(row_format(source, formats), []).append(index)
    shares = {**FORMAT_SHARES, "procedural": procedural_share}
    budgets = format_budgets({name: len(rows) for name, rows in groups.items()}, max_rows, shares)
    # group-preserving caps can fall a few rows short; the format with the most
    # spare rows goes last and takes up the difference
    last = max(groups, key=lambda name: len(groups[name]) - budgets[name])
    parts = [diverse_cap(dataset.select(groups[name]), budgets[name]) for name in sorted(groups) if name != last]
    parts.append(diverse_cap(dataset.select(groups[last]), max_rows - sum(map(len, parts))))
    return concatenate_datasets(parts)


def source_row_group(identifier):
    """The stable source-row prefix shared by direct and variant questions."""
    return ":".join(identifier.split(":", 3)[:3])


def exclude_publish_sources(
    dataset, prefixes=PUBLISH_EXCLUDED_PREFIXES, allowed_sources=None
):
    """Keep only current, allowed source tasks without deleting build shards."""
    if "source" not in dataset.column_names:
        return dataset
    sources = dataset.select_columns(["source"])[:]["source"]
    keep = [
        index for index, source in enumerate(sources)
        if not source.startswith(prefixes)
        and (allowed_sources is None or source in allowed_sources)
    ]
    return dataset if len(keep) == len(dataset) else dataset.select(keep)


def add_question_groups(dataset):
    """Backfill grouping metadata in preexisting Parquet shards."""
    if "group_id" in dataset.column_names:
        return dataset
    ids = dataset.select_columns(["id"])[:]["id"]
    group_ids = []
    question_ids = []
    for identifier in ids:
        parts = identifier.split(":")
        if len(parts) > 3 and parts[3].startswith("token-"):
            group_ids.append(source_row_group(identifier))
            question_ids.append(":".join(parts[3:]))
        else:
            group_ids.append(source_row_group(identifier))
            question_ids.append("decision" if len(parts) == 3 else ":".join(parts[3:]))
    return dataset.add_column("group_id", group_ids).add_column("question_id", question_ids)


def diversify_published_prompts(dataset):
    """Vary safe paired-field wording by source row without multiplying rows."""
    def convert(batch):
        states = []
        questions = []
        for group_id, state, question, options in zip(
            batch["group_id"], batch["state"], batch["question"], batch["options"]
        ):
            styled_state, styled_question = published_pair_style(
                state, question, stable_fraction(group_id, "published-pair-style")
            )
            styled_question = published_question_style(
                styled_question, options, styled_state,
                stable_fraction(group_id, "published-question-style"),
            )
            states.append(styled_state)
            questions.append(styled_question)
        return {"state": states, "question": questions}
    return dataset.map(convert, batched=True, batch_size=1_000)


def validate_decisions(rows, split):
    """Check the training target against its primitive before publication."""
    columns = rows.select_columns(["id", "kind", "options", "target"])
    for batch in columns.iter(batch_size=10_000):
        for identifier, kind, options, target in zip(
            batch["id"], batch["kind"], batch["options"], batch["target"]
        ):
            if kind == "noul":
                valid = (
                    not options and len(target) == 1
                    and math.isfinite(target[0]) and 0 <= target[0] <= 1
                )
            else:
                texts = [str(option).strip() for option in options if option is not None]
                valid = (
                    kind in {"choice", "score"} and len(options) >= 2
                    and len(texts) == len(options) and all(texts)
                    and len(set(texts)) == len(texts)
                    and len(options) == len(target)
                    and all(math.isfinite(value) and value >= 0 for value in target)
                    and math.isclose(sum(target), 1.0, abs_tol=1e-5)
                )
            if not valid:
                raise ValueError(
                    f"Invalid {kind} target in {split}, decision {identifier}"
                )


def publish_dataset(
    output, repo_id, pretty_rows=1_000, publish_rows=1_000_000, eval_rows=15_000,
    allowed_sources=None, procedural_share=0.1,
):
    data_files = {}
    for split in ("train", "validation", "test"):
        files = sorted((output / "data").glob(f"{split}-*.parquet"))
        if files:
            data_files[split] = [str(path) for path in files]
    dataset = load_dataset("parquet", data_files=data_files)
    dataset = DatasetDict({
        split: exclude_publish_sources(
            split_dataset, allowed_sources=allowed_sources
        )
        for split, split_dataset in dataset.items()
    })
    overlap_dropped = {}
    formats = {task: record.get("task_type") for task, record in latest_records(output / "build-report.jsonl").items()}
    if publish_rows:
        caps = {"train": publish_rows, "validation": eval_rows, "test": eval_rows}
        train_keys = None
        # Train first: eval rows whose content is in the published train go.
        for split in sorted(dataset, key=lambda name: name != "train"):
            cap = caps.get(split, 0)
            if split != "train" and train_keys is not None:
                dataset[split], overlap_dropped[split] = drop_train_overlap(
                    dataset[split], train_keys
                )
            available = len(dataset[split])
            dataset[split] = share_cap(dataset[split], cap, procedural_share, formats)
            if available >= cap and len(dataset[split]) != cap:
                raise ValueError(
                    f"Group-preserving cap produced {len(dataset[split])} "
                    f"rather than {cap} rows in {split}"
                )
            if (repo_id == "tasksource/tasksource-jev-typed-decisions"
                    and publish_rows >= 1_000_000 and len(dataset[split]) != cap):
                raise ValueError(
                    f"One-million-row release requires {cap} {split} rows, "
                    f"but only {len(dataset[split])} were selected"
                )
            if split == "train":
                train_keys = content_keys(dataset[split])
    dataset = DatasetDict({
        split: diversify_published_prompts(add_question_groups(split_dataset))
        for split, split_dataset in dataset.items()
    })
    if "train" in dataset:
        dataset["train"] = pretty_order(dataset["train"], pretty_rows)
    licenses = source_licenses(sorted({source for rows in dataset.values()
                                       for source in set(rows.select_columns(["source"])[:]["source"])}))
    dataset = DatasetDict({split: add_license_columns(rows, licenses) for split, rows in dataset.items()})
    release_audit = {
        "repo_id": repo_id, "requested_rows": publish_rows,
        "requested_eval_rows": eval_rows,
        "eval_rows_dropped_for_train_overlap": overlap_dropped, "splits": {},
    }
    multiple_choice_sources = {
        task for task, record in latest_records(output / "build-report.jsonl").items()
        if record.get("task_type") == "MultipleChoice"
    }
    for split, rows in dataset.items():
        validate_decisions(rows, split)
        check_gold_positions(rows, split, multiple_choice_sources)
        metadata = rows.select_columns(["id", "source", "split", "kind", "variant"])[:]
        if len(set(metadata["id"])) != len(rows):
            raise ValueError(f"Duplicate Jev decision IDs in {split}")
        if any(source.startswith(PUBLISH_EXCLUDED_PREFIXES) for source in metadata["source"]):
            raise ValueError(f"Excluded benchmark source in {split}")
        if allowed_sources is not None and not set(metadata["source"]) <= allowed_sources:
            raise ValueError(f"Unselected source in {split}")
        if set(metadata["split"]) != {normalized_split(split)}:
            raise ValueError(f"Mixed source split annotations in {split}")
        source_counts = Counter(metadata["source"])
        format_counts = Counter()
        for source, count in source_counts.items():
            format_counts[row_format(source, formats)] += count
        family_counts = Counter()
        for source, count in source_counts.items():
            family_counts[source_family(source)] += count
        release_audit["splits"][split] = {
            "rows": len(rows),
            "sources": dict(sorted(source_counts.items())),
            "formats": dict(format_counts.most_common()),
            "families": dict(sorted(family_counts.items())),
            "kinds": dict(sorted(Counter(metadata["kind"]).items())),
            "variants": dict(sorted(Counter(metadata["variant"]).items())),
            "license_use": dict(sorted(Counter(licenses[source]["license_use"] for source in metadata["source"]).items())),
        }
        print(f"Publish {split}: {len(rows)} rows, {len(set(metadata['source']))} sources", flush=True)
        if split == "train":
            preview = rows.select(range(min(pretty_rows, len(rows))))
            variants = preview.select_columns(["variant"])[:]["variant"]
            variant_counts = pd.Series(variants).value_counts().to_dict()
            print(f"Preview variants: {variant_counts}", flush=True)
    if repo_id == "tasksource/tasksource-jev-typed-decisions" and publish_rows >= 1_000_000:
        train = release_audit["splits"]["train"]
        missing = sorted(
            (set(JEV_TOKEN_TASKS) | {
                procedural.SOURCE_PREFIX + name for name in procedural.TASKS
            }) - set(train["sources"])
        )
        missing_kinds = sorted({"choice", "noul", "score"} - set(train["kinds"]))
        missing_variants = sorted({
            "direct", "label_verification", "criteria_permutation",
            "instruction_paraphrase", "paired_text_format",
        } - set(train["variants"]))
        if missing or missing_kinds or missing_variants:
            raise ValueError(
                "Required release coverage missing: "
                f"sources={missing}, kinds={missing_kinds}, variants={missing_variants}"
            )
    (output / "release-audit.json").write_text(
        json.dumps(release_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_sources_yaml(output / "sources.yaml", release_audit, latest_records(output / "build-report.jsonl"), licenses)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(output / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add tasksource-jev-typed-decisions dataset card",
    )
    dataset.push_to_hub(
        repo_id,
        max_shard_size="256MB",
        commit_message="Publish tasksource-jev-typed-decisions Parquet dataset",
    )
    for name in (
        "build-report.jsonl", "build-summary.json", "build-manifest.json",
        "failed-tasks.json", "outdated-datasets.json", "fixed-source-audit.json",
        "release-audit.json", "token-source-status.md", "sources.yaml",
    ):
        path = output / name
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=name,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add {name}",
            )


def source_provenance(source):
    """Hub provenance of one published source (see tasksource.task_provenance)."""
    if source.startswith(procedural.SOURCE_PREFIX):
        return {"dataset": procedural.REPO_ID, "config": source[len(procedural.SOURCE_PREFIX):],
                "generated": "procedural (tasksource.jev.procedural)"}
    if source.startswith("multilingual/"):
        return task_provenance(source[len("multilingual/"):], multilingual=True)
    return task_provenance(source)


# Hub card license ids (lowercase) whose terms allow commercial use; share-alike and
# copyleft included. Anything else (cc, other, no-derivatives, custom) stays unclassified.
PERMISSIVE_LICENSE = re.compile(
    r"^(apache|mit|bsd|cc0|cc-by-\d|cc-by-sa-\d|odc-by|odbl|pddl|afl|gpl|lgpl|agpl|artistic|cdla|ecl|epl|mpl"
    r"|isc|unlicense|wtfpl|bigscience|openrail|creativeml-openrail|llama2|llama3)")
NON_COMMERCIAL_LICENSE = re.compile(r"(^|-)nc(-|$)|academic|research")
DPI_LICENSE_USE = {"All": "commercial", "NC": "non-commercial", "Acad": "non-commercial"}


def card_license_use(license):
    if NON_COMMERCIAL_LICENSE.search(license):
        return "non-commercial"
    return "commercial" if PERMISSIVE_LICENSE.match(license) else None


def dpi_license_use(license):
    # "No License" means all rights reserved, whatever DPI's class says
    return None if license == "No License" else DPI_LICENSE_USE.get(DPI_USE.get(license))


def fetch_card_licenses(repos):
    """The ``license`` of each Hub dataset card, as a list (``unknown`` dropped)."""
    api = HfApi()

    def read(repo):
        for attempt in range(3):
            try:
                card = api.dataset_info(repo).card_data
                break
            except (GatedRepoError, RepositoryNotFoundError):  # no readable card
                return repo, []
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
        value = (card or {}).get("license") if card else None
        values = value if isinstance(value, list) else [value]
        return repo, [str(v).lower() for v in values if v and str(v).lower() != "unknown"]

    with ThreadPoolExecutor(16) as pool:
        return dict(pool.map(read, sorted(repos)))


def source_license(source, info, cards):
    """License of one source from the cards of the repos it loads (and their originals) and
    Data Provenance Initiative annotations. ``license_use`` takes the most restrictive
    classified license: non-commercial, else commercial, else unspecified."""
    repos = [r for r in [info.get("dataset"), *info.get("data_files_from", []), *info.get("originals", [])] if r]
    card = {repo: cards[repo] for repo in repos if cards.get(repo)}
    task = source.removeprefix("multilingual/")
    # DPI annotated older tasksource ids; a whole-dataset id also covers its configs
    dpi = sorted(set(DPI_TASKS.get(task, DPI_TASKS.get(task.split("/")[0], []))).union(
        *[DPI_REPOS.get(repo.lower(), []) for repo in repos]))
    uses = {card_license_use(license) for licenses in card.values() for license in licenses}
    uses |= {dpi_license_use(license) for license in dpi}
    use = "non-commercial" if "non-commercial" in uses else "commercial" if "commercial" in uses else "unspecified"
    names = sorted({license for licenses in card.values() for license in licenses}) + [f"{name} (DPI)" for name in dpi]
    return {"license": ", ".join(names) or "unspecified", "license_use": use,
            **({"card_licenses": card} if card else {}), **({"dpi_licenses": dpi} if dpi else {})}


def source_licenses(sources):
    provenance = {}
    for source in sources:
        try:
            provenance[source] = source_provenance(source)
        except KeyError:
            provenance[source] = {}
    cards = fetch_card_licenses({repo for info in provenance.values() for repo in
                                 [info.get("dataset"), *info.get("data_files_from", []), *info.get("originals", [])]
                                 if repo})
    return {source: source_license(source, info, cards) for source, info in provenance.items()}


def add_license_columns(rows, licenses):
    sources = rows.select_columns(["source"])[:]["source"]
    rows = rows.add_column("license", [licenses[source]["license"] for source in sources])
    return rows.add_column("license_use", [licenses[source]["license_use"] for source in sources])


def write_sources_yaml(path, release_audit, records=None, licenses=None):
    """Every source in the release with its rows, Hub dataset, revision and originals.

    Revisions are the commits each task was loaded from, recorded in the build report
    (``records``, the latest report record per task); a task built before revisions were
    recorded is marked ``revisions_unrecorded`` rather than given today's commit."""
    records = records or {}

    splits = release_audit["splits"]
    sources = {}
    for source in sorted({s for split in splits.values() for s in split["sources"]}):
        info = {"rows": {name: split["sources"][source] for name, split in splits.items() if source in split["sources"]}}
        try:
            info.update(source_provenance(source))
        except KeyError as error:
            info["provenance_error"] = str(error)
        recorded = (records.get(source) or {}).get("revisions")
        if recorded:
            info["revisions"] = {repo: sha for repo, sha in recorded.items() if sha}
        elif info.get("dataset") or info.get("data_files_from"):  # URL-only sources have no Hub revision
            info["revisions_unrecorded"] = True
        info.update((licenses or {}).get(source, {}))
        sources[source] = info
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except OSError:
        commit = None
    document = {
        "repo_id": release_audit["repo_id"],
        "tasksource_commit": commit,
        "datasets": sorted({repo for info in sources.values()
                            for repo in [info.get("dataset"), *info.get("data_files_from", []), *info.get("originals", [])]
                            if repo}),
        "sources": sources,
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")


def migrate_legacy_shards(
    data_dir, noul_rate=0.05, score_rate=0.0, permutation_rate=0.05,
    prompt_rate=0.05, paired_format_rate=0.05,
):
    """Upgrade resumable shards created before the public schema was finalized."""
    for path in sorted(data_dir.glob("*.parquet")):
        shard = load_dataset("parquet", data_files=str(path), split="train")
        if "id" not in shard.column_names:
            split = path.name.split("-", 1)[0]
            task_id = shard[0]["task"]
            upgraded = shard.map(
                to_training_row,
                with_indices=True,
                fn_kwargs={"task_id": task_id, "split": split},
                remove_columns=shard.column_names,
            )
        else:
            upgraded = shard
            if "variant" not in upgraded.column_names:
                upgraded = upgraded.add_column("variant", ["direct"] * len(upgraded))
            if "usage" in upgraded.column_names:
                upgraded = upgraded.remove_columns("usage")
            if "split" not in upgraded.column_names:
                physical_split = path.name.split("-", 1)[0]
                upgraded = upgraded.add_column(
                    "split", [normalized_split(physical_split)] * len(upgraded)
                )
        upgraded = augment_jev_internal(
            upgraded, noul_rate, score_rate, permutation_rate, prompt_rate,
            paired_format_rate,
        )
        if upgraded.column_names == shard.column_names and len(upgraded) == len(shard):
            continue
        temporary = path.with_suffix(".parquet.tmp")
        upgraded.to_parquet(temporary)
        temporary.replace(path)


def select_tasks(args):
    frame = list_tasks(instruct=True, soft=True, min_annotators=MIN_ANNOTATORS)
    frame["multilingual"] = False
    frame["source_id"] = frame.id
    if not args.english_only:
        multilingual = list_tasks(multilingual=True, instruct=True, soft=True, min_annotators=MIN_ANNOTATORS)
        multilingual["multilingual"] = True
        multilingual["source_id"] = "multilingual/" + multilingual.id
        frame = pd.concat([frame, multilingual], ignore_index=True)
    frame = frame[frame.task_type.isin(SUPPORTED_TYPES)]
    frame = pd.concat([frame, pd.DataFrame([
        {"id": source, "source_id": source, "task_type": "NativeJev", "multilingual": False}
        for source in NATIVE_SOURCES
    ])], ignore_index=True)
    frame = frame[
        ~frame.source_id.str.startswith(PUBLISH_EXCLUDED_PREFIXES, na=False)
    ]
    frame = frame[
        (frame.task_type != "TokenClassification")
        | frame.source_id.isin(JEV_TOKEN_TASKS)
    ]
    if args.tasks:
        wanted = set(args.tasks)
        frame = frame[frame.source_id.isin(wanted)]
        missing = wanted - set(frame.source_id)
        if missing:
            raise ValueError(f"Unknown or unsupported tasks: {sorted(missing)}")
    if args.limit:
        frame = frame.head(args.limit)
    # Two catalog aliases can resolve to the same source/config. Never run
    # either twice or let a later shard silently overwrite the first one.
    return frame.drop_duplicates("source_id", keep="first").reset_index(drop=True)


def build(args):
    output = args.output.resolve()
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_migrate:
        migrate_legacy_shards(
            data_dir, args.noul_rate, args.score_rate, args.permutation_rate,
            args.prompt_rate, args.paired_format_rate,
        )
    report_path = output / "build-report.jsonl"
    fingerprint = build_fingerprint(args)
    completed, stale = read_completed(report_path, data_dir, fingerprint)
    tasks = select_tasks(args)
    stale &= set(tasks.source_id)
    if stale and not (args.finalize_only or args.reuse_incompatible_shards):
        raise SystemExit(
            f"{len(stale)} task shards in {output} were built with other settings (build fingerprint "
            f"{fingerprint} differs), e.g. {sorted(stale)[:3]}. Use a fresh --output, or pass "
            "--reuse-incompatible-shards to keep them (build-manifest.json records their fingerprints).")
    if args.reuse_incompatible_shards:
        completed |= stale
    packing_budget = LengthBudget(args.pack_max_tokens, args.pack_tokenizer)
    request_budget = (
        LengthBudget(args.max_request_tokens, args.request_tokenizer)
        if args.request_tokenizer and args.max_request_tokens else None
    )
    print(f"Selected {len(tasks)} tasks; {len(completed)} already complete", flush=True)
    manifest_path = output / "build-manifest.json"
    manifest = build_manifest(args, tasks)
    if manifest_path.exists():  # keep earlier runs' provenance next to this one's
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["previous_runs"] = previous.pop("previous_runs", []) + [
            {key: previous.get(key) for key in ("git_commit", "git_diff_sha256", "fingerprint", "parameters")}]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for position, row in enumerate(
        () if args.finalize_only else tasks.itertuples(index=False), start=1
    ):
        task_id = row.source_id
        if task_id in completed:
            print(f"[{position}/{len(tasks)}] skip {task_id}", flush=True)
            continue
        started = time.time()
        print(f"[{position}/{len(tasks)}] build {task_id}", flush=True)
        partial = {}  # every split goes to a partial file, renamed only once all splits succeed
        try:
            max_rows = args.max_rows
            max_rows_eval = args.max_rows_eval
            if row.task_type == "TokenClassification":
                max_rows = max(1, max_rows // 2)
                max_rows_eval = max(1, max_rows_eval // 2)
            # pin every Hub repo to the commit its requested revision points to now, and record it
            provenance = source_provenance(task_id)
            pins = resolve_revisions(provenance)
            loaded = pins.get(provenance.get("dataset"))
            file_pins = {repo: pins.get(repo) for repo in provenance.get("data_files_from", []) if pins.get(repo)}
            if row.task_type == "NativeJev":
                dataset = load_native_task(task_id, max_rows, max_rows_eval, loaded, file_pins)
            else:
                dataset = load_jev_task(row, max_rows, max_rows_eval, loaded, file_pins)
            split_rows = {}
            request_length_dropped = {}
            for split, split_dataset in dataset.items():
                if row.task_type != "NativeJev":
                    split_dataset = split_dataset.map(
                        to_training_row,
                        with_indices=True,
                        fn_kwargs={"task_id": task_id, "split": split},
                        remove_columns=split_dataset.column_names,
                    )
                    pack_audit = []
                    split_dataset = add_packed_classification(
                        split_dataset, row.task_type, rate=args.pack_rate,
                        budget=packing_budget, max_items=args.pack_max_items,
                        audit=pack_audit,
                    )
                    write_pack_audit(output, split, task_id, pack_audit)
                    if row.task_type != "SoftLabeling":  # a distribution's questions are asked as authored
                        split_dataset = augment_jev_internal(
                            split_dataset, args.noul_rate, args.score_rate,
                            args.permutation_rate, args.prompt_rate,
                            args.paired_format_rate,
                        )
                split_dataset, dropped = filter_request_lengths(
                    split_dataset, args.max_request_bytes, request_budget)
                if dropped:
                    request_length_dropped[split] = dropped
                partial[split] = data_dir / f".{split}-{slug(task_id)}.parquet.partial"
                split_dataset.to_parquet(partial[split])
                split_rows[split] = len(split_dataset)
            for split in ("train", "validation", "test"):
                final = data_dir / f"{split}-{slug(task_id)}.parquet"
                if split in partial:
                    partial.pop(split).replace(final)
                else:
                    final.unlink(missing_ok=True)  # a split an earlier build had and this one lacks
            record = {
                "task": task_id,
                "task_type": row.task_type,
                "status": "ok",
                "rows": split_rows,
                "request_length_dropped": request_length_dropped,
                "fingerprint": fingerprint,
                "code": code_state(),
                "revisions": pins,
                "seconds": round(time.time() - started, 3),
            }
            completed.add(task_id)
        except Exception as error:
            for path in partial.values():
                path.unlink(missing_ok=True)
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
        shutil.copyfile(args.card.parent / "jev-token-source-status.md", output / "token-source-status.md")
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
            "excluded_source_families": list(PUBLISH_EXCLUDED_PREFIXES),
            # the code/settings fingerprints the selected shards were built with; more than
            # one means --reuse-incompatible-shards or --finalize-only kept older shards
            "build_fingerprint": fingerprint,
            "shard_fingerprints": dict(Counter(
                latest[task].get("fingerprint") or "unrecorded" for task in sorted(selected & set(latest))
                if latest[task]["status"] == "ok")),
        }
        (output / "build-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        fix_audit = fixed_source_audit(args.baseline_failures, selected, latest)
        if fix_audit is not None:
            (output / "fixed-source-audit.json").write_text(
                json.dumps(fix_audit, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print("Previously failed sources: " + json.dumps({
                key: len(value) for key, value in fix_audit.items()
            }), flush=True)
        print(json.dumps(summary), flush=True)

    if args.upload:
        publish_dataset(
            output, args.repo_id, args.pretty_rows, args.publish_rows,
            eval_rows=args.eval_rows,
            allowed_sources=set(tasks.source_id),
            procedural_share=args.procedural_share,
        )


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "build" / "tasksource-jev-typed-decisions")
    parser.add_argument("--card", type=Path, default=root / "dataset_cards" / "tasksource-jev.md")
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--english-only", action="store_true",
        help="Exclude the multilingual Tasksource catalog.",
    )
    parser.add_argument(
        "--reuse-incompatible-shards", action="store_true",
        help="Resume even when existing shards come from other code or shard settings.",
    )
    parser.add_argument("--max-rows", type=int, default=30_000)
    parser.add_argument("--max-rows-eval", type=int, default=3_000)
    parser.add_argument(
        "--noul-rate", type=float, default=0.05,
        help="Fraction of direct rows receiving a label-verification Noul variant.",
    )
    parser.add_argument(
        "--score-rate", type=float, default=0.0,
        help="Fraction receiving an ordered-rubric Score variant (off by default; use genuine ordinal sources).",
    )
    parser.add_argument(
        "--permutation-rate", type=float, default=0.05,
        help="Fraction receiving a deterministic criterion-order permutation.",
    )
    parser.add_argument(
        "--prompt-rate", type=float, default=0.05,
        help="Fraction receiving a vetted, meaning-preserving instruction variant.",
    )
    parser.add_argument(
        "--paired-format-rate", type=float, default=0.05,
        help="Fraction of paired-text rows receiving neutral field-label variation.",
    )
    parser.add_argument(
        "--pack-rate", type=float, default=0.10,
        help="Max fraction of each classification task/split packed into exact multi-question states.",
    )
    parser.add_argument(
        "--pack-max-tokens", type=int, default=4096,
        help="Length budget for a complete packed request (state, questions, criteria).",
    )
    parser.add_argument(
        "--pack-tokenizer",
        help="Hugging Face tokenizer for exact budgets; default is a conservative UTF-8 byte bound.",
    )
    parser.add_argument("--pack-max-items", type=int, default=4)
    parser.add_argument(
        "--max-request-bytes", type=int, default=131_072,
        help="Maximum UTF-8 size of a complete rendered request. Set 0 to disable.",
    )
    parser.add_argument(
        "--max-request-tokens", type=int, default=32_768,
        help="Exact token cap when --request-tokenizer is supplied. Set 0 to disable.",
    )
    parser.add_argument(
        "--request-tokenizer",
        help="Hugging Face tokenizer used for the optional exact token cap.",
    )
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument(
        "--finalize-only", action="store_true",
        help="Refresh reports and publish from checkpointed shards without retrying tasks.",
    )
    parser.add_argument(
        "--skip-migrate", action="store_true",
        help="Reuse checkpointed shards already written with the current schema.",
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id", default="tasksource/tasksource-jev-typed-decisions")
    parser.add_argument(
        "--baseline-failures", type=Path,
        help="Previous failed/outdated-datasets.json; writes fixed-source-audit.json.",
    )
    parser.add_argument(
        "--publish-rows", type=int, default=1_000_000,
        help="Published train rows; 0 disables all capping.",
    )
    parser.add_argument(
        "--eval-rows", type=int, default=15_000,
        help="Published rows in each of validation and test.",
    )
    parser.add_argument(
        "--pretty-rows", type=int, default=1_000,
        help="Deterministically interleave sources in this many leading train rows.",
    )
    parser.add_argument(
        "--procedural-share", type=float, default=0.1,
        help="Fraction of each capped split reserved for procedural-typed-decisions sources.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
