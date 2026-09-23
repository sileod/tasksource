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
import shutil
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, List, Value, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
import numpy as np
import pandas as pd
from tasksource import list_tasks, load_task
from tasksource import tasks as english_tasks
from tasksource.access import load_preprocessing
from tasksource.preprocess import sample_dataset
from tasksource.jev.augmentations import augment_jev_internal, stable_fraction
from tasksource.jev.prompt_augmentations import (
    published_pair_style, published_question_style,
)
from tasksource.jev import graded, procedural
from tasksource.jev.recast import recast_jev


SUPPORTED_TYPES = {"Classification", "MultipleChoice", "TokenClassification"}
PUBLISH_EXCLUDED_PREFIXES = ("bigbench/", "mmlu/", "blimp/")
JEV_TOKEN_TASKS = {
    "conll2003/ner_tags", "wnut_17/wnut_17",
}
TOKEN_SOURCE_MIRRORS = {
    # Data-only copies; each retains the original columns, ClassLabel names,
    # and train/validation/test boundaries used by the Tasksource annotation.
    "conll2003/ner_tags": ("tomaarsen/conll2003", None),
    "wnut_17/wnut_17": ("flaitenberger/wnut_17", None),
}


def load_jev_task(row, max_rows, max_rows_eval):
    """Use audited data-only mirrors for otherwise script-bound token tasks."""
    mirror = TOKEN_SOURCE_MIRRORS.get(row.source_id)
    if mirror is None:
        return load_task(
            row.id, recast="jev", multilingual=row.multilingual,
            max_rows=max_rows, max_rows_eval=max_rows_eval,
        )
    preprocessing = load_preprocessing(english_tasks, id=row.id)
    source = load_dataset(*mirror)
    standardized = preprocessing(source, max_rows, max_rows_eval)
    return recast_jev(standardized, task=row.source_id)


TRAINING_FEATURES = Features({
    "id": Value("string"), "kind": Value("string"), "options": List(Value("string")),
    "target": List(Value("float64")), "state": Value("string"), "question": Value("string"),
    "source": Value("string"), "variant": Value("string"), "split": Value("string"),
})


NATIVE_SOURCES = (
    [procedural.SOURCE_PREFIX + name for name in sorted(procedural.TASKS)]
    + [graded.SOURCE_PREFIX + name for name in graded.FAMILIES]
)


def load_native_task(source_id, max_rows, max_rows_eval):
    """Sources authored as typed Jev questions, grouped by state (no recast)."""
    if source_id.startswith(graded.SOURCE_PREFIX):
        rows = graded.load_family(source_id[len(graded.SOURCE_PREFIX):], max_rows, max_rows_eval)
    else:
        dataset = load_dataset(procedural.REPO_ID, source_id[len(procedural.SOURCE_PREFIX):])
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


def slug(task_id):
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:10]
    readable = "".join(c if c.isalnum() else "-" for c in task_id).strip("-")[:70]
    return f"{readable}-{digest}"


def read_completed(report_path, data_dir=None):
    completed = set()
    if not report_path.exists():
        return completed
    for line in report_path.read_text().splitlines():
        record = json.loads(line)
        if record["status"] == "ok":
            task = record["task"]
            splits = record.get("rows", {})
            if data_dir is None or (
                splits and all(
                    (data_dir / f"{split}-{slug(task)}.parquet").exists()
                    for split in splits
                )
            ):
                completed.add(task)
    return completed


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
    label = int(example["label"])
    target = [0.0] * len(options)
    target[label] = 1.0
    source_row = example.get("source_row", index)
    question_id = example.get("question_id", "decision")
    group_id = f"{slug(task_id)}:{split}:{source_row}"
    row_id = group_id if question_id == "decision" else f"{group_id}:{question_id}"
    return {
        "id": row_id,
        "kind": "choice",
        "options": options,
        "target": target,
        "state": example["state"],
        "question": example["instructions"],
        "source": task_id,
        "variant": "direct",
        "split": normalized_split(split),
    }


def pretty_order(dataset, first_rows=1_000):
    """Round-robin sources in a display prefix without shuffling the remainder."""
    first_rows = min(first_rows, len(dataset))
    if first_rows < 2 or "source" not in dataset.column_names:
        return dataset
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
        return dataset
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
    order = np.concatenate((np.asarray(prefix), np.flatnonzero(~selected)))
    return dataset.select(order)


def source_family(source):
    """Balance dataset families, not each configuration as an independent task."""
    parts = source.split("/")
    if parts[0] in {"multilingual", "graded", "procedural-jev"} and len(parts) > 1:
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
    quota = max(1, max_rows // len(buckets))
    family_sizes = {
        family: sum(len(indices) for indices in configs.values())
        for family, configs in buckets.items()
    }
    base_possible = sum(min(quota, size) for size in family_sizes.values())
    overflow_needed = max_rows - base_possible
    total_extra = sum(max(0, size - quota) for size in family_sizes.values())
    ids = (
        metadata["id"]
        if "id" in metadata
        else list(map(str, range(len(dataset))))
    )
    for family in sorted(buckets):
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


def share_cap(dataset, max_rows, procedural_share):
    """Reserve a fixed share of the cap for procedural sources.

    Under the per-source quota, seven generators would get under 2% of the
    rows although they are the only source of Score, graded Noul, and
    multi-question states. A fixed share also stays put as sources come and go.
    """
    sources = dataset.select_columns(["source"])[:]["source"]
    generated = [i for i, source in enumerate(sources) if source.startswith(procedural.SOURCE_PREFIX)]
    if max_rows is None or not generated or len(generated) == len(dataset):
        return diverse_cap(dataset, max_rows)
    generated_rows = min(len(generated), int(max_rows * procedural_share))
    other = sorted(set(range(len(dataset))) - set(generated))
    return concatenate_datasets([
        diverse_cap(dataset.select(other), max_rows - generated_rows),
        diverse_cap(dataset.select(generated), generated_rows),
    ])


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
                valid = (
                    kind in {"choice", "score"} and len(options) >= 2
                    and len(options) == len(target)
                    and all(math.isfinite(value) and value >= 0 for value in target)
                    and math.isclose(sum(target), 1.0, abs_tol=1e-5)
                )
            if not valid:
                raise ValueError(
                    f"Invalid {kind} target in {split}, decision {identifier}"
                )


def publish_dataset(
    output, repo_id, pretty_rows=1_000, publish_rows=1_000_000,
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
    if publish_rows:
        weights = {"train": 0.90, "validation": 0.05, "test": 0.05}
        for split in dataset:
            cap = int(publish_rows * weights.get(split, 0))
            available = len(dataset[split])
            dataset[split] = share_cap(dataset[split], cap, procedural_share)
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
    dataset = DatasetDict({
        split: diversify_published_prompts(add_question_groups(split_dataset))
        for split, split_dataset in dataset.items()
    })
    if "train" in dataset:
        dataset["train"] = pretty_order(dataset["train"], pretty_rows)
    release_audit = {"repo_id": repo_id, "requested_rows": publish_rows, "splits": {}}
    for split, rows in dataset.items():
        validate_decisions(rows, split)
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
        family_counts = Counter()
        for source, count in source_counts.items():
            family_counts[source_family(source)] += count
        release_audit["splits"][split] = {
            "rows": len(rows),
            "sources": dict(sorted(source_counts.items())),
            "families": dict(sorted(family_counts.items())),
            "kinds": dict(sorted(Counter(metadata["kind"]).items())),
            "variants": dict(sorted(Counter(metadata["variant"]).items())),
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
        "release-audit.json", "token-source-status.md",
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
    frame = list_tasks(instruct=True)
    frame["multilingual"] = False
    frame["source_id"] = frame.id
    if not args.english_only:
        multilingual = list_tasks(multilingual=True, instruct=True)
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
    completed = read_completed(report_path, data_dir)
    tasks = select_tasks(args)
    print(f"Selected {len(tasks)} tasks; {len(completed)} already complete", flush=True)
    manifest_path = output / "build-manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(build_manifest(args, tasks), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    for position, row in enumerate(
        () if args.finalize_only else tasks.itertuples(index=False), start=1
    ):
        task_id = row.source_id
        if task_id in completed:
            print(f"[{position}/{len(tasks)}] skip {task_id}", flush=True)
            continue
        started = time.time()
        print(f"[{position}/{len(tasks)}] build {task_id}", flush=True)
        try:
            max_rows = args.max_rows
            max_rows_eval = args.max_rows_eval
            if row.task_type == "TokenClassification":
                max_rows = max(1, max_rows // 2)
                max_rows_eval = max(1, max_rows_eval // 2)
            if row.task_type == "NativeJev":
                dataset = load_native_task(task_id, max_rows, max_rows_eval)
            else:
                dataset = load_jev_task(row, max_rows, max_rows_eval)
            split_rows = {}
            for split, split_dataset in dataset.items():
                if row.task_type == "NativeJev":
                    path = data_dir / f"{split}-{slug(task_id)}.parquet"
                    split_dataset.to_parquet(path)
                    split_rows[split] = len(split_dataset)
                    continue
                split_dataset = split_dataset.map(
                    to_training_row,
                    with_indices=True,
                    fn_kwargs={"task_id": task_id, "split": split},
                    remove_columns=split_dataset.column_names,
                )
                split_dataset = augment_jev_internal(
                    split_dataset, args.noul_rate, args.score_rate,
                    args.permutation_rate, args.prompt_rate,
                    args.paired_format_rate,
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
        help="Maximum published rows across a 90/5/5 train/dev/test allocation; 0 disables.",
    )
    parser.add_argument(
        "--pretty-rows", type=int, default=1_000,
        help="Deterministically interleave sources in this many leading train rows.",
    )
    parser.add_argument(
        "--procedural-share", type=float, default=0.1,
        help="Fraction of each capped split reserved for procedural-jev sources.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
