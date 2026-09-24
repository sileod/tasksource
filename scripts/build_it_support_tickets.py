#!/usr/bin/env python3
"""Ingest Zenodo's Classification of IT Support Tickets into Tasksource."""

import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi


ZENODO_RECORD_ID = "7648117"
SPLIT_FILES = {
    "train": ("X_train.csv", "y_train.csv"),
    "test": ("X_test.csv", "y_test.csv"),
}


def strip_accidental_index_columns(frame):
    """Remove CSV-export index columns, retaining the real issue ID for joins."""
    drop = [
        column for column in frame.columns
        if not str(column).strip()
        or str(column).lower().startswith("unnamed:")
        or str(column).lower() in {"index", "level_0"}
    ]
    return frame.drop(columns=drop)


def merge_source_split(x_frame, y_frame):
    """Join labels by source ID, preserving X row order and exact text values."""
    x_frame = strip_accidental_index_columns(x_frame)
    y_frame = strip_accidental_index_columns(y_frame)
    required_x = {"id", "text"}
    required_y = {"id", "category_truth"}
    if not required_x <= set(x_frame.columns):
        raise ValueError(f"X CSV missing required columns: {required_x - set(x_frame.columns)}")
    if not required_y <= set(y_frame.columns):
        raise ValueError(f"y CSV missing required columns: {required_y - set(y_frame.columns)}")
    if x_frame["id"].duplicated().any() or y_frame["id"].duplicated().any():
        raise ValueError("Source IDs must be unique within each split")
    if set(x_frame["id"]) != set(y_frame["id"]):
        raise ValueError("X/y source ID sets do not match; refusing a partial join")

    joined = x_frame[["id", "text"]].merge(
        y_frame[["id", "category_truth"]],
        on="id", how="left", sort=False, validate="one_to_one",
    )
    if joined["text"].isna().any() or joined["category_truth"].isna().any():
        raise ValueError("Missing ticket text or category after ID join")
    return joined


def build_dataset(split_frames):
    """Create train/test Parquet-ready splits without reshuffling."""
    joined = {split: merge_source_split(*frames) for split, frames in split_frames.items()}
    label_names = sorted({
        str(label)
        for frame in joined.values()
        for label in frame["category_truth"]
    })
    features = Features({
        "text": Value("string"),
        "label": ClassLabel(names=label_names),
    })
    result = {}
    for split, frame in joined.items():
        rows = [
            {"text": text, "label": str(label)}
            for text, label in zip(frame["text"], frame["category_truth"])
        ]
        result[split] = Dataset.from_list(rows, features=features)
    return DatasetDict(result)


def download_source_files(directory):
    endpoint = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    response = requests.get(endpoint, timeout=60)
    response.raise_for_status()
    record = response.json()
    meta = record["metadata"]
    if meta.get("doi") != "10.5281/zenodo.7648117":
        raise ValueError(f"Unexpected Zenodo DOI: {meta.get('doi')!r}")

    file_urls = {file["key"]: file["links"]["self"]
                 for file in record.get("files", [])}
    for names in SPLIT_FILES.values():
        for name in names:
            if name not in file_urls:
                raise FileNotFoundError(f"Zenodo record is missing {name}")
            result = requests.get(file_urls[name], timeout=120)
            result.raise_for_status()
            (directory / name).write_bytes(result.content)
    return meta


def upload(repo_id="tasksource/it-support-tickets", card_path=None):
    with TemporaryDirectory(prefix="it-support-tickets-") as temp:
        root = Path(temp)
        download_source_files(root)
        split_frames = {
            split: (pd.read_csv(root / x_name), pd.read_csv(root / y_name))
            for split, (x_name, y_name) in SPLIT_FILES.items()
        }
        dataset = build_dataset(split_frames)
        for split in dataset:
            print(f"{split}: {len(dataset[split])} rows; labels={Counter(dataset[split]['label'])}")

        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        dataset.push_to_hub(
            repo_id,
            commit_message="Add data-only IT support ticket classification dataset",
        )
        if card_path is None:
            card_path = Path(__file__).resolve().parents[1] / "dataset_cards" / "tasksource-it-support-tickets.md"
        api.upload_file(
            path_or_fileobj=str(card_path), path_in_repo="README.md",
            repo_id=repo_id, repo_type="dataset",
            commit_message="Add IT support tickets dataset card and citation",
        )
        print(f"Uploaded {repo_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="tasksource/it-support-tickets")
    parser.add_argument("--card", type=Path)
    args = parser.parse_args()
    upload(args.repo_id, args.card)


if __name__ == "__main__":
    main()
