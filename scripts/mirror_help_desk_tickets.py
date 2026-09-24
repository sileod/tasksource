#!/usr/bin/env python3
"""Publish tidy help-desk tables derived from Mendeley Help Desk Tickets v3."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import requests
from datasets import Dataset
from huggingface_hub import HfApi


DATASET_ID = "btm76zndnt"
DOI = "10.17632/btm76zndnt.3"
API_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"

CARD = """---
license: cc-by-4.0
pretty_name: Help Desk Tickets (Mendeley v3, processed tables)
configs:
- config_name: default
  data_files:
  - split: train
    path: reporter_messages/train.parquet
---

# Help Desk Tickets

Processed tables derived from version 3 of Mohammad Abdellatif's Mendeley Data
dataset. The source contains real helpdesk tickets and associated workflow
records from an international software company, covering January 2007 through
March 2023. Identifiers and message content were masked by the source authors
to protect privacy while retaining context.

The default config is **`reporter_messages`**, with one row per issue that has
reporter-authored text (357 rows). Utterances are kept in chronological order
and joined with newline separators. Assignee and other-participant replies are
excluded. Issue metadata is joined onto those rows; the separate full issues
table is not published.

These are derived data views, not supervised train/dev/test splits. Mendeley v3
does not provide a seven-class category field in `issues.csv`; its
`issue_type` field is a separate ticket/work-item type. Therefore no
classification task is defined here. The reporter-message view contains 357
text-bearing issues; their `issue_type` values are `Ticket` (339), `HD Service`
(9), and `Deployment` (9), not the requested seven-category target.

## Columns

### Default: `reporter_messages`

- `issue_id`: source issue identifier
- `text`: chronological concatenation of reporter-authored utterances, with
  source text otherwise left unchanged
- `issue_type`, `issue_priority`, `issue_resolution`, `issue_status`
- `issue_created`, `issue_resolution_date`, `issue_comments_count`
- `message_count`: number of included reporter utterances

## Source and license

- Original title: *Help Desk Tickets*
- Author: Mohammad Abdellatif
- DOI: [10.17632/btm76zndnt.3](https://doi.org/10.17632/btm76zndnt.3)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The source provides PII-masked records and messages. Its language mix includes
English and other languages; no language filtering or text correction is
applied here.

This mirror uses Mendeley v3 (DOI `10.17632/btm76zndnt.3`). The core issue,
snapshot, history, utterance, and sample files have identical SHA-256 hashes in
v2 and v3; v3 adds `holidays_by_country.csv` and updated documentation. We do not
duplicate the identical v2 data.

## Citation

Mohammad Abdellatif. *Help Desk Tickets*. Mendeley Data, version 3.
https://doi.org/10.17632/btm76zndnt.3
"""


def canonical_id(value):
    if pd.isna(value):
        return None
    number = float(value)
    return str(int(number)) if number.is_integer() else str(value)


def prepare_reporter_messages(issues, utterances):
    """Join issue metadata onto reporter-only text grouped by issue."""
    required_issues = {"id", "issue_num", "issue_type", "issue_priority",
                       "issue_resolution", "issue_status", "issue_created",
                       "issue_resolution_date", "issue_comments_count",
                       "wf_total_time", "processing_steps"}
    required_messages = {"issueid", "created", "comment_seq", "utr_seq",
                         "actionbody", "author_role"}
    if not required_issues <= set(issues.columns):
        raise ValueError(f"issues.csv missing columns: {required_issues - set(issues.columns)}")
    if not required_messages <= set(utterances.columns):
        raise ValueError(
            f"sample_utterances.csv missing columns: {required_messages - set(utterances.columns)}"
        )

    issues = issues.copy()
    issues["issue_id"] = issues["id"].map(canonical_id)
    issue_columns = [
        "issue_id", "issue_num", "issue_proj", "issue_type", "issue_priority",
        "issue_resolution", "issue_status", "issue_created",
        "issue_resolution_date", "issue_comments_count", "wf_total_time",
        "processing_steps",
    ]
    issue_table = issues[issue_columns].copy()

    reporter = utterances.loc[utterances["author_role"] == "reporter"].copy()
    reporter["issue_id"] = reporter["issueid"].map(canonical_id)
    reporter["_sort_created"] = pd.to_datetime(reporter["created"], utc=True, errors="coerce")
    reporter = reporter.sort_values(
        ["issue_id", "_sort_created", "comment_seq", "utr_seq", "id"],
        kind="stable",
    )
    grouped = reporter.groupby("issue_id", sort=False).agg(
        text=("actionbody", lambda values: "\n".join(
            str(value) for value in values if pd.notna(value) and str(value).strip()
        )),
        message_count=("actionbody", lambda values: sum(
            pd.notna(value) and bool(str(value).strip()) for value in values
        )),
    ).reset_index()
    grouped = grouped.loc[grouped["text"].str.strip().ne("")]
    reporter_messages = grouped.merge(
        issue_table, on="issue_id", how="inner", validate="one_to_one", sort=False
    )
    if reporter_messages.empty:
        raise ValueError("No reporter-authored utterances matched source issues")
    return reporter_messages


def download_source_files(directory):
    response = requests.get(API_URL, timeout=60)
    response.raise_for_status()
    record = response.json()
    if record.get("version") != 3 or record.get("doi", {}).get("id") != DOI:
        raise ValueError("Mendeley API did not return the expected v3 record/DOI")
    urls = {item["filename"]: item["content_details"]["download_url"]
            for item in record["files"]}
    for name in ["issues.csv", "sample_utterances.csv"]:
        if name not in urls:
            raise FileNotFoundError(f"Mendeley v3 record is missing {name}")
        download = requests.get(urls[name], timeout=180)
        download.raise_for_status()
        (directory / name).write_bytes(download.content)


def mirror(repo_id="tasksource/help-desk-tickets"):
    with TemporaryDirectory(prefix="help-desk-tickets-v3-") as temp:
        root = Path(temp)
        download_source_files(root)
        issues = pd.read_csv(root / "issues.csv")
        utterances = pd.read_csv(root / "sample_utterances.csv")
        reporter_messages = prepare_reporter_messages(issues, utterances)

        output = root / "upload"
        path = output / "reporter_messages" / "train.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        Dataset.from_pandas(reporter_messages, preserve_index=False).to_parquet(path)
        print(f"reporter_messages: {len(reporter_messages)} rows")
        (output / "README.md").write_text(CARD, encoding="utf-8")

        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(output), repo_id=repo_id, repo_type="dataset",
            commit_message="Publish processed Mendeley v3 help-desk tables",
        )
        # Replace only the earlier raw mirror files created during this repair.
        keep = {"README.md", ".gitattributes"}
        keep.add("reporter_messages/train.parquet")
        for path in api.list_repo_files(repo_id, repo_type="dataset"):
            if path not in keep:
                api.delete_file(
                    path, repo_id=repo_id, repo_type="dataset",
                    commit_message="Replace raw mirror with tidy processed tables",
                )
        print(f"Published processed v3 tables to {repo_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="tasksource/help-desk-tickets")
    args = parser.parse_args()
    mirror(args.repo_id)


if __name__ == "__main__":
    main()
