"""Repackage raw-file datasets as parquet under tasksource/ on the Hub.

These sources only exist as GitHub files or inconsistent JSON, which makes
them slow and brittle to load. Each function returns a clean DatasetDict.

    python scripts/upload_repackaged.py sharc numer_sense clutrr [--dry-run]
"""

import argparse
import ast
import json
import re
import urllib.request

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset

from tasksource.jev.augmentations import stable_fraction

SHARC_URL = "https://raw.githubusercontent.com/nikhilweee/neural-conv-qa/master/datasets/mod_{}.json"
NUMERSENSE_URL = "https://raw.githubusercontent.com/INK-USC/NumerSense/main/data/train.masked.tsv"
CLUTRR_URL = "hf://datasets/kendrivp/CLUTRR_v1_extracted/gen_train234_test2to10/CLUTRR_v1_gen_train234_test2to10_{}.json"


def _split_of(key, validation=0.1, test=0.1):
    fraction = stable_fraction(key, "split")
    return "test" if fraction < test else "validation" if fraction < test + validation else "train"


def _follow_up(turn):
    # the source mixes follow_up_* and followup_* keys
    return (turn.get("follow_up_question") or turn.get("followup_question"),
            turn.get("follow_up_answer") or turn.get("followup_answer"))


def sharc():
    """ShARC (modified by Verma et al.): answer a rule question given a user scenario and follow-up Q&A."""
    labels = ["Yes", "No", "Irrelevant", "Clarification needed"]
    splits = {}
    for source, name in [("train", "train"), ("dev", "test")]:
        with urllib.request.urlopen(SHARC_URL.format(source)) as response:
            rows = json.load(response)
        for row in rows:
            history = [_follow_up(turn) for turn in row["history"]]
            answer = row["answer"]
            label = answer if answer in labels[:3] else labels[3]
            # train and validation are split by rule tree so no rule text leaks
            split = name if name == "test" else ("validation" if stable_fraction(row["tree_id"], "split") < 0.08 else "train")
            splits.setdefault(split, []).append(dict(
                snippet=row["snippet"], scenario=row["scenario"], question=row["question"],
                history="\n".join(f"Q: {q}\nA: {a}" for q, a in history),
                label=label, follow_up_question=answer if label == labels[3] else None,
                tree_id=row["tree_id"], utterance_id=row["utterance_id"], source_url=row["source_url"]))
    dataset = DatasetDict({split: Dataset.from_list(rows) for split, rows in splits.items()})
    return dataset.cast_column("label", ClassLabel(names=labels))


def numer_sense():
    """NumerSense masked number words (0-10), with deterministic 80/10/10 splits."""
    raw = load_dataset("csv", data_files=NUMERSENSE_URL, delimiter="\t", column_names=["sentence", "target"])["train"]
    names = sorted(set(raw["target"]))
    splits = {}
    for row in raw:
        splits.setdefault(_split_of(row["sentence"]), []).append(row)
    dataset = DatasetDict({split: Dataset.from_list(rows) for split, rows in splits.items()})
    return dataset.cast_column("target", ClassLabel(names=names))


def clutrr():
    """CLUTRR v1 (gen_train234_test2to10): infer a kinship relation from a short story."""
    raw = load_dataset("json", data_files={split: CLUTRR_URL.format(split) for split in ["train", "validation", "test"]})
    names = sorted({label for rows in raw.values() for label in rows["target_text"]})

    def render(row):
        head, tail = ast.literal_eval(row["query"])
        return dict(story=re.sub(r"\[([^\]]+)\]", r"\1", row["story"]),
                    query=f"How is {tail} related to {head}?", label=row["target_text"],
                    head=head, tail=tail, hops=len(ast.literal_eval(row["edge_types"])))

    dataset = raw.map(render, remove_columns=raw["train"].column_names)
    return dataset.cast_column("label", ClassLabel(names=names))


BUILDERS = {"sharc": ("tasksource/sharc", sharc), "numer_sense": ("tasksource/numer_sense", numer_sense),
            "clutrr": ("tasksource/clutrr", clutrr)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", choices=BUILDERS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in args.names:
        repo, build = BUILDERS[name]
        dataset = build()
        print(repo, dataset, dataset["train"][0], sep="\n")
        if not args.dry_run:
            dataset.push_to_hub(repo)
