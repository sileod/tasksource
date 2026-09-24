"""Repackage hard-to-load datasets as parquet under tasksource/ on the Hub.

These sources only exist as GitHub files, loading scripts or inconsistent
JSON, which makes them slow and brittle to load. Each function returns a clean
DatasetDict; its docstring becomes the Hub card, which also lists the original
datasets from tasksource/metadata/originals.py (add the entry there first).

    python scripts/upload_repackaged.py sharc numer_sense clutrr wellformed [--dry-run] [--card-only]
"""

import argparse
import ast
import inspect
import json
import re
import urllib.request

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset

from huggingface_hub import DatasetCard

from tasksource.jev.augmentations import stable_fraction
from tasksource.metadata.originals import ORIGINALS

SHARC_URL = "https://raw.githubusercontent.com/nikhilweee/neural-conv-qa/master/datasets/mod_{}.json"
NUMERSENSE_URL = "https://raw.githubusercontent.com/INK-USC/NumerSense/main/data/train.masked.tsv"
WELLFORMED_URL = "https://raw.githubusercontent.com/google-research-datasets/query-wellformedness/master/{}.tsv"
HUMICROEDIT_URL = "https://cs.rochester.edu/u/nhossain/semeval-2020-task-7-dataset.zip"
ETHOS_URL = "https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Multi_Label.csv"
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


def wellformed():
    """Google query wellformedness: the share of 5 raters who judged a search query a well-formed question.

    The trailing " ?" the source adds to every query is attached as "?"; the queries are otherwise untouched.
    """
    splits = {}
    for source, split in [("train", "train"), ("dev", "validation"), ("test", "test")]:
        raw = load_dataset("csv", data_files=WELLFORMED_URL.format(source), delimiter="\t",
                           column_names=["content", "rating"], quoting=3)["train"]
        splits[split] = raw.map(lambda x: {"content": re.sub(r"\s+\?$", "?", x["content"])})
    return DatasetDict(splits)


def humicroedit():
    """Humicroedit (SemEval-2020 task 7): news headlines edited to be funny.

    subtask-1 grades one edited headline (meanGrade averages five 0-3 funniness grades); `headline` and
    `edited` render the original headline and its edit from the `<word/>` markup. subtask-2 compares two
    edits of the same headline (label 1 or 2 is the funnier one, 0 a tie).
    """
    import io, zipfile
    import pandas as pd
    with urllib.request.urlopen(HUMICROEDIT_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    splits = {}
    for source, split in [("train", "train"), ("dev", "validation"), ("test", "test")]:
        rows = pd.read_csv(archive.open(f"semeval-2020-task-7-dataset/subtask-1/{source}.csv"), dtype={"grades": str})
        rows["headline"] = rows.original.str.replace(r"<([^/>]+)/>", r"\1", regex=True)
        rows["edited"] = [re.sub(r"<[^/>]+/>", edit, original) for original, edit in zip(rows.original, rows.edit)]
        splits[split] = Dataset.from_pandas(rows, preserve_index=False)
    return DatasetDict(splits)


def ethos():
    """ETHOS multi-label: hateful comments with the share of raters who saw each aspect (violence, target, grounds).

    Only the source's single split is provided.
    """
    import pandas as pd
    rows = pd.read_csv(ETHOS_URL, sep=";")
    return DatasetDict(train=Dataset.from_pandas(rows, preserve_index=False))


BUILDERS = {"sharc": ("tasksource/sharc", sharc), "numer_sense": ("tasksource/numer_sense", numer_sense),
            "clutrr": ("tasksource/clutrr", clutrr), "wellformed": ("tasksource/google_wellformed_query", wellformed),
            "humicroedit": ("tasksource/humicroedit", humicroedit, "subtask-1"), "ethos": ("tasksource/ethos", ethos, "multilabel")}


def push_card(repo, build):
    card = DatasetCard.load(repo)
    card.data.source_datasets = ORIGINALS[repo]
    sources = ", ".join(f"[{name}](https://huggingface.co/datasets/{name})" for name in ORIGINALS[repo])
    card.text = (f"\n# {repo.split('/')[1]}\n\n{inspect.cleandoc(build.__doc__)}\n\nOriginal data: {sources}. "
                 "Repackaged as parquet for [tasksource](https://github.com/sileod/tasksource) by "
                 "[scripts/upload_repackaged.py](https://github.com/sileod/tasksource/blob/main/scripts/upload_repackaged.py).\n")
    card.push_to_hub(repo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", choices=BUILDERS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--card-only", action="store_true")
    args = parser.parse_args()
    for name in args.names:
        repo, build, *config = BUILDERS[name]
        assert repo in ORIGINALS, f"add {repo} to tasksource/metadata/originals.py"
        if not args.card_only:
            dataset = build()
            print(repo, dataset, dataset["train"][0], sep="\n")
            if args.dry_run:
                continue
            dataset.push_to_hub(repo, config_name=config[0] if config else "default")
        push_card(repo, build)
