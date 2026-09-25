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
MULTILINGUAL_SENTIMENTS_URL = "https://raw.githubusercontent.com/tyqiangz/multilingual-sentiment-datasets/main/data/all/{}.csv"
LEWIDI_URL = "https://raw.githubusercontent.com/Le-Wi-Di/le-wi-di.github.io/546eaab420adab3c658813ae3fc7ae9f73cf8f05/{}"
CHAOS_MNLI_URL = "hf://datasets/tasksource/chaos-mnli-ambiguity/chaos_mnli.jsonl"
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


def multilingual_sentiments():
    """Multilingual sentiments (tyqiangz): 3-way sentiment in 12 languages, merged from public corpora (`source`)."""
    splits = {}
    for source, split in [("train", "train"), ("valid", "validation"), ("test", "test")]:
        rows = load_dataset("csv", data_files=MULTILINGUAL_SENTIMENTS_URL.format(source))["train"]
        splits[split] = rows.cast_column("label", ClassLabel(names=["negative", "neutral", "positive"]))
    return DatasetDict(splits)


def mms():
    """MMS (Brand24): 79 sentiment datasets in 27 languages, one config per language.

    Labels are the source's -1/0/1 shifted to negative/neutral/positive; `original_dataset`, `domain` and the
    cleanlab self-confidence are kept. MMS ships a single split, so rows are split 90/5/5 by a hash of the text.
    The bundled datasets keep their own licenses; check each `original_dataset` before use.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files
    template = open(hf_hub_download("Brand24/mms", "_template.py", repo_type="dataset")).read()
    domains = dict(re.findall(r"'(\w+)': \"(\w+)_DOMAIN\"", template))
    files = sorted(f for f in list_repo_files("Brand24/mms", repo_type="dataset") if f.startswith("data/") and f.endswith(".tsv"))
    frames = {}
    for path in files:
        language, name = path.split("/")[1], path.split("/")[2][:-4]
        rows = pd.read_csv(hf_hub_download("Brand24/mms", path, repo_type="dataset"), sep="\t",
                           names=["label", "text", "cleanlab_self_confidence"])
        rows = rows[rows.label.isin([-1, 0, 1]) & rows.text.notna()]
        rows = rows.assign(label=rows.label + 1, original_dataset=name, domain=domains[name].lower(), language=language)
        frames.setdefault(language, []).append(rows)
    configs = {}
    for language, parts in frames.items():
        rows = pd.concat(parts, ignore_index=True)
        split = rows.text.map(lambda text: _split_of(str(text), validation=0.05, test=0.05))
        configs[language] = DatasetDict({
            name: Dataset.from_pandas(rows[split == name], preserve_index=False)
            .cast_column("label", ClassLabel(names=["negative", "neutral", "positive"]))
            for name in ("train", "validation", "test")})
    return configs


def _gini(shares):
    # Gini coefficient: mean absolute difference between shares over twice their mean
    return sum(abs(a - b) for a in shares for b in shares) / (2 * len(shares) * sum(shares))


def chaos_mnli_ambiguity():
    """ChaosNLI, MNLI portion: 1,599 MNLI pairs relabeled by 100 annotators each (Nie et al., 2020).

    `label_dist` and `label_count` follow the entailment/neutral/contradiction order, and `gini` is the Gini
    coefficient of `label_dist` (0 = annotators evenly split, 1 = unanimous). Built from the jsonl first uploaded
    here, which flattens the ChaosNLI release (https://github.com/easonnie/ChaosNLI) and adds `gini`; the
    variable-key `label_counter` (a duplicate of `label_count`) is dropped so the rows have a fixed schema.

    ```
    @inproceedings{nie2020chaosnli,
        title = {What Can We Learn from Collective Human Opinions on Natural Language Inference Data?},
        author = {Nie, Yixin and Zhou, Xiang and Bansal, Mohit},
        booktitle = {Proceedings of EMNLP},
        year = {2020}
    }
    @inproceedings{xzhou2022distnli,
        title = {Distributed NLI: Learning to Predict Human Opinion Distributions for Language Reasoning},
        author = {Zhou, Xiang and Nie, Yixin and Bansal, Mohit},
        booktitle = {Findings of the Association for Computational Linguistics: ACL 2022},
        year = {2022}
    }
    ```
    """
    rows = load_dataset("json", data_files=CHAOS_MNLI_URL, features=None)["train"].remove_columns("label_counter")
    for row in rows:
        assert abs(_gini(row["label_dist"]) - row["gini"]) < 1e-6, row["uid"]
    return DatasetDict(train=rows)


MHS_ITEMS = dict(sentiment=5, respect=5, insult=5, humiliate=5, status=5, dehumanize=5,
                 violence=5, genocide=5, attack_defend=5, hatespeech=3)


def measuring_hate_speech_votes():
    """Measuring Hate Speech (Kennedy et al., 2020; Sachdeva et al., 2022), one row per comment with vote counts.

    The source has one row per (comment, annotator). Each survey item becomes a list of vote counts over its
    ordinal codes, in code order: 0-4 for the nine Likert items, where a higher code is more hateful
    (sentiment: strongly positive to strongly negative; respect: strongly respectful to strongly disrespectful;
    insult, humiliate, dehumanize, violence, genocide: strongly disagree to strongly agree; status: strongly
    superior to strongly inferior; attack_defend: strongly defending to strongly attacking), and 0-2 for
    hatespeech (no, unclear, yes). Survey wording: Sachdeva et al. (2022), appendix table 1. Comments with fewer
    than three annotators are dropped; rows are split 90/5/5 by a hash of the comment id. License: CC BY 4.0,
    as the original.
    """
    import pandas as pd
    rows = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train").to_pandas()
    for item in MHS_ITEMS:
        rows[item] = pd.to_numeric(rows[item]).astype(int)
    grouped = rows.groupby("comment_id")
    comments = grouped.agg(text=("text", "first"), platform=("platform", "first"),
                           annotators=("annotator_id", "size"), hate_speech_score=("hate_speech_score", "first"))
    for item, levels in MHS_ITEMS.items():
        comments[item] = grouped[item].agg(lambda codes, n=levels: [int((codes == level).sum()) for level in range(n)])
    comments = comments[comments.annotators >= 3].reset_index()
    comments["hate_speech_score"] = pd.to_numeric(comments.hate_speech_score)
    split = comments.comment_id.map(lambda key: _split_of(str(key), validation=0.05, test=0.05))
    return DatasetDict({name: Dataset.from_pandas(comments[split == name], preserve_index=False)
                        for name in ("train", "validation", "test")})


# Learning with Disagreements: (edition folder, file prefix, text fields, label order)
LEWIDI = {
    "md_agreement": ("LeWiDi_2-2023/MD-Agreement_dataset", "MD-Agreement", ["text"], ["0", "1"]),
    "hs_brexit": ("LeWiDi_2-2023/HS-Brexit_dataset", "HS-Brexit", ["text"], ["0", "1"]),
    "armis": ("LeWiDi_2-2023/ArMIS_dataset", "ArMIS", ["text"], ["0", "1"]),
    "conv_abuse": ("LeWiDi_2-2023/ConvAbuse_dataset", "ConvAbuse", ["prev_agent", "prev_user", "agent", "user"], ["0", "1"]),
    "csc": ("LeWiDi_3-2025/CSC", "CSC", ["context", "response"], ["1", "2", "3", "4", "5", "6"]),
    "mp": ("LeWiDi_3-2025/MP", "MP", ["post", "reply"], ["0", "1"]),
    "varierrnli": ("LeWiDi_3-2025/VariErrNLI", "VariErrNLI", ["context", "statement"], None),
}


def _lewidi_text(text):
    if isinstance(text, dict):
        return text
    try:  # ConvAbuse stores its dialogue as a JSON string
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"text": text}
    except (TypeError, ValueError):
        return {"text": text}


def lewidi():
    """Learning with Disagreements (LeWiDi, SemEval-2023 Task 11 and its 2025 edition): soft labels from every annotator.

    One config per dataset: md_agreement (offensiveness, 5 annotators), hs_brexit (hate speech, 6), armis
    (Arabic misogyny and sexism, 3), conv_abuse (abuse in user turns of chatbot dialogues, 3 or more), csc
    (sarcasm rated 1-6), mp (MultiPICo irony, multilingual) and varierrnli (NLI where each annotator may accept
    several labels). ``soft_label`` lists the share of annotators per label in ``labels`` order; varierrnli
    instead gives, per NLI label, the share of annotators who accepted it. Text fields are unpacked from the
    harmonized JSON. The Paraphrase set (an 11-level scale over 500 items) is left out. The original datasets'
    licenses and terms apply unchanged; see the LeWiDi repository (https://github.com/Le-Wi-Di/le-wi-di.github.io)
    and each dataset's paper.
    """
    configs = {}
    for name, (folder, prefix, fields, labels) in LEWIDI.items():
        splits = {}
        for source, split in [("train", "train"), ("dev", "validation"), ("test", "test")]:
            with urllib.request.urlopen(LEWIDI_URL.format(f"{folder}/{prefix}_{source}.json")) as response:
                items = json.load(response)
            rows = []
            for item_id, item in items.items():
                text = _lewidi_text(item["text"])
                row = {"id": str(item_id), **{field: str(text[field]) for field in fields},
                       "lang": item.get("lang", ""), "annotations": item.get("number of annotations")}
                soft = item["soft_label"]
                if labels is None:
                    for relation in ("entailment", "neutral", "contradiction"):
                        row[relation] = float(soft[relation]["1"])
                else:
                    # MP writes its labels as "0.0"/"1.0" in train
                    soft = {str(int(float(key))): value for key, value in soft.items()}
                    shares = [float(soft[label]) for label in labels]
                    row["soft_label"] = [share / sum(shares) for share in shares]  # CSC test shares are rounded
                rows.append(row)
            splits[split] = Dataset.from_list(rows)
        configs[name] = DatasetDict(splits)
    return configs


BUILDERS = {"sharc": ("tasksource/sharc", sharc), "numer_sense": ("tasksource/numer_sense", numer_sense),
            "clutrr": ("tasksource/clutrr", clutrr), "wellformed": ("tasksource/google_wellformed_query", wellformed),
            "humicroedit": ("tasksource/humicroedit", humicroedit, "subtask-1"), "ethos": ("tasksource/ethos", ethos, "multilabel"),
            "multilingual_sentiments": ("tasksource/multilingual-sentiments", multilingual_sentiments),
            "mms": ("tasksource/mms", mms, "per-language"),
            "chaos_mnli_ambiguity": ("tasksource/chaos-mnli-ambiguity", chaos_mnli_ambiguity),
            "measuring_hate_speech_votes": ("tasksource/measuring-hate-speech-votes", measuring_hate_speech_votes),
            "lewidi": ("tasksource/lewidi", lewidi, "per-config")}


# license metadata for repackaged sets, as the originals state it
LICENSES = {"tasksource/measuring-hate-speech-votes": "cc-by-4.0", "tasksource/lewidi": "other"}


def push_card(repo, build):
    card = DatasetCard.load(repo)
    card.data.source_datasets = ORIGINALS[repo]
    if repo in LICENSES:
        card.data.license = LICENSES[repo]
    sources = ", ".join(f"[{name}](https://huggingface.co/datasets/{name})" for name in ORIGINALS[repo])
    sources = f"Original data: {sources}. " if sources else ""  # an empty entry: the original is not on the Hub
    card.text = (f"\n# {repo.split('/')[1]}\n\n{inspect.cleandoc(build.__doc__)}\n\n{sources}"
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
            # a builder returns one DatasetDict, or {config: DatasetDict} for per-config repos
            configs = dataset if config in (["per-language"], ["per-config"]) else {config[0] if config else "default": dataset}
            for config_name, splits in configs.items():
                print(repo, config_name, splits, splits["train"][0], sep="\n")
                if not args.dry_run:
                    splits.push_to_hub(repo, config_name=config_name)
            if args.dry_run:
                continue
        push_card(repo, build)
