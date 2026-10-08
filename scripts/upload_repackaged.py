"""Repackage hard-to-load datasets as data-only mirrors under tasksource/ on the Hub.

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
from pathlib import Path

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


# Fixed files from the Wikimedia/Figshare releases. The older aggression file
# 7383748 contains binary labels only; 7394506 also retains ordinal scores.
WIKIPEDIA_DETOX_FILES = {
    "attack": (7554634, 7554637),
    "aggression": (7038038, 7394506),
    "toxicity": (7394542, 7394539),
}


def _wikipedia_detox_votes(comments, annotations, dimension):
    """Join canonical comments to aggregated human votes, validating the join."""
    import pandas as pd
    comments = comments.copy()
    annotations = annotations.copy()
    for frame in (comments, annotations):
        ids = pd.to_numeric(frame["rev_id"], errors="raise")
        if ids.isna().any() or (ids % 1 != 0).any():
            raise ValueError("Detox rev_id must be an integer")
        frame["rev_id"] = ids.astype("int64")
    if comments.rev_id.duplicated().any():
        raise ValueError("Duplicate Detox comments")
    if set(comments.rev_id) != set(annotations.rev_id):
        raise ValueError("Detox comments and annotations have unmatched rev_id values")
    binary = pd.to_numeric(annotations[dimension], errors="raise")
    if not binary.isin([0, 1]).all():
        raise ValueError("Invalid Detox binary label")
    annotations[dimension] = binary.astype(int)
    levels = range(-3, 4) if dimension == "aggression" else range(-2, 3)
    if dimension != "attack":
        scores = pd.to_numeric(annotations[f"{dimension}_score"], errors="raise")
        if not scores.isin(levels).all():
            raise ValueError("Invalid Detox ordinal score")
        if not ((scores < 0).astype(int) == binary).all():
            raise ValueError("Detox binary label disagrees with ordinal score")
        annotations[f"{dimension}_score"] = scores.astype(int)
    grouped = annotations.groupby("rev_id", sort=False)
    votes = grouped.size().to_frame("annotators")
    binary_counts = pd.crosstab(annotations.rev_id, annotations[dimension]).reindex(columns=[0, 1], fill_value=0)
    votes[f"{dimension}_votes"] = binary_counts.apply(lambda row: row.tolist(), axis=1)
    if dimension != "attack":
        score_counts = pd.crosstab(annotations.rev_id, annotations[f"{dimension}_score"]).reindex(
            columns=levels, fill_value=0)
        votes["score_votes"] = score_counts.apply(lambda row: row.tolist(), axis=1)
    rows = comments[["rev_id", "comment", "sample", "year", "ns", "split"]].merge(
        votes, on="rev_id", how="left", validate="one_to_one").rename(columns={"comment": "text"})
    rows["text"] = rows.text.str.replace("NEWLINE_TOKEN", "\n", regex=False).str.replace("TAB_TOKEN", "\t", regex=False)
    if rows.text.isna().any() or not rows.split.isin(["train", "dev", "test"]).all():
        raise ValueError("Invalid Detox text or canonical split")
    return DatasetDict({split: Dataset.from_pandas(rows[rows.split == source].drop(columns="split"),
                                                 preserve_index=False)
                        for source, split in [("train", "train"), ("dev", "validation"), ("test", "test")]})


def wikipedia_detox_votes():
    """Wikipedia Detox: human vote counts, with attack, aggression and toxicity configs.

    No new annotation. One row per Wikipedia comment. Human annotations are aggregated
    to vote counts after joining *_annotated_comments.tsv and *_annotations.tsv on rev_id.
    Original train/dev/test assignments are retained; dev is renamed validation.
    Binary votes use [not attack/aggression/toxicity, attack/aggression/toxicity] order.
    Aggression score_votes follow [-3, -2, -1, 0, +1, +2, +3]; toxicity score_votes follow
    [-2, -1, 0, +1, +2]. Negative scores mean aggressive/toxic. annotators records the
    actual number of human annotations, rather than assuming ten for every comment.
    Text NEWLINE_TOKEN and TAB_TOKEN placeholders are restored to newlines and tabs.
    rev_id, sample, year and ns are retained as provenance; worker IDs and demographics
    are omitted. Source: [Wikipedia Detox / Wikimedia](https://meta.wikimedia.org/wiki/Research:Detox/Data_Release)
    and Figshare: [attack](https://doi.org/10.6084/m9.figshare.4054689),
    [aggression](https://doi.org/10.6084/m9.figshare.4267550),
    [toxicity](https://doi.org/10.6084/m9.figshare.4563973). Released datasets are CC0.
    """
    import pandas as pd
    configs = {}
    for dimension, file_ids in WIKIPEDIA_DETOX_FILES.items():
        frames = []
        for file_id in file_ids:
            with urllib.request.urlopen(f"https://ndownloader.figshare.com/files/{file_id}") as response:
                frames.append(pd.read_csv(response, sep="\t"))
        configs[dimension] = _wikipedia_detox_votes(*frames, dimension)
    return configs


def webinstruct(bad_examples=None, repairs=None):
    """WebInstruct verified: single-answer multiple choice and explicit yes/no or true/false answers.

    Full preprocessing documentation is in dataset_cards/webinstruct.md.
    """
    from scripts.build_webinstruct import build
    output = Path('build/webinstruct-release')
    counts = build(output, bad_examples, repairs)
    return {config: DatasetDict({split: Dataset.from_parquet(str(output / config / f'{split}.parquet'))
                                for split in ['train', 'test']}) for config in sorted({key.split('/')[0] for key in counts})}


def view2space():
    """VIEW2SPACE's native training MCQs, grouped by ordered source images.

    Source: Pokerme/view2space-train at 1af36aa39a18143db6599a049da6de657e18215c.
    Only explicit multiple-choice questions are included; counting and detection are
    excluded. Original PNG bytes, image order, question IDs, input boxes and reasoning
    are preserved. Reasoning is metadata, never question input. Related QAs share one
    image group to avoid repeating image bytes for every question. The source has only
    a training split; no evaluation split is fabricated. The source README states CC BY 4.0.
    Duplicate distractors are deduplicated with gold indices remapped; ambiguous duplicate
    gold options are excluded and recorded in excluded-questions.jsonl.
    """
    import hashlib
    import zipfile
    import shutil
    from huggingface_hub import hf_hub_download

    repo, revision = 'Pokerme/view2space-train', '1af36aa39a18143db6599a049da6de657e18215c'
    root = Path('build/view2space-original')
    root.mkdir(parents=True, exist_ok=True)
    paths = {name: hf_hub_download(repo, name, repo_type='dataset', revision=revision, local_dir=root)
             for name in ('overall.jsonl', 'images.zip')}
    groups, excluded, deduplicated = {}, [], 0
    with open(paths['overall.jsonl']) as handle:
        for line in handle:
            row = json.loads(line)
            if row['q_type'] != 'mcq':
                continue
            options = row['options']
            if not 2 <= len(options) <= 26 or row['answer'] not in options:
                raise ValueError(f"Invalid source options: {row['q_idx']}")
            gold = options[row['answer']]
            if list(options.values()).count(gold) != 1:
                excluded.append({'id': row['q_idx'], 'reason': 'ambiguous duplicate gold option'})
                continue
            kept = {}
            for key, value in options.items():
                if value not in kept.values():
                    kept[key] = value
            deduplicated += len(kept) != len(options)
            supporting = row.get('supporting') or {}
            groups.setdefault(tuple(row['image_paths']), []).append({
                'inputs': '\n'.join(filter(None, [row['question'], row['question_prompt'],
                    'Image files in order: ' + json.dumps(row['image_paths']) if supporting.get('draw_boxes') else '',
                    'Input boxes: ' + json.dumps(supporting['draw_boxes'], sort_keys=True) if supporting.get('draw_boxes') else ''])),
                'choices_list': list(kept.values()), 'labels': list(kept).index(row['answer']),
                'metadata': json.dumps({'id': row['q_idx'], 'image_paths': row['image_paths'],
                    'source_answer': row['answer'], 'option_keys': list(kept), 'source_options': options,
                    'draw_boxes': supporting.get('draw_boxes'),
                    'reasoning': supporting.get('chain_of_thought'), 'source_revision': revision},
                    ensure_ascii=False, sort_keys=True)})
    output = Path('build/view2space-release-imagefolder')
    output.mkdir(parents=True, exist_ok=True)
    conversion = hashlib.sha256(inspect.getsource(view2space).encode()).hexdigest()
    # Keep each original PNG once. Native ImageFolder supports ZIP + metadata.jsonl.
    packaged = output / 'data.zip'
    if not packaged.exists() or packaged.stat().st_size != Path(paths['images.zip']).stat().st_size:
        shutil.copyfile(paths['images.zip'], packaged)
    with zipfile.ZipFile(packaged, 'a') as archive:
        members = {name[name.index('images/'):]: name for name in archive.namelist()
                   if 'images/' in name and not name.endswith('/')}
        needed = {name for images in groups for name in images}
        # Append one canonical manifest while retaining the original compressed images.
        manifest = zipfile.ZipInfo('metadata.jsonl')  # fixed timestamp for reproducible bytes
        manifest.compress_type = zipfile.ZIP_DEFLATED
        with archive.open(manifest, 'w', force_zip64=True) as handle:
            for images, qas in groups.items():
                handle.write((json.dumps({'file_names': [members[name] for name in images],
                    'image_group_id': hashlib.sha256('\x1f'.join(images).encode()).hexdigest(),
                    'qa': qas}, ensure_ascii=False) + '\n').encode())
    report = {'source': repo, 'revision': revision, 'rows': len(groups), 'images': len(needed),
              'questions': sum(len(qas) for qas in groups.values()), 'license': 'cc-by-4.0',
              'format': 'imagefolder ZIP: ordered images + grouped canonical MC QAs', 'format_version': 1,
              'conversion_sha256': conversion, 'excluded_questions': len(excluded),
              'deduplicated_distractor_rows': deduplicated}
    output.joinpath('provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    output.joinpath('excluded-questions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in excluded))
    # Native imagefolder maps file_names to Sequence(Image); no custom loader.
    output.joinpath('README.md').write_text('---\nlicense: cc-by-4.0\nconfigs:\n'
        '- config_name: default\n  data_files:\n  - split: train\n'
        '    path: data.zip\n---\n')
    return output


BUILDERS = {"view2space": ("tasksource/view2space", view2space),
            "webinstruct": ("tasksource/webinstruct", webinstruct, "per-config"),
            "sharc": ("tasksource/sharc", sharc), "numer_sense": ("tasksource/numer_sense", numer_sense),
            "clutrr": ("tasksource/clutrr", clutrr), "wellformed": ("tasksource/google_wellformed_query", wellformed),
            "humicroedit": ("tasksource/humicroedit", humicroedit, "subtask-1"), "ethos": ("tasksource/ethos", ethos, "multilabel"),
            "multilingual_sentiments": ("tasksource/multilingual-sentiments", multilingual_sentiments),
            "mms": ("tasksource/mms", mms, "per-language"),
            "chaos_mnli_ambiguity": ("tasksource/chaos-mnli-ambiguity", chaos_mnli_ambiguity),
            "measuring_hate_speech_votes": ("tasksource/measuring-hate-speech-votes", measuring_hate_speech_votes),
            "lewidi": ("tasksource/lewidi", lewidi, "per-config"),
            "wikipedia_detox_votes": ("tasksource/wikipedia-detox-votes", wikipedia_detox_votes, "per-config")}


# license metadata for repackaged sets, as the originals state it
LICENSES = {"tasksource/view2space": "cc-by-4.0", "tasksource/measuring-hate-speech-votes": "cc-by-4.0", "tasksource/lewidi": "other",
            "tasksource/wikipedia-detox-votes": "cc0-1.0",
            "tasksource/webinstruct": "apache-2.0"}


def push_card(repo, build):
    card = DatasetCard.load(repo)
    card.data.source_datasets = ORIGINALS[repo]
    if repo in LICENSES:
        card.data.license = LICENSES[repo]
    sources = ", ".join(f"[{name}](https://huggingface.co/datasets/{name})" for name in ORIGINALS[repo])
    sources = f"Original data: {sources}. " if sources else ""  # an empty entry: the original is not on the Hub
    storage = 'native ImageFolder' if repo == 'tasksource/view2space' else 'parquet'
    card.text = (f"\n# {repo.split('/')[1]}\n\n{inspect.cleandoc(build.__doc__)}\n\n{sources}"
                 f"Repackaged as {storage} for [tasksource](https://github.com/sileod/tasksource) by "
                 "[scripts/upload_repackaged.py](https://github.com/sileod/tasksource/blob/main/scripts/upload_repackaged.py).\n")
    if repo == 'tasksource/webinstruct':
        prepared_card = Path('build/webinstruct-release/README.md')
        source_card = prepared_card if prepared_card.exists() else Path(__file__).resolve().parents[1] / 'dataset_cards/webinstruct.md'
        card.text = source_card.read_text().split('---', 2)[2]
    card.push_to_hub(repo)
    if repo == 'tasksource/view2space':
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj='build/view2space-release-imagefolder/provenance.json',
            path_in_repo='provenance.json', repo_id=repo, repo_type='dataset',
            commit_message='Record pinned VIEW2SPACE conversion provenance')
    if repo == 'tasksource/webinstruct':
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj='build/webinstruct-release/provenance.json', path_in_repo='provenance.json',
                           repo_id=repo, repo_type='dataset', commit_message='Record WebInstruct preprocessing provenance')
        provenance = json.loads(Path('build/webinstruct-release/provenance.json').read_text())
        for filename, field in [('bad-examples.jsonl', 'removal_manifest_sha256'), ('repairs.jsonl', 'repairs_sha256')]:
            if field in provenance:
                HfApi().upload_file(path_or_fileobj=Path('build/webinstruct-release') / filename, path_in_repo=filename,
                                   repo_id=repo, repo_type='dataset', commit_message='Record WebInstruct audit decisions')
        if 'audit' in provenance:
            for filename in ['audit-prompt.txt', 'audit-settings.json', 'screen-verdicts.jsonl', 'confirm-verdicts.jsonl']:
                path = Path('build/webinstruct-release') / filename
                if path.exists():
                    HfApi().upload_file(path_or_fileobj=path, path_in_repo=filename, repo_id=repo,
                                       repo_type='dataset', commit_message='Document WebInstruct presentation audit')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", choices=BUILDERS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--card-only", action="store_true")
    parser.add_argument("--bad-examples", type=Path, help="WebInstruct only: confirmed-removal manifest")
    parser.add_argument("--repairs", type=Path, help="WebInstruct only: verified presentation repairs")
    args = parser.parse_args()
    if (args.bad_examples or args.repairs) and args.names != ['webinstruct']:
        parser.error('--bad-examples and --repairs apply only to webinstruct')
    for name in args.names:
        repo, build, *config = BUILDERS[name]
        assert repo in ORIGINALS, f"add {repo} to tasksource/metadata/originals.py"
        if not args.card_only:
            dataset = build(args.bad_examples, args.repairs) if name == 'webinstruct' else build()
            if isinstance(dataset, Path):  # native imagefolder: retain one copy of each image
                print(repo, json.loads((dataset / 'provenance.json').read_text()))
                if args.dry_run:
                    continue
                from huggingface_hub import HfApi
                api = HfApi()
                api.create_repo(repo, repo_type='dataset', exist_ok=True)
                api.upload_large_folder(repo_id=repo, repo_type='dataset', folder_path=dataset,
                    allow_patterns=['data.zip', 'README.md', 'provenance.json', 'excluded-questions.jsonl'],
                    num_workers=8)
                push_card(repo, build)
                continue
            # a builder returns one DatasetDict, or {config: DatasetDict} for per-config repos
            configs = dataset if config in (["per-language"], ["per-config"]) else {config[0] if config else "default": dataset}
            for config_name, splits in configs.items():
                print(repo, config_name, splits, sep="\n")
                if not args.dry_run:
                    splits.push_to_hub(repo, config_name=config_name)
            if args.dry_run:
                continue
        push_card(repo, build)
