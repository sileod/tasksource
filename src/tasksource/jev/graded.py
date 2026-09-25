"""Graded and multi-attribute sources as native Jev questions over one state.

The choice recast needs named classes, so sources labelled with ratings,
annotator fractions, or vote distributions never reached tasksource-jev.
Each family names a dataset, the fields that form the state, and one typed
question per label column; sibling annotations of one input share a state.
"""

import ast
import math
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_dataset

from ..preprocess import fix_splits, sample_dataset
from ..tasks import render_dialogue, render_helpsteer_prompt

SOURCE_PREFIX = "graded/"


@dataclass
class Question:
    kind: str
    column: str
    instructions: str
    criteria: list = None
    low: float = 0.0
    high: float = 1.0
    step: float = 1.0

    def target(self, value):
        """The Jev target, or None for a missing or hidden (out-of-range) label."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if isinstance(value, (list, tuple)):  # annotator vote counts
            return [v / sum(value) for v in value] if sum(value) else None
        if self.kind == "noul":
            return [(value - self.low) / (self.high - self.low)] if self.low <= value <= self.high else None
        index = round((value - self.low) / self.step)
        return [float(i == index) for i in range(len(self.criteria))] if 0 <= index < len(self.criteria) else None


def score(column, instructions, criteria, low=0, step=1):
    return Question("score", column, instructions, criteria, low=low, step=step)


def noul(column, instructions, low=0, high=1):
    return Question("noul", column, instructions, low=low, high=high)


def choice(column, instructions, criteria):
    return Question("choice", column, instructions, criteria)


def likert(low, high, n=5, start=0, step=1):
    levels = [f"{start + i * step:g}" for i in range(n)]
    return [f"{levels[0]}: {low}", *levels[1:-1], f"{levels[-1]}: {high}"]


@dataclass
class Family:
    dataset: str
    state: dict  # displayed field name -> column
    questions: dict
    config: str = None
    keep: object = None  # row filter applied before splitting
    prepare: object = None  # row map adding the state columns
    dedupe: str = None  # column whose repeats are dropped (one row per annotator upstream)
    load_kwargs: dict = None  # e.g. the Hub parquet export of a script-only dataset
    pre_process: object = None  # DatasetDict -> DatasetDict, e.g. one row per item from per-annotator rows


HELPSTEER = dict(
    helpfulness=score("helpfulness", "Rate the overall helpfulness of the response to the prompt.",
                      likert("not helpful", "extremely helpful")),
    correctness=score("correctness", "Rate whether the response includes all pertinent facts without errors.",
                      likert("mostly incorrect", "fully correct and complete")),
    coherence=score("coherence", "Rate the consistency and clarity of expression of the response.",
                    likert("incoherent", "perfectly clear")),
    complexity=score("complexity", "Rate the intellectual depth required to write the response.",
                     likert("basic competency", "deep domain expertise")),
    verbosity=score("verbosity", "Rate the amount of detail relative to what the prompt asks for.",
                    likert("very terse", "very verbose")),
)
CIVIL = dict(toxicity="toxic", severe_toxicity="severely toxic", obscene="obscene", threat="threatening",
             insult="insulting", identity_attack="an identity attack", sexual_explicit="sexually explicit")
OASST_RATINGS = ["quality", "helpfulness", "creativity", "humor", "toxicity", "violence"]
OASST_FLAGS = dict(spam="spam", fails_task="failing the task", not_appropriate="inappropriate",
                   hate_speech="hate speech", sexual_content="sexual content",
                   pii="revealing personal information", lang_mismatch="in the wrong language")
ESSAY = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
PAIR = {"Sentence 1": "sentence1", "Sentence 2": "sentence2"}
SIMILARITY = "How similar in meaning are the sentences, from 0 (unrelated) to 1 (equivalent)?"
AITA = dict(AUTHOR="the author is in the wrong", OTHER="the other party is in the wrong",
            EVERYBODY="everyone is in the wrong", NOBODY="no one is in the wrong", INFO="more information is needed")
HATE_VOTES = ["hate_speech_count", "offensive_language_count", "neither_count"]
# RuyuanWan disagreement sets: what the annotators judged
DISAGREEMENT = dict(Dynasent="the sentiment of the text", Politeness="the politeness of the text",
                    SBIC="whether the text is offensive", SChem="whether the rule of thumb is acceptable",
                    Dilemmas="which of the two actions is less ethical")

PARQUET_EXPORT = "refs/convert/parquet"
PREMISE = {"Premise": "premise", "Hypothesis": "hypothesis"}
AGREEMENT = ["strongly disagree", "disagree", "neither disagree nor agree", "agree", "strongly agree"]
TARGETED = "the group(s) it targets"
# Measuring Hate Speech survey items (Sachdeva et al., 2022, table 1); vote counts in code order, where a
# higher code is more hateful (checked against the dataset's hate_speech_score)
MHS = dict(
    sentiment=("How would you describe the sentiment of this comment?",
               ["strongly positive", "somewhat positive", "neutral", "somewhat negative", "strongly negative"]),
    respect=(f"Is this comment respectful towards {TARGETED}?",
             ["strongly respectful", "respectful", "neutral", "disrespectful", "strongly disrespectful"]),
    insult=(f"Do you agree that this comment is insulting towards {TARGETED}?", AGREEMENT),
    humiliate=(f"Do you agree that this comment is humiliating towards {TARGETED}?", AGREEMENT),
    status=(f"This comment states that {TARGETED} is:",
            ["strongly superior", "superior", "neither superior nor inferior", "inferior", "strongly inferior"]),
    dehumanize=(f"Do you agree that this comment dehumanizes {TARGETED} (e.g., by comparing them to an animal)?",
                AGREEMENT),
    violence=(f"Do you agree that this comment calls for using violence against {TARGETED}?", AGREEMENT),
    genocide=(f"Do you agree that this comment calls for the deliberate killing of a large group of people "
              f"from {TARGETED}?", AGREEMENT),
    attack_defend=(f"Is this comment attacking or defending {TARGETED}?",
                   ["strongly defending", "defending", "neither defending nor attacking", "attacking",
                    "strongly attacking"]),
)
HATE_SPEECH_DEFINITION = ("Does this comment contain hate speech, defined as bias-motivated, hostile and malicious "
                          "language targeted at a person or group because of their actual or perceived innate "
                          "characteristics?")
DYNASENT = ["positive", "negative", "neutral", "mixed"]


def _dynasent_votes(example):
    annotators = example["label_distribution"]  # label -> annotator ids, a dict or its repr
    annotators = ast.literal_eval(annotators) if isinstance(annotators, str) else annotators
    return {"votes": [len(annotators.get(label, [])) for label in DYNASENT]}


def _beavertails_votes(dataset):
    """One row per (prompt, response) with the share of its three annotators who judged it unsafe."""
    def votes(rows):
        frame = rows.to_pandas()
        frame["unsafe_share"] = frame.groupby(["prompt", "response"]).is_safe.transform(lambda safe: (~safe).sum() / len(safe))
        return Dataset.from_pandas(frame.drop_duplicates(["prompt", "response"]), preserve_index=False)
    return DatasetDict(train=votes(dataset["330k_train"]), test=votes(dataset["330k_test"]))


def _share_of_yes(example):
    return {"share": example["soft_label"][1]}


def _conv_abuse(example):
    turns = [("Agent", "prev_agent"), ("User", "prev_user"), ("Agent", "agent"), ("User", "user")]
    # the source writes "_" for turns before the conversation started
    return {**_share_of_yes(example), "conversation": "\n".join(
        f"{who}: {example[column]}" for who, column in turns if example[column].strip() not in ("", "_"))}


def _lewidi_noul(config, state, question, prepare=_share_of_yes):
    return Family("tasksource/lewidi", state, dict(label=noul("share", question)), config=config, prepare=prepare)


FAMILIES = {
    "helpsteer": Family("nvidia/HelpSteer", {"Prompt": "prompt", "Response": "response"}, HELPSTEER),
    "helpsteer2": Family("nvidia/HelpSteer2", {"Prompt": "prompt", "Response": "response"}, HELPSTEER,
                         prepare=lambda x: {"prompt": render_helpsteer_prompt(x["prompt"])}),
    "helpsteer3": Family("nvidia/HelpSteer3", {"Conversation": "dialogue", "Response 1": "response1",
                                               "Response 2": "response2"}, dict(
        preference=score("overall_preference", "Which response is the better next assistant reply, and by how much?",
                         ["-3: Response 1 is much better", "-2: Response 1 is better", "-1: Response 1 is slightly better",
                          "0: About the same", "1: Response 2 is slightly better", "2: Response 2 is better",
                          "3: Response 2 is much better"], low=-3)),
        config="preference", prepare=lambda x: {"dialogue": render_dialogue(x["context"])}),
    "oasst2": Family("tasksource/oasst2_dense_flat", {"Prompt": "parent_text", "Response": "text"}, {
        **{c: noul(c, f"Mean reviewer rating of the response's {c}, from 0 (lowest) to 1 (highest).")
           for c in OASST_RATINGS},
        **{c: noul(c, f"What fraction of reviewers flagged the response as {label}?")
           for c, label in OASST_FLAGS.items()},
    }, keep=lambda x: x["lang"] == "en" and x["role"] == "assistant"),
    "civil_comments": Family("google/civil_comments", {"Comment": "text"}, {
        c: noul(c, f"What fraction of annotators rated the comment as {label}?") for c, label in CIVIL.items()}),
    "english_grading": Family("tasksource/english-grading", {"Essay": "full_text"}, {
        c: score(c, f"Grade the {c} of this English-learner essay.",
                 likert("very weak", "native-like", n=9, start=1, step=0.5), low=1, step=0.5)
        for c in ESSAY}),
    "essay_scoring": Family("tasksource/AES2-essay-scoring", {"Essay": "full_text"}, dict(
        score=score("score", "Give the essay's holistic score.", likert("very weak", "excellent", n=6, start=1), low=1))),
    "app_reviews": Family("app_reviews", {"Review": "review"}, dict(
        stars=score("star", "How many stars did the reviewer give the app?", likert("worst", "best", start=1), low=1))),
    "joci": Family("pietrolesci/joci", {"Context": "context", "Hypothesis": "hypothesis"}, dict(
        likelihood=score("original_label", "How likely is the hypothesis, given the context?",
                         ["impossible", "technically possible", "plausible", "likely", "very likely"], low=1))),
    "stsb": Family("nyu-mll/glue", PAIR, dict(similarity=noul("label", SIMILARITY, high=5)), config="stsb"),
    "sts_companion": Family("tasksource/sts-companion", PAIR, dict(similarity=noul("label", SIMILARITY, high=5))),
    "sick": Family("tasksource/sick", {"Sentence 1": "sentence_A", "Sentence 2": "sentence_B"}, dict(
        relatedness=noul("relatedness_score", "How related in meaning are the sentences, "
                                              "from 0 (unrelated) to 1 (closely related)?", low=1, high=5))),
    "chaos_mnli": Family("tasksource/chaos-mnli-ambiguity", {"Premise": "premise", "Hypothesis": "hypothesis"}, dict(
        relation=choice("label_count", "How would annotators label the relation of the hypothesis to the premise?",
                        ["entailment", "neutral", "contradiction"]))),
    "hate_speech_offensive": Family("tdavidson/hate_speech_offensive", {"Tweet": "tweet"}, dict(
        votes=choice("votes", "How would annotators classify the tweet?",
                     ["hate speech", "offensive but not hate speech", "neither"])),
        prepare=lambda x: {"votes": [x[c] for c in HATE_VOTES]}),
    "scruples": Family("tasksource/scruples", {"Title": "title", "Post": "text"}, dict(
        verdict=choice("votes", "How would Reddit readers judge who is in the wrong in this story?", list(AITA.values()))),
        prepare=lambda x: {"votes": [x["label_scores"][k] for k in AITA]}),
    **{f"{name.lower()}_disagreement": Family(f"RuyuanWan/{name}_Disagreement", {"Text": "text"}, dict(
        disagreement=noul("disagreement_rate", f"How much would annotators disagree about {topic}, "
                                               "from 0 (all agree) to 1 (maximal disagreement)?")),
        dedupe="text") for name, topic in DISAGREEMENT.items()},
    "beavertails": Family("PKU-Alignment/BeaverTails", {"Prompt": "prompt", "Response": "response"}, dict(
        unsafe=noul("unsafe_share", "What fraction of annotators judged the assistant response unsafe?")),
        pre_process=_beavertails_votes),
    "unli": Family("Zhengping/UNLI", PREMISE, dict(
        probability=noul("label", "How likely is the hypothesis to be true, given the premise?"))),
    **{f"dynasent_{r}": Family("dynabench/dynasent", {"Sentence": "sentence"}, dict(
        sentiment=choice("votes", "How would annotators label the sentiment of the sentence?", DYNASENT)),
        prepare=_dynasent_votes, load_kwargs=dict(revision=PARQUET_EXPORT, data_dir=f"dynabench.dynasent.{r}.all"))
       for r in ("r1", "r2")},
    "hatexplain": Family("Hate-speech-CNERG/hatexplain", {"Post": "post"}, dict(
        label=choice("votes", "How would annotators classify the post?", ["hate speech", "normal", "offensive"])),
        prepare=lambda x: {"post": " ".join(x["post_tokens"]),
                           "votes": [x["annotators"]["label"].count(label) for label in range(3)]},
        load_kwargs=dict(revision=PARQUET_EXPORT, data_dir="plain_text")),
    "measuring_hate_speech": Family("tasksource/measuring-hate-speech-votes", {"Comment": "text"}, {
        **{item: score(item, question, levels) for item, (question, levels) in MHS.items()},
        "hatespeech": choice("hatespeech", HATE_SPEECH_DEFINITION, ["no", "unclear", "yes"])}),
    "lewidi_md_agreement": _lewidi_noul("md_agreement", {"Tweet": "text"},
                                        "What fraction of annotators find the tweet offensive?"),
    "lewidi_hs_brexit": _lewidi_noul("hs_brexit", {"Tweet": "text"},
                                     "What fraction of annotators consider the tweet hate speech?"),
    "lewidi_armis": _lewidi_noul("armis", {"Tweet": "text"},
                                 "What fraction of annotators consider the tweet misogynistic or sexist?"),
    "lewidi_conv_abuse": _lewidi_noul("conv_abuse", {"Conversation": "conversation"},
                                      "What fraction of annotators consider the user's last message abusive?",
                                      prepare=_conv_abuse),
    "lewidi_mp": _lewidi_noul("mp", {"Post": "post", "Reply": "reply"},
                              "What fraction of annotators consider the reply ironic?"),
    "lewidi_csc": Family("tasksource/lewidi", {"Context": "context", "Response": "response"}, dict(
        sarcasm=score("soft_label", "How sarcastic is the response, given the context?",
                      ["1: not sarcastic", "2", "3", "4", "5", "6: very sarcastic"])), config="csc"),
    "lewidi_varierrnli": Family("tasksource/lewidi", {"Context": "context", "Statement": "statement"}, {
        relation: noul(relation, f'What fraction of annotators accept "{relation}" as a label for the '
                                 f"statement, given the context?")
        for relation in ("entailment", "neutral", "contradiction")}, config="varierrnli"),
    "acceptability": Family("tasksource/acceptability-prediction", {"Sentence": "text"}, dict(
        acceptability=noul("normalized_score", "How acceptable do native speakers find the sentence, "
                                               "from 0 (unacceptable) to 1 (fully acceptable)?"))),
}

# tasksource task ids (substrings) that these families replace in the Jev build. Regression tasks
# (stsb, sick relatedness, oasst2 ...) are not listed: recast.improper_labels already excludes them.
COVERED_TASKS = ("HelpSteer/", "HelpSteer2/", "HelpSteer3/preference", "civil_comments/",
                 "english-grading/", "AES2-essay-scoring", "app_reviews", "joci",
                 "hate_speech_offensive", "scruples", "_Disagreement")


def jev_rows(example, family, source, split, index):
    """One source row as grouped decisions; unlabeled questions are skipped."""
    state = "\n\n".join(f"{name}:\n{example[column]}" for name, column in family.state.items())
    rows = []
    for qid, question in family.questions.items():
        target = question.target(example[question.column])
        if target is not None:
            rows.append({
                "id": f"{source.replace('/', '-')}:{split}:{index}:{qid}",
                "kind": question.kind,
                "options": list(question.criteria or []),
                "target": target,
                "state": state,
                "question": question.instructions,
                "source": source,
                "variant": "direct",
                "split": split,
            })
    return rows


def load_family(name, max_rows=None, max_rows_eval=None, revision=None, data_file_pins=None):
    """Grouped Jev rows per split, split and sampled like any Tasksource source.

    ``revision`` pins the source commit, overriding the family's requested ref;
    ``data_file_pins`` ({repo: commit}) pins ``hf://`` data files."""
    from ..access import pin_hf_urls
    family = FAMILIES[name]
    kwargs = {**pin_hf_urls(family.load_kwargs or {}, data_file_pins or {}),
              **({"revision": revision} if revision else {})}
    dataset = DatasetDict(load_dataset(family.dataset, family.config, **kwargs))
    if family.pre_process:
        dataset = family.pre_process(dataset)
    if family.dedupe:
        dataset = DatasetDict({split: rows.select(
            rows.to_pandas().drop_duplicates(family.dedupe).index.tolist()) for split, rows in dataset.items()})
    if family.keep:
        dataset = dataset.filter(family.keep)
    if family.prepare:
        dataset = dataset.map(family.prepare)
    dataset = sample_dataset(fix_splits(dataset), max_rows, max_rows_eval)
    return {split: [row for i, example in enumerate(rows)
                    for row in jev_rows(example, family, SOURCE_PREFIX + name,
                                        "dev" if split == "validation" else split, i)]
            for split, rows in dataset.items() if split in ("train", "validation", "test")}
