"""Graded and multi-attribute sources as native Jev questions over one state.

The choice recast needs named classes, so sources labelled with ratings,
annotator fractions, or vote distributions never reached tasksource-jev.
Each family names a dataset, the fields that form the state, and one typed
question per label column; sibling annotations of one input share a state.
"""

import math
from dataclasses import dataclass

from datasets import DatasetDict, load_dataset

from ..preprocess import fix_splits, sample_dataset
from ..tasks import render_dialogue

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

FAMILIES = {
    "helpsteer": Family("nvidia/HelpSteer", {"Prompt": "prompt", "Response": "response"}, HELPSTEER),
    "helpsteer2": Family("nvidia/HelpSteer2", {"Prompt": "prompt", "Response": "response"}, HELPSTEER),
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
    "acceptability": Family("tasksource/acceptability-prediction", {"Sentence": "text"}, dict(
        acceptability=noul("normalized_score", "How acceptable do native speakers find the sentence, "
                                               "from 0 (unacceptable) to 1 (fully acceptable)?"))),
}


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


def load_family(name, max_rows=None, max_rows_eval=None):
    """Grouped Jev rows per split, split and sampled like any Tasksource source."""
    family = FAMILIES[name]
    dataset = DatasetDict(load_dataset(family.dataset, family.config))
    if family.keep:
        dataset = dataset.filter(family.keep)
    if family.prepare:
        dataset = dataset.map(family.prepare)
    dataset = sample_dataset(fix_splits(dataset), max_rows, max_rows_eval)
    return {split: [row for i, example in enumerate(rows)
                    for row in jev_rows(example, family, SOURCE_PREFIX + name,
                                        "dev" if split == "validation" else split, i)]
            for split, rows in dataset.items() if split in ("train", "validation", "test")}
