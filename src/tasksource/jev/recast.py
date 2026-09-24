"""Canonical Tasksource-to-Jev recasts and typed-decision (System One) request rendering."""

import html
import re
from collections import OrderedDict

import ftfy
from datasets import ClassLabel, DatasetDict, List, Sequence

from .augmentations import stable_fraction
from .options import permute_choices
from .token_labels import (
    MAX_JEV_TOKENS_PER_SEQUENCE,
    normalize_token_label,
    readable_token_labels,
)


JEV_CLASSIFICATION_INSTRUCTIONS = "Choose the criterion that best describes the state."
JEV_MULTIPLE_CHOICE_INSTRUCTIONS = "Choose the criterion that best answers the question."


_LINE_BREAK = re.compile(r"</?br\s*/?>", re.IGNORECASE)
_ENTITY = re.compile(r"&(?:amp|lt|gt|quot|apos|#39|#x27);")


def clean_text(text):
    """Undo encoding damage left in source text: mojibake (``â€™``, ``Ã³``),
    HTML escaping in tweets, and ``<br>`` breaks in WikiHow contexts. Other markup stays."""
    if not isinstance(text, str):
        return text
    text = _LINE_BREAK.sub("\n", ftfy.fix_encoding(text))
    for _ in range(2):  # tweets are sometimes escaped twice (&amp;amp;)
        text = _ENTITY.sub(lambda m: html.unescape(m.group(0)), text)
    return text


def _strip(text):
    return text.strip() if isinstance(text, str) else text


def _choice_columns(features):
    """Return choice columns in numeric order (choice2 before choice10)."""
    choices = [name for name in features if name.startswith("choice")]

    def key(name):
        suffix = name[len("choice"):]
        return (0, int(suffix)) if suffix.isdigit() else (1, name)

    return sorted(choices, key=key)


def _valid_criteria(criteria):
    """Jev criteria are runtime names: present, non-blank, and unique."""
    texts = [str(c).strip() for c in criteria if c is not None]
    return len(texts) == len(criteria) and all(texts) and len(set(texts)) == len(texts)


def _jev_task_type(features):
    if "sentence1" in features:
        return "Classification"
    if _choice_columns(features) and "inputs" in features:
        return "MultipleChoice"
    if "tokens" in features and "labels" in features:
        labels = features["labels"]
        if not isinstance(labels, (Sequence, List)) or not isinstance(labels.feature, ClassLabel):
            raise NotImplementedError(
                "TokenClassification labels must be Sequence(ClassLabel) or List(ClassLabel) for JEV"
            )
        if not readable_token_labels(labels.feature.names):
            raise NotImplementedError(
                "TokenClassification labels are not semantically readable for JEV"
            )
        return "TokenClassification"
    raise NotImplementedError(
        "Jev recasting currently supports Classification, MultipleChoice, and "
        "readable TokenClassification tasks"
    )


def _token_question_indices(tokens, labels, names, identifier):
    """Select at most two deterministic token judgments from one sequence."""
    valid = [index for index in range(min(len(tokens), len(labels))) if 0 <= int(labels[index]) < len(names)]
    if not valid:
        return []
    default = names.index("O") if "O" in names else None
    non_default = [index for index in valid if default is None or int(labels[index]) != default]
    ranked_non_default = sorted(
        non_default,
        key=lambda index: stable_fraction(f"{identifier}:{index}", "token-primary"),
    )
    selected = ranked_non_default[:1]
    remaining = [index for index in valid if index not in selected]
    remaining.sort(
        key=lambda index: stable_fraction(f"{identifier}:{index}", "token-secondary")
    )
    selected.extend(remaining[: MAX_JEV_TOKENS_PER_SEQUENCE - len(selected)])
    return selected


def _token_state(tokens, target_index):
    marked = list(tokens)
    marked[target_index] = f"[TARGET: {tokens[target_index]}]"
    sentence = " ".join(tokens)
    return (
        f"Sentence: {sentence}\n"
        f"Target token at position {target_index}: {tokens[target_index]}\n"
        f"Marked sentence: {' '.join(marked)}"
    )


def recast_jev(dataset, task=None):
    """Recast a standardized Tasksource dataset as runtime-defined choices.

    The output is model- and wire-format-independent. ``criteria`` contains
    the runtime choices, ``label`` their zero-based index, and ``answer``
    the matching criterion. Multiple-choice criteria are permuted
    deterministically per source row so the gold slot carries no signal;
    augmentation is otherwise deliberately separate.
    """
    if not isinstance(dataset, DatasetDict):
        raise TypeError("recast_jev expects a datasets.DatasetDict")
    if "train" not in dataset:
        raise ValueError("recast_jev expects a train split")

    features = dataset["train"].features
    task_type = _jev_task_type(features)
    labels = features.get("labels")

    if task_type == "Classification":
        if not hasattr(labels, "names"):
            raise TypeError("Classification labels must use datasets.ClassLabel")
        criteria = list(labels.names)
        if not _valid_criteria(criteria):
            raise ValueError(
                f"Classification label names must be present and unique: {criteria}"
            )

        def convert(example):
            state = clean_text(example["sentence1"])
            if "sentence2" in example:
                state = f"text_A: {state}\ntext_B: {clean_text(example['sentence2'])}"
            label = int(example["labels"])
            return {
                "state": state,
                "instructions": JEV_CLASSIFICATION_INSTRUCTIONS,
                "criteria": criteria,
                "label": label,
                "answer": criteria[label],
                "task": task or "",
            }

    elif task_type == "MultipleChoice":
        choices = _choice_columns(features)

        def convert(example, index, split):
            # Jev preprocessing pads variable-length choice lists with None.
            present = [name for name in choices if example[name] is not None]
            label = int(example["labels"])
            if 0 <= label < len(choices) and example[choices[label]] is None:
                present = choices  # missing gold: keep None so the row is filtered
            if 0 <= label < len(choices):
                label = present.index(choices[label])
            criteria, label = permute_choices(
                [_strip(clean_text(example[name])) for name in present], label,
                f"{task or ''}:{split}:{index}",
            )
            return {
                "state": clean_text(example["inputs"]),
                "instructions": JEV_MULTIPLE_CHOICE_INSTRUCTIONS,
                "criteria": criteria,
                "label": label,
                "answer": criteria[label],
                "task": task or "",
            }

        converted = DatasetDict({
            split: rows.map(convert, with_indices=True, fn_kwargs={"split": split})
            .filter(lambda row: _valid_criteria(row["criteria"]))
            for split, rows in dataset.items()
        })

    else:
        raw_names = list(labels.feature.names)
        criteria = [normalize_token_label(name) for name in raw_names]
        if len(criteria) != len(set(criteria)):
            raise NotImplementedError(
                "TokenClassification labels collapse to duplicate readable JEV criteria"
            )

        def convert_batch(batch, indices):
            output = {
                "state": [], "instructions": [], "criteria": [], "label": [],
                "answer": [], "task": [], "shared_state": [],
                "question_id": [], "source_row": [], "target_index": [],
                "target_token": [],
            }
            for tokens, token_labels, source_index in zip(
                batch["tokens"], batch["labels"], indices
            ):
                if len(tokens) != len(token_labels):
                    raise ValueError(
                        f"Token and label lengths differ at source row {source_index}"
                    )
                identifier = f"{task or 'token-task'}:{source_index}"
                for token_index in _token_question_indices(
                    tokens, token_labels, raw_names, identifier
                ):
                    label = int(token_labels[token_index])
                    output["state"].append(_token_state(tokens, token_index))
                    output["instructions"].append(
                        "Choose the criterion that best labels the target token."
                    )
                    output["criteria"].append(criteria)
                    output["label"].append(label)
                    output["answer"].append(criteria[label])
                    output["task"].append(task or "")
                    output["shared_state"].append(f"Sentence: {' '.join(tokens)}")
                    output["question_id"].append(f"token-{token_index}")
                    output["source_row"].append(source_index)
                    output["target_index"].append(token_index)
                    output["target_token"].append(tokens[token_index])
            return output

        return dataset.map(
            convert_batch,
            batched=True,
            batch_size=1_000,
            with_indices=True,
            remove_columns=dataset["train"].column_names,
        )

    if task_type == "Classification":
        converted = dataset.map(convert)
    keep = {"state", "instructions", "criteria", "label", "answer", "task"}
    remove = [name for name in converted["train"].column_names if name not in keep]
    if remove:
        converted = converted.remove_columns(remove)
    return converted


def render_typed_decision(example, question_id="decision", model=None):
    """Render one canonical Jev row as a System One choice request."""
    criteria = list(example["criteria"])
    if len(criteria) != len(set(criteria)):
        raise ValueError("System One choice criterion names must be unique")
    request = OrderedDict()
    if model is not None:
        request["model"] = model
    request["state"] = example["state"]
    request["questions"] = {
        question_id: {
            "type": "choice",
            "instructions": example["instructions"],
            "criteria": {criterion: None for criterion in criteria},
        }
    }
    return dict(request)


def render_typed_decision_group(examples, model=None):
    """Render related decisions over one source state as a multi-question request."""
    examples = list(examples)
    if not examples:
        raise ValueError("At least one decision is required")
    state = examples[0].get("shared_state", examples[0]["state"])
    request = OrderedDict()
    if model is not None:
        request["model"] = model
    request["state"] = state
    questions = {}
    for example in examples:
        if example.get("shared_state", example["state"]) != state:
            raise ValueError("Grouped decisions must share one state")
        question_id = example.get("question_id", "decision")
        if question_id in questions:
            raise ValueError(f"Duplicate question id: {question_id}")
        criteria = list(example["criteria"])
        if len(criteria) != len(set(criteria)):
            raise ValueError("System One choice criterion names must be unique")
        instructions = example["instructions"]
        if "target_index" in example:
            instructions += (
                f" Target token at position {example['target_index']}: "
                f"{example['target_token']}"
            )
        questions[question_id] = {
            "type": "choice", "instructions": instructions,
            "criteria": {criterion: None for criterion in criteria},
        }
    request["questions"] = questions
    return dict(request)


# Former names, kept for existing callers.
render_systemone = render_typed_decision
render_systemone_group = render_typed_decision_group
