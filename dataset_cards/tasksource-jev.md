---
pretty_name: tasksource-jev-typed-decisions
language:
- en
- multilingual
license: other
task_categories:
- text-classification
- question-answering
- token-classification
tags:
- tasksource
- jev
- system-one
- runtime-defined-decisions
- decision-models
- multiple-choice
size_categories:
- 1M<n<10M
---

# tasksource-jev-typed-decisions

**Two million human-labeled decisions from 600+ tasks, in one format for models
that read their answer criteria at runtime.**

Most instruction data teaches a model to *write*. This dataset teaches it to
*decide*: given a state and a question, pick among the options it is handed,
rate on a scale it is handed, or give a calibrated probability. The options
change from row to row, so a model has to read them rather than memorize a label set.

## Why use it

- **Real supervision.** Labels, ratings, and annotator votes come from
  established datasets, not a teacher model. Every row names its `source`.
- **Breadth.** Over 300 dataset families: NLI and reasoning, QA and
  commonsense, sentiment, intent and topic, toxicity and safety, preference
  pairs, fact checking, entity tagging, and dozens of languages. GLUE,
  SuperGLUE, HellaSwag, PIQA, ScienceQA, Banking77, CoNLL-2003, MasakhaNEWS,
  HelpSteer, ChaosNLI, and many more, with no task allowed to dominate.
- **Three decision types in one schema.** `choice` (pick one option), `score`
  (an ordered scale), and `noul` (the probability that a statement is true).
  Soft targets are kept wherever the source has votes or ratings.
- **Built so position, repeated eval data, and question choice give nothing away.**
  - Multiple-choice options are shuffled per row, so the answer's position carries no signal.
  - Validation and test rows whose content appears in train are removed.
  - Derived questions are chosen without looking at their answers.
  - Annotations were reviewed task by task. Inverted, unanswerable, and garbled labels were fixed or dropped.
- **Multi-question states.** Related decisions share a `group_id` and can be
  asked together. Packed states test reasoning over several items at once, and
  [procedural-typed-decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions)
  adds exact counting, arithmetic, retrieval, and state tracking.

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("tasksource/tasksource-jev-typed-decisions")
row = ds["train"][0]
print(row["state"], row["question"], row["options"], row["target"])
```

```json
{"state": "My body cast a shadow over the grass. What was the cause of this?",
 "question": "Choose the criterion that best answers the question.",
 "kind": "choice", "options": ["The sun was rising.", "The grass was cut."],
 "target": [1.0, 0.0], "source": "super_glue/copa"}
```

## Format

| field | meaning |
|---|---|
| `state` | The text to decide about |
| `question` | What to decide |
| `kind` | `choice`, `score`, or `noul` |
| `options` | Runtime criteria; empty for `noul` |
| `target` | Distribution over `options`, or `[p]` for `noul` |
| `id`, `group_id`, `question_id` | Link decisions over the same source example |
| `source`, `split`, `variant` | Originating task, original split, and recast variant |

Splits: 2,000,000 train, 15,000 validation (`dev` in `split`), and 15,000 test,
following each source's own train/dev/test splits where it has them.

## How it is built

- **Canonical recasts.** Each Tasksource task is converted deterministically.
  - Criteria are the source's own label names and answer options.
  - Multiple-choice rows keep every option in a per-row order.
  - A final "all/none of the above" reads "all/none of the other options".
  - Options that cite other options by letter or number keep their order.
  - The question is the task's own when its inputs alone do not say what to predict
    ("What stance does the tweet take on feminism?"), and a generic instruction otherwise.
    Label-verification and packed questions carry it too.
- **Variants.** Low-frequency, deterministic variants cover label verification as `noul`, criterion order, and instruction wording.
- **Packing.** Up to 10% of each classification task's examples are packed, two to four at a time, into `packed_derived` states. Their questions (an item's label, agreement, existence, counts) follow exactly from the gold labels.
- **Mixing.** Formats get fixed shares of the train rows (47% classification, 30% multiple choice, 3% token labeling, 10% graded, 10% procedural). Within a format, dataset families get equal shares, scaled by [hand-set weights](https://github.com/sileod/tasksource/blob/main/src/tasksource/metadata/weights.py) (more for adversarial NLI, long documents and preference pairs; less for templated probes). Related questions are kept together.
  - The first 1,000 train rows are interleaved to show variety in the Dataset Viewer; the rest is shuffled.
  - Evaluation benchmarks (BIG-bench, MMLU, BLiMP, MATH test, ...) are left out so they stay clean for evaluation.
- **Sources.** [sources.yaml](sources.yaml) lists every source with its rows, the Hub dataset and revision it was loaded from, and the original dataset behind each tasksource copy.
- **Audit trail.** The [source mix](release-audit.json), [failed source list](failed-tasks.json), and [build manifest](build-manifest.json) ship with the data.
- **Reproducible.** The [build runbook](https://github.com/sileod/tasksource/blob/main/docs/jev/README.md) rebuilds the release from [Tasksource](https://github.com/sileod/tasksource)'s [task catalog](https://github.com/sileod/tasksource/blob/main/tasks.md).

## License and scope

Tasksource harmonizes datasets from many publishers; their original licenses
and terms still apply, hence `license: other`. This recast is independent of
TypeSafe and OpenJev.

## Citation

```bibtex
@inproceedings{sileo-2024-tasksource,
  title = {tasksource: A Large Collection of {NLP} tasks with a Structured Dataset Preprocessing Framework},
  author = {Sileo, Damien},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  year = {2024},
  pages = {15655--15684},
  url = {https://aclanthology.org/2024.lrec-main.1361/}
}
```
