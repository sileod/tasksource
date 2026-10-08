---
pretty_name: tasksource-jev-typed-decisions
language:
- en
- multilingual
license: other
task_categories:
- zero-shot-classification
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

**2.5 million typed decisions (choices, ratings and probabilities) from 670 sources.**

## Why use it

- **Real supervision.** Labels, ratings, and annotator votes come from
  established datasets, not a teacher model. Every row names its `source`.
- **Breadth.** Over 300 dataset families: NLI and reasoning, QA and
  commonsense, sentiment, intent and topic, toxicity and safety, preference
  pairs, fact checking, entity tagging, and dozens of languages. GLUE,
  SuperGLUE, HellaSwag, PIQA, ScienceQA, Banking77, CoNLL-2003, MasakhaNEWS,
  HelpSteer, ChaosNLI, and many more, with no task allowed to dominate.
- **Three decision types in one schema.** `choice` (pick one option), `score`
  (an ordered scale), and `noul` (the probability that the answer to a yes/no
  question is yes). `noul` holds only probabilities: entailment likelihoods, and the
  share of annotators who answered yes. Mean ratings and similarity are `score`
  distributions whose expected level is the mean (3.4 on 1–5 puts 0.6 on 3 and 0.4
  on 4). Ordinal label sets appear as both `choice` and `score`, split
  deterministically per row, so a model learns both requests for the same scale.
  Soft targets are kept wherever the source has mean ratings or votes from at
  least five annotators per item (vote shares from fewer are too noisy): STS,
  ChaosNLI, civil_comments, Measuring Hate Speech, WouldYouRather, ProtoQA,
  LeWiDi and more. They make up the graded share.
- **Built so position, repeated eval data, and question choice give nothing away.**
  - Multiple-choice options are shuffled per row, so the answer's position carries no signal.
  - Validation and test rows whose content appears in train are removed.
  - Derived questions are chosen without looking at their answers.
  - Annotations were reviewed task by task. Inverted, unanswerable, and garbled labels were fixed or dropped.
- **Multi-question states.** Related decisions share a `group_id` and can be
  asked together. Packed states test reasoning over several items at once, and
  [procedural-typed-decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions)
  adds exact counting, arithmetic, retrieval, state tracking, routing among
  up to 60 options, and exact posteriors when a policy applies to a requester
  whose role is uncertain.

## Quick start

Coding agent? Read [AGENTS.md](AGENTS.md): row semantics, rebuilding multi-question requests from `group_id`, filtering, and evaluation caveats.

```python
from datasets import load_dataset

ds = load_dataset("tasksource/tasksource-jev-typed-decisions")          # steered 1M-row mix
# full = load_dataset("tasksource/tasksource-jev-typed-decisions", "full")  # every row of the build
row = ds["train"][0]
print(row["state"], row["question"], row["options"], row["target"])
```

```json
{"state": "My body cast a shadow over the grass. What was the cause of this?",
 "question": "Choose the criterion that best answers the question.",
 "kind": "choice", "options": ["The sun was rising.", "The grass was cut."],
 "target": [1.0, 0.0], "source": "super_glue/copa"}
```

## Configs

- `default`: a steered mix of about 1M train rows. Sources are first gated on label correctness, then weighted by how interesting they are and how close they sit to the zone of proximal development (judged by decision models). Two-option tasks get fewer rows, and procedural generators get 12%. No row is repeated. Validation and test are the full eval splits, restricted to the mixed sources. Buckets and shares are in [`jev_mixes.py`](https://github.com/sileod/tasksource/blob/main/src/tasksource/metadata/jev_mixes.py) and per-source scores in `jev_source_scores.csv`.
- `full`: every row that passed the build, with per-source caps (about 2.5M train rows).
- `vision`: a multimodal pilot covering NLVR2, SNLI-VE, A-OKVQA, ScienceQA-IMG, AI2D, and FigureQA. It uses the same decision fields plus ordered `images` and a JSON-string `metadata` column. Load it with `load_dataset("tasksource/tasksource-jev-typed-decisions", "vision")`. Images remain encoded in the dataset and decode on access; the model input adapter must consume them alongside `state`. Native source splits are preserved, and evaluation rows sharing a training image are excluded. See [vision/sources.yaml](vision/sources.yaml) for pinned sources and license evidence.

## Format

| field | meaning |
|---|---|
| `state` | The text to decide about |
| `question` | What to decide |
| `kind` | `choice`, `score`, or `noul` |
| `options` | Runtime criteria; empty for `noul` |
| `target` | Distribution over `options`, or `[p]` for `noul` |
| `id`, `group_id`, `question_id` | Link decisions over the same source example |
| `example_id` | Stable hash of the source example's input and gold; the same across releases unless the label changes |
| `source`, `split`, `variant` | Originating task, original split, and recast variant |
| `license`, `license_use` | The source's license(s), and `commercial`, `non-commercial` or `unspecified` (see below) |

Splits: train (about 1M rows in `default`, 2.5M in `full`), 15,000 validation (`dev` in `split`), and 15,000 test,
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
- **Mixing (`full`).** Formats get fixed shares of the train rows (47% classification, 30% multiple choice, 3% token labeling, 10% graded (soft-label sources), 10% procedural). Within a format, dataset families get equal shares, scaled by [hand-set weights](https://github.com/sileod/tasksource/blob/main/src/tasksource/metadata/weights.py) (more for adversarial NLI, long documents and preference pairs; less for templated probes), times [audit weights](https://github.com/sileod/tasksource/blob/main/src/tasksource/metadata/audit_weights.py) from a per-task check of Jev on 200 examples: ×1.5 for hard tasks whose gold is right by construction (synthetic logic, theory of mind, spatial reasoning), ×0.5 for near-solved tasks and for hard tasks whose gold is a judgment call (ratings, preferences, crowd sentiment). Sources with many options get slightly more room. Related questions are kept together.
- **Mixing (`default`).** About 1M rows drawn from `full`, never repeating a row. Buckets of related sources get set shares ([jev_mixes.py](https://github.com/sileod/tasksource/blob/main/src/tasksource/metadata/jev_mixes.py): 15% logic, 10% NLI, 10% knowledge QA, 9% long documents and fact-checking, 8% intent and routing, 12% procedural, ...). Inside a bucket, sources get rows by score × √size. The [score](https://github.com/sileod/tasksource/blob/main/src/tasksource/metadata/jev_source_scores.csv) multiplies:
  - correctness: sources that failed label review get 0;
  - the zone of proximal development: how much probability the decision models (Jev, Liquid D1) give the gold answer. Near-solved sources and sources the models miss outright both get less;
  - interest and cleanliness, from Jev yes/no checks for transferable skills, trivial examples and malformed rows;
  - ×0.6 for two-option tasks (about a third of the rows);
  - ×2 for sources picked by reading them.
- **Order and coverage.**
  - The first 1,000 train rows are interleaved to show variety in the Dataset Viewer; the rest is shuffled. Questions of a group stay adjacent throughout.
  - Evaluation benchmarks (BIG-bench, MMLU, BLiMP, MATH test, ...) are left out so they stay clean for evaluation.
- **Sources.** [sources.yaml](sources.yaml) lists every source with its rows, the Hub dataset and revision it was loaded from, the original dataset behind each tasksource copy, and its licenses.
- **Audit trail.** The [source mix](release-audit.json), [failed source list](failed-tasks.json), and [build manifest](build-manifest.json) ship with the data.
- **Reproducible.** The [build runbook](https://github.com/sileod/tasksource/blob/main/docs/jev/README.md) rebuilds the release from [Tasksource](https://github.com/sileod/tasksource)'s [task catalog](https://github.com/sileod/tasksource/blob/main/catalog_english.md).

## License and scope

Tasksource harmonizes datasets from many publishers; their original licenses
and terms still apply, hence `license: other`.

Each row carries its source's license, to help filter:

```python
ds = ds.filter(lambda use: use == "commercial", input_columns="license_use")
```

- `license` lists the `license` of the Hub dataset card the source was loaded from,
  and of the original dataset behind a tasksource copy. It also lists licenses recorded
  by the [Data Provenance Initiative](https://www.dataprovenance.org/), marked `(DPI)`.
- `license_use` takes the most restrictive of those: `non-commercial` if any is
  non-commercial or academic-only, `commercial` if one allows commercial use (share-alike
  and copyleft included), and `unspecified` otherwise. That covers missing licenses and
  `other`, bare `cc`, and no-derivatives licenses.
- [sources.yaml](sources.yaml) records each card and DPI license per source.

This is a best-effort aid, not legal advice. Licenses on cards can be wrong or
incomplete, and a source's terms may differ from its card's. Check the
original terms before relying on them. This recast is independent of
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
