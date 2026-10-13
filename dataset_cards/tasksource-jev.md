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
- decision-model
size_categories:
- 1M<n<10M
---

# tasksource-jev-typed-decisions

Typed decisions from hundreds of curated [Tasksource](https://github.com/sileod/tasksource) tasks: classification, multiple choice, ordered ratings, probabilities and visual reasoning. Targets come from source annotations or deterministic transformations of them. Native annotation noise and subjective judgments remain.

## Load

```python
from datasets import load_dataset

text = load_dataset("tasksource/tasksource-jev-typed-decisions")
vision = load_dataset("tasksource/tasksource-jev-typed-decisions", "vision")
row = text["train"][0]
print(row["state"], row["question"], row["options"], row["target"])
```

| Config | Contents |
|---|---|
| `default` | Steered text training mix, plus reviewed incremental additions; validation and test included. |
| `full` | Larger text pool with per-source caps, including incremental additions; validation and test included. |
| `filtered-full` | Filtered, training-only historical snapshot. Later additions are not included. See [selection policy and counts](filtered-full/README.md). |
| `vision` | Multimodal decisions with ordered images and text choices. See [visual sources, sampling and provenance](vision/README.md). |

Exact split sizes are in the Dataset Viewer metadata. Task coverage and counts are in [sources.yaml](sources.yaml) and [vision/sources.yaml](vision/sources.yaml).

## Row format

| Field | Meaning |
|---|---|
| `state` | Text to reason about. |
| `question` | The decision to make. |
| `kind` | `choice`, `score` or `noul`. |
| `options` | Criteria in their presented order; empty for `noul`. |
| `target` | Distribution aligned with `options`, or `[p_true]` for `noul`. |
| `id`, `group_id`, `question_id` | Decision identity and grouping of related questions. |
| `example_id` | Content-derived example identity used by the build. |
| `source_index` | Original row of the example in its source dataset, as `split:row` (the split before any re-splitting); empty for packed and generated rows. |
| `source`, `split`, `variant` | Task identity, split annotation and recast variant. |
| `license`, `license_use` | Recorded source terms and their usage classification. |
| `images` | **Vision only:** ordered Hugging Face image features, decoded on access. |
| `metadata` | **Vision only:** JSON-string source identity, geometry, augmentation and license evidence. |

```json
{"state": "My body cast a shadow over the grass. What was the cause of this?",
 "question": "Choose the criterion that best answers the question.",
 "kind": "choice", "options": ["The sun was rising.", "The grass was cut."],
 "target": [1.0, 0.0], "source": "super_glue/copa"}
```

`choice` selects a criterion; `score` uses an ordered scale. Soft targets preserve annotator disagreement or interpolate numerical ratings between neighboring scale anchors. `noul` represents the probability that a proposition is true. Scale meaning is determined by the question and criteria.

Rows sharing `group_id` can be assembled into a multi-question request. Vision images stay separate from text and must be passed through the model's image adapter; answer options remain text. See [AGENTS.md](AGENTS.md) for request assembly and row semantics.

## Construction and evaluation

Tasksource standardizes native labels and choices, adds task-specific questions where needed, and applies deterministic option permutations and low-frequency text variants. Some text rows pack related decisions over a shared state. A decision's majority label keeps at most three rows per row of its other labels (graded answers count as yes above 0.5; a question with a single label is dropped), so skewed binary sources such as answer selection or toxicity no longer reward always answering the common label; procedural questions keep each label under 60%. The visual config contains direct decisions and shares sampling budgets across related task views.

Native source partitions are retained where available; derived group-based holdouts and overlap exclusions are documented per source. These checks do not establish absence of all cross-dataset or benchmark contamination. For example, Mind2Web text tasks use derived holdouts from public training trajectories, not official benchmark tests; alignment with private Decision Index evaluation rows is unverified. Mind2Web is excluded from the visual config.

Source and recast reviews are coarse checks, not corpus-wide correctness guarantees. [English examples](https://github.com/sileod/tasksource/blob/main/docs/examples/english/README.md) illustrate the exported and canonical representations.

## Licenses

This collection has **mixed source terms**, hence `license: other`. Original annotation and image terms still apply. `license_use` is a best-effort classification: `commercial`, `non-commercial` or `unspecified`; it is not legal clearance. Missing or scoped evidence is retained explicitly.

```python
text = text.filter(lambda use: use == "commercial", input_columns="license_use")
```

Visual annotation licenses may not cover third-party images. The visual license review records unresolved image rights and explicit restrictions, including FigureQA's prohibition on redistribution. See [component-level evidence](vision/license-review.json) and the [vision documentation](vision/README.md) before reuse.

## Documentation and provenance

- [Text sources and revisions](sources.yaml), [release audit](release-audit.json), [build manifest](build-manifest.json) and [failed sources](failed-tasks.json).
- [Vision documentation](vision/README.md) and [filtered-full selection evidence](quality/filtered-full-v1/README.md).
- [Build runbook](https://github.com/sileod/tasksource/blob/main/docs/jev/README.md), [base build recipe](BUILD.md) and [task catalogs](https://github.com/sileod/tasksource#task-catalogs).
- [Release history](CHANGELOG.md). Addition manifests and sampled reviews retain their original snapshot context.

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
