---
pretty_name: tasksource-jev-typed-decisions
language:
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

One million decisions from **500+ Tasksource tasks across 300+ dataset families**,
in a single format for models that receive their answer criteria at runtime.
The value is breadth with traceable supervision: most rows inherit labels,
ratings, or annotator votes from existing datasets, not labels invented by a
teacher model. The `source` field identifies the originating task; existing
train/dev/test boundaries are retained where the source provides them.
The [Tasksource repository](https://github.com/sileod/tasksource) and
[task catalog](https://github.com/sileod/tasksource/blob/main/tasks.md) document
the source preprocessings.

The coverage is deliberately wide: GLUE and SuperGLUE inference and language
understanding; SNLI and XNLI; HellaSwag, PIQA, and ScienceQA; AG News,
Banking77, and real support-ticket classification; CoNLL-2003 and WNUT-17
entity tagging; MasakhaNEWS and other multilingual tasks; and graded sources
such as HelpSteer and ChaosNLI. These are different decision problems with
different criteria, made usable through one schema. A small, tagged procedural
component adds controlled multi-question states.

## Format

Each Parquet row has a `state`, a `question`, runtime `options`, and a `target`.
`kind` is `choice`, `noul` (one truth probability), or `score` (ordered levels).
For `choice` and `score`, `target` is a distribution aligned with `options`;
for `noul`, `options` is empty. `id`, `group_id`, and `question_id` let related
decisions share a source example without requiring nested rows. `variant`
marks deterministic subrecasts; `source` and `split` preserve provenance.

```python
from datasets import load_dataset

ds = load_dataset("tasksource/tasksource-jev-typed-decisions")
row = ds["train"][0]
print(row["state"], row["question"], row["options"], row["target"])
```

Most classification targets are one-hot because the source annotations are
hard labels. Sources with vote distributions or ratings retain softer or
ordinal targets where justified. Low-frequency, deterministic variants cover
label verification, criterion order, instruction wording, and paired-text
field wording. The first 1,000 training rows are interleaved to show task
variety in the Dataset Viewer; no rows are added by that display order.

The release has 900,000 train, 50,000 validation (`dev` in the `split` field),
and 50,000 test decisions. Publication balances dataset families while
sampling their configurations and keeping related questions together. The
full [source mix](release-audit.json), [failed source list](failed-tasks.json),
and [build manifest](build-manifest.json) are published alongside the data.
BIG-bench, MMLU, and BLiMP are not included.

Tasksource harmonizes datasets from many publishers; their original licenses
and usage terms still apply. The aggregate is marked `license: other` because
there is no single license for every source. The
[build runbook](https://github.com/sileod/tasksource/blob/main/docs/jev/README.md)
describes the resumable pipeline. This recast is independent of TypeSafe and
OpenJev.

## Citation

Please cite the Tasksource collection and preprocessing framework:

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
