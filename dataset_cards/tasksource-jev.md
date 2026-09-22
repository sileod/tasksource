---
pretty_name: tasksource-jev
language:
- multilingual
license: other
task_categories:
- text-classification
- question-answering
tags:
- tasksource
- jev
- system-one
- runtime-defined-decisions
- multiple-choice
size_categories:
- 1M<n<10M
---

# tasksource-jev

`tasksource-jev` recasts Tasksource classification and multiple-choice datasets
as runtime-defined decisions. Each example supplies its state and candidate
criteria at inference time. The dataset is intended for training and evaluating
bounded decision models; it is not tied to one Jev implementation.

This is an independent data transformation. It is not an official TypeSafe Jev
dataset and is not produced by or affiliated with TypeSafe or OpenJev.

> **Preview release:** the first 105,000 examples are published now so the schema
> and loading path can be tested while the full 575-task build runs. The preview
> is versioned from completed tasks and will be replaced by the complete build,
> which also includes compatible tasks from Tasksource's multilingual catalog.

## Schema

| field | type | meaning |
|---|---|---|
| `id` | string | Stable identifier derived from task, split, and row index. |
| `kind` | string | System One primitive: `choice`, `noul`, or `score`. |
| `options` | list of strings | Candidate labels or answers supplied at runtime. Order is significant. |
| `target` | list of floats | One-hot target distribution aligned with `options`. |
| `state` | string | Text or question on which the decision is based. Paired classification inputs are marked `text_A` and `text_B`. |
| `question` | string | The decision requested from the model. |
| `source` | string | Tasksource task identifier used to load the source data. |
| `variant` | string | `direct`, `label_verification`, or `ordered_rubric`. |
| `split` | string | Source split normalized to `train`, `dev`, or `test`. |

Classification criteria are the source task's label names. Multiple-choice
criteria are the answer choices. Every source row is retained as a direct
`choice`. A deterministic augmentation pass adds label-verification `noul`
questions to about 5% of rows. Ordered-rubric `score` augmentation is available
for genuinely ordinal sources but is disabled by default. Native regression and
ordinal recasting will be used for score examples rather than imposing an order
on nominal classification labels.
For `noul`, `target` contains the scalar truth probability and `options` is empty;
for `choice` and `score`, `target` is aligned with `options`. These lower-frequency
variants exercise all three Jev primitives without paraphrasing or shuffling the
canonical decision.

```python
from datasets import load_dataset

dataset = load_dataset("tasksource/tasksource-jev")
row = dataset["train"][0]
answer = row["options"][max(range(len(row["target"])), key=row["target"].__getitem__)]
```

With Tasksource installed, the same representation can be produced directly:

```python
from tasksource import load_task, render_systemone

dataset = load_task("glue/rte", recast="jev")
request = render_systemone(dataset["train"][0], model="openjev")
```

## Construction

Tasksource standardizes heterogeneous datasets into common classification and
multiple-choice templates. This release applies `recast_jev` to compatible
English and multilingual tasks, retains the standard train/validation/test splits, and records the
Tasksource identifier in every row. Tasks that fail to download or preprocess
are recorded by the build report rather than silently represented as complete.
To keep very large sources balanced, the build caps each task at 30,000
training rows and 3,000 validation or test rows using Tasksource's deterministic
sampling (seed 0). The published release is capped at 500,000 rows using a
source-balanced 90/5/5 train/dev/test allocation; selection preserves relative
row order.

For a useful Dataset Viewer preview, only the first 1,000 training rows are
ordered round-robin by `source`. This is a deterministic permutation, not a
random shuffle. After that display prefix, all remaining examples retain their
original relative order.

The selected catalog includes 100 BIG-bench task configurations and all 57 MMLU
subjects currently registered in Tasksource. Their original split identity is
preserved in the row-level `split` field, with `validation` normalized to `dev`.
This makes source/split exclusion explicit when constructing a training mixture.

The repository includes `failed-tasks.json` and `outdated-datasets.json`.
The latter specifically tracks upstream datasets that still depend on loading
scripts no longer supported by current Hugging Face Datasets, so they can be
migrated to data-only Parquet repositories and incorporated in a later build.

The build is reproducible from the Tasksource repository:

```bash
python scripts/build_jev_dataset.py --output build/tasksource-jev --finalize
```

## Licensing and provenance

Tasksource is a preprocessing framework and catalog, not the original publisher
of the constituent datasets. Copyright, license, and usage restrictions remain
those of each upstream dataset. Users should consult the upstream dataset card
identified by `source` before redistributing or using a subset. The aggregate is
therefore marked `license: other`; no single license is asserted over all rows.

## Citation

If this recast is useful, cite Tasksource, which provides the task collection and
harmonization framework:

```bibtex
@inproceedings{sileo-2024-tasksource,
    title = "tasksource: A Large Collection of {NLP} tasks with a Structured Dataset Preprocessing Framework",
    author = "Sileo, Damien",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1361/",
    pages = "15655--15684",
    abstract = "The HuggingFace Datasets Hub hosts thousands of datasets, offering exciting opportunities for language model training and evaluation. However, datasets for a specific task type often have different structures, making harmonization challenging which prevents the interchangeable use of comparable datasets. As a result, multi-task training or evaluation necessitates manual work to fit data into task templates. Several initiatives independently tackle this issue by releasing harmonized datasets or providing harmonization codes to preprocess datasets into a consistent format. We identify patterns in such preprocessings, such as column renaming, or more complex patterns. We then propose an annotation framework that enables concise, readable, and reusable preprocessing annotations. tasksource annotates more than 600 task preprocessings and provides a backend to automate dataset alignment. We fine-tune a multi-task text encoder on all tasksource tasks, outperforming every publicly available text encoder of comparable parameter count according to an external evaluation."
}
```
