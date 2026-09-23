---
pretty_name: tasksource-jev
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
- 100K<n<1M
---

# tasksource-jev

`tasksource-jev` recasts Tasksource classification, multiple-choice, and selected
token-classification datasets
as runtime-defined decisions. Each example supplies its state and candidate
criteria at inference time. The dataset is intended for training and evaluating
bounded decision models; it is not tied to one Jev implementation.

This is an independent data transformation. It is not an official TypeSafe Jev
dataset and is not produced by or affiliated with TypeSafe or OpenJev.

The current release is capped at 500,000 decisions across English and
multilingual tasks. Build reports identify completed and failed source tasks.

## Schema

| field | type | meaning |
|---|---|---|
| `id` | string | Stable identifier derived from task, split, and row index. |
| `group_id` | string | Identifies decisions derived from the same source row. |
| `question_id` | string | Identifies a question within that group. |
| `kind` | string | System One primitive: `choice`, `noul`, or `score`. |
| `options` | list of strings | Candidate labels or answers supplied at runtime. Order is significant. |
| `target` | list of floats | One-hot target distribution aligned with `options`. |
| `state` | string | Text or question on which the decision is based. Token decisions include the sentence, marked target, and target index. |
| `question` | string | The decision requested from the model. |
| `source` | string | Tasksource task identifier used to load the source data. |
| `variant` | string | Direct decision or a named deterministic subrecast. |
| `split` | string | Source split normalized to `train`, `dev`, or `test`. |

Classification criteria are the source task's label names. Multiple-choice
criteria are the answer choices. Every source row is retained as a direct
`choice` for classification and multiple choice. For selected token tasks, at
most two token questions are sampled deterministically from each source
sequence; one non-`O` token is preferred when available. BIO/BILOU boundaries
and compact POS or dependency labels are expanded into readable criteria.
Only `Sequence(ClassLabel)` or equivalent `List(ClassLabel)` ontologies with
2–32 readable labels qualify. This release includes CoNLL-2003 NER and WNUT-17
token decisions from checked data-only mirrors; see
[token-source-status.md](token-source-status.md) for the source audit and backlog.
The `group_id` lets several questions share one source row without a costly
global grouping pass. A deterministic augmentation pass adds label-verification `noul`
questions to about 5% of rows. Ordered-rubric `score` augmentation is available
for genuinely ordinal sources but is disabled by default. Native regression and
ordinal recasting will be used for score examples rather than imposing an order
on nominal classification labels.

In the canonical Tasksource recast, related token rows also carry
`shared_state`, `source_row`, and distinct `question_id` values. Pass rows from
one `source_row` to `render_systemone_group(rows)` to obtain a single Jev
request with several questions over the same sentence. The Parquet view keeps
one decision per row so it can be shuffled, sampled, or streamed normally.

### Deterministic subrecasts

The release adds conservative, low-frequency variants while retaining every
direct row:

| variant | default rate | purpose |
|---|---:|---|
| `label_verification` | 5% | A `noul` judgement asking whether a deterministically proposed label is correct. Correct and incorrect proposals are balanced. |
| `criteria_permutation` | 5% | The same `choice` decision with options and targets permuted together, reducing option-position shortcuts. |
| `instruction_paraphrase` | 5% | The same decision with a manually vetted equivalent instruction. Common NLI and sentiment label groups receive specific wording; other tasks use conservative generic alternatives. |
| `paired_text_format` | 5% of paired rows | Neutral alternatives to repeated `text_A`/`text_B` field labels, without assuming a task-specific relation between the texts. |

All transformations are derived exactly from the source target and introduce no
teacher-generated claims. Candidate-subset decisions are a possible later
addition. Synthetic uncertainty, abstention, and nominal-to-ordinal conversions
are intentionally excluded because one-hot classification labels do not justify
them.
For `noul`, `target` contains the scalar truth probability and `options` is empty;
for `choice` and `score`, `target` is aligned with `options`. The direct canonical
decision is unchanged; lower-frequency variants are identified explicitly by
the `variant` field.

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

BIG-bench, MMLU, and BLiMP are excluded from this release. Original split
identity is preserved in `split`, with `validation` normalized to `dev`.

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
