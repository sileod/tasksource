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
---

# tasksource-jev-typed-decisions

Hundreds of source tasks, one typed-decision format. This dataset recasts
Tasksource's broad collection of NLP datasets as decisions whose criteria are
supplied with each example, rather than fixed in a model head. Much of that
breadth comes from human-written or human-annotated English and multilingual
data: classification, question answering, and selected token tasks. The
source catalog now also includes newly sourced
[IT support tickets](https://huggingface.co/datasets/tasksource/it-support-tickets)
with seven categories assigned by support professionals. A separate procedural
slice adds multi-question states and calibrated targets; it is identified by its
`procedural-jev/` source prefix, not presented as human-authored data.

This is useful when training a decision model to handle *new label sets at
runtime*: the same columns cover sentiment, inference, intent, entity tags,
ticket routing, graded judgments, and more. Tasksource supplies the breadth;
the Jev recast supplies a consistent `state + question + criteria → target`
interface. The project is independent of TypeSafe and OpenJev.

## Data at a glance

| column | meaning |
|---|---|
| `state` | Text or structured input to judge. |
| `question` | What to decide about that state. |
| `kind` | `choice`, `noul` (truth probability), or `score` (ordered levels). |
| `options` | Runtime criteria for `choice`/`score`; empty for `noul`. |
| `target` | Distribution aligned with `options`, or one probability for `noul`. |
| `source`, `split`, `variant` | Provenance, original train/dev/test identity, and transformation. |
| `id`, `group_id`, `question_id` | Decision identity and questions sharing a source example. |

```python
from datasets import load_dataset

ds = load_dataset("tasksource/tasksource-jev-typed-decisions")
row = ds["train"][0]
print(row["state"], row["question"], row["options"], row["target"])
```

Most Tasksource classification targets are one-hot. Native graded sources add
soft distributions or ordinal scores where the original annotation supports
them; no uncertainty is invented from a hard label. Selected CoNLL-2003 and
WNUT-17 token tasks ask at most two readable BIO-tag questions per sentence.
Several questions can share a `group_id`; Parquet keeps one decision per row
for easy streaming and shuffling. The
[procedural source](https://huggingface.co/datasets/tasksource/procedural-jev)
contributes `choice`, `noul`, and `score` questions over shared states.

Deterministic, low-frequency variants include label verification, criterion
permutation, vetted instruction paraphrases, and paired-text field wording.
The `variant` column identifies them. The source's direct decision remains in
the data. The first 1,000 training rows are interleaved for a more useful
Dataset Viewer preview; this changes order, not membership.

The release builder targets up to one million rows with a 90/5/5
train/dev/test allocation. It balances source *families* while sampling their
configs, keeps multi-question groups intact, and excludes BIG-bench, MMLU,
and BLiMP. Consult the Hub split counts for the currently published size.

## Provenance and limits

Tasksource harmonizes the source datasets; it does not own their content.
Human-authored, crowd-annotated, and procedural rows should not be treated as
interchangeable evidence. Use `source` to inspect the original dataset and
its license before reuse. The aggregate has `license: other` because no single
upstream license covers every row. The ticket source, for example, is the
[Benitez Pereira collection](https://doi.org/10.5281/zenodo.7648117), not
synthetic ticket text.

Some upstream tasks fail to load or preprocess; the build records those
failures instead of implying complete coverage. The builder writes
`failed-tasks.json`, `fixed-source-audit.json`, `build-manifest.json`, and
`release-audit.json`. The
[build runbook](https://github.com/sileod/tasksource/blob/main/docs/jev/README.md)
documents the resumable Parquet pipeline, validation, and reproducibility
boundary. For an uncapped canonical recast, use
`load_task("glue/rte", recast="jev")` in Tasksource.

## Citation

If this recast is useful, cite the Tasksource collection and preprocessing
framework:

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
