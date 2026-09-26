---
pretty_name: tasksource-instruct
language:
- en
license: other
size_categories:
- 1M<n<10M
task_categories:
- text-generation
- text-classification
- token-classification
- zero-shot-classification
tags:
- instructions
- instruction-tuning
- instruction-finetuning
- flan
- promptsource
- tasksource
---

# tasksource-instruct

**Instruction-tuning data recast from the ~480 English classification, multiple-choice
and token-classification tasks of [tasksource](https://github.com/sileod/tasksource).**

Every example comes from a human-built dataset (NLI, logical reasoning, sentiment,
hate speech, discourse, argumentation, ...), not from a teacher model. Each task is
capped at 30k training examples, so no task dominates. Many tasks aren't in FLAN v2,
for example DynaSent, DynaHate, discriminative bAbI, epistemic logic, RuleTaker,
veridicality and dozens of NLI datasets.

```python
from datasets import load_dataset

ds = load_dataset("tasksource/tasksource-instruct", split="train")
ds = ds.filter(lambda use: use == "commercial", input_columns="license_use")  # optional
```

## Format

| column | content |
|---|---|
| `inputs` | the instruction, the example, and the answer options |
| `targets` | the answer: an option (`entailment.`), a letter (`B.`), or `word: TAG` lines for token tasks |
| `task` | the tasksource task id |
| `license`, `license_use` | the source's licenses, see below |

Prompts ask for the answer with no explanation, so the short targets don't teach a
model to stop explaining in general. Tasks are interleaved round-robin, so any
slice of the split mixes them. Validation and test keep up to 500 examples per task.

`tasksource-instruct` works well mixed with FLAN v2 or other instruction data. It
covers discriminative reasoning tasks that those sets cover less.

For preference pairs built from the same rows, see
[tasksource_dpo_pairs](https://huggingface.co/datasets/tasksource/tasksource_dpo_pairs).
For soft labels, ratings and multi-question requests, see
[tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions).

## Reproducibility

The dataset is built by
[`scripts/build_instruct_dataset.py`](https://github.com/sileod/tasksource/blob/main/scripts/build_instruct_dataset.py):

```bash
PYTHONPATH=.:src python scripts/build_instruct_dataset.py --finalize
```

Sources are loaded at pinned Hub revisions. [sources.yaml](sources.yaml) records, per
task, the Hub dataset, revision, original dataset, licenses and row counts, and
`build-report.jsonl` records the code commit of each task's build. MMLU, BIG-bench and
BLiMP are left out, so they stay clean for evaluation. Other public benchmarks (GLUE,
SuperGLUE, HellaSwag, PIQA, ...) are **in** the data through their training splits.

## License and scope

Tasksource harmonizes datasets from many publishers; their original licenses
and terms still apply, hence `license: other`.

- `license` lists the `license` of the Hub dataset card the task was loaded from,
  and of the original dataset behind a tasksource copy. It also lists licenses recorded
  by the [Data Provenance Initiative](https://www.dataprovenance.org/), marked `(DPI)`.
- `license_use` takes the most restrictive of those: `non-commercial` if any is
  non-commercial or academic-only, `commercial` if one allows commercial use (share-alike
  and copyleft included), and `unspecified` otherwise.

This is a best-effort aid, not legal advice. Check the original terms before relying on them.

## Citation

```bibtex
@inproceedings{sileo-2024-tasksource,
    title = "tasksource: A Large Collection of {NLP} tasks with a Structured Dataset Preprocessing Framework",
    author = "Sileo, Damien",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1361/",
    pages = "15655--15684",
}
```
