---
pretty_name: synthetic-typed-decisions
language:
- en
license: apache-2.0
task_categories:
- text-classification
tags:
- tasksource
- jev
- system-one
- synthetic
- multi-question
- soft-labels
---

# synthetic-typed-decisions

LLM-written items (emails, chats, forum threads, field reports, case notes, reviews...) with **several
typed questions over the same item**, following the Jev / System One request shape: `choice`, `noul`
(yes/no probability), and `score` (an ordered rubric). It covers judgment that cannot be computed
by rules: meaning, intent, tone, stance, plausibility, urgency, fuzzy categorization and routing.
Anything formalizable (rules, thresholds, deadlines, counting, lookups) is left to
[procedural-typed-decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions),
whose labels are exact.

This is an independent dataset. It is not an official TypeSafe Jev dataset and is not produced by or
affiliated with TypeSafe or OpenJev.

## How it is made

1. **Workflows.** For a domain and a kind of incoming item, DeepSeek V4 Flash designs a reusable
   decision application: 3 to 8 short judgment questions over 32 skills, with natural option counts
   (2 to 8, and 12 to 40 for one routing or categorization question in some workflows).
2. **Items.** About 30 items are written per workflow. One or two questions per item get a sampled
   intended reading (a clear option, a borderline one, or a yes/no probability), so answers are balanced
   and some uncertainty is deliberate; the other questions are answered by whatever the item says.
3. **Check.** A separate blind pass answers every question with probabilities and flags ill-posed ones,
   which are dropped.
4. **Labels.** Jev 1.13 (typesafe/jev-1.13 on OpenRouter) labels every kept question; questions where Jev
   and the check confidently disagree are dropped. `answers` averages the probabilities of the
   decision models that accept the question: Jev, upstage/solar-decide (up to 26 options) and, for
   yes/no questions, respan/span-01. Each model's probabilities are in `teachers`.

The soft labels carry real uncertainty (see the confidence shares below); train on the full
distributions rather than the argmax.

## Fields

| field | content |
|---|---|
| `state` | the item text |
| `questions` | JSON Jev request: `{id: {type, instructions, criteria}}` |
| `answers` | JSON answers in Jev format with the averaged probabilities (`noul`, or `probabilities` over the criteria) |
| `teachers` | JSON `{id: {model: probabilities}}` for each decision model |
| `skills` | JSON `{id: skill}` |
| `checker` | JSON `{id: probabilities}` from the blind check, for diagnostics |
| `workflow`, `application`, `domain`, `source` | the decision application; `domain` is a loose tag |

## Size and splits

Splits are by workflow, so validation and test use decision applications never seen in training.

<!-- size -->
<!-- /size -->

## Caveats

Items are synthetic and the labels come from a model. A few questions still lean on an unstated standard
(for example "standard safety procedures"). The generator code is
`src/tasksource/jev/synthetic/workflows.py` in [tasksource](https://github.com/sileod/tasksource).

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
    url = "https://aclanthology.org/2024.lrec-main.1361",
    pages = "15655--15684",
}
```
