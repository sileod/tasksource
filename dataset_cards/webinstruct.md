---
license: apache-2.0
language:
- en
task_categories:
- multiple-choice
- text-classification
configs:
- config_name: mc
  data_files:
  - split: train
    path: mc/train.parquet
  - split: test
    path: mc/test.parquet
- config_name: binary
  data_files:
  - split: train
    path: binary/train.parquet
  - split: test
    path: binary/test.parquet
---

# WebInstruct: multiple choice and binary

Prepared subsets of [TIGER-Lab/WebInstruct-verified](https://huggingface.co/datasets/TIGER-Lab/WebInstruct-verified),
source revision `3e8a350b3a935d68fe70bdf379692500abc9ff51`.
The original `train` and `test` assignments are preserved; `train_legacy` is omitted.
Original fields (`id`, `question`, `answer`, `answer_type`, `category`, `difficulty`) are retained.
The source answers are used as supplied, without checking their factual correctness.

## Multiple-choice preprocessing

Only `answer_type="Multiple Choice"` rows are considered. Option markers are
whitespace-delimited letters written as `A.`, `a)`, `a).`, `(a)`, or `a:`.
Markers must be unique and cover consecutive letters starting at A, with at least
two nonempty options. Options stay in their order in the source question, including
nonalphabetical orders such as A, C, B, D. `prompt` is the nonempty text before the
first marker; `options` contains the text between markers, with outer whitespace removed.
The source `question` retains the complete original text.

`answer` is lowercased and stripped of surrounding whitespace, parentheses, and
periods. It must identify exactly one of the parsed letters. `gold` is its zero-based
index in `options`; `method="regex"` records this deterministic extraction.
Unmarked options, duplicate or missing markers, textual answers, multiple-answer
annotations, and answers outside the option list are excluded. This is a conservative
subset: valid examples with unsupported formatting are also excluded. Option text
is copied from the source, so trailing source commentary can remain in the last option.

## Binary preprocessing

Only `answer_type="Boolean"` rows with an explicit `yes`, `no`, `true`, or `false`
answer are retained, ignoring case and outer whitespace. `label=0` means no/false
and `label=1` means yes/true; `method="exact"` records this normalization.
Letter-only answers and explanatory or unexpected answers are excluded rather
than assigned a guessed label. Questions and source answers are preserved verbatim.

## Reproduction and Tasksource

Run `PYTHONPATH=src python scripts/build_webinstruct.py` in the
[Tasksource repository](https://github.com/sileod/tasksource).
Publish with `PYTHONPATH=.:src python scripts/upload_repackaged.py webinstruct`.
`provenance.json` records the pinned source revision and retained counts by split.
The initial deterministic pass retains 19,087 MC and 10,934 binary training rows.

The Hub release retains every parsed option in source order. When loaded through
Tasksource, its default MC preprocessing shuffles options and keeps the gold plus
up to three distractors. Use `gold_first=False, max_options=None` when calling the
MC annotation directly to retain source order and all options. Tasksource also
creates a validation split from the prepared training set using its standard seed.
