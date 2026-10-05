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
- config_name: mc-unfiltered
  data_files:
  - split: train
    path: mc-unfiltered/train.parquet
  - split: test
    path: mc-unfiltered/test.parquet
- config_name: binary-unfiltered
  data_files:
  - split: train
    path: binary-unfiltered/train.parquet
  - split: test
    path: binary-unfiltered/test.parquet
---

# WebInstruct: multiple choice and binary

Prepared subsets of [TIGER-Lab/WebInstruct-verified](https://huggingface.co/datasets/TIGER-Lab/WebInstruct-verified),
source revision `3e8a350b3a935d68fe70bdf379692500abc9ff51`.
The original `train` and `test` assignments are preserved; `train_legacy` is omitted.
Original fields (`id`, `question`, `answer`, `answer_type`, `category`, `difficulty`) are retained.
The source answers are used as supplied, without checking their factual correctness.

`mc-unfiltered` and `binary-unfiltered` are the deterministic parsed baseline described below.
`mc` and `binary` apply confirmed presentation exclusions and verified source-grounded repairs.
Uncertain judgments and unrepaired examples remain; this is a model-assisted presentation audit,
not a guarantee of factual correctness or complete context. Original source questions and answers
are retained. Tasksource uses the canonical `prompt` in both filtered configs.

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
than assigned a guessed label. Questions and source answers are preserved verbatim;
`prompt` initially copies `question`, and only verified presentation repairs change it.

## Reproduction and Tasksource

Run `PYTHONPATH=src python scripts/build_webinstruct.py` in the
[Tasksource repository](https://github.com/sileod/tasksource).
Publish with `PYTHONPATH=.:src python scripts/upload_repackaged.py webinstruct`.
That command prepares the unfiltered configs. To include filtered configs, pass
`--bad-examples PATH` and optionally `--repairs PATH` from the audit pipeline.
`provenance.json` records the pinned source revision and retained counts by split.
The initial deterministic pass retains 19,087 MC and 10,934 binary training rows.

## Presentation audit

Run `scripts/audit_webinstruct.py --output build/webinstruct-presentation-audit-v2`,
then the same command with `--stage confirm`, then `--stage repair`.
Requests use litlm with four interchangeable API keys, per-key pacing, resumable
checkpoints, and `deepseek-v4-flash-0731`. Screening checks the complete original
question against the parsed prompt and options; it checks essential missing context,
referenced figures and target spans, broken option boundaries, and binary casting.
Ordinary domain knowledge and harmless markup are allowed. Source answers are not re-solved.
The prompt also distinguishes checking arithmetic from validating absent setup or rules.

Screening is batched; flags are challenged individually before removal. Confirmation
uses the same model and is not an independent correctness assessment. Repairs use
only source information, retain labels and option order, and undergo source checks
and a presentation recheck. Unvalidated edits are discarded. The source questions
remain available for comparison. `bad-examples.jsonl`, `repairs.jsonl`, and
`provenance.json` record the release decisions, counts, and manifest hashes.

The Hub release retains every parsed option in source order. When loaded through
Tasksource, its default MC preprocessing shuffles options and keeps the gold plus
up to three distractors. Use `gold_first=False, max_options=None` when calling the
MC annotation directly to retain source order and all options. Tasksource also
creates a validation split from the prepared training set using its standard seed.
