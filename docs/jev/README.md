# tasksource-jev-typed-decisions build

The builders live in [`scripts/`](../../scripts/); canonical recasts, token
label handling, procedural generators, and augmentations live in
[`src/tasksource/jev/`](../../src/tasksource/jev/). The public
`tasksource.recast_jev` and `load_task(..., recast="jev")` APIs are unchanged.
The release card is [`dataset_cards/tasksource-jev.md`](../../dataset_cards/tasksource-jev.md).

## Environment and inputs

Run commands from the repository root with Tasksource's Python dependencies,
`huggingface_hub`, and `pyarrow` installed. Set `PYTHONPATH=.:src` when
working from a checkout. The release was last exercised with
`datasets==4.8.4`, `huggingface_hub==0.36.2`, `pyarrow==21.0.0`,
`numpy==1.26.4`, and `pandas==2.2.3`; these are recorded environment
versions, not a complete lockfile.

The builder reads the current English and multilingual Tasksource catalogs
plus the native graded and procedural Jev sources. Publish
[`procedural-typed-decisions`](../../dataset_cards/procedural-typed-decisions.md) first if it should
be included; the default public cap reserves up to 10% for those sources
(`--procedural-share`); graded sources get 10%, and the remaining rows go 47:30:3 to
classification, multiple choice and token tasks (`FORMAT_SHARES` in the builder), with
family weights from `src/tasksource/metadata/weights.py` inside each format. The builder excludes BIG-bench, MMLU, and BLiMP and
only enables token tasks whose labels and source rows have been checked.
Per-task defaults are 30,000 train
and 3,000 evaluation source rows (half as many source sequences for token
tasks, which emit up to two decisions each). The builder writes one resumable
Parquet shard per task and split and records every attempt in
`build-report.jsonl`.
For source datasets that expose integer labels without `ClassLabel` metadata,
Tasksource annotations can supply a source-verified `label_values` mapping.
The adapter checks every observed value before assigning readable criteria;
it does not infer label meanings from integer order. See
[`migration-notes.md`](migration-notes.md) for the 2026-09-23 repairs and
MetaEval namespace transfer.

## Smoke test

Build a small, non-publishing example in a fresh output directory:

```bash
PYTHONPATH=.:src python scripts/build_jev_dataset.py --output build/jev-smoke \
  --tasks glue/rte --max-rows 50 --max-rows-eval 20 --finalize
PYTHONPATH=.:src pytest -q -c /dev/null tests/test_recast_jev.py
```

Inspect the resulting Parquet rows and `build-report.jsonl`. Do not pass
`--upload` to a smoke test: it would replace the public dataset with a
task-limited release.

## Build and publish

Use a fresh output directory for a new full release. The same command resumes
an interrupted build: successful tasks are skipped, failed tasks are retried,
and shards are preserved. A task is considered complete only if its reported
Parquet shards still exist. Reuse an output directory only with the same code
and settings; use a new directory when preprocessing changes.

```bash
PYTHONPATH=.:src python scripts/build_jev_dataset.py \
  --output build/tasksource-jev-typed-decisions --publish-rows 1000000 \
  --baseline-failures /path/to/previous/outdated-datasets.json --finalize
```

For an unattended build that publishes after automated target/schema/coverage
checks, add `--upload` to that command. A failed task is recorded rather than
silently omitted; `failed-tasks.json` is the complete current failure list,
and `fixed-source-audit.json` compares previously failing task IDs. Native
procedural configs, both enabled token sources, all three Jev primitives, and
all safe augmentation variants are required before a 1M-row production upload.

Before publishing, inspect `build-summary.json`, `failed-tasks.json`,
`outdated-datasets.json`, `fixed-source-audit.json`, and sample rows from
every newly migrated source. The optional `--baseline-failures` argument
compares this build with a prior failure list; `build-manifest.json` records
the selected sources, parameters, code revision, dirty-file list, and package
versions.
Check the `state`, readable `options`, `target`, original split, and any
multiple questions sharing `group_id`. The release caps train at 1,000,000 rows
and dev/test at 15,000 rows each (`--eval-rows`). It balances dataset families, samples
their configs, and keeps complete source-row groups;
the first 1,000 train rows are ordered for source and prompt variety, and the
rest of train is shuffled deterministically. Dev/test source-row groups and packs
whose normalized content (state plus options; each item of a packed state) also
occurs in the published train split are dropped before the eval caps apply;
`release-audit.json` records how many. Publication rejects decisions with
missing, blank, or duplicate options.

Authenticate with the Hugging Face Hub, then publish the checked checkpoint:

```bash
PYTHONPATH=.:src python scripts/build_jev_dataset.py \
  --output build/tasksource-jev-typed-decisions --publish-rows 1000000 \
  --skip-migrate --finalize-only --finalize --upload
```

`--skip-migrate` is appropriate only when the existing shards already use
the current schema. Omit `--finalize-only` to retry failed or newly added
tasks before publishing. The upload writes the dataset card, train/dev/test
Parquet, and build/failure/audit reports to
`tasksource/tasksource-jev-typed-decisions`; use
`--repo-id` for another destination. Verify the remote split counts, source
coverage, exclusions, and several rendered rows after upload.

## Reproducibility boundary

Sampling, augmentation, grouping, and preview order are deterministic for
fixed inputs and code. Upstream dataset repositories and Tasksource
annotations are not revision-pinned here; a fresh build at a later date can
fetch changed data or encounter newly unsupported loaders. Save the Git
commit, environment versions, build report, and upstream revisions when an
exactly repeatable snapshot matters. Existing shards can reproduce the
publication step without refetching upstream sources.

The separate procedural corpus is built by
[`scripts/build_procedural_jev.py`](../../scripts/build_procedural_jev.py);
its schema and provenance are documented in the
[`procedural-typed-decisions` card](../../dataset_cards/procedural-typed-decisions.md). The synthetic
generation pipeline is a package entry point:
`python -m tasksource.jev.synthetic.run --config <config.yaml>`.

## GUI decisions

The visual catalog includes six annotations of the pinned
[`osunlp/Multimodal-Mind2Web`](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web/tree/1b4c6a8cf9f77b7a5e0d641959935c80c4a05889)
source: `mind2web/action`, `mind2web/element`, `mind2web/x10`, `mind2web/y10`,
`mind2web/grid5`, and `mind2web/grid7`. Load them with `vision=True`.
The source's `train`, `test_task`, `test_website`, and `test_domain` splits remain
separate; evaluation splits are never fabricated.

All six views use one eligibility filter and deterministic sampling. An action
needs one explicitly marked original target with a valid box inside the encoded
screenshot, plus at least one valid negative candidate. Parent-only positives,
ambiguous targets, and invalid boxes are excluded. Element decisions keep the gold
and up to 23 negatives, shuffle deterministically, and render every candidate with
the same attributes. Inputs contain the confirmed task and actions strictly before
the current action. Current/future action descriptions and current operation values
are supervision, never input text.

Coordinates are target-box centers, not recorded pointer positions. Bins cover the
full screenshot: x increases rightward and y downward. Axis annotations request
score decisions on all rows through `score_only=True`; other ordinal tasks retain
their existing mixture of score and choice. Fixed grids use 25 or 49 classification
labels, with all labels retained by Jev. Instruction recasting still uses the existing
sampled classification distractors; use Jev for the full grid decision.

```python
from tasksource import load_task, render_typed_decision_group

x = load_task('mind2web/x10', vision=True, recast='jev', max_rows=10, max_rows_eval=5)
y = load_task('mind2web/y10', vision=True, recast='jev', max_rows=10, max_rows_eval=5)
request = render_typed_decision_group([x['train'][0], y['train'][0]])
```

The canonical request keeps ordered `images` alongside `state` and `questions`.
Grouping verifies the state, screenshot sequence, source namespace and action
identity. This representation does not establish that a hosted API accepts those
image objects; the caller must supply its model's image transport. JSON `metadata`
retains trajectory/action IDs, source element ID, original operation, image dimensions,
box, and derived point. Original HTML remains available in the pinned source by ID.
The Hub card reports `openrail`; existing license evidence handling retains that value
without inferring unrestricted reuse.

For comparisons, use the same eligible action IDs and decode an axis bin to its
center `(index + 0.5) / 10`. Audit gold-center point-in-box accuracy before training:
coarse grids can miss small elements on tall screenshots even with a correct label.
Hierarchical crops and GUI execution are outside this initial integration.

A native-source smoke check read the first 32 source rows of each split and validated
10 eligible training rows plus 5 per evaluation split for every annotation. On those
10 training rows, gold-center point-in-box counts were 0/10 for 5×5, 2/10 for 7×7,
and 1/10 for paired 10-bin axes. This small sample checks the mapping and demonstrates
quantization loss; it is not a dataset-wide accuracy estimate.

A local GUI-only pilot can be built with:

```bash
PYTHONPATH=.:src python scripts/build_jev_dataset.py --vision \
  --output build/jev-gui --tasks mind2web/action mind2web/element \
  mind2web/x10 mind2web/y10 mind2web/grid5 mind2web/grid7 \
  --max-rows 1000 --max-rows-eval 100
```

The vision publisher replaces the entire `vision` config. Include every intended
visual source when rebuilding it for publication.

## Additional visual reasoning sources

The existing visual templates also cover `m3cot`, `exams-v`, `visualsphinx`,
`muslr/tfu`, `muslr/mc`, and `iconqa/text`. M3CoT, EXAMS-V and VisualSphinx
load their pinned Parquet releases directly. MuSLR uses its native Hugging Face
image-folder dataset. IconQA reuses the existing Cauldron `iconqa` upload and keeps
only explicit text-choice QAs; image-choice and open-answer QAs are excluded.
MuSLR truth questions have an explicit True/False/Unknown ontology, while its
MC questions retain their source options. Rationales and explanations remain in
JSON metadata and are excluded from model inputs. M3CoT rows missing images are
filtered before sampling. Path-only images are embedded after sampling using
Datasets' file reader, preserving their encoded bytes without decoding pixels.

`view2space/mcq` uses the pinned [tasksource/view2space](https://huggingface.co/datasets/tasksource/view2space) mirror.
VIEW2SPACE's JSON and image archive require preparation. Reproduce its MCQ mirror
with `python scripts/upload_repackaged.py view2space` (`--dry-run` builds locally).
The script pins the original release, reads the original PNG bytes, and groups
QAs by their ordered image set. Its data-only schema is `images`,
`image_group_id`, and `qa`, with canonical `inputs`, `choices_list`, `labels`,
and JSON `metadata` inside each QA. The native ImageFolder ZIP stores each
original PNG once and appends a canonical `metadata.jsonl`; `load_dataset` handles its images without a custom script.
The conversion retains 425,494 training MCQs, input box annotations and source
option order, with reasoning in metadata. It records 506 excluded questions whose
gold answer text appears more than once; repeated distractors are deduplicated
with gold indices remapped. Counting and detection questions are excluded.
No evaluation splits are synthesized. The mirror card and `provenance.json` record the original
release, conversion code hash, included counts and source license.

Capped visual streaming loads use uniform reservoir sampling over the **complete
eligible split**, with deterministic seeds and bounded memory. This removes the
bias toward the first shuffle buffer. It still reads the entire source; use the
prepared mirrors and a local cache for repeated runs. Native source smoke checks
above deliberately inspect small source slices and are schema checks, not uniform
population samples. Text streaming and instruction option shuffling keep their
existing behavior; padding removal applies only to visual instruction recasts.
