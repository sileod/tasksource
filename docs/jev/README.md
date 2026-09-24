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
