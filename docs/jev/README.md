# tasksource-jev build

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
[`procedural-jev`](../../dataset_cards/procedural-jev.md) first if it should
be included; the default public cap reserves up to 10% for those sources
(`--procedural-share`). The builder excludes BIG-bench, MMLU, and BLiMP and
only enables token tasks whose labels and source rows have been checked.
Per-task defaults are 30,000 train
and 3,000 evaluation source rows (half as many source sequences for token
tasks, which emit up to two decisions each). The builder writes one resumable
Parquet shard per task and split and records every attempt in
`build-report.jsonl`.

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
and shards are preserved.

```bash
PYTHONPATH=.:src python scripts/build_jev_dataset.py --output build/tasksource-jev --finalize
```

Before publishing, inspect `build-summary.json`, `failed-tasks.json`,
`outdated-datasets.json`, and sample rows from every newly migrated source.
Check the `state`, readable `options`, `target`, original split, and any
multiple questions sharing `group_id`. The release cap is 500,000 rows,
allocated 90/5/5 to train/dev/test. It samples complete source-row groups;
the first 1,000 train rows are ordered for source and prompt variety without
changing membership.

Authenticate with the Hugging Face Hub, then publish the checked checkpoint:

```bash
PYTHONPATH=.:src python scripts/build_jev_dataset.py --output build/tasksource-jev \
  --skip-migrate --finalize-only --finalize --upload
```

`--skip-migrate` is appropriate only when the existing shards already use
the current schema. Omit `--finalize-only` to retry failed or newly added
tasks before publishing. The upload writes the dataset card, train/dev/test
Parquet, and build/failure reports to `tasksource/tasksource-jev`; use
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
[`procedural-jev` card](../../dataset_cards/procedural-jev.md). The synthetic
generation pipeline is a package entry point:
`python -m tasksource.jev.synthetic.run --config <config.yaml>`.
