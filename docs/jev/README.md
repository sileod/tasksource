# tasksource-jev-typed-decisions build

[`scripts/build_jev_dataset.py`](../../scripts/build_jev_dataset.py) builds and publishes
[tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions)
from the Tasksource catalogs and the [procedural sources](../../dataset_cards/procedural-typed-decisions.md).
Recasts, token labels, procedural generators and augmentations live in [`src/tasksource/jev/`](../../src/tasksource/jev/);
the release card is [`dataset_cards/tasksource-jev.md`](../../dataset_cards/tasksource-jev.md).

- [Label audit](label-audit.md): screening published examples for wrong gold labels.
- [Source notes](sources.md) and [specialized text sources](specialized-text.md): per-source mappings, filtering and provenance.
- [Audit records](audits/): machine-readable checks of source batches.

## Adding a source

Declare the task in the catalog (`src/tasksource/tasks.py`, or a mirror under
`scripts/repackage_dataset/` first when loading needs repairs). The next build includes it,
pins its Hub revision, reads its license and gives it a provisional mix score; nothing else is manual.

## Build and publish

```bash
PYTHONPATH=.:src python scripts/build_jev_dataset.py --output build/tasksource-jev-typed-decisions \
  --max-rows 30000 --max-rows-eval 3000 --publish-rows 2500000 --eval-rows 15000 --procedural-share 0.1 \
  --skip-migrate --finalize --upload
```

The build writes one Parquet shard per task and split and records every attempt in `build-report.jsonl`.
Rerunning resumes: completed tasks are kept, failed and new ones are built. To rebuild a task after
changing its mapping, append `{"task": ..., "status": "invalidated"}` to `build-report.jsonl`.

Publication balances labels, drops bad examples (`src/tasksource/metadata/jev_bad_examples.csv`),
caps sources, removes evaluation rows overlapping train, and pushes `full` (every row that passed) and
`default` (the steered mix of `src/tasksource/metadata/jev_mixes.py`). Rows are ordered by a hash of their
`example_id`, so row-level annotations survive rebuilds. `build-manifest.json`, `release-audit.json`,
`failed-tasks.json` and `sources.yaml` (source revisions and licenses) are published with the data.

Each publish is a release: add a `## tasksource-jev-typed-decisions vYYYY.MM.DD[.n]` entry to
[`DATA_CHANGELOG.md`](../../DATA_CHANGELOG.md) first; the build publishes the dataset's entries and tags the Hub revision. Never pass `--upload` to a partial build (`--tasks`): it would replace the release.

To add new sources without republishing, append them to `full` (text) or `vision`:

```bash
python -m scripts.append_jev --config full --tasks TASK ... --push --release vYYYY.MM.DD.n
```

The append builds the sources with this builder, applies the release row pipeline (bad examples, label
balance, prompt styling, licenses) and drops rows already published in train or evaluation. Existing
shards stay byte-identical; `default` is re-mixed at the next full rebuild. Changing a published source
needs a full rebuild.

Use `preserve_choices=True` on multiple-choice annotations that need every native option. This also keeps source order and pads shorter rows when loading the ordinary Tasksource view.

Smoke test, without publishing:

```bash
PYTHONPATH=.:src python scripts/build_jev_dataset.py --output build/jev-smoke --tasks glue/rte \
  --max-rows 50 --max-rows-eval 20 --finalize
```

## Task-specific prompts

Most Jev rows phrase their decision with a few generic instructions over generic fields (`text_A`/`text_B`).
[`src/tasksource/jev/task_prompts.json`](../../src/tasksource/jev/task_prompts.json) adds, per source, natural field
names and reviewed task-specific instructions tagged by style (`directive`, `question`, `explained`, `terse`, `conversational`, `annotation`,
`grounded`, `system`) plus label definitions for fixed label sets. `task_style` renders a row with them, inline after
the instruction or as a JSON state, and `describe_options` renders options as `label: definition`.
Consumers choose the styles, layouts and share of rows, so phrasings can be compared in controlled runs. The published
data is unchanged.

Regenerate after a release that adds sources (any OpenAI-compatible endpoint, see `tasksource.llm`):

```bash
PYTHONPATH=../litlm:.:src python scripts/build_jev_task_prompts.py --data <dir of train-*.parquet> --revision <sha>
```

An LLM drafts the prompts from a seeded sample of published rows; a review pass renders real rows with each
instruction and keeps only those that ask what the gold option answers, fit every option, match their style and do not
leak the answer. The sidecar records the data revision, model and prompt hash; `build/jev-task-prompts/reviews.jsonl`
keeps every verdict.

## Vision

`--vision` builds the multimodal `vision` config. `--license-use commercial` excludes unresolved and
non-commercial source families. New visual tasks are added to the published training split without
resampling existing rows:

```bash
python -m scripts.append_jev --config vision --tasks TASK ... --cap 30000 --push --release vYYYY.MM.DD.n
```

Existing training and evaluation files stay byte-identical; evaluation images and duplicate option
permutations are excluded from additions. FigureQA is not expanded: its original terms prohibit redistribution.
