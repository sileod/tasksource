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
image-folder dataset. IconQA uses a prepared `tasksource/iconqa-text` mirror and keeps
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

## Prepared source mirrors

Source-specific builders live in [scripts/repackage_dataset/](../../scripts/repackage_dataset/).
Run `python -m scripts.repackage_dataset ai2d iconqa_text mind2web` to reproduce
and publish the mirrors; `--dry-run` builds locally. The existing
`scripts/upload_repackaged.py` command remains available.

AI2D and IconQA parse their Cauldron prompts into canonical grouped QAs once,
retaining original image bytes and option order. Mind2Web prepares one eligible
action pool with source IDs, history, geometry and deterministic candidate options.
Its six annotations share the prepared mirror instead of rerunning candidate
validation and screenshot-header inspection on every load. Native evaluation
splits are preserved. Conversion hashes, source revisions, license evidence and
exclusion manifests accompany the mirrors.

SNLI-VE is parked as unsound and removed from the published vision pilot. Its
caption-derived labels do not reliably supervise image entailment; reannotation
of neutral evaluation pairs does not repair this source's training labels. See
[E-ViL](https://openaccess.thecvf.com/content/ICCV2021/papers/Kayser_E-ViL_A_Dataset_and_Benchmark_for_Natural_Language_Explanations_in_ICCV_2021_paper.pdf).


## Native Cauldron tasks

`visual7w`, `tqa`, and `intergps` use explicit native multiple-choice answers.
`clevr/yesno`, `mapqa/yesno`, and `hateful-memes` use named yes/no labels;
`clevr/color`, `clevr/shape`, `clevr/size`, and `clevr/material` use separate,
disjoint attribute vocabularies. They all load the same pinned Cauldron revision
`847a98a779b1652d65111daf20c972dfcd333605` directly, with only its labeled
training split. MC preprocessing rejects malformed options, duplicate options,
and invalid gold letters; it creates no distractors. Metadata retains the native
question and answer, an ordered-image hash in `image_group_id`, and a stable QA ID.

The declarations default to streaming. A capped visual load scans the complete
eligible source split using uniform reservoir sampling; a cap limits memory and
output rows, not download volume. Images stay encoded until accessed.

[QA audit](cauldron-audit.json) records full-source retained and excluded counts,
with reasons. [Native-image smoke results](cauldron-smoke.json) exercise all ten
views through the public loader and Jev recast on the first 32 native image groups
per config, injected at the loader boundary without modifying their records.
| Task | Retained QAs | Excluded QAs | Reason |
|---|---:|---:|---|
| `visual7w` | 69,817 | 0 | None |
| `clevr/yesno` | 282,834 | 417,155 | outside_vocabulary |
| `mapqa/yesno` | 138,519 | 344,897 | outside_vocabulary |
| `tqa` | 6,473 | 9 | duplicate_options |
| `hateful-memes` | 8,500 | 0 | None |
| `clevr/color` | 62,838 | 637,151 | outside_vocabulary |
| `clevr/shape` | 63,152 | 636,837 | outside_vocabulary |
| `clevr/size` | 62,929 | 637,060 | outside_vocabulary |
| `clevr/material` | 62,830 | 637,159 | outside_vocabulary |
| `intergps` | 1,753 | 7 | duplicate_options |

These smoke fixtures are bounded checks, not representative training samples.
TQA and InterGPS additionally passed direct, complete-split Hub loading with
`load_task(id, vision=True, recast='jev', max_rows=10)`.
Reproduce the audit and fixtures with:

```bash
python scripts/audit_cauldron.py --output docs/jev/cauldron-audit.json \
  --samples build/cauldron-samples
python scripts/audit_cauldron.py --smoke-only --samples build/cauldron-samples \
  --output docs/jev/cauldron-smoke.json
```

Original data license evidence is separate from software licensing. CLEVR's
[original release](https://cs.stanford.edu/people/jcjohns/clevr/) states CC BY 4.0;
[MapQA](https://github.com/OSU-slatelab/MapQA#citation) and AllenAI's
[TQA registry entry](https://registry.opendata.aws/allenai-tqa/) state CC BY-SA 4.0.
The MIT licenses on the Visual7W toolkit and InterGPS software do not establish
an image or annotation redistribution license; their data terms remain
`unspecified` in metadata, with original source links and separate code-license evidence.

HatefulMemes remains a direct Cauldron mapping with its native prompt and answers.
Its [original dataset agreement](https://huggingface.co/datasets/emily49/hateful-memes/blob/390eaf2f1a31eed27275b49c9bafcfc8ae721733/LICENSE.txt)
is recorded as custom terms in metadata, with `source_redistribution: restricted`.
The Apache license on supplementary annotation code does not cover the images.
No Tasksource image mirror is published for it. The vision publisher rejects
rows marked restricted before making any upload; omit `hateful-memes` when
building a public vision config. Loading this direct mapping does not replace
the original agreement or establish permission for Cauldron's redistribution.

## Counting, spatial relations, and widget grounding

`clevr/count`, `tallyqa/count`, and `vsr/yesno` use the existing pinned Cauldron
release and shared closed-answer preprocessing. Counts use explicit vocabularies
(CLEVR 0–10; TallyQA 0–15). They become Jev `choice` decisions: neither vocabulary
fits Jev's current 2–10-level `score` constraint. VSR uses explicit No/Yes labels.
Only Cauldron's native training split is included; evaluation splits are not invented.
TallyQA data licensing remains unspecified, with the original project recorded in
metadata. VSR records its project's Apache 2.0 terms and notes that individual
COCO image licenses still apply.

`rico-widget/grid7` directly uses the packaged RICO Widget Captioning source at
`6ec57b56bebd722b9c646c78d0f34e1199b6d7a9`. Each nonempty human caption yields a
49-class decision at the normalized widget-box center. Invalid boxes and missing
images are excluded. Encoded screenshots are preserved; no crop or resize runs
in task preprocessing. Native `val` is exposed as `validation`. Screen, widget,
and caption identities remain separate in metadata alongside normalized boxes
and target centers. The source Parquet columns are projected to screenshot,
caption, box, and screen ID, omitting semantic renderings, icons, and DOM trees.
The full-source screen overlap audit is reproducible with:

```bash
PYTHONPATH=src python scripts/audit_rico_widget.py --output docs/jev/rico-widget-audit.json
PYTHONPATH=src python scripts/audit_cauldron.py \
  --tasks clevr/count tallyqa/count vsr/yesno \
  --samples build/cauldron-samples --output docs/jev/visual-expansion-audit.json
PYTHONPATH=src python scripts/audit_cauldron.py --smoke-only \
  --tasks clevr/count tallyqa/count vsr/yesno \
  --samples build/cauldron-samples --output docs/jev/visual-expansion-smoke.json
```

### Refinement views need repackaging

`mind2web/grid7/refine` and `rico-widget/grid7/refine` are deferred: screenshot
pixel decoding, exact parent-cell cropping, and crop encoding belong in
`scripts/repackage_dataset/vision.py`, followed by separate pinned data-only
mirrors. Loading either existing stage-1 task should not regenerate crops or
increase its download size. Native crop resolution and lossless encoding are
preferred; model adapters handle resizing. A future builder must record its
configuration/code hashes, actual integer pixel crop rectangles, original
geometry, source revision, and both original action/widget and stage-specific
row identities. Labels must refer to the displayed crop. Integer pixel rounding
must be included in coordinate round-trip tests, since pixel crop boundaries
need not coincide exactly with ideal normalized seventh boundaries.

Stage-1 and stage-2 images belong in separate Jev requests. Refinement data stays
out of the published training mix until experiments compare single-pass,
recursive stage-1-only, and refinement-trained decoding by target-box hit rate,
including the oracle-parent-cell upper bound. No model experiment or accuracy
improvement is claimed by these task additions.

The full RICO screen audit found 41,221 native train widget rows on 14,878 screens,
3,483 validation rows on 1,292 screens, and 3,621 test rows on 1,265 screens.
No `screenId` crosses those split boundaries. See
[rico-widget-audit.json](rico-widget-audit.json) and the bounded native-image
[smoke](rico-widget-smoke.json). Three recast train examples per new family were
opened for a coarse visual pass. Counting and RICO golds were coherent with the
images. VSR showed a source-caption naming issue: “banana is on the orange” has
a native Yes label although the depicted sliced citrus appears to be a lemon.
That native answer is preserved; correct recasting does not establish that all
VSR source labels or object names are correct. These additions have not rebuilt
the published Jev vision config.

RICO caption eligibility retained 109,359 training, 9,416 validation, and 9,794
test descriptions; one empty training caption was excluded. No invalid boxes
were found. All 49-cell label distributions are in the audit JSON. The smoke
loads retain three recast rows per native split, with image decoding and gold
criterion checks. Original widget-caption terms are
[CC BY 4.0](https://github.com/google-research-datasets/widget-caption).

The full pinned Cauldron [answer audit](visual-expansion-audit.json) records:

| Task | Retained training QAs | Excluded QAs | Reason |
|---|---:|---:|---|
| `clevr/count` | 165,406 | 534,583 | Other CLEVR answer vocabularies |
| `tallyqa/count` | 183,986 | 0 | All native answers are in 0–15 |
| `vsr/yesno` | 3,354 | 0 | All native answers are yes/no |

Per-label distributions are retained in the JSON. All three views passed
[native-image loader/Jev smoke checks](visual-expansion-smoke.json); the
[coarse visual review](visual-expansion-quality.json) records source-quality
observations separately from recast correctness.

## Grounding and Set-of-Mark

Grounding uses the existing visual templates. `tasksource.grounding.grounding_row`
projects an image, instruction and normalized `xyxy` target box to grid labels, or
to MC when supplied real candidate boxes and descriptions. MC requires 2–26
unique boxes and exactly one matching target; no target box is inserted and no
negative boxes are invented. Geometry is relative to the displayed image, so a
separate refinement mirror can supply crop-relative boxes with original geometry
kept in metadata.

```python
from tasksource import load_task, list_tasks

excluded = ["osunlp/Multimodal-Mind2Web", "tasksource/multimodal-mind2web"]
tasks = list_tasks(vision=True, excluded_sources=excluded)
ds = load_task(
    "rico-widget/element", vision=True, recast="jev", max_rows=1000,
    max_rows_eval=100, seed=42, excluded_sources=excluded,
    grounding={"probabilities": {"plain": 0.5, "som+text": 0.25, "som-only": 0.25}},
)
```

Augmentation runs only on sampled rows. `plain` retains encoded images and native
candidate descriptions with candidate geometry. `som+text` adds numbered boxes
and keeps the descriptions; `som-only` keeps the instruction and uses `Mark N`
criteria. All marks have identical styling. Stable mark IDs are content, distinct
from Jev's shuffled option indices; the gold criterion continues to name the same
mark. Configure `mark_size` (fraction of the shorter image dimension) and
`line_width` in `grounding`. Metadata records weights, seed, source identity,
normalized candidate geometry, original image hash, rendering version and Pillow
version. Images and metadata stay separate from the rendered textual request.

The loader rejects explicitly excluded sources before downloading and checks
cross-split screenshot hashes and trajectory/image-group IDs before augmentation.
The publisher also checks original image hashes when variants change the pixels.
For a candidate-only release, the builder accepts `--excluded-sources`,
`--grounding-probabilities '{"plain":0.5,"som+text":0.25,"som-only":0.25}'`
and `--seed`. Use that option only with grounding tasks providing candidate boxes.

RICO uses native semantic leaf boxes, normalized by their annotation canvas. The
captioned target must already occur in those candidates. Native splits have a
previous full screen-ID disjointness audit; the candidate pilot and three visual
checks are recorded in [rico-grounding-audit.json](rico-grounding-audit.json).
AndroidControl includes screenshot-aligned serialized accessibility trees, but
requires a TFRecord/protobuf conversion before it is usable as a canonical Hub
source. That belongs under `scripts/repackage_dataset/`, with episode-level native
splits, candidate validity checks, and original revisions/terms recorded; it is
not yet registered. [Official AndroidControl format](https://github.com/google-research/google-research/blob/master/android_control/README.md).
