# Static procedural decisions

These builders export ordinary grouped visual MC data: `images`, `image_group_id`,
and `qa` records containing `inputs`, `choices_list`, integer `labels` and JSON
`metadata`. They use the existing upload entry point and visual templates.
No generator dependencies are added to Tasksource's normal installation.

| Builder | Original source | Construction |
|---|---|---|
| `tron-mcq` | [TRON](https://github.com/YangTianze009/TRON) | Freeze seeded generator/verifier instances. Keep native explicit text choices; otherwise draw answers from other generations of the same environment and reject answer-equivalent distractors with the current instance's verifier. Opaque image-only answer letters are excluded. |
| `trace-mcq` | [TRACE Hub release](https://huggingface.co/datasets/maveryn/trace) / [code](https://github.com/maveryn/trace) | Reuse native train/validation images and typed answers. Same-task, same-answer-type training answers supply distractors; the pinned exact-JSON scorer checks every option. Preserve the original image bytes. |
| `mentis-oculi-paper-fold` | [MentisOculi](https://github.com/Jana-Z/mentis-oculi) | Generate fresh paper-fold instances, retain the complete source question/options panel, and recheck distinguishability of all five hole patterns after final unfolding. No published benchmark rows are ingested. |

## Build

TRON needs its documented optional `matplotlib`, `numpy` and Pillow dependencies.
TRACE's source verifier uses its declared package dependencies; its generator
code is Apache-2.0, while the released dataset is CC BY 4.0. MentisOculi declares
Python 3.13 and its own plotting/geometry dependencies. Install these in an isolated
environment; pinned code and source hashes are recorded by the builders.

```bash
# TASKSOURCE_GENERATOR_CACHE selects a fast scratch filesystem for source checkouts.
TASKSOURCE_GENERATOR_CACHE=/tmp python scripts/upload_repackaged.py tron-mcq \
  --dry-run --max-rows 100 --max-rows-eval 20 \
  --generator-task aquarium_grid --generator-task angle_bisector \
  --generator-task bar_chart_aggregate --generator-task before_after \
  --generator-task overlapping_shape_count
TASKSOURCE_GENERATOR_CACHE=/tmp python scripts/upload_repackaged.py trace-mcq \
  --dry-run --max-rows 5000 --max-rows-eval 100
TASKSOURCE_GENERATOR_CACHE=/tmp python scripts/upload_repackaged.py mentis-oculi-paper-fold \
  --dry-run --max-rows 20 --max-rows-eval 5
```

Without `--generator-task`, TRON attempts its complete registered environment set;
this can be expensive. The initial validated build uses the five environments above,
not all 520. Row arguments are candidate caps: unsupported formats, failed generators,
insufficient distractor diversity and ambiguous choices can reduce retained counts.

Builds write Parquet data, provenance and exclusions under `build/<builder>-release/`.
Publish the reviewed split data without regenerating examples:

```bash
python scripts/upload_repackaged.py trace-mcq --prepared build/trace-mcq-release
```

`--prepared` verifies the mirror identity and all split checksums from provenance
before uploading. Keep source revisions and configuration fixed between validation
and publication. `--card-only` updates only documentation for an existing mirror.

## Guarantees and limits

All emitted choices are unique and have exactly one accepted gold under the
recorded verifier. TRON numeric pools exclude MC-letter answers from other modes;
native option letters are replaced with their actual text. TRACE removes only
recognized output-contract blocks and JSON demonstrations, preserving answer units.
Single-letter image labels are displayed as `Label A`, etc., so Jev option letters
cannot be confused with labels in the source image. Toggle-grid questions explicitly
state the transition rule from the pinned generator; it was absent from the native prompt.

Images are grouped without byte duplication. Held-out images take priority over
training copies. TRACE also excludes copies of **all** native validation images,
including validation examples outside the exported sample. Fresh generated splits
use different seed namespaces; this does not establish generalization to unseen
puzzle families. MentisOculi additionally checks any source example images present
in the pinned repository. These checks cannot cover unpublished benchmark instances.

Verifier correctness does not establish that every image/question is unambiguous.
Visual spot checks remain necessary, especially for generator tasks outside the
initial reviewed scope. Original prompts, seeds, donor IDs, verifier results and
geometry stay in metadata, separate from model inputs. Licenses are source-specific:
TRON states research-use terms; TRACE includes its original asset credits; MentisOculi
paper-fold uses its Apache-licensed procedural shapes rather than photographic assets.

## Larger generated releases

The [scale runner](scale.py) generates bounded chunks in parallel, resumes checksummed
chunks, groups shared images, removes repeated image/question/gold combinations and
keeps held-out images out of training. It stops at the requested number of distinct,
verified training QAs rather than padding a candidate cap. Use a fast scratch disk.

```bash
TASKSOURCE_GENERATOR_CACHE=/tmp python -m scripts.repackage_dataset.procedural.scale \
  tron-mcq --output /tmp/tasksource-procedural/tron-100k --train 100000 --workers 8 --push
# Run in MentisOculi's isolated Python 3.13 environment:
TASKSOURCE_GENERATOR_CACHE=/tmp python -m scripts.repackage_dataset.procedural.scale \
  mentis-oculi-paper-fold --output /tmp/tasksource-procedural/mentis-100k \
  --train 100000 --workers 8 --push
# TRACE already has native static data: use all eligible training rows, not fresh duplicates.
python scripts/upload_repackaged.py trace-mcq --max-rows 100000 --max-rows-eval 2000
```

Each generated release includes per-chunk seeds and exclusions in
`generation-checkpoints.json`; `status.json` in the output directory reports progress
and an observed ETA. Resume with exactly the same source, seed, cap and chunk size.
Changing worker count is safe. A candidate-budget failure leaves checkpoints intact
and does not publish an undersized release. Add the source release to
`DATA_CHANGELOG.md` before using `--push`.

### TRACE at 100k

`trace-expanded-mcq` retains the pinned native mirror and adds fresh verified
training generations to reach 100,000 rows. It uses the original generator API,
keeps the native validation partition and rejects decoded-pixel validation copies.
Every generated instance is cached with an image checksum; the recipe cache key
includes source/helper hashes, the native release, seed and rendering versions.

```bash
TASKSOURCE_GENERATOR_CACHE=/tmp python scripts/upload_repackaged.py trace-expanded-mcq \
  --max-rows 100000 --max-rows-eval 2000 --workers 8
```

For a local native base, set `TASKSOURCE_TRACE_BASE` to its prepared release folder.
The builder checks it against the pinned Hub release's provenance before using it.
The [builder](trace_generate.py) otherwise downloads that immutable mirror through
the normal Hub client. Native images and labels remain intact; recognized JSON
output directives are removed from the question text. Original prompts stay in metadata.

### SPHINX and VisGym

The [static suite builders](suites.py) add two narrowly scoped sources using the
same grouped mirror format. SPHINX covers pie ordering, transformation inference,
transformation result selection and shape counting. Native choices refer to the
same scene; count alternatives are within three of gold. SPHINX's constructive
labels and native image-distinctness guards are retained, rather than claiming an
independent semantic verifier. Code is MIT; shipped icons are CC BY 4.0.

VisGym exports fresh 2D Wilson mazes at sizes 11, 15 and 19. A reachable intermediate
state supplies a unique shortest-path next move from Right/Up/Left/Down. A native
solver and a separate distance calculation check the decision; tied and trivial
states are excluded. This scope uses no photographic, MuJoCo or benchmark assets.

Use an isolated environment: SPHINX currently needs Pillow 11.3.0 because its
renderer calls the removed `ImageMath.eval` API. VisGym uses its pinned Gymnasium
fork, plus pygame. Set `SDL_VIDEODRIVER=dummy` on headless machines. The source
checkouts are added only inside builders; normal Tasksource installs are unchanged.

```bash
TASKSOURCE_GENERATOR_CACHE=/tmp SDL_VIDEODRIVER=dummy python -m scripts.repackage_dataset.procedural.scale \
  sphinx-mcq --output /tmp/tasksource-procedural/sphinx-100k --train 100000 --workers 8 --push
TASKSOURCE_GENERATOR_CACHE=/tmp SDL_VIDEODRIVER=dummy python -m scripts.repackage_dataset.procedural.scale \
  visgym-mcq --output /tmp/tasksource-procedural/visgym-100k --train 100000 --workers 8 --push
```

Gym-V is excluded at the user's request. VisualSphinx already has a direct
`visualsphinx` task using its curated RL-20K release. Its fresh-generation pipeline
requires LLM code synthesis and verification; it is not this deterministic static
export path. The unfiltered raw release is not substituted for curated training
merely to reach a row target.

## Other static VisGym views

[The shared builders](visgym.py) add `visgym/equation`, `visgym/rotation` and
`visgym/block-legality` using the same visual MC template. Equation distractors
are true equations requiring more than one match transfer; rotation alternatives
differ by 15 or 30 degrees; block alternatives collide despite staying in bounds.
Native verifiers check every choice, with independent segment/occupancy checks
for equations and blocks. These are fresh procedural images, not source benchmarks.

```bash
TASKSOURCE_GENERATOR_CACHE=/tmp SDL_VIDEODRIVER=dummy \
  python -m scripts.repackage_dataset.procedural.scale visgym-rotation-mcq \
  --output build/visgym-rotation-100k --train 100000 --eval 100 --workers 8 --push
```

Replace the builder with `visgym-equation-mcq` or `visgym-block-mcq` for the other
views. Use an isolated VisGym generator environment as above. Source/recipe hashes,
seeds, geometry and exclusion counts accompany each release. The scale command
fails rather than padding or publishing an undersized target. Gym-V is excluded.
