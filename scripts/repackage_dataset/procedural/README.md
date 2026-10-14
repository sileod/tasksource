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
