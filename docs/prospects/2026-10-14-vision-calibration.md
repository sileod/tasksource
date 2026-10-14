# Vision discovery calibration — 2026-10-14

Replay of the saved English image discovery judgments, without new model calls or task implementation. These are repository leads, with remaining mirror/family duplicates; counts are not unique datasets or approved additions.

| Stage | Reviewed | Strict handoffs | Additional investigation leads |
|---|---:|---:|---:|
| HF multiple-choice | 185 | 1 | 4 |
| HF visual-question-answering | 1,881 | 12 | 42 |
| HF image-classification | 1,910 | 8 | 177 |
| Papers with Code | 349 | 0 | 2 |

The strict handoff keeps the existing 0.8 consensus gates and annotation policy. One former handoff now stays unresolved because its sampled image is null. Investigation requires at least 0.8 support from every reviewer for training and decision fit, at least 0.5 resource and requested-language support, and at least 0.5 support for allowed annotation origins when the stricter origin check fails. Confidently contradicted requirements and known unverified model labels are not recovered. Missing evidence elsewhere remains `needs_evidence`, rather than becoming a rejection or an investigation priority automatically.

[Investigation leads and specific gaps](2026-10-14-vision-calibration.jsonl) are separate from strict handoffs. The [manifest](2026-10-14-vision-calibration.manifest.json) records input and policy hashes. Replay each saved batch using `python -m tasksource.task_discovery.discover_task_prospects BATCH/prospects.jsonl --from-reviews --allow-verified-synthetic --output NEW_DIRECTORY`. Existing snapshots are not overwritten.

## Small card/schema spot-check

This checks pipeline decisions, not image-level answer accuracy. Findings are advisory; they do not alter the authoritative implemented/parked index.

| Source | Finding | Consequence |
|---|---|---|
| [HW1 paper triage](https://huggingface.co/datasets/ishaanamahajan/hw1-paper-triage-multimodal) | 70 original training examples augmented into 10,070; hypothetical paper cards, one narrow reader profile. The profile must be supplied for the judgment to be answerable. | A strict-filter false positive; hold for quality/context review, not a recommended general-purpose training addition. |
| [MiraBest unofficial](https://huggingface.co/datasets/kwazzi-jack/mirabest-radio-astronomy-unofficial) | Sampled Image feature is null; label text in captions does not establish usable visual input. | Removed from strict handoff; inspect actual image distribution before considering it. |
| [GlyphNet](https://huggingface.co/datasets/Akshat4112/Glyphnet) | Procedural homoglyph substitutions with linked genuine/spoofed pairs; rendering font differs from paper. | Plausible specialized task; preserve pair grouping and inspect whether rendering supports the label. Procedural does not mean model-generated. |
| [CrisisMMD](https://huggingface.co/datasets/pranitchawla/crisismmd-mteb) | Native train/validation/test configs and image/text labels are observable. | Credible lead; verify per-config label semantics, source grouping and original coverage. |
| [WorldCuisines](https://huggingface.co/datasets/worldcuisines/vqa) | Source description explicitly reports a 1M training partition; multilingual card includes English. | Useful evidence-gap lead. Verify annotation origin and English subset instead of dismissing it for incomplete metadata. |
| [IQUAD](https://github.com/danielgordon10/thor-iqa-cvpr-2018) | Interactive environment and training code exist; static image decision projection remains uncertain. | Keep unresolved; resource availability alone does not establish a suitable static task. |

Two Cats vs Dogs mirrors also survived the identity screen. The strict handoff is still provisional, and this replay does not fix cross-repository dataset-family identity matching.

Future reviews now ask separately about supervision quality: target-defining context, defensible label derivation, and independent example volume versus repeated augmentation. Documented concerns hold a source out of the strict handoff. Old reviews did not answer this new question and are not retroactively treated as quality approvals. For image-only classification, the English-language question now permits language-neutral images with an explicit vocabulary renderable in English.
