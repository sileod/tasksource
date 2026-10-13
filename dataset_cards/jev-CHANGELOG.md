# Changelog

Each release is tagged on the Hub: `load_dataset("tasksource/tasksource-jev-typed-decisions", revision="v2026.10.13")`.

## v2026.10.13.4

- `source_index` (`split:row` of the original dataset row) on every row, to trace label errors back to their source.
- Stable row order across rebuilds: shuffled by `example_id`, so row-level annotations stay valid.
- Removed after manual review of random rows: `dair-ai/emotion`, CrowdFlower `text_emotion` and IndicGLUE ACTSA (many wrong labels). `app_reviews` drops reviews under four words.
- DiscoSense states include the discourse connective; PropSegmEnt question fixed; PROST removed (test-only).
- New: CEBaB, FABSA and TAB.
- Restores 57 sources missing from v2026.10.13 (MAUD, multidogo, WikiQA, OLID, TabFact and others), now built with the rest instead of appended.

## v2026.10.13.3

- `vision`: GameQA and Valen Game grow to 20,000 training rows each, retaining the original 1,000-row pilots (106,085 vision training rows total). All new Valen rows are English; its preserved bilingual pilot leaves 19,493 English and 507 Chinese rows.
- New rows are shuffled together with seed 0; original shards and evaluation splits are unchanged. GameQA question identities now include the native record to avoid collisions.

## v2026.10.13.1

- `vision`: game decisions, 1,000 training rows each from [GameQA](https://huggingface.co/datasets/OpenMOSS-Team/GameQA-140K) multiple-choice questions and [Valen](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) optimal-action judgments (68,085 vision training rows). Existing rows and evaluation splits are unchanged.

## v2026.10.13

- 18 procedural sources, including `record_eligibility`; procedural rows are label-balanced (at most 60% per answer, yes/no at most a third of questions).
- Label balance: per source and question, the majority label is at most three times the rarest; single-label questions are dropped.
- `default` mix weighs sources by option count, so binary decisions are 15% of rows.
- Known issue: 58 sources from earlier appends are missing (MAUD, multidogo, WikiQA, OLID, TabFact and others).

## Earlier releases

Untagged notes from the dataset card before its cleanup. Counts and policies describe their snapshots.

### Incremental filtered WebInstruct addition

Filtered `tasksource/webinstruct` configs `mc` and `binary`, pinned at `05e8b528d68c628c1af682d47bdb53f85535876a`, are appended as standalone shards to both `default` and `full`. The original train/test split assignment is preserved; validation is unchanged. All choices of included rows are retained, with canonical Jev option permutation and one-hot targets; 47 MC rows with duplicate option text are omitted under the existing choice contract. binary questions use explicit no/false and yes/true choices. No repairs or derived variants are included. Existing shards and mixture sampling are unchanged. See `webinstruct-addition.json` for exact counts, transformations and provenance.


### Reusable selection evidence

The [exclusion evidence resource](quality/filtered-full-v1/README.md) records historical filtered-full selection membership separately from licensing and benchmark policies. Remaining candidates are unattributed exclusions, not verified errors. Canonical decision keys support evidence reuse across recasts.

### Vision config

Load `load_dataset("tasksource/tasksource-jev-typed-decisions", "vision")` for the multimodal pilot (66,085 train, 475 test, 972 validation).
It covers ai2d, aokvqa, bapps/preference, clevr/color, clevr/count, clevr/material, clevr/shape, clevr/size, clevr/yesno, coco/panoptic-region, doclaynet/region, figureqa, intergps, lvis/region, mapqa/yesno, nlvr2, rico-widget/element, rico-widget/grid7, scienceqa-img, spair71k/grid7, superclevr/color, superclevr/count, superclevr/material, superclevr/shape, superclevr/size, superclevr/yesno, tallyqa/count, tqa, visual7w, vsr/yesno, with genuine source labels.
`state`, `question`, `kind`, `options`, and `target` retain their existing decision semantics.
`images` is an ordered sequence of Hugging Face Image features. Source bytes are preserved
unless `metadata.augmentation` records rendering (SoM, region highlighting or source keypoint marks); images decode on access.
Feed the images alongside `state` through your model's image adapter.
NLVR2 retains its two images in source order. Answer options remain text.

`metadata` is a JSON string preserving available source identifiers, pinned provenance,
and separate Hub-card/DPI license evidence. `license` and `license_use` are also available
as columns; missing or conflicting evidence is retained rather than replaced by an inferred license.
See [vision/sources.yaml](vision/sources.yaml), [vision/release-audit.json](vision/release-audit.json),
[vision/build-manifest.json](vision/build-manifest.json), and [vision/quality-audit.json](vision/quality-audit.json)
when the materialized release was audited.

Sampling: full-source uniform reservoirs, shared per-family export caps, with per-task checkpoint provenance. Sampling caps and checkpoint provenance are recorded in the manifests.
Visual task views share one export budget per dataset family, including CLEVR, TallyQA, and RICO;
individual task IDs remain in `source`, and family totals appear in `release-audit.json`.
Only native labeled splits are used. Source action identities group related GUI decisions;
other decisions sharing the same ordered image set and text state share `group_id`.
`metadata.image_group_id` links source image groups across questions and augmentation variants.
Evaluation rows sharing any image with training are excluded.
This initial config contains direct decisions; text packing and derived variants are disabled.
Typed request renderers preserve images; model-specific image transport remains the caller's responsibility.

The 2026-10-09 selective update adds at most 1,000 train rows per new family while preserving the existing rows and caps. Mind2Web remains excluded. HTML-only browser tasks are not part of this config.

### Vision prompt and license correction (2026-10-09)

Known Cauldron answer-style suffixes were removed from 18,353 training rows;
complete original questions remain in metadata. Twelve identical MapQA requests
with identical gold answers were deduplicated. Images and retained gold targets
are unchanged. There are **66,085 train, 972 validation and 475 test** decisions.

Original dataset terms for [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/),
[MapQA](https://github.com/OSU-slatelab/MapQA/blob/main/license), and
[TQA](https://registry.opendata.aws/allenai-tqa/) fill missing Hub-card evidence.
Training license-use counts are **26,987 commercial, 7,000 non-commercial, and
32,098 unspecified**. This remains a **mixed-license research pilot**, not a
commercial-cleared dataset. Unspecified does not grant permission. NLVR2 photo
rights and other unresolved sources remain unresolved; repository software
licenses do not establish image rights. LVIS/COCO classification conservatively
reflects the NC images included in their source mirrors.

Filter `license_use` when loading; new Tasksource vision builds also accept
`--license-use commercial` to exclude unresolved/NC sources before loading them.
See [repair-audit.json](vision/repair-audit.json) and the reproducible
[repair script](vision/repair.py); both record the pinned parent release.

### Component-level license review (2026-10-09)

This review supersedes earlier license-use counts. All sources remain included.
Training: **26,987 commercial, 13,000 non-commercial,
26,098 unresolved complete-row coverage**. Counts describe scoped
license evidence, not legal clearance of each image.

**FigureQA's official archive contains Microsoft Research Open Data License terms:
non-commercial research/testing only; dataset redistribution and standalone hosting
are prohibited.** Its generator's MIT license covers code. The retained FigureQA
rows are explicitly flagged `redistribution: prohibited`; their inclusion here does
not grant redistribution rights. BAPPS training patches inherit Adobe / Adobe-MIT
research image terms; its software BSD license does not clear the images.

NLVR2 annotations and VSR annotation cards specify CC BY 4.0. TallyQA and A-OKVQA
repositories specify Apache 2.0; InterGPS's data/code repository specifies MIT.
These terms do not clear third-party images. COCO per-image grants, NLVR2 photo
rights, imported TallyQA QAs and SPair PASCAL/Flickr terms still need row-level
resolution. No explicit Visual7W annotation grant was found in checked original
project/toolkit documentation. These are documented coverage gaps, not a claim
that every upstream component has no license.

Every reviewed row has `metadata.licenses.license_review` with separate annotation
and image terms, evidence URLs, review status and unresolved reasons. See
[license-review.json](vision/license-review.json), [sources.yaml](vision/sources.yaml)
and [the reproducible update script](vision/update_vision_license_metadata.py).
Images, questions, choices, targets, IDs and source splits are unchanged.

### English task correction and benchmark exclusion (2026-10-09)

CLadder is evaluation-only and removed from `default`, `full` and `filtered-full`.
Missing-item decisions now separate the list and query and explicitly ask about
membership; the previous query was at the end of the state. WikiHow order now
asks which step comes first. QuaRel uses native answer texts as MC options with
permuted, correctly remapped golds, replacing fixed A/B classes. Its old
classification-only packs and obsolete generic instruction-paraphrase variants
were removed. Source examples and splits remain; unrelated tasks and vision
are unchanged. Current counts are in the config metadata. See
[rebuild audit](english-task-rebuild.json) and [reproducible script](english-task-rebuild.py).


### HTML/browser text additions

WebLINX action/DOM-element and WebSRC yes/no/element tasks retain their source partitions. WebLINX named test subsets are combined into the hosted test split, with native split names in row provenance. `mind2web/action` and `mind2web/dom-element` use only the original public training trajectories. Tasksource-derived train/validation/test holdouts keep whole trajectories and identical DOM snapshots together (80/10/10 deterministic group hashing). These are not the official benchmark test partitions. The same partition is used for both annotations. Original annotation/action IDs and page hashes are recorded in [row provenance](additions/167c5ad6fb79/rows.json). Only past actions enter context; four-way choices use native DOM candidates. Direct decisions are appended to default/full; existing shards, vision and filtered-full are unchanged. Alignment with the Decision Index private evaluation rows has not been independently verified.
SEntFiN entity sentiment, WANDS ordered relevance grades and SciRepEval Search click-derived scores are added as direct decisions to default/full. Headline/query groups stay within one split. SEntFiN and WANDS use derived 80/10/10 holdouts; Search uses native train/validation with overlapping queries removed and no benchmark evaluation data. Search numeric scores are interpolated over 0,2,...,14 anchors, preserving their expectation; they are not human mean relevance ratings. RELISH and SciNUP are excluded as evaluation-only. Source code/provenance retain original score semantics and license evidence; Search text licensing remains unspecified.

Source-row metadata: additions/2d27d40ae2fe/rows.json.


Legal, biomedical and structured-text additions use native labels and source partitions; ACORD relevance uses explicit attorney grades. Evidence Inference joins annotations by prompt and article. Complete evidence is retained; oversized requests are excluded rather than answer-biased truncation. Source/license details and counts are recorded in the addition manifest.

Source-row metadata: additions/b3e70284a0c4/rows.json.


### Dialogue and evidence additions

Eleven reviewed decision views from ten sources retain native holdouts and source identities. QuAC contributes answerability only; MultiWOZ contributes dialogue acts and categorical state slots. QASPER requires consistent binary annotations and all cited evidence in the supplied text. SLURP uses native scenario/action intent identities. Repackaged source cards link the original data and reproducible converters. A coarse source/recast review sampled 1–3 examples per task; it does not establish corpus-wide label accuracy. Unresolved source licenses remain explicitly marked.


Source-row metadata: additions/aebc640a1ba7/rows.json.

Coarse model review, sampled row IDs and limitations: [review](additions/aebc640a1ba7/review.json).


Stance, argument preference, discourse and biomedical sentence-role additions retain native judgments and source identities. UKPConvArg uses documented debate-disjoint internal holdouts; FNC-1 and DiscoGeM do not fabricate validation partitions. DiscoGeM targets are native annotator distributions, including disagreement. PubMed RCT preserves PMID-disjoint source partitions and uses original plaintext evaluation files. Unresolved source terms remain marked unspecified. Each mirror card links the original data and exact conversion script.

Source-row metadata: additions/b221677e3faf/rows.json.

Coarse source and final-export review, with sample IDs and limitations: [review](additions/b221677e3faf/review.json).


Human-curated linguistic additions: event triggers and arguments, speaker-reported emotions, hierarchical offensive-language labels, author-intended sarcasm, idiomaticity, annotated coreference, movie preferences, sentiment, SQL decisions, word senses and supersenses. Native source partitions are preserved; source-document overlaps in MAGPIE are removed with holdout precedence. SemCor is training-only. Only native supervision is used, with source terms retained per task. Reproducible packaging is limited to Spider schema joins, the SemCor/WordNet 1.6 join and STREUSLE JSON conversion.

Source-row metadata: additions/a9a3d118186b/rows.json.

Coarse source and exact-export review, sample IDs and limitations: [review](additions/a9a3d118186b/review.json). OLID targeted/target and SemEval 2023 Task 3 are withheld; reasons are recorded in Tasksource.


Spider aggregation was withdrawn from default/full as a low-value standalone projection. `spider/column` remains included. Historical addition manifests describe their original snapshots.
