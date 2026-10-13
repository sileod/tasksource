# Dataset prospects

Potential additions to Tasksource, separate from the implemented [English](../catalog_english.md),
[multilingual](../catalog_multilingual.md) and [vision](../catalog_vision.md) catalogs.
These are screening results, **not approved training sources or completed integrations**.

## Published screening results

- [October 13 exhaustive HF category pass](prospects/2026-10-13.md): 142 shortlisted
  sources from 11,776 screened candidates; includes downloadable review records.
- [October 12 exhaustive HF multiple-choice pass](prospects/2026-10-12-multiple-choice.md):
  9 eligibility passes, including 2 provenance-filter passes, from 208 screened sources.
- October 12 mixed-source pass: the 15 candidates summarized below.

- [October 13 OpenAlex general/medical/legal passes](prospects/2026-10-13-openalex.md):
  73 paper leads reviewed; no confirmed English shortlist additions.

- [October 13 HF Papers pilot](prospects/2026-10-13-hf-papers.md): 21 paper leads
  reviewed; 1 eligibility pass, withheld for unresolved provenance.

## October 12 snapshot

Snapshot: October 12, 2026. A 5,000-candidate discovery pass produced 42 eligibility
passes; 15 remained after annotation-provenance screening. All 15 appear below,
including candidates subsequently flagged for overlap or other holds. The order
reflects follow-up priority rather than a measured quality ranking.

## Inspect next

Workloads and label origins below are provisional readings of source metadata and
screening results. Each needs example inspection, license review and split checks.

| Source | Potential decision workload | Label origin indicated by screening | Next check |
|---|---|---|---|
| [DrugProt](https://huggingface.co/datasets/OpenMed/drugprot-parquet) | Chemical–protein relation classification | Human | Verify marked entity pairs, negative-label policy and original partitions. |
| [HumAID event type](https://huggingface.co/datasets/QCRI/HumAID-event-type) | Disaster event-type classification | Human | Check target definitions and event-level split overlap. |
| [FABSA](https://huggingface.co/datasets/jordiclive/FABSA) | Aspect-conditioned sentiment | Human | Verify aspect grounding and faithful labels in the converted source. |
| [Steam review constructiveness](https://huggingface.co/datasets/abullard1/steam-reviews-constructiveness-binary-label-annotations-1.5k) | Constructive / nonconstructive review | Human | Inspect annotation rubric, agreement and game/review overlap. |
| [POPIA compliance NLI](https://huggingface.co/datasets/labrat-aiko/popia-compliance-nli) | Legal compliance entailment | Human | Check author expertise, clause context and example diversity. |
| [US bank transaction categories](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories) | Transaction category | Procedural | Inspect label derivation, newer releases and merchant/template leakage. |
| [CFIR stability/intervention](https://huggingface.co/datasets/ClarusC64/cfir-stability-intervention-geometry-v0.1) | System stability / intervention decisions | Procedural | Verify generator correctness, input sufficiency and designated training use. |
| [Trilemma of Truth](https://huggingface.co/datasets/carlomarxx/trilemma-of-truth) | True / false / neither | Human or procedural; reviewers differ | Resolve label semantics and provenance before choosing a projection. |
| [Twitter misinformation](https://huggingface.co/datasets/roupenminassian/twitter-misinformation) | Misinformation-related classification | Derived/procedural | Check constituent datasets, label compatibility and existing coverage. |

## Overlap and other holds

Passing automated eligibility does not override these concerns. Keep these visible
so discovery does not repeatedly present them as straightforward additions.

| Source | Potential workload | Hold / unresolved check |
|---|---|---|
| [Counterfactual SNLI](https://huggingface.co/datasets/sagnikrayc/snli-cf-kaushik) | Human-labeled NLI | Likely overlaps the existing counterfactually augmented SNLI task; establish identity first. |
| [Arena human preference 55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) | Paired response preference | Check overlap with existing Arena sources and their splits. |
| [KEV code verification](https://huggingface.co/datasets/jtatman/kev-code-verify-data) | Execution-derived correctness | HumanEvalPack lineage: benchmark contamination risk; withhold from training pending overlap checks. |
| [AIME router](https://huggingface.co/datasets/ymoslem/AIME-24-25-26-router) | Correctness of generated answers | AIME benchmark lineage: withhold from training pending evaluation-overlap policy. |
| [Manglish NLP](https://huggingface.co/datasets/vexccz/manglish-nlp-dataset) | Sentiment / emotion | Human labels, but code-switched Malay/English; lower priority for English-focused expansion. |
| [HiFi-KPI Lite](https://huggingface.co/datasets/AAU-NLP/hifi-kpi-lite) | Financial KPI entity classification | Expert labels, but evaluation-oriented documentation; establish an intended training partition. |

## How this list is produced

The [discovery tools](../src/tasksource/task_discovery/README.md) retrieve English-focused
HF/Papers with Code candidates and compare names and source identities against
implemented tasks, recorded exclusions and the internal curated queue. GitHub and
paper matches flag possible overlap; identity coverage remains incomplete.

Screening checks training-split metadata and a small schema/row preview, then reviews
bounded source documentation. For this snapshot, native Jev and Perplexity reviews
both had to assign at least 0.8 to documented training availability and decision
fit. Resource availability requires an observed preview or reviewer support.
Model judgments are not calibrated measures of annotation accuracy.

The provenance policy accepts human or procedural labels, and can also accept
model-derived labels with strong evidence of independent human or executable
verification covering all labels. None of these 15 entered through that exception.
Procedural labels can still accompany generated inputs, and human labels can be noisy.

Saved run artifacts retain source/evidence hashes, review dates, exact returned model
versions, per-model distributions, disagreements and failures. This page is a public
summary, not a dump of the internal implementation queue or raw review inputs.
Deeper processing must confirm answerability, label correctness, novelty, licenses
and leakage before a source enters a training catalog.

The exhaustive English-tagged HF category queue completed on October 13; its
[separate shortlist](prospects/2026-10-13.md) preserves the results without changing
these earlier follow-up notes. These pages are dated snapshots, not live counters.
