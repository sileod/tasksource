# English image multiple-choice prospects — October 13, 2026

**Discovery only. No prospects were implemented, repackaged or published as training data.**

The complete English-tagged, image-tagged HF multiple-choice listing contained
191 repositories. Six matched the current catalog/history; 185 were screened.
Nine passed automated training/resource/decision-fit checks, 104 need more evidence,
and 72 were judged unsuitable. No model requests failed. The broader unfiltered
image-MC inventory contained 218 repositories; this pass specifically uses English tags.

Only one of the nine also passed the human/procedural or independently verified
synthetic-label filter. A pipeline spot check put that source on hold: its previewed
MC configuration is evaluation-oriented, despite the Viewer calling its split `train`.
**These are follow-up leads, not nine sound training datasets.**

| Source | Potential workload | Provenance screening / follow-up |
|---|---|---|
| [ShipBench](https://huggingface.co/datasets/Anonymous1383/ship-dataset) | Procedural engineering-drawing decisions | Procedural filter passes; hold until MC/classification labels are confirmed in the separate `capability_decomp_train` corpus. Other configs include evaluation rows exposed as `train`. |
| [JEE questions — eQOURSE](https://huggingface.co/datasets/eQOURSE/jee-main-questions) | Exam MCQA | Label origin unresolved. Check original exams, training intent and image-bearing option rows. |
| [JEE questions — Grass-G](https://huggingface.co/datasets/Grass-G/jee-main-questions) | Exam MCQA | Label origin unresolved; likely overlaps the preceding mirror. These are two repositories, not established distinct datasets. |
| [INOI](https://huggingface.co/datasets/combviz/inoi) | Visual competition questions | Label origin unresolved; verify native options and designated training use. |
| [GeoGuessr MCQA](https://huggingface.co/datasets/raphaell/geoguesserMCQA) | Image-based location selection | Label origin unresolved; verify location ground truth, option construction and geographic splits. |
| [PMC-VQA-2](https://huggingface.co/datasets/CAIR-M3LLM/PMC-VQA-2) | Medical visual decisions | Label origin unresolved; identify original supervision and whether decisions actually require the image. |
| [Dermatology40k](https://huggingface.co/datasets/CAIR-M3LLM/Dermatology40k) | Dermatology visual decisions | Label origin unresolved; verify source diagnoses, image rights and patient-level partitioning. |
| [MedThinkVQA](https://huggingface.co/datasets/CAIR-M3LLM/MedThinkVQA) | Medical visual decisions | Label origin unresolved; separate native gold labels from generated reasoning. |
| [RSRCC](https://huggingface.co/datasets/google/RSRCC) | Visual relation decisions | Model-derived label judgment; withheld because independent verification covering retained labels is not established. |

The live identity check includes English, multilingual, soft-label and vision catalogs,
original repositories behind mirrors, parked/evaluation history and the curated queue.
The vision GitHub/paper index was refreshed from **38 source cards covering 57 task
views**, including recent additions; Semantic Scholar linking reported no request errors.
The MC inventory was matched again against that refreshed index. Shared papers and
GitHub links are overlap hints, not proof that two datasets are identical. Untagged
sources, undocumented mirrors and incomplete source cards remain limitations.

Screening uses the existing native Jev workflow, English dataset-language checks,
fresh reviewer controls and saved metadata/row previews. Discovery does not establish
label accuracy or split disjointness. The ShipBench audit demonstrates why a named
training split must not be equated with a suitable training configuration. The
training-config preview follow-up hit a Viewer rate limit; that uncertainty remains
explicit rather than being interpreted as rejection or approval.

[Review records](2026-10-13-vision-multiple-choice.jsonl) preserve eligibility and
provenance distributions, revisions and evidence hashes without publishing raw rows
or source-card bodies. The [manifest](2026-10-13-vision-multiple-choice.manifest.json)
records inventory, screening and refreshed-index hashes.

The detached queue continues with **English image VQA → image classification →
the pinned Papers with Code image pass**. This page is the completed MC snapshot,
not a claim that those later stages have finished. See the
[replay commands](../../src/tasksource/task_discovery/README.md#english-image-discovery-queue).
