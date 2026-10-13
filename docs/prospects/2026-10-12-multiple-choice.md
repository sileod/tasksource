# Multiple-choice discovery results — October 12, 2026

The earlier exhaustive **English-tagged HF multiple-choice** pass found 856
repositories. After catalog, queue and prior-run exclusions, 208 remained for
screening. Nine passed eligibility checks; **two also passed the annotation
provenance filter**. This is separate from the subsequent classification/ranking
run and the earlier mixed-source sample.

These are automated screening results, not manual approval or implemented tasks.
MC tags and model judgments can produce false positives: actual explicit options,
gold alignment, source language, training intent and split safety still need review.
In particular, the MicPie sources require checking that usable option-bearing rows
exist; a preview without options is not itself an MC training example.

| Source | Leading label-origin judgment | Provenance filter |
|---|---|---|
| [MicPie/unpredictable_cluster00](https://huggingface.co/datasets/MicPie/unpredictable_cluster00) | procedural | Withheld; provenance unresolved or insufficiently verified |
| [MicPie/unpredictable_support-google-com](https://huggingface.co/datasets/MicPie/unpredictable_support-google-com) | procedural | Passed; awaiting source review |
| [yusuke1997/mCSQA](https://huggingface.co/datasets/yusuke1997/mCSQA) | mixed | Withheld; provenance unresolved or insufficiently verified |
| [Neura-parse/quantum-optimization](https://huggingface.co/datasets/Neura-parse/quantum-optimization) | unknown | Withheld; provenance unresolved or insufficiently verified |
| [QCRI/ArabicCulturalQA](https://huggingface.co/datasets/QCRI/ArabicCulturalQA) | mixed | Withheld; provenance unresolved or insufficiently verified |
| [TorpedoSoftware/RobloxQA-v2.0](https://huggingface.co/datasets/TorpedoSoftware/RobloxQA-v2.0) | model | Withheld; provenance unresolved or insufficiently verified |
| [raphaell/geoguesserMCQA](https://huggingface.co/datasets/raphaell/geoguesserMCQA) | unknown | Withheld; provenance unresolved or insufficiently verified |
| [MicPie/unpredictable_baseball-fantasysports-yahoo-com](https://huggingface.co/datasets/MicPie/unpredictable_baseball-fantasysports-yahoo-com) | procedural | Passed; awaiting source review |
| [swj0419/qa-pairs-mvd-10k_split2](https://huggingface.co/datasets/swj0419/qa-pairs-mvd-10k_split2) | unknown | Withheld; provenance unresolved or insufficiently verified |

## Screening scope

The final results use Jev-only judgments after the second reviewer failed an
opposite-answer control. Viewer rate-limit failures were rechecked separately;
request failures were not treated as proof of unsuitable data. Seven eligibility
passes were withheld by the human/procedural plus verified-synthetic policy. A
leading origin category alone does not establish the required probability threshold.

[Machine-readable records](2026-10-12-multiple-choice.jsonl) preserve the accepted
reviewer’s judgments, source revisions and evidence hashes. No dataset rows or raw
card bodies are included. See the [screening methodology](../../src/tasksource/task_discovery/README.md)
and the [prospects index](../prospects.md).
