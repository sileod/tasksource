# Aspect and privacy decisions — 2026-10-13

Three existing Hugging Face sources supply seven training views. No new mirror, template or loader is needed.

| Family | Views | Representation |
|---|---|---|
| [CEBaB](https://huggingface.co/datasets/EleutherAI/CEBaB) | `cebab/rating`, `cebab/food`, `cebab/ambiance`, `cebab/service`, `cebab/noise` | Direct field mappings. Ratings are the cleaned copy's median of native human votes; aspect labels retain the source's unknown category. |
| [FABSA](https://huggingface.co/datasets/jordiclive/FABSA) | `fabsa/sentiment` | One decision per annotated, single-polarity aspect. Unmentioned aspects are not assigned neutral. Multi-polarity aspects are omitted, without discarding other aspects of the review. |
| [TAB](https://huggingface.co/datasets/doshimit3015/text-anonymization-benchmark) | `tab/identifier` | One native annotator judgment per marked mention, with full document and person-to-anonymize instructions. Direct/quasi identifiers require masking; explicit no-mask annotations supply the third class. |

CEBaB preserves original-review IDs and counterfactual flags. Its cleaned copy omits edit IDs and individual votes; all median ratings were checked against the original vote distributions. Native original-ID splits are disjoint. Repeated text across splits is assigned to holdouts.

FABSA has 105 reviews containing multiple polarities for an aspect and repeated reviews across native partitions. Holdouts take precedence; duplicate or conflicting review/aspect prompts are removed. Neutral labels are retained.

TAB retains annotator IDs, native offsets, entity IDs and coreference links in metadata. Disagreement is retained as separate native judgments, not replaced with a consensus label. All source spans match their document text, all mention IDs are unique, and document partitions are disjoint. Small mapping batches bound memory while retaining full documents.

CEBaB declares CC BY 4.0; TAB declares MIT. FABSA has no explicit license in its dataset card and remains marked **unspecified**.

Validation: 32 focused tests and manual review of two staged Jev examples per view. The [machine-readable audit](aspect-privacy-audit.json) records counts, label distributions, revisions, overlap checks and the reviewed rows. This is a coarse preprocessing/source check, not an audit of every annotation.

Published to Jev `default` and `full` in [Hub commit 8fea8388](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/8fea8388f24f0ead40c9364fa6a73dd675df33ba): **7,000 train / 700 validation / 700 test** rows per config. Existing Parquet blobs and root card body were preserved; new shards were downloaded and verified against the reviewed build. The English gallery includes three published examples per new view.
