# Retrieval evaluation sources

These sources live in `tasksource.eval_only` and never enter `list_tasks()` or the Jev training builders. Load them explicitly:

```python
from tasksource import eval_only

bright = eval_only.load_bright('biology')
followir = eval_only.load_followir('robust04')  # also core17, news21
rag25 = eval_only.load_trec_rag_2025()

support = eval_only.garage__answer_support.load()
evidence = eval_only.garage__evidence_type.load()
```

The three retrieval loaders return native tables, not classification examples. They retain retrieval evaluation semantics rather than generate negative labels or construct artificial multiple-choice questions.

| Source | Returned tables | Verified size |
|---|---|---|
| BRIGHT | `queries` with native `gold_ids`/`excluded_ids`, matching `corpus` | Biology: 103 queries, 57,359 documents; other domains selectable |
| FollowIR | `queries` with original/changed instructions, `corpus`, `qrels_og`, `qrels_changed`, `top_ranked` | Robust04: 52 queries, 18,465 judgments per instruction view; Core17: 20 / 4,740; News21: 32 / 4,277 |
| TREC RAG 2025 | `requests` with query and top-100 candidate passages, separate manual `qrels` | 105 requests, 10,500 candidates; 10,284 manual judgments |
| GaRAGe | Two canonical `Classification` annotations | 2,366 source records; 25,949 labeled passages per view |

## Native retrieval judgments

BRIGHT loads the original `examples` config, not the model-rewritten query configs. Match its `gold_ids` to corpus `id` within the selected domain, and honor `excluded_ids`. Gold reasoning/answers remain native evaluation annotations; use only `query` as model input. No unjudged document is assigned a negative label.

FollowIR preserves both original and changed instructions and their separate qrels, with original query/document IDs. Native scores remain intact: News21 includes values up to 16, not binary labels. The synthetic `jhu-clsp/FollowIR-train` source is explicitly deferred in the discovery decisions. For comparison, pair query `instruction_og` with `qrels_og` and `instruction_changed` with `qrels_changed`.

TREC RAG 2025 retains the organizer baseline's ranking scores independently of the manual relevance grades 0–4. Only 22 of the 105 requests have manual qrels. The remaining candidates stay unjudged. The loader uses `qrels.rag25.test.txt`, not UMBRELA model judgments or participant predictions. Qrels and candidate documents remain separate, so consumers must match `(query_id, document_id)` and handle missing judgments explicitly.

## GaRAGe decision views

- `garage/answer-support`: `ANSWER-THE-QUESTION` → `yes`; the other three annotated evidence types → `other evidence`. This is a collapse of native categories, not a claim that unknown evidence is factually false.
- `garage/evidence-type`: preserves `ANSWER-THE-QUESTION`, `RELATED-INFORMATION`, `UNKNOWN`, and `OUTDATED`.

Of 35,351 source passages, 9,402 have blank evidence-type annotations and are excluded from both views. Blanks are never negative examples. Passage/annotation lengths and citation-text alignment are checked before flattening.

Inputs contain the question, question date, passage, and an explicit passage date when available. Metadata retains `sample_id`, zero-based passage index, citation marker, relevance/citation judgments, passage age/date/provider, and all question annotations including complexity. Generated/reference answers are excluded from inputs. Stable sample IDs identify related passages.

The original Hub split is named `train`; this name is preserved for compatibility with the existing preprocessing interface. It is still an evaluation benchmark, not a designed training partition. No validation/test split is fabricated.

## Sources and terms

All upstream revisions are pinned in [source_revisions.py](../src/tasksource/metadata/source_revisions.py). No new mirrors or dependencies are needed.

- [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT): CC BY 4.0.
- [FollowIR](https://github.com/orionw/FollowIR): official human evaluation uploads ([Robust04](https://huggingface.co/datasets/jhu-clsp/robust04-instructions), [Core17](https://huggingface.co/datasets/jhu-clsp/core17-instructions), [News21](https://huggingface.co/datasets/jhu-clsp/news21-instructions)) declare MIT; underlying news/document corpus terms remain applicable. Tasksource does not redistribute them.
- [TREC RAG 2025](https://trec-rag.github.io/annoucements/2025-track-guidelines/): organizer baseline and manual qrels from Castorini; document use remains subject to MS MARCO/original-source terms. No new license grant is inferred for qrels.
- [GaRAGe](https://huggingface.co/datasets/AmazonScience/GaRAGe): CC BY-NC 4.0 according to the repository README/license; the Hub's generic `cc` tag is incomplete.

[Source and preprocessing checks](jev/retrieval-evaluation-audit.json) include live source loads, reference integrity and coarse example reviews. This change does not publish data or modify training releases.
