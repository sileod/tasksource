# Dataset discovery

Retrieve candidates, then compare them against Tasksource **before** checking train splits, licenses, annotation quality or implementation cost. This is a small offline-friendly tool, not a task registry or automatic integration system.

```bash
python -m tasksource.task_discovery retrieve --output build/discovery/pwc.jsonl
python -m tasksource.task_discovery match build/discovery/pwc.jsonl --output build/discovery/matches.jsonl
# Optionally keep only unmatched names for subsequent review:
python -m tasksource.task_discovery match build/discovery/pwc.jsonl --unmatched-only --output build/discovery/candidates.jsonl
```

Retrieval defaults to classification, native multiple-choice and decision-oriented structured annotations. `--task 'event detection' --task 'semantic parsing'` narrows by task tags; `--all` disables the decision-tag screening. Tags are discovery hints: they do not establish usable supervision. QA/semantic parsing sources still need a faithful decision projection.

[Papers with Code's export documentation](https://github.com/paperswithcode/paperswithcode-data) points to the [archived Hub files](https://huggingface.co/datasets/pwc-archive/files). We pin the July 2025 export by commit and SHA-256. This is historical coverage, not a live search of newer papers. `--snapshot path/to/jul-28-datasets.json.gz` replays the same checksum-verified export locally. PWC metadata carries its own CC BY-SA terms, separate from each dataset's terms.

Each JSONL record contains a dataset `name`, `aliases`, `sources` (source identifier and record URL), `papers` (titles and URLs), `dataset_urls`, `github`, `huggingface`, task tags, modalities and languages. Links are extracted from the source record, not guessed; missing links remain empty. Their presence does not establish that a repository is official or loadable. `train_status` stays `unchecked` until a separate verification step records evidence for a labeled training split. No remote dataset code is executed and no training corpus is downloaded.

Matching uses exact names after case/punctuation normalization, plus explicit Hub repository links. It includes English, multilingual, vision and soft-label catalogs and original repositories behind mirrors. Reports retain match evidence and task IDs; a source match does not imply every possible annotation is implemented. Versions and task projections need review. Shared paper/GitHub links are not sufficient to merge distinct datasets, and fuzzy names are not silently discarded.

Every output has a `.manifest.json` recording source or input hashes, selection settings, catalog fingerprint and output hash. For another discovery source, emit the same candidate JSONL and reuse `match`; no provider framework is needed.

## Interchangeable retrieval sources

Use `--source paperswithcode`, `--source hf_hub`, or `--source openalex`. Repeat `--source` to combine retrievals into one deduplicated JSONL; all outputs feed the same `match` and `link` commands.

```bash
python -m tasksource.task_discovery retrieve --source hf_hub --source openalex \
  --query 'event detection dataset' --limit 100 \
  --cache build/discovery/cache --output build/discovery/combined.jsonl
python -m tasksource.task_discovery match build/discovery/combined.jsonl \
  --unreviewed-only --output build/discovery/new-candidates.jsonl
# Replay the captured searches, without querying either service:
python -m tasksource.task_discovery retrieve --source hf_hub --source openalex \
  --query 'event detection dataset' --limit 100 --offline \
  --cache build/discovery/cache --output build/discovery/replayed.jsonl
```

[`hf_hub.py`](hf_hub.py) uses [`HfApi.list_datasets()`](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_datasets). Without queries it searches text/token/image classification and question-answering categories. `--task` selects exact Hub task categories; with `--query` alone it searches repository names, including untagged sources. Captured metadata includes repository revision, task tags, paper links, declared license and popularity. Names use full repository IDs, with basename aliases, so unrelated uploads sharing a basename are not merged. License tags are evidence, not a completed original-source license review.

[`openalex.py`](openalex.py) uses the [OpenAlex works search API](https://help.openalex.org/api/) with bounded pagination. Defaults search classification datasets, multiple-choice datasets and decision benchmarks. Results carry `candidate_kind: paper`, paper title, OpenAlex ID, DOI when available, publication year and explicit repository links found in locations/abstract metadata. Search results can include papers that use datasets rather than introduce them. We do not invent a dataset name from a title or assume the absence of a Hub mirror. Dataset identification and linking remain a later step. OpenAlex's CC0 metadata terms do not establish the underlying dataset's license.

`--query` is repeatable: scholarly search for OpenAlex, repository-name search for HF Hub, and local metadata screening for the pinned PWC export. `--task` screens PWC tags or selects Hub categories; OpenAlex uses the query text. `--limit` bounds each API query/category (PWC still uses its pinned export). API results are cached with timestamps and hashes; `--offline` replays them, and a fresh cache directory captures updated searches. PWC can replay its local snapshot or cached pinned Hub file. These are candidate retrievals: train status stays `unchecked`, quality is not inferred from popularity, and the human queue is never edited automatically.

The [human-curated queue](todo_batch.md) is authoritative. Generated candidates live in separate output files and never rewrite it. Promote suggestions to that queue only after review; retrieval alone does not approve a training source.

## Recorded decisions

[`decisions.json`](decisions.json) stores prior assessments independently of Python task annotations. Each record identifies the dataset name/aliases and known `huggingface` repos, `github` repos and `papers` (URLs or `arxiv:` IDs), with `scope` (`dataset` or `task`), affected `tasks`, `status`, `reason_code`, reason text and evidence. New reviews also record `checked_on`, source revisions and `reconsider_if`. Unknown identifiers stay empty. Existing parked/evaluation annotations are indexed with their original reasons; a test guards against losing those reasons.

Discovery attaches matching records as `previous_decisions`, including matches through GitHub and paper identity. Any recorded assessment is marked `previously_reviewed` when there is no catalog match. `--unmatched-only` retains these records; `--unreviewed-only` excludes them to avoid repeating past work. A task-scoped decision records only the affected projections, not a whole-dataset ban. The human queue remains separate: a deferral records a failed or uncertain projection, not permission to remove the whole source from that queue. To retry, inspect the reason and evidence, verify the reconsideration condition, and update the record explicitly.

Keep model assessments in a `reviews` array rather than overwriting a previous review. Each review records `reviewer` (kind, model identifier and how it was identified), date, scope, sampled example IDs and recast hashes, verdict, findings, source revision and limitations. Preserve disagreements between models and human reviewers. A model verdict is evidence for a decision, not automatic approval. Historical assessments with unknown reviewers stay unattributed. The [dialogue expansion audit](../../../docs/jev/dialogue-expansion-audit.json) shows sampled source/recast reviews; its model identifier is user-provided.

## Reusable linking

```bash
# Set SEMANTIC_SCHOLAR_API_KEY in the environment if available.
python -m tasksource.task_discovery link build/discovery/candidates.jsonl \
  --cache build/discovery/link-cache --limit 20 --output build/discovery/linked.jsonl
# Reproduce the same enrichment from the saved responses:
python -m tasksource.task_discovery link build/discovery/candidates.jsonl \
  --cache build/discovery/link-cache --limit 20 --offline --output build/discovery/replayed.jsonl
```

`identifiers.py` normalizes Hub/GitHub repos, arXiv/DOI identifiers and Semantic Scholar paper IDs; `link.py` queries the [Semantic Scholar Graph API](https://api.semanticscholar.org/api-docs/graph) and [Hub metadata API](https://huggingface.co/docs/hub/api). These functions accept the same candidate records regardless of discovery source. Paper resolution uses an explicit identifier or a unique exact normalized title; approximate matches remain unresolved. Hub name matches are recorded separately in `link_candidates` with `verified: false`; they are not treated as established dataset identities. Existing GitHub links are retained, not inferred from paper titles.

Each saved response records its request URL, retrieval time and checksum. Headers and API keys are never written to the cache. The first `--limit` records are enriched; other records are preserved. Requests have bounded timeouts, Semantic Scholar requests are paced, and failures remain explicit in `link_errors`. Reuse the cache on subsequent runs. Linking does not perform quality review or automatically edit the human queue.
