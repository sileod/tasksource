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

Matching uses exact names after case/punctuation normalization, plus explicit Hub repository links. It includes English, multilingual, vision and soft-label catalogs and original repositories behind mirrors. Reports retain match evidence and task IDs; a source match does not imply every possible annotation is implemented. Versions and task projections need review. Source names from the human-curated queue tables are also checked, preserving their requested views and status as `queue_matches`. Matching is exact after case/punctuation normalization: `SST-5` and `sst5` match, while `SST-2` remains distinct. A queue match is reported as `queued` unless catalog or prior-review evidence takes precedence. Shared paper/GitHub links are not sufficient to merge distinct datasets, and fuzzy names are not silently discarded.

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

[`hf_hub.py`](hf_hub.py) uses [`HfApi.list_datasets()`](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_datasets). Without queries it searches multiple-choice, text/token/image classification and question-answering categories. `--task` selects exact Hub task categories; with `--query` alone it searches repository names, including untagged sources. Captured metadata includes repository revision, task tags, paper links, declared license and popularity. Names use full repository IDs, with basename aliases, so unrelated uploads sharing a basename are not merged. License tags are evidence, not a completed original-source license review.

[`openalex.py`](openalex.py) uses the [OpenAlex works search API](https://help.openalex.org/api/) with bounded pagination. Defaults search classification datasets, multiple-choice datasets and decision benchmarks. Results carry `candidate_kind: paper`, paper title, OpenAlex ID, DOI when available, publication year and explicit repository links found in locations/abstract metadata. Search results can include papers that use datasets rather than introduce them. We do not invent a dataset name from a title or assume the absence of a Hub mirror. Dataset identification and linking remain a later step. OpenAlex's CC0 metadata terms do not establish the underlying dataset's license.

`--query` is repeatable: scholarly search for OpenAlex, repository-name search for HF Hub, and local metadata screening for the pinned PWC export. `--task` screens PWC tags or selects Hub categories; OpenAlex uses the query text. `--limit` bounds each API query/category (PWC still uses its pinned export). API results are cached with timestamps and hashes; `--offline` replays them, and a fresh cache directory captures updated searches. PWC can replay its local snapshot or cached pinned Hub file. These are candidate retrievals: train status stays `unchecked`, quality is not inferred from popularity, and the human queue is never edited automatically.

The [human-curated queue](todo_batch.md) is authoritative. Generated candidates live in separate output files and never rewrite it. Promote suggestions to that queue only after review; retrieval alone does not approve a training source. Novelty is a screening step, not a reason to integrate: require sound supervision, enough evidence in the input, usable labeled training data and a worthwhile contribution to existing coverage. A small useful shortlist is preferable to more weak datasets.

For now, novelty against the catalog and queue relies mainly on names and explicit Hub repository links. A durable cross-source identity index should eventually connect verified Hub, GitHub, PWC and paper identifiers. Existing links and review identities are retained for that work, but unresolved aliases and search suggestions do not establish a new dataset.

## Recorded decisions

[`decisions.json`](decisions.json) stores prior assessments independently of Python task annotations. Each record identifies the dataset name/aliases and known `huggingface` repos, `github` repos and `papers` (URLs or `arxiv:` IDs), with `scope` (`dataset` or `task`), affected `tasks`, `status`, `reason_code`, reason text and evidence. New reviews also record `checked_on`, source revisions and `reconsider_if`. Unknown identifiers stay empty. Existing parked/evaluation annotations are indexed with their original reasons; a test guards against losing those reasons.

Discovery attaches matching records as `previous_decisions`, including matches through GitHub and paper identity. Any recorded assessment is marked `previously_reviewed` when there is no catalog match. `--unmatched-only` retains these records; `--unreviewed-only` excludes them and human-queue matches to avoid repeating past work. The manifest fingerprints the queue contents alongside the catalog and decision history. A task-scoped decision records only the affected projections, not a whole-dataset ban. The human queue remains separate: a deferral records a failed or uncertain projection, not permission to remove the whole source from that queue. To retry, inspect the reason and evidence, verify the reconsideration condition, and update the record explicitly.

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

## Catalog link index

Mine README cards of all implemented source repositories (including originals behind mirrors), without downloading dataset contents:

```bash
python -m tasksource.task_discovery index --output src/tasksource/task_discovery/catalog_links.jsonl
python -m tasksource.task_discovery match build/discovery/hf_candidates.jsonl \
  --unreviewed-only --output build/discovery/new_candidates.jsonl
```

`index --limit 20` bounds the scan; `--offline` replays locally cached cards. Each record retains existing task IDs, resolved card revision, SHA-256 and a pinned card URL; failed or inaccessible cards are recorded explicitly. The bundled index is partial and can be refreshed independently. `match --index path.jsonl` uses another snapshot. Its manifest fingerprints the effective reference index.

GitHub repository and paper identifiers (arXiv, DOI, Semantic Scholar, ACL Anthology) appear as `related_matches` with the associated task IDs. They flag possible mirrors or new views of a known source. They do not silently discard candidates or merge datasets: a card may cite related work, and a repository or paper may introduce several datasets. Exact name/Hub matches remain separate. Common dataset-loader infrastructure links are ignored. This index records card claims, not a verification of authorship, license or annotation quality.

The initial snapshot scans 510 source repositories: 496 readable cards, 14 recorded failures, 246 distinct GitHub identities and 330 paper identities. The discovery regression suite passes 15 checks; the cached PWC run reports 63 related-source records alongside 149 direct catalog matches. Related-source records are review hints and remain eligible when unmatched by name/repository.

To standardize catalog paper references through Semantic Scholar, reuse the same linker:

```bash
python -m tasksource.task_discovery link src/tasksource/task_discovery/catalog_links.jsonl \
  --cache build/discovery/link-cache --no-hub-search --limit 1000 \
  --output build/discovery/catalog-links-s2.jsonl
python -m tasksource.task_discovery match build/discovery/hf_candidates.jsonl \
  --index build/discovery/catalog-links-s2.jsonl --output build/discovery/matched.jsonl
```

Resolved references retain original URLs and add Semantic Scholar IDs and external arXiv/DOI IDs. Unresolved references and request failures remain explicit. Authentication uses `SEMANTIC_SCHOLAR_API_KEY`, falling back to `~/.config/tasksource/semantic_scholar_api_key` (keep it private, mode 0600). Neither credentials nor request headers enter snapshots. Rate-limited requests get bounded backoff; cached successes are reused.

## Two phases: discover, then process

`discover_task_prospects` retrieves and screens potential sources cheaply. It does not convert datasets, create task annotations, publish mirrors or rewrite curated decisions. Its handoff is input for the separate, heavier `process_task_prospect` phase (implementation and deeper source validation), which is deliberately outside this discovery command.

```bash
python -m tasksource.task_discovery retrieve --source hf_hub \
  --task multiple-choice --task text-classification --language en --limit 5000 \
  --output build/discovery/prospects-hf.jsonl
python -m tasksource.task_discovery retrieve --source paperswithcode \
  --output build/discovery/prospects-pwc.jsonl
python -m tasksource.task_discovery.discover_task_prospects \
  build/discovery/prospects-hf.jsonl build/discovery/prospects-pwc.jsonl \
  --output build/discovery/prospects-pass --limit 100 --seed 0 --language en \
  --model openrouter/perplexity/pplx-decider-v1.1-27b \
  --model openrouter/~typesafe/jev-latest
```

The pass matches implemented catalogs, mirrors, structured prior decisions and the curated queue, retaining GitHub/paper overlap hints. It draws a deterministic source-stratified sample of unmatched candidates, checks Viewer splits and one training-row/schema preview, and reviews bounded card/README evidence. GitHub/paper-only sources remain unknown when those documents do not establish training availability; the command does not download full corpora or papers. Missing Viewer results do not prove that training data is absent.

`jev_review` uses native Decisions requests through optional tool dependency [litlm](https://github.com/sileod/litlm) (`c3340b1` or newer). All eligibility questions share one evidence state. Repeat `--model` for an ensemble; native probabilities, returned model versions, member disagreements and failures are preserved. Raw requests/results are separate from the final handoff, successful requests are cached, and credentials stay outside artifacts.

The output directory retains `matched.jsonl`, `checked.jsonl`, per-model review artifacts, `prospects.jsonl` and a `handoff.jsonl` of promising candidates. A promising source needs explicit evidence and all configured reviewers assigning at least 0.8 to each eligibility criterion; HF sources additionally need observed training-split metadata and no failed training checks. Missing evidence or reviewer disagreement stays `needs_evidence`. These are screening judgments, not verified annotation quality. The manifest records input/catalog hashes, seed, model routes and prompt/code hashes. The human-curated queue remains separate.

For English-focused discovery, HF retrieval passes `--language en` directly to the Hub API, including multilingual datasets tagged English. The screening command also accepts `--language en`: declared English is preferred within each source sample, known non-English sources are excluded, and missing language metadata remains eligible for evidence review. PWC language names such as `English` are normalized. OpenAlex paper language is not treated as dataset language. Filters are recorded in cache keys and run manifests; metadata tags do not establish example-level language quality.

Training availability and split safety are separate. Screening records `source_splits` and `split_policy` (`train_only`, `multiple_splits`, `unknown`). A Hub split named `train` is not sufficient positive evidence: the train criterion requires source documentation of a designated training partition or a corpus intended for training. Train-only dumps may include evaluation material; even multiple split names do not prove disjointness. The processing phase must verify source partitions and leakage before implementation.
