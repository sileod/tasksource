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

The public [prospects list](../../../docs/prospects.md) summarizes screened sources
and outstanding checks. `todo_batch.md` is an internal maintainer work queue,
not the public discovery results. Discovery never rewrites it.

## Ranked prospect search

The default way to look for missing tasks. Exhaustive tag screening spent model reviews on the Hub's long tail and on re-uploads of implemented tasks; `prospect` spends cheap signals first and reviews only the head of a ranking:

```bash
python -m tasksource.task_discovery.fingerprint                  # content index from Jev `full`
python -m tasksource.task_discovery.prospect --output build/discovery/prospect-YYYY-MM-DD --probe 6000 --review 600
```

1. **Identity.** Beyond names, Hub links and the catalog paper/GitHub index, two indexes catch re-uploads: hashed text segments of every implemented source's Jev rows (`fingerprint.py`), and dataset signatures from the viewer `/info` of catalog repositories: two split sizes of one configuration, a split size with its label vocabulary, or a large label vocabulary.
2. **Prior.** Downloads, likes, a linked paper and age (capped), minus generated/placeholder names and tags and bulk uploaders whose datasets nobody uses. These weights separate catalog sources from the tag inventory at cross-validated AUC 0.96.
3. **Structure.** For the top `--probe`: a train split of at least `--min-train` rows, a discrete label column checked on the 100 preview rows (not free text or continuous scores), no content or signature match, and no duplicate of a higher-ranked candidate.
4. **Review.** Jev-style decision models answer the existing evidence questions plus a sampled-row quality question. Ranking uses the least favourable reviewer and penalizes model-made labels.

`report.md` and `ranked.jsonl` are a reading list; implementation still follows "What enters the catalog" in `AGENTS.md`.

## Interchangeable retrieval sources

Use `--source paperswithcode`, `--source hf_hub`, `--source hf_papers`, or `--source openalex`. Repeat `--source` to combine retrievals into one deduplicated JSONL; all outputs feed the same `match` and `link` commands.

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

The internal human-curated queue is authoritative for requested implementation work. Generated candidates live in separate output files and never rewrite it. Promote suggestions to that queue only after review; retrieval alone does not approve a training source. Novelty is a screening step, not a reason to integrate: require sound supervision, enough evidence in the input, usable labeled training data and a worthwhile contribution to existing coverage. A small useful shortlist is preferable to more weak datasets.

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

The output directory retains `matched.jsonl`, `checked.jsonl`, per-model review artifacts, `prospects.jsonl` and a `handoff.jsonl` of promising candidates. A promising source needs explicit evidence and all configured reviewers assigning at least 0.8 to training and decision fit; resources must be demonstrated by a fetched Viewer row or supported at 0.8 by all reviewers; HF sources additionally need observed training-split metadata and no failed training checks. Missing evidence or reviewer disagreement stays `needs_evidence`. `investigate.jsonl` separately retains strongly supported training/decision leads with specific evidence gaps; it does not relax the handoff. Null Image previews and documented supervision-quality concerns hold sources out of the handoff. Unknown annotation origin alone is not enough to prioritize investigation, and known unverified model labels remain excluded under the origin filter. See the [vision calibration](../../../docs/prospects/2026-10-14-vision-calibration.md). These are screening judgments, not verified annotation quality; older reviews lack the new quality question. The manifest records input/catalog hashes, seed, model routes and prompt/code hashes. The human-curated queue remains separate.

For English-focused discovery, HF retrieval passes `--language en` directly to the Hub API, including multilingual datasets tagged English. The screening command also accepts `--language en`: declared English is preferred within each source sample, known non-English sources are excluded, and missing language metadata remains eligible for evidence review. PWC language names such as `English` are normalized. OpenAlex paper language is not treated as dataset language. Retrieval accepts
`--source openalex --language en` to filter English papers; this does not establish
that their datasets are English. Abstracts are reconstructed from the supplied
inverted index and retained as review evidence, with missing abstracts left empty. Filters are recorded in cache keys and run manifests; metadata tags do not establish example-level language quality.

When screening with `--language`, an additional native decision checks the actual
labeled dataset language from source evidence. Paper language and repository tags
alone do not satisfy it; unknown language remains unresolved and explicit mismatches
are unsuitable for that language-scoped pass. Earlier snapshots without this
question retain their original judgments and require language checks before use.

Training availability and split safety are separate. Screening records `source_splits` and `split_policy` (`train_only`, `multiple_splits`, `unknown`). A Hub split named `train` is not sufficient positive evidence: the train criterion requires source documentation of a designated training partition or a corpus intended for training. Train-only dumps may include evaluation material; even multiple split names do not prove disjointness. The processing phase must verify source partitions and leakage before implementation.

Annotation provenance is reviewed through four additional native Jev choice questions over the same evidence: `annotation_origin` (human/model/procedural/mixed/unknown), `human_annotators` (experts/crowd/authors/unspecified/none/mixed/unknown), `annotation_checks` (independent review/agreement/executable/model review/multiple/none/unknown), and `verification_scope` (all/sample/partial/none/unknown). Each model’s distributions remain in `reviews[].data.answers`, linked to its evidence-state hash and saved source documents. These descriptive judgments do not alter the three eligibility votes or imply that human/procedural labels are necessarily correct. Generated inputs are distinguished from generated labels, missing descriptions remain unknown, and annotation quality still requires source/example inspection during processing.

For another screening batch, increase `--limit` and repeat `--exclude path/to/previous/checked.jsonl` for prior runs. Exclusions compare normalized names and exact Hub identities, preserve shared-paper/GitHub hints, and record input hashes. `selected.jsonl` is saved before metadata checks; evidence collection uses eight workers, while litlm retains the bounded model request schedule. Progress and raw per-model results are saved throughout the review.

Model review uses litlm adaptive concurrency with a ceiling of 32 and no imposed RPM cap. It starts conservatively and backs off on congestion; returned request failures stay explicit for cached reruns. This tunes concurrency rather than discovering a guaranteed provider RPM allowance. Scheduling settings are recorded in the review manifest.


## Filter annotation origins and check screening recall

```bash
# Re-filter saved results without new metadata requests or model calls:
python -m tasksource.task_discovery.discover_task_prospects \
  build/discovery/prospects-en-1000/prospects.jsonl --from-reviews \
  --annotation-origin human --annotation-origin procedural \
  --output build/discovery/prospects-en-1000/non-model
# Run the same screening on canonical records for implemented sources:
python -m tasksource.task_discovery.discover_task_prospects \
  build/discovery/calibration/input.jsonl --include-known --limit 30 \
  --model openrouter/perplexity/pplx-decider-v1.1-27b \
  --model openrouter/~typesafe/jev-latest \
  --output build/discovery/calibration/baseline
```

`--annotation-origin` restricts only the handoff: every current reviewer must assign at least 0.8 total probability to the allowed origins. Repeat it for a union. Human + procedural excludes model-derived, mixed and unresolved provenance; explicitly add `unknown` to retain uncertain origins. Mixed provenance is withheld conservatively because it can contain model labels. This describes label origin, not whether inputs are synthetic or annotations are correct. All records and original judgments remain in `prospects.jsonl`; the manifest records the filter and withheld count.

`--include-known` bypasses implemented/parked/queued matching exclusions for an explicit calibration input; it does not change eligibility thresholds, source evidence or model prompts. Avoid interpreting this as proof that every implemented task is sound. Compare criterion confidence, missing evidence, native split names and request failures before deciding whether a gate is too strict. The model is not told that a source is an expected positive.

Calibration identified three targeted recall fixes: recognize separated training split names such as `train_r1`; respect an explicitly supplied config when fetching a preview; and treat a successfully fetched, checksummed Viewer row as direct evidence of an available resource rather than requiring a model to rediscover it. The decision-fit prompt explicitly allows deterministic projections of native structured annotations (such as span presence → answerability). Training and decision-fit thresholds remain unchanged, missing resources still require evidence, and multiple splits do not prove leakage safety.


To allow verified model-derived labels while excluding unverified ones, add `--allow-verified-synthetic` to the human/procedural origin filter. Every current reviewer must either support an allowed origin, or assign at least 0.8 to model/mixed origin, at least 0.8 total to human review/agreement/executable checking, and at least 0.8 to coverage of **all retained labels**. Model-only checking, sampled audits and unspecified coverage do not qualify for this strict option. Their distinctions remain in the saved distributions rather than being discarded. Old reviews lacking `verification_scope` cannot establish full verification; re-review only prospective sources when that information is needed.

```bash
python -m tasksource.task_discovery.discover_task_prospects \
  build/discovery/prospects-en-5000/verification-review/reviews.jsonl --from-reviews \
  --annotation-origin human --annotation-origin procedural --allow-verified-synthetic \
  --output build/discovery/prospects-en-5000/verified-handoff
```

Used alone, `--allow-verified-synthetic` defaults to human/procedural origins plus the verified model/mixed exception; it does not silently admit unverified model labels. The 30-source calibration improved from 8 initial passes to 22 after the targeted fixes and a cached recheck of failed requests; the remaining eight need evidence or stronger reviewer support, and none is classified unsuitable. This is a deliberately chosen diagnostic sample, not an unbiased recall estimate.

## Exhaustive scheduled queue

```bash
python -m tasksource.task_discovery.scheduled \
  --output build/discovery/exhaustive-queue \
  --exclude build/discovery/prospects-en-5000/checked.jsonl
```

The fixed order is **classification → zero-shot classification → sentence similarity → ranking**. Classification searches `text-classification` and `token-classification`; ranking searches `text-ranking` and `text-retrieval`. Every search uses the Hub's `en` language filter and consumes the complete paginated listing (`limit=None`). Coverage is exhaustive for these tags at retrieval time; untagged sources are outside that inventory. Inventories, retrieval hashes and matching reports are saved independently of screening.

The queue processes 128-source batches, excludes implemented/parked/curated sources plus repeated `--exclude` inputs, and deduplicates sources across stages. Its default reviewer is Jev. Before each unfinished batch, a fresh true/false/unknown control must pass at 0.8 for each retained reviewer; repeat `--model` to request additional reviewers. Failed controls are recorded and excluded. If no reviewer passes, the queue stops with `reviewer_unavailable`. Perplexity is not a default reviewer because its route failed this control during the latest MC runs.

A lock prevents duplicate runners. Rerunning the exact command skips completed stages and batches; cached successful model calls survive an interrupted batch. Fatal batch processes get one retry, then stop with `batch_failed`. Source-level uncertainty, inaccessible/gated sources and request failures remain explicit in batch artifacts. Viewer requests are paced to one uncached request per second per process, with bounded 429 backoff and atomic cache writes. This is separate from litlm's adaptive model concurrency.

`status.json` and `report.md` track stage progress; the root `handoff.jsonl` accumulates the shortlist after each completed batch. All judgments remain in each batch's `prospects.jsonl`, with inputs, prompts, controls and logs beside them. The human/procedural plus independently verified-synthetic policy is enabled. The queue does not implement datasets or modify the human-curated queue. Closing the interactive session does not stop a detached runner; machine/process termination requires rerunning the saved command.


## Bounded OpenAlex domain passes

```bash
# Retrieve four searches per domain (100 papers/query), then review 12/domain:
python scripts/discover_openalex.py --pilot-only
# After inspecting the pilot, review resource-linked leads (up to 188/domain):
python scripts/discover_openalex.py
```

The script records general, medical and legal query sets explicitly and uses
English-paper filtering. It reuses discovery matching, prior-screening exclusions,
reviewer controls and the human/procedural plus verified-synthetic policy. Domains
exclude earlier passes so shared papers are not knowingly reviewed twice under the
same identity. Cross-source aliases remain imperfect. Paper-only records can remain
`needs_evidence`: abstracts do not always document train partitions or resource
links, and an English paper can describe a non-English corpus. No papers are treated
as datasets merely because of their titles. Saved commands, source snapshots,
evidence hashes and review judgments support replay. Successful steps are skipped
on resume; use a fresh output directory if changing query settings. This produces
prospects only and never edits task annotations or the internal implementation queue.
Expansion prioritizes papers with explicit HF/GitHub links so cards/READMEs can
provide evidence absent from abstracts. Other retrieved leads remain in the saved
inventory, unresolved rather than rejected. The combined final shortlist is in
`english-confirmed/handoff.jsonl`; any legacy eligibility passes are re-reviewed
with the dataset-language criterion before appearing there.


## Hugging Face Papers

```bash
python -m tasksource.task_discovery retrieve --source hf_papers --limit 20 \
  --cache build/discovery/hf-papers/cache --output build/discovery/hf-papers/inventory.jsonl
python -m tasksource.task_discovery.discover_task_prospects \
  build/discovery/hf-papers/inventory.jsonl --limit 12 --language en --hold-related \
  --allow-verified-synthetic --model openrouter/~typesafe/jev-latest \
  --exclude build/discovery/hf-papers/previous-identities.jsonl --output build/discovery/hf-papers/pilot
```

[`hf_papers.py`](hf_papers.py) uses Hugging Face paper search, paper metadata and
dataset repositories tagged with each paper's arXiv ID. Default searches cover
English classification, multiple choice, relevance, biomedical and legal data.
`--query` overrides them; `--limit` is per query (1–120). Search reaches indexed
paper pages, not only the Daily Papers selection, and is not an exhaustive inventory.
Responses are checksummed and replayable with `--offline`. Native abstracts are
retained; AI-generated summaries are not substituted as source evidence. Explicit
linked repositories can cite a paper without being its official data distribution.

Paper leads use the same canonical records and screening as other sources. HF
Papers has no dataset-language search filter: apply `--language en` during screening,
which verifies the actual dataset language from evidence. Before review, build a
prior-identities JSONL from previous checked records, retaining `name`, `huggingface`
and `papers`; pass it through `--exclude`. Paper-ID exclusion applies to paper leads,
not distinct dataset records sharing a paper. `--hold-related` also withholds catalog
paper/GitHub overlap hints for identity review; held rows stay in `matched.jsonl` and
counts remain in the manifest. These holds do not declare a source unsuitable or
prove every projection is already implemented. Titles/aliases and links remain
imperfect, and unresolved train/resource evidence does not approve a prospect.

## English image discovery queue

```bash
python -m tasksource.task_discovery.scheduled \
  --stage multiple-choice --stage visual-question-answering --stage image-classification \
  --modality image --batch-size 64 --output build/discovery/vision/hf-queue
```

This consumes each complete English-tagged, image-tagged Hub inventory in the
requested order, deduplicating against the catalog, history, human queue and earlier
stages. Tags only retrieve candidates: screening separately checks labeled training
intent, accessible resources, native decision supervision, dataset language and label
provenance. Free-form VQA needs a faithful closed-set projection; generated distractors
or guessed labels do not qualify. Untagged sources remain outside this inventory.

After the Hub stages, use the pinned PWC archive for explicitly image-tagged leads:

```bash
python -m tasksource.task_discovery retrieve --source paperswithcode \
  --task 'visual question answering' --task 'image classification' --modality image \
  --output build/discovery/vision/pwc.jsonl
cat build/discovery/vision/hf-queue/*/matched.jsonl > build/discovery/vision/hf-identities.jsonl
python -m tasksource.task_discovery.discover_task_prospects build/discovery/vision/pwc.jsonl \
  --language en --hold-related --allow-verified-synthetic --limit 10000 \
  --exclude build/discovery/vision/hf-identities.jsonl \
  --model 'openrouter/~typesafe/jev-latest' --output build/discovery/vision/pwc-screen
```

PWC modality filtering uses its explicit metadata (`Images` → `image`); missing
modality declarations are withheld and counted, not interpreted as non-image data.
Its July 2025 archive cannot discover later releases. Results are prospects only;
source identity, split safety, licenses and example correctness still need processing.
