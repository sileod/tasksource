# Adding sources to tasksource

The brief for anyone (person, Codex, Claude subagent) given a candidate source, a prospect list or a
shortlist: decide whether each source belongs in tasksource, and if so add it. It extends `AGENTS.md`
with what earlier rounds got wrong; add a line here when a round teaches something new. Shortlists
written by LLMs are leads: verify every claim (existence, sizes, splits, labels, license) yourself.

## Relevance

Add a source only if it passes the gates below and brings something the catalog lacks:
- not already covered: check every catalog (`list_tasks()`, `list_tasks(vision=True)`,
  `list_tasks(multilingual=True)`, `list_tasks(extractive=True)`), `parked.py`, `eval_only.py` and
  `task_discovery/decisions.json`; a known source can still gain a new view (span extraction of a
  dataset used for classification);
- not a re-upload or near-duplicate of a source in (bionlp2004 is JNLPBA; MASSIVE en-US is SLURP);
- a new domain, genre, skill or task format, rather than another copy of a saturated one (ATIS, SNIPS);
- valid but low-signal sources get `max_examples`, systematically unreliable ones `parked.py`.
Its kind decides where it goes: classification and multiple choice in `tasks.py` (vision in
`vision_tasks.py`), token labeling in `tasks.py` (and the `NER` dict of the extraction builder),
extractive QA in `extractive_tasks.py`.

## Workflow

- Work in your own git worktree (`git worktree add ../tasksource-<topic> -b <topic> origin/main`);
  other agents keep uncommitted work in the main checkout. Do not push, publish to the Hub or edit
  `DATA_CHANGELOG.md`: the maintainer merges, rebuilds and writes the release entry.
- Per source: verify gates, declare, prove `load_task` works, smoke-build, inspect rows, add a test
  for non-trivial preprocessing. Write a report: per candidate, verdict (added / skipped + reason),
  task id, source and revision, license, split sizes, label-origin evidence, smoke notes.
- Finish sources one by one; a half-implemented source is worse than a documented skip. Agents can
  stop mid-run (quota, timeouts): the report and tests are the proof a source is done.

## Gates

- Genuine training split; test-only sources are eval-only candidates (`eval_only.py`).
- Human, expert or procedural labels. Silver, distant (TriviaQA, SearchQA) and unverified model
  labels fail; model-written text verified by experts can pass (LegalLensNER), say why.
- Licenses: unspecified/unknown is fine, record `unspecified`. Non-commercial and research-only are
  fine, record `license_use: non-commercial` (FiNER-ORD, GUM, VUA, FCE, QAConv). Skip only data that
  may not be redistributed (e.g. Yelp review terms), gated data you cannot read, and sources whose
  labels are documented as unreliable (tner/fin).
- English (other languages go to the multilingual catalog).

## Sources and pins

- Pin full 40-character SHAs read from the Hub API (`HfApi().dataset_info(repo).sha`); never type
  or guess one. Script-only datasets (no longer loadable) are read from their
  `refs/convert/parquet` commit (`HfApi().list_repo_refs(repo, repo_type='dataset').converts`).
- Check the repo is what its name says before using it: `plncmm/wl-procedure` is a Spanish clinical
  corpus, not WLP. Read rows, not just the card.
- Prefer the original or author upload; re-uploads can drop labels (disi-unibo bc2gm vs
  spyysalo/bc2gm_corpus). Reuse a loader tasksource already has for another view of the same source
  (BioRED, MAVEN, SLURP, QuAC, CUAD, Qasper).
- No Hub upload, or archives/joins needed: repackage under `scripts/repackage_dataset/` and point the
  task at a `tasksource/` mirror (see `AGENTS.md`); the maintainer publishes the mirror.

## Splits

- Keep native splits (`complete_splits=False`). Check that splits do not overlap: TweetNER7's default
  view merges overlapping files, read the native 2020 files.
- Hidden test labels (all `O`, all empty answers) are not evaluation data; the extraction builder
  drops such splits, other views should too.

## Spans

- Every span must be verbatim at its offset. Never turn an unlocated positive answer into a null;
  keep native unanswerables as null. Count and report dropped rows.
- Encodings differ: MRQA char spans are end-inclusive; QAConv marks unanswerable as an empty string
  at offset 0; DROP validation lists every annotator's answer; Quoref starts can be off by one
  (`offset_tolerance=1`); multi-span answers that are sets are `LIST_QA`, not alternatives.
- Character spans into tokens: tokenize deterministically, convert to BIO, drop and count spans that
  cut through tokens (HiFi-KPI could not be aligned at all), reject overlaps explicitly, split
  discontinuous mentions into contiguous runs.
- Tokenization artifacts become entities: Broad Twitter Corpus tagged `@` and the handle separately,
  merge them. Long documents need windows that never cut a gold span (CUAD).

## Checks before handing over

- Smoke build with the relevant builder (extraction: `PYTHONPATH=.:src python scripts/build_jev_extraction.py --output build/smoke --tasks <ids> --max-rows 300 --max-rows-eval 50`;
  decisions: `scripts/build_jev_dataset.py` with `--tasks`); look at a few rows per source: readable field names and instructions
  (`NAMES` in `jev/extraction.py`), correct spans, not all null.
- `task_licenses([...])` reports the intended `license_use`.
- `PYTHONPATH=.:src python -m pytest -q tests/test_task_catalogs.py tests/test_catalog_naming.py tests/test_licenses.py`
  plus the tests of the area you touched (e.g. `tests/test_extraction.py`).
