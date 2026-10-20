# What enters the catalog

A source becomes a training task only if all three hold:
- it has a genuine training split (test-only benchmarks go to `eval_only.py`);
- it is a decision: classification, multiple choice or token labeling, not open-ended answers;
- its labels are human, procedural (solver, verifier, construction) or otherwise demonstrably reliable; unverified model labels do not qualify.

Valid but low-signal sources get `max_examples`; systematically unreliable ones go to `parked.py`.

When adding sources (including from prospect lists or LLM shortlists), follow [docs/implementing-sources.md](docs/implementing-sources.md): relevance, gates, pins, splits and span checks learned from earlier rounds.

# Dataset sources

Reading an existing dataset with a mapping or a little code is an annotation: declare it in the task catalog on the original Hugging Face dataset. Heavy preprocessing is a repackage. Renaming or dropping columns is not a reason to mirror.

Use existing Hugging Face datasets directly when `datasets.load_dataset` provides usable data. Keep task mappings declarative and reuse the existing templates and shared loader.

For sources that require archive extraction, joins, broken-loader repairs, expensive pixel transformations such as crop generation, or have no usable Hub upload, add a reproducible conversion under `scripts/repackage_dataset/` (the upload entry point remains `scripts/upload_repackaged.py`) and publish a data-only `tasksource/` mirror. Pin upstream revisions, preserve native splits and original image bytes, record provenance and license evidence, and point the task declaration at the mirror. For generated crops, also record the actual pixel rectangle, encoding/configuration, and original action/widget identity. Avoid putting download or filesystem setup into task preprocessing.

Keep visual mirrors consistent: ordered `images` plus canonical `inputs`, `choices_list` (for MC), `labels`, and JSON-string `metadata`. When many QAs share images, group those canonical QA records in `qa` and store the images once. Keep rationales and provenance in metadata, separate from model inputs.

Each repackaged dataset README must link to the original data distribution and the specific reproducible packaging script, with source revisions, hashes and conversion settings recorded in linked provenance. A link to the packaging directory alone is insufficient.

# Dataset documentation

Keep Hub root dataset cards focused on current usage, configs, schema and license scope. Put release notes in CHANGELOG.md or addition manifests, and config-specific details in that config’s README. Incremental publication scripts must preserve the root card body rather than append batch summaries.

# Jev releases

The text configs of `tasksource/tasksource-jev-typed-decisions` (`default`, `full`) are rebuilt by `scripts/build_jev_dataset.py`, which replaces all their shards. New sources enter through the catalog, not through one-off publication scripts.

To add sources without a rebuild, declare and pin them in the catalog, add the changelog entry, then run `python -m scripts.append_jev --config full|vision --tasks TASK ... --push --release vYYYY.MM.DD.n` (see `docs/jev/README.md`). It appends training rows only; published shards stay byte-identical, and `default` changes at the next full rebuild.

# Changelog

Every change that affects published data (new or removed sources, label fixes, sampling, schema) gets an entry in the root [`DATA_CHANGELOG.md`](DATA_CHANGELOG.md), newest first: `## <dataset> <version>` and a few concise, user-facing bullets on what changed in the data, not how it was done. Code-only changes need no entry. Keep history out of dataset cards.

`tasksource-jev-typed-decisions` versions are `vYYYY.MM.DD[.n]`: add the entry before publishing; the build publishes that dataset's entries as its Hub `CHANGELOG.md` and tags the revision.
