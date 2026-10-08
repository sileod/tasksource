# Dataset sources

Use existing Hugging Face datasets directly when `datasets.load_dataset` provides usable data. Keep task mappings declarative and reuse the existing templates and shared loader.

For sources that require archive extraction, joins, broken-loader repairs, expensive pixel transformations such as crop generation, or have no usable Hub upload, add a reproducible conversion under `scripts/repackage_dataset/` (the upload entry point remains `scripts/upload_repackaged.py`) and publish a data-only `tasksource/` mirror. Pin upstream revisions, preserve native splits and original image bytes, record provenance and license evidence, and point the task declaration at the mirror. For generated crops, also record the actual pixel rectangle, encoding/configuration, and original action/widget identity. Avoid putting download or filesystem setup into task preprocessing.

Keep visual mirrors consistent: ordered `images` plus canonical `inputs`, `choices_list` (for MC), `labels`, and JSON-string `metadata`. When many QAs share images, group those canonical QA records in `qa` and store the images once. Keep rationales and provenance in metadata, separate from model inputs.
