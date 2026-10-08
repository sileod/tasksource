# Dataset repackaging

Source-specific preparation lives in `text.py` and `vision.py`. Publish a pinned,
data-only mirror with either command:

```bash
python -m scripts.repackage_dataset ai2d iconqa_text mind2web
python scripts/upload_repackaged.py ai2d iconqa_text mind2web
```

`--dry-run` builds locally; `--card-only` updates the published source card.
Visual conversions write Parquet, `provenance.json`, and an exclusion manifest
under `build/<mirror-name>-release/`. VIEW2SPACE uses a native ImageFolder ZIP.

AI2D and IconQA store ordered `images`, `image_group_id`, and grouped canonical
`qa` records (`inputs`, `choices_list`, integer `labels`, JSON-string `metadata`).
Mind2Web stores one screenshot per eligible action with canonical MC fields and
additional `action`, `x10`, `y10`, `grid5`, and `grid7` labels. All six task views
reuse this one prepared action pool. Only task intent and past actions enter its
input; original target flags and future action descriptions are excluded.

Preserve native splits, encoded image bytes and stable source IDs. Keep source
interpretation, eligibility checks, candidate sampling and geometry calculations
here. Runtime task declarations should select canonical fields and ontologies;
a short grouped-QA expansion is retained to avoid repeating stored image bytes.
