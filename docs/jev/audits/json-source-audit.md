# English JSON-source audit — 2026-10-13

The English catalog contained ten JSON-backed task views across seven declarations. All nine views with genuine source training data loaded successfully. A JSON loader is a file format, not evidence that a dataset needs repackaging: the catalog now links the underlying repositories alongside the loader name.

| Source | Finding | Action |
|---|---|---|
| RelBERT lexical relations (five views) | Hub JSON and converted Parquet match exactly, including splits, labels and row order. | Keep the working JSON mappings; no mirror needed. |
| Headline Cause / English simple | Titles and labels match the Hub Parquet export. Only timestamp representations differ. | Keep the explicit JSON split selection. |
| PropSegmEnt / NLI | GitHub JSON and the author's Hub Parquet export match exactly. The hypothesis is the marked proposition; the premise is the document. | Keep the loader; fix the question to ask whether `text_B` entails the proposition in `text_A`. |
| Agent Action Safety | Explicit Hub JSON files select the classification data rather than mixing in SFT files. | Keep the existing file selection. |
| DocRED | Explicit Hub JSON files select annotated training and development data, excluding distant supervision and unlabeled test data. | Keep the mapping; no immediate mirror needed. |
| PROST | The publisher explicitly provides test examples only. | Move to `eval_only.py`; remove from training discovery and mix selection. |

Validation used up to 200 training rows per task and small evaluation samples. Seven complete JSON/Parquet comparisons covered the lexical-relation and PropSegmEnt views plus Headline Cause. Small Jev smoke tests checked PROST's evaluation interface and the corrected PropSegmEnt question. These checks establish loader and recast consistency; they do not certify every source annotation.

The [machine-readable report](json-source-audit.json) records comparison counts and source evidence. Existing published Jev exports have not been rebuilt by this audit, so moving PROST and correcting the prompt take effect when those exports are rebuilt.
