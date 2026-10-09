# Dataset repackaging

Source-specific preparation lives in the Python modules in this directory. Publish a pinned,
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

## Archive-only HTML and Super-CLEVR

```bash
python -m scripts.repackage_dataset websrc superclevr --max-rows 1000 --max-rows-eval 100 --seed 42
```

`browser.py` prepares WebSRC's native yes/no and answer-element views as separate
configs. It preserves original HTML and website splits, verifies native span
offsets, and records every rejected annotation. Element candidates are all
nonempty native nodes on eligible pages; the 128-candidate cap excludes whole
pages independently of the target. Public test answers are not used. Cleaned
WebLINX is already loadable directly and needs no mirror or new HTML cleaner.

`superclevr.py` joins pinned native question tables, functional programs and
the image archive without extracting thousands of files. It stores original
PNG bytes once per sampled image and canonical QAs tagged by program-derived
view. Row caps count questions across the whole family, not per view. The full
native ontology and image split disjointness are checked before sampling.

## Region, preference and correspondence sources

```bash
python -m scripts.repackage_dataset coco_regions doclaynet_region bapps spair71k_grid \
  --max-rows 5000 --max-rows-eval 100 --seed 42
```

Use `--dry-run` to inspect the local Parquet and provenance before publishing.
The caps bound eligible examples per native split; selection uses a stable hash
over the complete annotation pool, independent of labels and source ordering.
Mirrors record that they contain a bounded sample, the full source ontology,
rendering parameters, image hashes, original geometry and exclusion reasons.

`coco_regions` publishes two configs, `lvis` and `panoptic`, in
`tasksource/coco-regions`. Both use one pinned COCO image cache. LVIS polygons
and native panoptic segment masks receive the same red highlight, without
class-dependent styling. Training requires native training annotations and
COCO train2017 images; validation requires both native validation annotations
and COCO val2017 images. Conflicting partitions and crowd regions are excluded.
COCO's per-image licenses and attribution remain in metadata; NoDerivatives
images are excluded. Annotation licensing does not replace image licensing.

`doclaynet_region` reuses the official v1.1 Parquet source, preserving its
native train/val/test partitions and eleven layout classes. It checks page and
document identity across splits. Red rectangles identify regions; PDF text
cells and annotation labels never enter model inputs. Conflicting labels for
an identical rectangle and out-of-frame boxes are excluded.

`bapps` reads official 2AFC archives. Images remain ordered reference, p0, p1;
text choices refer to images 2 and 3, including after Jev option permutation.
The native judgment is the fraction preferring p1. Ties are excluded; vote
fractions and counts remain in metadata. The original code's BSD license is
not asserted as the dataset's license; dataset terms remain unspecified.

`spair71k_grid` preserves native image-disjoint partitions. A red cross marks
the source keypoint, while the target image keeps its encoded bytes. Gold
labels are target-image cells in a 7×7 grid. Original keypoint coordinates,
image identities and geometry remain in metadata, separate from inputs.
Official archive content hashes are verified before either archive-based
conversion runs. SPair's PASCAL/Flickr source terms remain applicable.
