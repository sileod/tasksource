# Token source status for tasksource-jev

Checked on 23 September 2026 with the current `datasets` loader and a small
end-to-end Tasksource recast. The release admits token tasks only when their
source splits and label names can be checked and their ontology has 2–32
readable choices. The build report covers other Tasksource source failures.

| Tasksource source | Status | Work needed |
|---|---|---|
| `conll2003/ner_tags` | Included via [tomaarsen/conll2003](https://huggingface.co/datasets/tomaarsen/conll2003). | The original loader uses a forbidden script. The data-only copy retains the 14,041/3,250/3,453 train/dev/test counts and nine BIO NER labels. |
| `wnut_17/wnut_17` | Included via [flaitenberger/wnut_17](https://huggingface.co/datasets/flaitenberger/wnut_17). | The original loader uses a forbidden script. The data-only copy retains the 3,394/1,009/1,287 splits and 13 BIO labels. |
| `conll2003/pos_tags` | Excluded. | The original loader uses a forbidden script. The checked copy has 47 POS labels, above the current 32-choice gate. A curated smaller ontology would change the task and needs a separate design. |
| `ncbi_disease/ncbi_disease` | Excluded. | The original loader uses a forbidden script. A checked data-only copy has 2,930/489/539 rows rather than the source card's 5,433/924/941, so equivalence is unverified. Find or make a faithful data-only export. |
| `jnlpba/jnlpba` | Excluded. | The original loader uses a forbidden script. Find or make a faithful data-only export and verify its labels and splits. |
| `ontonotes_english/SpeedOfMagic--ontonotes_english` | Excluded. | Tasksource requests an obsolete config name. The current default config has raw integer NER labels without `ClassLabel.names`; restore and verify the ontology before recasting. |
| `few-nerd/supervised` | Excluded. | It loads, but its fine-grained ontology exceeds the 32-choice gate. |
| `multilingual/xglue/ner`, `multilingual/xglue/pos` | Excluded. | Their original loader uses a forbidden script. Find faithful data-only exports with readable label metadata. |
| `multilingual/universal_dependencies/pos` and English dependency-relation configurations | Excluded. | The registered `universal_dependencies` source is unavailable. Find a faithful export and verify POS/dependency label order. |
| `multilingual/multiconer_v2` | Excluded. | Its original loader uses a forbidden script. Find a faithful data-only export with readable label metadata. |
| FINER-139, MAPA, acronym identification, chunking | Not enabled. | Inspect label semantics, ontology size, and source compatibility before adding them. |

These are release decisions, not claims that the underlying datasets are
incorrect. Where a mirror is used, the original Tasksource identifier remains
in `source`; this table records the actual data repository.
