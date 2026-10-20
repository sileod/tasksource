# NER suites source audit — 2026-10-20

Added 15 training tasks and six evaluation-only MRQA views. Seven more UNER configs are converted and verified locally, pending a published data-only mirror and a real Hub pin. No publication or changelog edit was performed.

Counts below are **after cross-split duplicate / invalid-span removal, before training caps**, in train / validation / test order. A dash means no native split. Exact sentence overlap is removed with test > validation > train precedence; native membership is otherwise preserved. Native counts, exclusions, full revisions and smoke evidence are in [the machine-readable audit](ner-suites-2026-10-20.json). Caps apply to training rows.

## Added training tasks

| Task ID | Catalog | Train / validation / test | Cap | License |
|---|---|---:|---:|---|
| `bc4chemd` | English | 30,484 / 30,442 / 26,364 | 3,000 | cc0-1.0 |
| `harveyner` | English | 3,595 / 1,227 / 1,303 | 3,000 | unspecified |
| `msra-ner` | multilingual | 45,000 / — / 3,443 | 3,000 | unspecified |
| `weibo-ner` | multilingual | 1,272 / 262 / 270 | 2,000 | unspecified |
| `bionlp-cg/entities` | English | 300 / 100 / 200 | 2,000 | cc-by-sa-3.0 |
| `bioinfer/entities` | English | 894 / — / 206 | 2,000 | cc-by-2.0 |
| `pubmedqa/labeled-fold0` | English | 450 / 50 / 500 | 2,000 | mit |
| `uner/da_ddt` | multilingual | 4,380 / 564 / 565 | 5,000 | cc-by-sa-4.0 |
| `uner/en_ewt` | English | 12,192 / 1,924 / 2,077 | 5,000 | cc-by-sa-4.0 |
| `uner/hr_set` | multilingual | 6,913 / 960 / 1,136 | 5,000 | cc-by-sa-4.0 |
| `uner/pt_bosque` | multilingual | 7,015 / 1,172 / 1,167 | 5,000 | cc-by-sa-4.0 |
| `uner/sk_snk` | multilingual | 8,475 / 1,057 / 1,061 | 5,000 | cc-by-sa-4.0 |
| `uner/sr_set` | multilingual | 3,328 / 536 / 520 | 5,000 | cc-by-sa-4.0 |
| `uner/sv_talbanken` | multilingual | 4,303 / 504 / 1,219 | 5,000 | cc-by-sa-4.0 |
| `uner/zh_gsd` | multilingual | 3,997 / 500 / 500 | 5,000 | cc-by-sa-4.0 |

`unspecified` is retained explicitly, as allowed by docs/implementing-sources.md; the B2NERD MIT card and loader software licenses are never inherited as constituent data licenses. Original text rights remain applicable. All three new BigBIO tasks have caps of 2,000.

## Sources and label evidence

- **B2NERD:** reviewed the [constituent lists](https://github.com/UmeanNever/B2NER/tree/main/configs/ml_configs/bilingual_full) and [source card](https://huggingface.co/datasets/Umean/B2NERD) at Hub `290e7f34c3b7e13962765e1f7a0c777695c28f03`. Imported no aggregate, relabelled ontology, pruned selection or model labels.
- **CHEMDNER / bc4chemd:** chemistry-curator gold annotations, sentence token view on existing `disi-unibo-nlp/bc4chemd`; original [corpus publication](https://doi.org/10.1186/1758-2946-7-S1-S2) specifies CC0 for its data. BigBIO CHEMDNER has no actual Parquet shards at its conversion ref.
- **HarveyNER:** [original authors](https://github.com/brickee/HarveyNER) manually annotated fine-grained disaster-tweet locations. The existing Hub copy includes its reproducible source parser and native files. No original data license grant was established; third-party MIT claims were not adopted.
- **MSRA / Weibo:** original Hub loaders [MSRA](https://huggingface.co/datasets/levow/msra_ner) and [Weibo](https://huggingface.co/datasets/hltcoe/weibo_ner) declare human / expert annotations. Their converted Parquet commits load directly. Weibo keeps named and nominal tags separately. Licenses are explicitly unknown on the original cards.
- **UNER:** [original project](https://huggingface.co/datasets/universalner/universal_ner) documents volunteer annotation and transfers of human annotations, with CC BY-SA 4.0. Its original Hub upload is script-only, without a conversion ref; the existing [data-only copy](https://huggingface.co/datasets/BramVanroy/universal_ner) preserves tokens, ClassLabel names, annotator IDs and native splits. These eight configs cover the available v1 portion of the overall v2 suite, not the new v2 treebanks or later GitHub corrections.
- **BioNLP CG:** expert human cancer-genetics entities; the [original distribution license](https://github.com/openbiocorpora/bionlp-st-2013-cg/blob/master/LICENSE) explicitly licenses annotations CC BY-SA 3.0, despite BigBIO naming GENIA terms. Native train/dev/test documents loaded directly.
- **BioInfer:** [BigBIO source card](https://huggingface.co/datasets/bigbio/bioinfer) explicitly records CC BY 2.0 for the human protein/gene/RNA corpus. Native train/test loaded directly.
- **PubMedQA:** [original publisher](https://github.com/pubmedqa/pubmedqa) MIT release; expert-labeled PQA-L, official fold 0 only (450/50/500). Never use artificial PQA-A, unlabeled PQA-U, predictions, conclusions / long answers, or overlapping copies of all ten folds. Biomedical evidence and the research question are the only inputs.

Every active source SHA was read from the Hub API and is stored in `src/tasksource/metadata/source_revisions.py`. Script-only sources use actual converted Parquet commit SHAs. All added tasks use existing Hub datasets; no new mirror is needed for these active tasks.

## B2NERD constituents skipped

| Constituents | Verdict / reason |
|---|---|
| AnatEM, bc2gm, bc5cdr, Broad Tweet Corpus, CoNLL 2003, FabNER, NCBI disease, English OntoNotes, TweetNER7 | Already in the English catalog. |
| GENIA_NER | Near-duplicate biomedical GENIA coverage already provided by JNLPBA; no second recollection added. |
| CrossNER AI / literature / music / politics / science; MIT movie / restaurant | Already present with native training data, even though B2NERD uses them as held-out sets. |
| ACE 2004 / 2005; Chinese OntoNotes | Restricted original distributions; no blanket permission inherited from the recollection. |
| MultiNERD, PolyglotNER, WikiANN, WikiNeural; PileNER in the released code configs | Automatic / distant / model labels, not qualifying human gold. Recorded in `parked.NOT_ANNOTATED`. |
| FindVehicle | Source uses constructed vehicle descriptions and multiple overlapping tag views; no gate-complete human-gold source and data-license scope established. The B2NERD / generic re-upload does not establish those gates. |
| Chinese MultiCoNER 2023 | Already present as `multiconer_v2/Chinese (ZH)`; no recollection duplicate. |
| Boson; bank; CCKS 2017 / 2018 / 2019 / 2020 / 2021-address; Chinese literature; finance-sina; GAIIC 2022; IMCS21; NLPCC 2018; PeopleDaily2014; RuijinMCC; Taobao; Wanchuang; Microbiota; Resume; Youku; CAIL2021; CBLUE; math-high; Chinese MultiCoNER 2022; CLUENER | No gate-complete original distribution/config was established in this audit. Exact-name Hub searches mostly expose no dataset or third-party/modified uploads, not enough to prove corpus identity, native splits and original human-label/access scope. These are unresolved skips, not claims that the corpora are intrinsically invalid. No labels or licenses were inferred from B2NERD. Search evidence is retained in the JSON audit; archive-only candidates require their own reproducible conversion and publication before activation. |

The four added constituents are CHEMDNER, HarveyNER, MSRA and Weibo. The standardized B2NERD aggregate is skipped independently of these native annotations.

## UNER configs requiring a mirror

These configs were rebuilt by `scripts/repackage_dataset/uner.py`, through `python -m scripts.upload_repackaged uner_missing --dry-run`. Local Parquet data, source-file hashes, upstream commit SHAs, license URLs and per-config cards are in `build/uner-missing-release/`. All seven were loaded through TokenClassification and span-checked locally. Planned IDs below are **not active catalog tasks**.

| Planned ID | Native train / dev / test | Retained train / dev / test | License |
|---|---:|---:|---|
| `uner/qaf_arabizi` | 1,003 / 139 / 145 | 1,002 / 139 / 145 | cc-by-sa-4.0 |
| `uner/el_gdt` | 1,662 / 403 / 456 | 1,648 / 401 / 456 | unspecified |
| `uner/he_htb` | 5,241 / 484 / 491 | 5,241 / 484 / 491 | cc-by-sa-4.0 |
| `uner/nno_ndt` | 14,174 / 1,890 / 1,511 | 14,110 / 1,879 / 1,511 | cc-by-sa-4.0 |
| `uner/nob_ndt` | 15,696 / 2,409 / 1,939 | 15,628 / 2,398 / 1,939 | cc-by-sa-4.0 |
| `uner/sl_ssj` | 10,903 / 1,250 / 1,282 | 10,887 / 1,250 / 1,282 | cc-by-sa-4.0 |
| `uner/sv_lines` | 3,176 / 1,032 / 1,035 | 3,172 / 1,032 / 1,035 | cc-by-sa-4.0 |

Norwegian and Slovenian are original-author conversions of human annotations; Slovenian documents manual correction of differences in annotation guidelines. Greek documents two human annotators but has no dataset-specific license file, so is recorded unspecified. No task declaration points to a nonexistent Hub revision. The maintainer must publish `tasksource/uner-missing`, read its real Hub SHA, and then activate these configs.

Skipped UNER configs: `zh_gsdsimp` duplicates `zh_gsd` by simplified-script conversion; PUD treebanks, Cebuano GJA, Tagalog TRG and Ugnayan have no training split; Romanian LegalNERo is distributed as one unsplit file, without a native train split. Post-v2 Indonesian GSD / CSUI additions are outside the requested v2 release.

## MRQA

SQuAD is skipped because the existing `squad-v2` extraction view covers the SQuAD v1 answerable data. TriviaQA-web and SearchQA are skipped for distant span supervision, explicitly excluded by the repository gates; recorded in `parked.NOT_ANNOTATED`. Existing NaturalQuestionsShort, NewsQA and HotpotQA mappings are unchanged.

The six held-out subsets are added only to `eval_only.py`. The normal loader interface exposes source test data in its internal train slot (as existing eval-only tasks do), but these mappings never enter a training catalog; metadata retains `source_split=test`. The loaded source is pinned `mrqa-workshop/mrqa` at `c0499a54c1e32765044572e1d685bd16cf126a80`.

[Official MRQA normalization](https://github.com/mrqa/MRQA-Shared-Task-2019#datasets) permits case, punctuation, articles and whitespace differences. Supplied end-inclusive offsets are checked against the detected gold text under those exact rules; output answers are the verbatim source slices. No relocation or answer search is performed. Invalid occurrences are removed; positive rows without a remaining occurrence are dropped, never turned into nulls. These are checked subsets of the evaluation data, not unfiltered benchmark scores.

| Evaluation ID | Native test | Retained test | Rows dropped | Occurrences dropped |
|---|---:|---:|---:|---:|
| `mrqa-eval/BioASQ` | 1,504 | 1,481 | 23 | 34 |
| `mrqa-eval/DROP` | 1,503 | 1,462 | 41 | 183 |
| `mrqa-eval/DuoRC.ParaphraseRC` | 1,501 | 392 | 1,109 | 6,053 |
| `mrqa-eval/RACE` | 674 | 673 | 1 | 2 |
| `mrqa-eval/RelationExtraction` | 2,948 | 2,948 | 0 | 1 |
| `mrqa-eval/TextbookQA` | 1,503 | 1,482 | 21 | 42 |

All MRQA evaluation views have no cap and unspecified aggregate license scope: constituent terms continue to apply; the MRQA repository MIT software license is not a blanket data license.

## BigBIO selection and span audit

Added three diverse bounded sources: cancer genetics NER, BioInfer protein/gene/RNA NER, and expert biomedical research-question classification. Skipped overlapping MedMentions, Chia, ChemProt, Hallmarks of Cancer, AnatEM, BC5CDR, JNLPBA, LINNAEUS, BioRED, EBM-PICO and MQP (already cataloged). CHEMDNER is added once through its usable token upload.

Other inspected candidates: `tmvar_v3` explicitly has no native train/evaluation split; `medical_data` and `psytar` are not public per their BigBIO cards; MEDIQA RQE human-label provenance was not established in this audit; LitCovid mixes manual/model annotation descriptions without a row-level provenance check here. These were not added. The family is deliberately not exhaustive.

Every native BigBIO passage and retained entity segment was verified against exact character offsets. Discontinuous BioInfer annotations are split into contiguous runs; all members of an overlap are explicitly excluded and counted. No input row is dropped by this flattening.

| Task | Overlapping runs dropped train / dev / test | Discontinuous mentions split train / dev / test |
|---|---:|---:|
| `bionlp-cg/entities` | 258 / 42 / 165 | 0 / 0 / 0 |
| `bioinfer/entities` | 85 / — / 18 | 106 / — / 16 |

## Verification

- All 15 new catalog tasks were loaded through `load_task`, five rows per available native split. All six eval-only tasks were loaded, including full retained-span validation. All seven staged UNER configs were locally loaded, five rows per split.
- Original split sizes and exact cross-split overlap were audited across the full sources before capping; all newly standardized training-task splits are disjoint by exact token sequence / document identity.
- Extraction smoke builds: `scripts/build_jev_extraction.py --tasks bionlp-cg/entities bioinfer/entities bc4chemd uner/en_ewt --max-rows 300 --max-rows-eval 50` and a separate HarveyNER run. Each output was span-checked. The first build yielded 1,200/150/200 rows; HarveyNER yielded 300/50/50.
- Classification smoke: `scripts/build_jev_dataset.py --tasks pubmedqa/labeled-fold0 --max-rows 20 --max-rows-eval 5 --pack-rate 0 --english-only` succeeded; Jev variants yielded 24/6/6 rows. No upload flag was used.
- `PYTHONPATH=.:src python -m pytest -q tests/test_ner_suites.py tests/test_extraction.py tests/test_task_catalogs.py tests/test_catalog_naming.py tests/test_licenses.py tests/test_parked.py tests/test_offline_import.py`: **87 passed**.
- `git diff --check` passed. `DATA_CHANGELOG.md` is unchanged.
