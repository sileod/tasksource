# Loading-script migration notes (outdated-datasets.json, 125 entries / 63 families)

## MetaEval namespace transfer (2026-09-23)

All 27 datasets still owned by the `metaeval` Hub organization were moved to
`tasksource/` with their basenames unchanged. No destination name collided;
the old Hub URLs redirect to the new canonical repos. The transferred names
are `acceptability-prediction`, `ambient`, `chaos-mnli-ambiguity`,
`children-tom`, `cnli`, `ethics`, `figurative-nli`, `mega`,
`mega-acceptability-v2`, `mindpixels`, `multilingual-persuasion`,
`nli-veridicality-transitivity`, `nli4wills`, `offensive-humor`,
`path-naturalness-prediction`, `rankme-nlg-acceptability`,
`reqeval-ambiguity-detection`, `scruples`,
`semantic-feature-production-norms`, `shinet-2hop`,
`syntactic-augmentation-nli`, `twentyquestions`, `universal-joy`,
`utilitarianism`, `wouldyourather`, `x-fact`, and `xnli`.

`defeasible-nli`, `equate`, and `reclor` were already at `tasksource/` behind
old MetaEval redirects; their active annotations now use the canonical IDs.
`metaeval/disrpt` resolves to `multilingual-discourse-hub/disrpt` and was not
part of this two-organization transfer. The active annotation uses that
canonical URL. Dataset basenames (hence Tasksource task IDs) are unchanged.
Three transferred repositories remain script-only on the Hub: `mega`,
`utilitarianism`, and `xnli`. Active Tasksource annotations do not execute
those scripts: utilitarianism reads the original Hendrycks CSVs with the
script's seeded comparison orientation, and XNLI reads `facebook/xnli`.
`mega` has no active Tasksource annotation; its source files remain in the
transferred repo for a later data-only conversion.

## Jev failure repair after the one-million-row release

The first release completed 538/571 selected tasks. Thirty failures were
integer labels without `ClassLabel` names, not failed downloads. Tasksource's
`label_values` annotation now records an explicit, source-checked
integer-to-name mapping and rejects any unmapped value. This covers the NLI
families, HoVer, GitHub issue similarity, Amazon review stars, and
persuasion-score levels without guessing a label order at recast time. The
GitHub issue polarity was checked against positive and negative source pairs;
the NLI mappings come
from the [pietrolesci source cards](https://huggingface.co/pietrolesci/datasets);
the persuasion scale is the source's signed shift in support, not a binary
persuaded/not-persuaded label.

The other three failures had different causes: `TroFi` declares an empty
validation split in its card, so the annotation now loads its actual
train/test Parquet files; `CogALexV` has no validation JSONL, so its annotation
loads train/test and fixes its complete five-relation ontology before sampling;
`twentyquestions` pointed at a missing repo, and now uses the transferred
`tasksource/twentyquestions` data while omitting rows with no source answer.
The old 33-task failure set passed a bounded Jev smoke after these repairs.

Policy: prefer popular HF data-only repos with the same format (same basename to
preserve Tasksource ids); preserve preprocessing, label meaning, and
train/dev/test boundaries; keep BIG-bench, MMLU, BLiMP excluded.
Rule: only push a `tasksource/` copy when no good loadable alternative exists;
never delete others' datasets (only my own prototypes: `tasksource/piqa`,
`tasksource/AfriSenti-twitter-sentiment`, `tasksource/NusaX-senti`,
`tasksource/hope_edi`, `tasksource/numer_sense`,
`tasksource/multilingual-sentiments`, `tasksource/TuringBench`,
`tasksource/summarize_from_feedback`, `tasksource/health_fact`,
`tasksource/mc_taco`, `tasksource/phrase_similarity`, `tasksource/head_qa`,
`tasksource/wiki_hop`, `tasksource/prost`, `tasksource/discosense`,
`tasksource/clutrr`, `tasksource/docred`,
`tasksource/lexical_relation_classification`, `tasksource/propsegment`,
`tasksource/sharc_modified`, `tasksource/ethics`, `tasksource/dream`, and
`tasksource/tweet_sentiment_multilingual` — removed after direct/loadable
alternatives were wired into Tasksource. No user-owned datasets were deleted.
Mirrors stay full-scale; capping happens in the JEV build (`max_rows`).

## Fixed: generic raw-file alternatives (no mirror copies)

`SharedFields` now supports `load_dataset_kwargs` and a Tasksource `task_id`
override. This lets Tasksource use the generic CSV/JSON builders against public
raw files while preserving the original Tasksource task id:

| task | source / handling |
|---|---|
| `hope_edi/english` | Official Google Drive TSVs, loaded via generic CSV builder (train/validation, three original labels) |
| `numer_sense` | Official `INK-USC/NumerSense` `train.masked.tsv` via generic CSV builder; masked test files remain excluded as original Tasksource mapping did |
| `args_me` | Existing `webis/args_me/args-me.jsonl` via generic JSON builder; decode `premises[0]` into argument/stance, keep conclusion. No copied repo |
| `multilingual-sentiments/all` | Official raw GitHub CSVs via generic CSV builder; exact `source` filter removes `amazon_reviews`; no copied repo |
| `emo` | Loadable `oneonlee/cleansed_emocontext`, turns concatenated; label meaning retained. Task ID becomes `cleansed_emocontext` |
| `ethos_binary` | `SetFit/ethos_binary` (existing data-only alternative), original train+test pool concatenated before Tasksource's deterministic train/dev/test split |
| `xglue/qam`, `xglue/qadsm`, `xglue/wpr` | Official XGLUE archive distributed by `forresty/xglue` rebuilt to `tasksource/xglue` English train/dev/test with original fields/class names; no other XGLUE tasks copied |
| `miam` | Official `eusip/MIAM` DIHANA CSV train/dev/test through generic CSV loader; original 11 ClassLabel names restored |
| `mms` | `Brand24/mms/data/**/*.tsv` loaded directly by generic CSV builder in streaming mode; script's -1/0/1 → negative/neutral/positive filter and mapping retained; bounded materialization happens at the caller's `max_rows` |
| `rumoureval_2019/RumourEval2019` | Existing Hub CSVs through generic CSV loader (no copy) |
| `Touche23-ValueEval` | Zenodo argument TSVs through generic CSV loader (no copy); exact task fields Premise/Conclusion/Stance |
| `webgpt_comparisons` | `heegyu/webgpt_comparisons_ko` has the exact English question/answers/scores plus Korean translations; parsed stringified question struct; no copy |
| `hope_edi/english`, `numer_sense`, `multilingual-sentiments/all` | Official Google Drive/GitHub CSV/TSV data through generic CSV loader (no copy); source filter and split/label semantics preserved |
| `emo/emo2019` | Loadable `oneonlee/cleansed_emocontext` alternative (turns joined; original 4 emotion labels); task ID is `cleansed_emocontext` |
| `blog_authorship_corpus/job` | Existing tasksource mirror's `topic` column is the original job/industry label; mapped without uploading another copy |

## Fixed: XL-WiC pair-format mirror

`multilingual/xlwic/{xlwic_de_de,xlwic_en_ko,xlwic_fr_fr,xlwic_it_it}` now uses
`tasksource/xlwic`, rebuilt from the official `xlwic_datasets.zip`. It keeps the
target word, both contexts, POS, locations and boundaries; ClassLabel names are
`different/same` (`1` means same sense). No good data-only pair-format HF
alternative covered these four pairs, so this is an allowed tasksource mirror.

## UD loader source

The multilingual POS and English dependency-relation token tasks now use
`universal-dependencies/universal_dependencies`, the existing multilingual
Parquet release. Dependency-relation label vocabularies are derived per
dataset/config from its actual train/dev/test labels, avoiding stale hard-coded
English inventories.

## Fixed: tail wave 2 (few_rel, relbert, propsegment, sharc, scicite)

| task | change |
|---|---|
| few_rel/default | `tasksource/few_rel` from `thunlp/FewRel` GitHub (`train_wiki/val_wiki/val_nyt` + `pid2name.json`, exact script schema). Also fixed a latent recast gap: `_fewrel_relation_match` emitted bare ints (the original could never recast) — now `ClassLabel[negative, positive]` |
| lexical_relation_classification/×5 | Existing Hub-hosted `relbert/…/dataset/*/*.jsonl` loaded through generic JSON (`head`/`tail`/`relation`; BLESS/CogALexV/EVALution/K&H+N/ROOT09); no mirror |
| propsegment/nli | `schen149/PropSegmEnt` raw `propnli.*.jsonl` through generic JSON (`n/e/c` → entailment/neutral/contradiction verified); no mirror |
| sharc_modified/mod | `nikhilweee/neural-conv-qa` raw `mod_train/dev.json` via streaming JSON; only required fields materialized; no mirror |
| scicite | `tasksource/scicite` from the s3 tarball (`background/method/result` verified; NaN `sectionName` guarded) |

## Fixed: tail wave 1 (ethos, dream, hate_speech18, social_i_qa, wiqa, humicroedit, scifact)

| task | change |
|---|---|
| ethos_binary | `ethos` → `SetFit/ethos_binary` (`text` + `label`→`no hate speech/hate speech`; renamed var `ethos___binary`→`ethos` so id is `ethos_binary`; `/multilabel` untouched, out of scope) |
| dream | `nlpdata/dream` GitHub JSONs via generic JSON and a batched flatten (`[dialogue, QAs, id?]`; train/dev/test); no mirror |
| hate_speech18 | `tasksource/hate_speech18` from Vicomtech repo archive (`noHate/hate/idk-skip/relation`; the `Intuit-GenSRF` mirror's vote lists are all empty) |
| social_i_qa | `tasksource/social_i_qa` from ai2-mosaic zip (labels lst 1-based → 0-based; train/validation) |
| wiqa | `tasksource/wiqa` from aristo zip (flattened `question.{stem,choices,answer}`; train/validation/test) |
| humicroedit/subtask-2 | `tasksource/humicroedit` from rochester zip (CSV `0/1/2` → ClassLabel `equal/sentence1/sentence2`) |
| scifact_entailment | `tasksource/scifact_entailment` from s3 `data.tar.gz` replicating the script's (claim, cited_doc) pairing incl. `NEI` (test claims carry no evidence → train/validation only, as original) |

# Hub hosting policy

Mirrors stay full-scale; capping happens in the JEV build (`max_rows`/`max_rows_eval`),
not on the Hub. (An earlier trim-to-caps pass was fully reverted, including a
git-history restore of dynahate's train split.) Removed outright, not capped:
silicone `swda`+`mrda` configs (unused by catalog) and repair-session tasksource
duplicates whenever an existing data-only alternative was verified. Only those
copies were deleted; user-owned datasets were left in place.

## Fixed: alternatives group (massive, x-stance, finphrasebank, blog/gender, offenseval, xglue_nc, dynasent)

| task | change |
|---|---|
| MassiveIntentClassification/×51 | `AmazonScience/massive` → `mteb/MassiveIntentClassification` (`text`/`label` strings; ids change to 2-letter codes) |
| xstance | `strombergnlp/x-stance` → `michiel/xstance` (`question_<lang>`/`comment`/`stance_label` AGAINST/FAVOR; id `xstance`) |
| financial-phrasebank-all-agree-classification | `financial_phrasebank` → `ghbacct/…` (`text`→`sentence`; train+test concatenated to reproduce the 2264-row single pool) |
| blog_authorship_corpus/gender/job | Existing `tasksource/blog_authorship_corpus` CSV; `topic` is the original job/industry label and is ClassLabel-cast using the full vocabulary before sampling |
| offenseval_2020/ar | → `khalidalt/offenseval_2020_ar` (same `text`/`subtask_a` mapping) |
| offenseval_2020/da/gr/tr | new `tasksource/offenseval_2020` (ids preserved): da/gr from the dead repo's raw TSVs, tr parsed from `stefan-it` fastText txt; uniform `text`/`subtask_a(NOT/OFF)` schema |
| xglue_nc | `xglue/nc` → `SetFit/xglue_nc` (10 readable categories; labels are complete on the JEV-scale sample) |
| dynasent/r1+r2 | `dynabench/dynasent` → `tasksource/dynasent` rebuilt from `cgpotts/dynasent-v1.1.zip` with ternary filter (ids shortened to `dynasent/r1+r2`; the `HelloWorld2307` mirror merges rounds) |

## Fixed: data-source batch 2 (head_qa, wiki_hop, prost, discosense, trec, liar, math_qa, clutrr, docred, tweet_sentiment)

Data-only alternatives are used directly where available; tasksource mirrors remain for TREC, LIAR and MathQA.

| task | source |
|---|---|
| head_qa/en | `EleutherAI/headqa` en (answer structs/`ra`/`qid` ints; no copy) |
| wiki_hop/original | `MoE-UNC/wikihop` (`query`→`question`; train/validation, no test; no copy) |
| prost | `corypaik/prost` `data/default.jsonl` via generic JSON; no copy |
| discosense | `prajjwal1/discosense` GitHub JSONs via generic JSON; no copy |
| trec | CogComp `train_5500.label`/`TREC_10.label` with the original 50-name order (the `KushT` mirror's ints follow a different order) |
| liar | `cs.ucsb.edu` `liar_dataset.zip` (all 14 cols, 6-way order `false…pants-fire`) |
| math_qa | `math-qa.github.io` `MathQA.zip` (full 29,837 train — better than the `Calc` subset) |
| clutrr/gen_train234_test2to10 | `kendrivp/CLUTRR_v1_extracted` exact JSON config (`story`/`query`/`target_text`; no copy) |
| docred | `YufeiHFUT/DocRED_origin` + `_yufei_docred_to_columnar` adapter (`h/t/r` → head/tail/relation_id; generic JSON, no copy) |
| tweet_sentiment_multilingual | `mteb` per-language JSONL files via generic JSON; `0/1/2`→`negative/neutral/positive`; no copy |

## Fixed: batch 2 data sources (7 entries; only missing-source cases mirrored)

Rebuilt from original sources (not lossy third-party reformats):

| task | source |
|---|---|
| TuringBench | original AA CSVs via `jana4/turingbench-humanized` generic CSV loader; 20 generator-string labels preserved. A first attempted binary re-sample was discarded; no tasksource copy remains |
| contract-nli/contractnli_a/seg + contractnli_b/full | `cognitivplus/contract-nli` zips (script's own `MAIN_PATH`) → `tasksource/contract-nli` with both configs, `contradiction/entailment/neutral`; seg=span / full=document distinction kept (the `presencesw` mirror is B-only merged) |
| summarize_from_feedback/comparisons | `vwxyzjn/summarize_from_feedback_oai_preprocessing` exact data-only schema; no tasksource copy |
| health_fact | `marcov/health_fact_promptsource` core claim/label cols (`false/mixture/true/unproven` verified); no copy |
| mc_taco | `marcov/mc_taco_promptsource` validation/test (`no/yes` verified); no copy |
| phrase_similarity | `Deehan1866/processed_phrase_similarity` (`negative/positive` verified); no copy |

`open_question_type`/`TuringBench`/`mtop` note: tiny `max-rows` smokes can miss labels on
10-/20-/100-way tasks; full-scale loads verified.

## Fixed: mtop + dynahate (2 entries)

| task | change |
|---|---|
| multilingual/mtop | `tasksource/mtop` rebuilt in place from original `mtop.zip` (`dl.fbaipublicfiles.com`, via `fb.me` redirect): all 6 langs, `idx/intent/spans/question/domain/lang/logical_form/tokenized_question`, train/validation/test; `mtop.py` deleted. No code change. Full-scale load verified (73k train, `IN:SEND_MESSAGE` recast) |
| dynahate | `aps/dynahate` → `tasksource/dynahate` (basename same, id preserved); repo fixed in place (12 round jsonl had a null/string schema clash → replaced with clean concatenated train/validation/test parquet, same rows/columns; labels `hate/nothate` verified in recast) |

## Fixed: multilingual and MetEval-family failures (29 entries)

| family | entries | change |
|---|---|---|
| AfriSenti-twitter-sentiment | 12 | `mteb/AfriSentiClassification` (popular data-only parquet; same 12 configs/splits; ints re-attached as `positive/neutral/negative`, order verified vs upstream TSVs). Own `tasksource/AfriSenti-twitter-sentiment` mirror DELETED per no-bloat rule. Ids become `AfriSentiClassification/<lang>` |
| NusaX-senti | 12 | `mteb/NusaX-senti` (popular data-only parquet; same basename so ids preserved; ints re-attached as `negative/neutral/positive`, order verified). Own `tasksource/NusaX-senti` mirror DELETED per no-bloat rule |
| ethics | 4 | `hendrycks/ethics` CSV files are read directly through the generic CSV loader with their original train/test/test_hard boundaries and virtue `[SEP]` split. An earlier MetaEval mirror now lives at `tasksource/ethics`, but these annotations use the original files. |
| xnli | 1 | `metaeval/xnli` → `facebook/xnli` English config (official data-only; `premise/hypothesis/label` and entailment/neutral/contradiction order checked; original `multilingual/xnli` task ID preserved) |

`indonlp/NusaX-senti` is script-only, so the loadable `mteb/NusaX-senti` source is
used with the original label names/order restored.

## Fixed: script-rebuild batch (25 entries, ids preserved, done first per instruction)

Rebuilt from the legacy scripts' own archives, preserving schema/splits/labels;
scripts deleted so the repos are data-only parquet:

| family | entries | repo | method |
|---|---|---|---|
| pragmeval | 8 (`emergent,gum,mrda,pdtb,sarcasm,stac,switchboard,verifiability`) + 12 more configs preserved | `sileod/pragmeval` (yours; `pragmeval` resolves to it) | Dropbox `pragmeval.zip` → per-config `train/validation/test.parquet` via script's csv logic + `TASK_TO_LABELS`; README `configs:` declared; card updated with LREC 2022 citation (https://aclanthology.org/2022.lrec-1.255/, `sileo-etal-2022-pragmatics`) plus 2019 arXiv entry |
| silicone | 8 (`dyda_da,dyda_e,iemocap,maptask,meld_e,meld_s,oasis,sem`; `mrda,swda` also preserved) | `tasksource/silicone` (new; `silicone` itself is `eusip/silicone`, not ours) → `tasks.py` now `dataset_name="tasksource/silicone"` (basename same, ids preserved) | `eusip/SILICONE-benchmark` CSVs → script's pandas logic (`Label`/`Idx`); README `configs:` declared |
| crowdflower | 9 (all) | `tasksource/crowdflower` (in place; `crowdflower.py` deleted) | Dropbox `crowdflower.zip` → per-config `train.parquet` via script's latin-1 csv logic + `TASK_TO_LABELS`; README `configs:` declared |

Verification: `load_task` + `recast_jev` spot-checked
(`pragmeval/emergent` NLI pair, `verifiability`, `switchboard`/`mrda` dialog acts;
`silicone/meld_e` surprise, `oasis` reqInfo; `crowdflower/airline-sentiment` negative,
`text_emotion` love). No `tasks.py` mapping change needed for pragmeval/crowdflower
(same repo names); silicone needed the one-line `dataset_name` pointer above.

## Fixed in this pass (7 entries, ids preserved)

| task (outdated id) | old source | new data-only source | verification |
|---|---|---|---|
| piqa | `piqa` (script) | `baber/piqa` (parquet, `goal/sol1/sol2/label` ClassLabel 0/1) | `load_task('piqa')` + `recast_jev` 3 examples checked; JEV smoke ok |
| cosmos_qa | `cosmos_qa` | `Samsoup/cosmos_qa` (parquet, `context/question/answer0-3/label`; test `-1` filtered as before) | recast checked; JEV smoke ok |
| riddle_sense | `riddle_sense` | `jeggers/riddle_sense` (parquet; `choices` stringified list with `X:` prefixes stripped via `_parse_jeggers_riddle_choices`) | recast checked (water/cellout examples); JEV smoke ok |
| banking77 | `PolyAI/banking77` | `legacy-datasets/banking77` (parquet, `text/label` 77 intents) | recast checked; JEV smoke ok |
| logiqa | `lucasmccabe/logiqa` | `fireworks-ai/logiqa` (parquet, `context/question/options(list)/answer(letter)`; `_logiqa_options` strips `X.` prefixes, maps letter→int, `question`→`query`; no validation split in source so val is split from train) | recast checked; JEV smoke ok |
| moral_stories/full | `demelin/moral_stories` | `LabHC/moral_stories` (parquet train-only 12k; same `situation/intention/moral_action/immoral_action` as `moral_stories_full.jsonl`; val/test auto-split as before since full file had no splits) | recast checked; JEV smoke ok |
| open_question_type | `launch/open_question_type` | `Korea-MES/open_question_type` (parquet, `question/resolve_type` 10 types) | recast checked with full splits; note: tiny `max-rows` smoke (50) can miss labels — full build (30k/3k) keeps all 10 |

Helpers: `_parse_jeggers_riddle_choices`, `_strip_option_prefix`, `_logiqa_options`.
Tests: `tests/test_migrated_loaders.py` (helpers + data-only source assertions).
Smoke: `scripts/build_jev_dataset.py --tasks glue/rte` ok; `pytest -q -c /dev/null tests/test_recast_jev.py` 17 passed;
migrated 7-task JEV smoke: 6 ok, `open_question_type` needs full-size splits (see note), 0 script errors.

No unresolved failed task families remain from the recorded 125 entries. `ethos/multilabel`
and `humicroedit/subtask-1` are catalog siblings, not entries in the outdated list,
and were left unchanged.


Exclusions kept: `bigbench/`, `mmlu/`, `blimp/` (untouched in `scripts/build_jev_dataset.py`).
Full release rebuild/upload intentionally not run. Per `scripts/build_jev_dataset.py`,
build a small smoke first, inspect Parquet rows + `build-report.jsonl`, then run a
full `--finalize` without `--upload` before any publish.
