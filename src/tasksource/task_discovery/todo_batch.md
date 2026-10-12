# Human-curated dataset queue

This is the user-approved queue, in batch order. Keep it separate from machine-discovered candidates. Discovery may suggest additions for review; it must not rewrite or silently promote entries here. Update statuses only after checking code and publication.

## Batch 1 — implemented and published

15 sources, 28 task views; Git `140072e`; Jev Hub commit `9d0d1c7dd5f88e1295f2f2bfb954200c26c08e10`, added to `default` and `full`.

MAUD (six views), ACORD, Evidence Inference, DDI, Hallmarks of Cancer, Patent Classification (abstract), CUAD answerability, TabFact, SQuAD v2 answerability, MultiDoGO intent (six domains), InfoTabS, WikiSQL (aggregation/column/operator), EDOS, SemEval 2014 ABSA (laptops/restaurants).

## Batch 2 — reviewed; accepted views implemented

| Source | Requested views | Status / remaining checks |
|---|---|---|
| FEVEROUS | Evidence-conditioned supports/refutes/NEI | Deferred: available Hub evidence joins omit necessary table columns/context; concrete examples and reconsideration condition recorded in `decisions.json` |
| HateXplain | Majority classification; optional votes | Majority view published; existing vote view retained |
| QuAC | Answerability, yes/no/neither, follow-up | Answerability published; yes/no and follow-up deferred after ambiguous native labels in coarse review |
| SLURP | Intent; possible slot tagging | Text-only mirror and 60-class scenario/action intent published; native-token-aligned slot types implemented and reviewed |
| MultiWOZ 2.2 | Intent, dialogue act, categorical slots | Dialogue act and categorical slots published; intent deferred after stale active-intent labels in coarse review; single-act rows only |
| Schema-Guided Dialogue | Intent and categorical slot values | Intent published; categorical-slot projection deferred after ambiguous economical → moderate state in native dialogue 1_00000, turn 12 |
| HotpotQA | Native supporting-fact membership | Published; explicit native sentence identities |
| MCScript2.0 | Native two-choice comprehension | Original XML mirror and native MC view published; license still unspecified |

Eleven views across batches 2–3 were added to Jev `default` and `full` in [Hub commit f2529310](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/f2529310ce099771d5d6911e94ba24f4b1ce3ec4): 10,272 train, 1,060 validation and 795 test rows. One exact exported training row per task was inspected, alongside source/recast samples; [review findings](../../../docs/jev/dialogue-expansion-audit.json) preserve the model identifier, sample IDs, revisions and limitations. Existing Parquet shards were preserved.

## Batch 3 — reviewed; accepted views implemented

| Source | Requested views | Status / remaining checks |
|---|---|---|
| UKPConvArg1Strict | More convincing argument, native pair preferences | Implemented from original Strict TSVs; 8,819/1,303/1,528 rows in documented topic-disjoint derived train/validation/test |
| VAST | Topic-conditioned pro/con/neutral | Deferred after native row 18506 contradicts its omnivore topic; mirror records the quality finding |
| FNC-1 | Headline/body stance | Implemented original CSV join with complete article bodies; 49,753 train / 25,413 native test, body-disjoint |
| DiscoGeM | Discourse relation / annotator votes | Implemented native annotator votes; 1,282 train / 1,288 test after removing paragraph overlap; no validation |
| UKP Sentential Argument Mining | Topic-conditioned pro/con/no argument | Deferred: official archive requires authentication; sentence reconstruction and separate text rights must be verified |
| CoQA | Native exact yes/no with prior dialogue | Published to Jev default/full |
| QASPER | Native scientific yes/no; possible evidence selection | Binary consensus view published; all cited evidence must occur in supplied text; answer style is not used as a target; optional evidence projection deferred because nonselected paragraphs lack explicit negative judgments |
| TOPv2 | Root intent; possible slot types | Root intent published to Jev default/full; leaf slot-type projection implemented and reviewed |
| PERSPECTRUM | Claim/perspective supports/opposes | Implemented primary native perspectives and official claim partitions; 2,939/910/1,204 rows |
| PubMed 200k RCT | Sentence rhetorical role | Implemented pinned plaintext mirror; 2,211,861/28,932/29,493 sentences, native PMID-disjoint partitions; original eval files repair a malformed Hub dev copy |

Five further views were published to Jev `default` and `full` in [Hub commit 71f31c69](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/71f31c69db4325a92f8639093362578b696ba62f): FNC-1 stance, PERSPECTRUM stance, UKPConvArg convincingness, DiscoGeM relation votes and PubMed RCT role. Added 4,998 train / 300 validation / 500 test rows; previous Parquet blobs were preserved. [Source and final-export review](../../../docs/jev/stance-expansion-audit.json) records examples, native label caveats, source revisions and reviewer identity. Git implementation: `c4d35fb`; regression checks: 205 passed, 44 skipped. VAST and UKP Sentential are deferred with source-indexed reasons in `decisions.json`; they were not added to training.

## Batch 4 — reviewed and published

12 accepted sources, 13 active training views (Spider aggregation subsequently withdrawn). Nine sources load directly from existing Hub repositories; only Spider (schema join), SemCor (matching WordNet 1.6 inventory) and STREUSLE (original JSON) require repackaging. [Source and exact-export audit](../../../docs/jev/linguistic-expansion-audit.json) records native annotation caveats, sampled rows, pins, reviewer identity and split checks.

| Source | Requested views | Status / remaining checks |
|---|---|---|
| MAVEN | Sentence + marked trigger → event type | Direct `mneb/maven`; all 96,897 native event mentions verified by sentence/type; original document partitions match MAVEN-Arg |
| EmpatheticDialogues | Speaker situation → emotion | Direct canonical Parquet export; one unique situation, speaker-reported label; duplicate/conflicting prompts excluded |
| Spider | Column selection; aggregation parked as low value | Native parsed-SQL/schema join; no gold SQL in input; multi-output/arithmetic/set operations and equivalent foreign-key column targets excluded |
| OLID | Offensive, targeted, target type | Direct canonical Parquet export; offensive view retained; targeted/target parked after ambiguous native rows 88918 and 27540 |
| iSarcasmEval | Author-intended sarcasm | Direct English Hub copy with explicit files restoring original train/test; excludes custom test file and one empty post converted to nan |
| SemCor | WordNet sense-definition selection | Native keys joined to WordNet 1.6; 190,865 eligible training examples; no fabricated evaluation partitions |
| MAGPIE | Idiomatic/literal use | Direct filtered Hub source; native offsets/context/confidence; shared documents assigned to holdouts; original CC BY 4.0 evidence recorded |
| PreCo | Mention-pair coreference | Direct native clusters; bounded deterministic positive/negative pairs from explicit annotations; five train copies of validation documents removed |
| ReDial | Movie-conditioned human judgments | Direct native dialogues; movie names resolved; seeker preference labels require agreement by both participants |
| MAVEN-Arg | Event-conditioned argument role | Direct `mneb/maven-arg`; all 3,623 documents and 466,720 expanded argument spans match the original. [Detailed audit](../../../docs/jev/maven-arg-source-audit.json) |
| SemEval 2023 Task 3 | Annotated persuasion techniques | Deferred: vetted source access required; native task is paragraph-level multilabel. Source-indexed reason recorded in `decisions.json` |
| STREUSLE | Marked expression → supersense | Original versioned JSON; noun and verb views, including discontinuous expressions; native review partitions |
| SST-5 | Five-level sentence sentiment | Direct `SetFit/sst5`; native splits with duplicate/conflicting texts excluded; ordered score-only decisions |

Published 14,000 train, 1,000 validation and 800 test rows to Jev `default` and `full` in [Hub commit bf1e4a87](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/bf1e4a874575dfb16852af90f62eb739a737fd17), up to 1,000 train and 100 per existing evaluation split per task. Prior Parquet blobs were preserved. Implementation Git `cad17f7`; regression run: 517 passed, 44 skipped, followed by 46 passing focused checks after the final parking changes. SemEval 2023 Task 3 and the two fine-grained OLID views are explicitly deferred rather than silently forgotten.

## Selection rules

Use genuine labeled training data, existing templates and modest declarative mappings. Preserve native choices, complete evidence and source identities; do not fabricate distractors. Keep evaluation-only sources out of training. Prefer directly usable Hugging Face sources; archive joins, loader repairs and expensive transformations justify a reproducible data-only mirror. Record source terms even when unspecified. Reject or defer sources whose supervision cannot be mapped faithfully, recording the reason here.

Spider aggregation was subsequently withdrawn at user request in [Hub commit d4529cf8](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/d4529cf8f123010f243424486ff6527952d94b2a): 1,000 train and 100 validation rows removed from both `default` and `full`. Column selection and all unrelated Parquet shards are unchanged. The original publication counts above remain historical.

## Batch 5 — reviewed; accepted views implemented

| Source | Requested decision views | Status / remaining checks |
|---|---|---|
| MAVEN-ERE | Temporal and causal relations between marked events | Causal view implemented directly from Nofing; 53,358 native train / 13,624 labeled validation pairs. Entire Hub copy verified against original. Temporal view parked after a confirmed native chronological error. |
| REFinD | Financial relation classification | Deferred: competition terms prohibit redistribution/modification despite naming CC BY-NC; structured reason in `decisions.json` |
| BioRED | Biomedical relation classification | Direct `gyorilab/BioRED`; 3,828 / 1,113 / 990 eligible entity-pair relations, native document splits; full abstracts and grounded mentions preserved. |
| MATRES | Event start-point temporal relation | Implemented reproducible original TimeML join: 12,736 train / 837 test, 4 missing event instances excluded. [Mirror](https://huggingface.co/datasets/tasksource/matres), [recipe](../../../scripts/repackage_dataset/relations.py). |
| Causal News Corpus | Native binary event-causality classification | Direct thagen/CNCv2; all 3,075 train / 340 dev texts and labels match original V2 files. Native labeled splits only; no fabricated test. |

These are human-approved additions, not automatic promotions from discovery.
Training availability does not replace a source/recast example audit.

Published the three accepted relation views to Jev `default` and `full` in [Hub commit 07526e54](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/07526e54ee01c52d144cae72500bd8b84360fcf3): 3,000 train / 200 validation / 200 test rows appended, all previous Parquet blobs preserved. [Source and final-export audit](../../../docs/jev/relation-expansion-audit.json) records 2–3 reviewed examples per view and the two deferrals. Causal News Corpus was subsequently verified and implemented, as recorded below.

## Queue completion review — 2026-10-12

Every source in these five human-curated batches now has an implemented view or an explicit, source-indexed deferral. Optional projections are not silently promoted: QASPER evidence selection remains deferred; its binary view stays active. The last three accepted views are `causal-news`, `slurp/slot-type` and `topv2/slot-type`. Their [source and exact-export audit](../../../docs/jev/remaining-queue-audit.json) records counts, reviewed rows and limitations. SLURP’s existing mirror now retains original tokens in metadata for accurate slot alignment; all existing utterances, intent labels and splits were verified unchanged.
