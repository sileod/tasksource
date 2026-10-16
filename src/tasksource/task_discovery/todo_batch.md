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

Published these three completion views to Jev `default` and `full` in [Hub commit 2a17dec8](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/2a17dec8035da914f2e8b05712402d702216045d): 3,000 train / 300 validation / 200 test rows appended, all previous Parquet blobs preserved. Implementation Git `b39953c`; 39 focused regression checks passed. The English gallery now includes three actual published examples for each new view.

## Batch 6 — native relevance decisions (user-approved)

| Source | Requested view | Status |
|---|---|---|
| ANTIQUE | Four native relevance grades; all ~27k judged training pairs | Original qrel/query/answer join implemented and mirrored; all 27,422 train / 6,589 native test judgments retained, four ordered grades; original research-use terms recorded |
| WikiQA | Binary question/candidate relevance; include all-negative questions | Direct microsoft/wiki_qa; all 20,360 training pairs, including 1,245 all-negative questions; unique IDs preserve duplicate native judgments; original custom terms recorded |
| NevIR | Native contrastive training pairs sensitive to negation | Direct orionweller/NevIR; 948 native train pairs → 1,896 two-query MC decisions; original-document partitions are disjoint |
| MuSiQue | Optional supporting-paragraph selection; 5k–10k training questions | Deferred after coarse final-export review: question 2hop__512807_69291 asks when Puerto Rico became independent from Spain, a false premise. No MuSiQue rows published; optional projection requires further source-quality filtering. |

## Batch 7 — causal reasoning sources (user-approved)

| Source | Requested view | Status |
|---|---|---|
| COLD (2024, ~9M total) | Native causal MCQs | Excluded at user request: Exploration-Lab/COLD is gated; recorded in decisions.json |
| Causal Reasoning Benchmarks (2026, 100k train) | Native yes/no causal classification | Transitivity implemented directly with native holdouts; cyclic graphs/self-queries excluded. D-separation deferred: 2,069/50,000 labels disagree with standard graph checks. |

Batch 6 source/recast review: [audit](../../../docs/jev/retrieval-expansion-audit.json). Batch 7 graph-label review: [audit](../../../docs/jev/causal-reasoning-audit.json). Accepted views use existing templates; 23 focused regression checks pass.

Batch 6/7 source and exact-export audits: [relevance](../../../docs/jev/retrieval-expansion-audit.json), [causal reasoning](../../../docs/jev/causal-reasoning-audit.json). Three actual export examples reviewed per accepted view; 23 focused regression checks passed.

Published the four accepted views to Jev `default` and `full` in [Hub commit f8ff7e81](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/f8ff7e81ed820eeac1ad7afd63449fe99c8a76c3): **99,082 train / 200 validation / 400 test** rows appended. Training: ANTIQUE 27,422; WikiQA 20,360; NevIR 1,896; causal transitivity 49,404. Native evaluation pools remain available through `load_task`; the export caps each existing evaluation split at 100 rows per task. Every prior Parquet blob was preserved; implementation Git `65d82da`. No COLD, d-separation or MuSiQue rows were published.

Published the four accepted Batch 6/7 views to Jev `default` and `full` in [Hub commit f8ff7e81](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/f8ff7e81ed820eeac1ad7afd63449fe99c8a76c3): 99,082 train / 200 validation / 400 test rows. All 140 previous Parquet blobs verified unchanged. Implementation Git `65d82da`, metadata cleanup `8120bc0`. COLD skipped at user request; d-separation and optional MuSiQue remain explicitly deferred.

## Batch 8 — aspect decisions and contextual privacy (user-approved)

| Source | Requested view | Status |
|---|---|---|
| CEBaB | Restaurant rating and aspect-conditioned sentiment; preserve original/counterfactual identity | Implemented five direct views from EleutherAI/CEBaB; inclusive training set once, original-review groups/counterfactual flags retained; median ratings match every original vote distribution |
| FABSA | Aspect-conditioned sentiment, preserving neutral vs unmentioned | Implemented direct jordiclive/FABSA; single-polarity annotated aspects only, neutral retained, repeated review holdouts and conflicting prompts handled |
| TAB | Context-dependent identifier/masking decisions | Implemented existing Hub zipped JSON; native document partitions, person-to-anonymize instruction, full context, offsets and annotator/coreference IDs retained; disagreements remain separate native judgments |

[Source and staged-export audit](../../../docs/jev/aspect-privacy-audit.md): no new mirrors; two manually reviewed examples per view; 32 focused tests passed. FABSA's license remains explicitly unspecified.

Published to Jev `default` and `full` in [Hub commit 8fea8388](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/8fea8388f24f0ead40c9364fa6a73dd675df33ba): **7,000 train / 700 validation / 700 test** rows per config. Existing Parquet blobs and root card body were preserved; new shards were downloaded and verified against the reviewed build. The English gallery includes three published examples per new view.

## Batch 9 — game decisions (user-approved)

| Source | Implemented view | Status |
|---|---|---|
| OpenMOSS-Team/GameQA-140K | gameqa/mcq | Full native archive join hosted; 80,379 training / 10,493 test MCQs after format and held-out-image filtering |
| Valen-Team/Valen-Eval-Game | valen-game/optimal-action | Full native archive join hosted; 30,000 train / 500 test states, four action judgments per state, all tied optimal actions retained |

[Source/recast audit](../../../docs/jev/game-expansion-audit.md). Pilot published to the Jev vision config in [0100b447](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/0100b447ed4ee8b55bb9f89e725b62420ac71d2f): 1,000 training rows per task; existing rows and evaluation preserved.

Expanded the two game families to **20,000 training rows each** in [v2026.10.13.3](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/bfb636a73eb3250e073fafbbb79b989141ca5517). All 38,000 additions passed native-gold/image-hash checks; new Valen rows are English, new rows are shuffled with seed 0, and previous data/evaluation are preserved.

## Batch 10 — target/aspect sentiment (user-curated)

| Source | Task | Outcome |
|---|---|---|
| NewsMTSC | newsmtsc | Direct official converted Parquet, real-world view; author framing of the exact marked mention |
| UK election / TDParse | uk-election/sentiment | Native Figshare join mirrored reproducibly; parked after coarse label review found unsupported topic sentiment |
| SentiHood | sentihood/sentiment | Direct existing Hub JSONL; native entity/aspect Positive/Negative only; one conflicted dev pair excluded |
| MAMS-ATSA | mams/atsa | Direct existing Hub JSONL; all native sentences and aspect/polarity pairs verified against original XML |

[Mapping and coarse review](../../../docs/jev/audits/target-sentiment-audit.json). Active tasks enter through the catalog and canonical text build; no one-off Jev append. Original source splits are preserved, with repeated/conflicting inputs and sentence-group leakage excluded.

## Batch 11 — image safety (user-curated, processed)

| Source | Intended task | Status / next canonical build |
|---|---|---|
| OpenSafetyLab/t2i_safety_dataset | t2isafety/toxicity | Implemented and mirrored: 30,781 train / 366 test judgments; all native gold labels and encoded image hashes verified, zero train/test image overlap. Registered in vision_tasks.py for the next canonical vision build (20,000 train-row family cap). No privacy/demographic projection. |
| AIML-TUDA/LlavaGuard | Policy-conditioned safe/unsafe | Parked, source-access blocker: rechecked annotation CSV returns authenticated 403. Recorded in decisions.json; no pending modeling decision. Reconsider only after permitted data access. |

## Batch 12 — memes (user-curated, processed)

| Dataset | View | Outcome |
|---|---|---|
| MemeQA (ACL 2025) | Native visual MCQA | Implemented memeqa/mcq via reproducible tasksource/memeqa mirror; 3,000/1,244/1,194 native single-answer QAs, source meme-level splits, human-verified source distractors. Native multiselect excluded. |
| MET-Meme | Metaphor, sentiment, intention, offensiveness | Implemented four English views directly from Anthony3456347095/MET-Meme; native emotion, intention, offensiveness and metaphor labels. 2,391 train / 794 validation / 800 test per view after exact split-copy and conflicting-image exclusions; source license explicitly unspecified. |
| Memotion 2.0 | Humor, sarcasm, offense, motivation, sentiment | Parked, source unavailable: official download requires shared-task registration; referenced public Drive link is 404. Existing 7K Hub derivatives are not substituted for version 2.0. Evidence and reconsideration condition in decisions.json. |

Batches 11–12 have no unreviewed entries remaining. Accepted tasks enter the catalog and the documented scripts.append_jev_vision build/publication path (20k per source family, existing rows preserved). Access-blocked sources live in decisions.json with evidence and explicit reconsideration conditions.

## Batch 13 — explicitly requested procedural visual sources

These are user requests, separate from automatically discovered prospects.

| Source | Requested view | Status |
|---|---|---|
| TRON (YangTianze009/TRON) | Static MC exports from seeded generator/verifier pairs | Hosted tasksource/tron-mcq and registered tron/mcq: 100,000/83 train/validation QAs from five reviewed environments. Final materialization counts checked; rounded-measurement MC instructions repaired. |
| TRACE (maveryn/trace) | Static MC view of native procedural training examples | Hosted tasksource/trace-mcq and registered trace/mcq: 100,000 train / 2,000 validation QAs, retaining 64k native training rows and adding 36k fresh verified generations. |
| TACIT (tylerxdurden/TACIT-benchmark; danielxmed/tacit-benchmark) | Investigate reproducible procedural decision training | Access blocked: supplied GitHub repository and authenticated Hub metadata return 404. Recorded in decisions.json; reconsider after source access is restored. |
| MentisOculi (Jana-Z/mentis-oculi) | Investigate reproducible procedural decision training | Hosted tasksource/mentis-oculi-paper-fold and registered mentis-oculi/paper-fold; 100,000/100 train/validation QAs hosted and pinned. Other families remain outside this adapter; planning outputs need multiple-valid-plan handling. |


## Batch 14: additional procedural image sources (2026-10-14)

| Source | Requested view | Status |
|---|---|---|
| SPHINX | Static visual MC | `sphinx/mcq` registered; 100,000 training QAs hosted across four families. Native same-scene choices and nearby count distractors. |
| VisGym | Static visual MC | One `visgym/mcq` mirror: 100,000 total training QAs, balanced 25,000 per family (maze/equation/rotation/blocks), plus 100 held-out QAs. Separate family registrations removed. |
| VisualSphinx | Visual puzzle MC | Existing `visualsphinx` uses curated RL-20K directly. Fresh generation needs LLM code synthesis/verification; no blind expansion from its raw corpus. |
| Gym-V | Excluded | Excluded by explicit user request; recorded in decisions.json. No catalog registration or scale job. |


## Batch 15: human-curated reading, dialogue and clinical sources (2026-10-15)

| Source | Requested view | Status |
|---|---|---|
| QuALITY | Full-article native four-option MC | Added `quality` directly from emozilla/quality; 2,523 train / 2,086 validation QAs, zero-based native gold indices. |
| CLEAR | Human-fitted reading-ease regression | Added `clear/readability` directly from casey-martin/CommonLit-Ease-of-Readability; preserve raw BT scores and native 3,543/708/473 splits. |
| ESConv | Eight supporter strategies | Added `esconv/strategy` using the official-file tasksource/esconv mirror; consecutive differently annotated turns remain separate. |
| ProsocialDialog | Five ordinal safety judgments | Added `prosocial-dialog/safety` directly; reconstruct earlier dialogue without current response/rationales. Inputs are GPT-3-generated; labels are human. |
| MedNLI | Clinical NLI | Blocked: PhysioNet credentialing and signed data-use agreement; no public redistribution through Tasksource. Recorded in decisions.json. |
| GAP | A/B/neither pronoun resolution | Added `gap` directly from coref-data/gap_raw; upstream development is the HF train split, allowed for model development. Official test remains test. |
| TRIP (sled-umich/TRIP) | Physical-state plausibility MC | Added `trip/cloze` and `trip/order` directly from the official Hub Parquet conversion; 799 and 2,330 train pairs. Native story groups are split-disjoint, including across views. License unspecified. |
| QA4MRE (community-datasets/qa4mre) | English main-track native MC | Queued next: verify genuinely labeled training partitions versus released evaluation questions; resolve original terms. |

The seven catalog additions enter the next canonical text build automatically. Partial smoke builds are not publications and must not be uploaded over `full`/`default`.

See the [source and recast audit](../../../docs/jev/audits/reading-dialogue-batch-audit.json) for native examples, caveats and the five-task build smoke results.

TRIP checks and source caveats: [audit](../../../docs/jev/audits/trip-audit.json).


## Batch 16: SBIC quality aggregation (2026-10-15)

| Source | Requested view | Status |
|---|---|---|
| Social Bias Frames | Offensiveness and perceived intent | Added `sbic/offensiveness` (11,692 train) and `sbic/intent` (8,647 train) via tasksource/sbic. Three distinct workers and 75% categorical agreement; native ratings and worker/source-row provenance retained in metadata. Split conflicts with existing SBIC_Disagreement are excluded. |
| Social Bias Frames | Identity-group targeting | Withheld: a questionable unanimous negative in the coarse source audit. Original group annotations remain in metadata. |
| ErnestSDavis/winograd_wsc | Pronoun resolution MC | Excluded from training: only 273/285-example native test sets; no training split. |

[Source and recast audit](../../../docs/jev/audits/sbic-audit.json). The mirror is published; Jev `full`/`default` require the canonical text rebuild. No partial text append.


## Batch 17: grounded dialogue and reasoning (2026-10-15)

Human or procedural labels required; a model may generate inputs when humans supply the labels. Screened native examples and training partitions. FaithDial, RAGTruth and ComVE are implemented; the ComVE mirror is published. See the [recast checks](../../../docs/jev/audits/grounded-reasoning-recast.json).

| Source | Requested view | Status |
|---|---|---|
| FaithDial | Knowledge-grounded original-response classification | Added `faithdial/grounding` via official HF Parquet. 18,357 native train wizard turns; 17,262 strict Hallucination/pure-Entailment candidates. BEGIN belongs to the original response; exclude Generic/Uncooperative cases initially. |
| RAGTruth | Human-span-derived hallucination classification | Added `ragtruth/grounding` via usable wandb/RAGTruth-processed. 15,090 native train, 14,942 quality=good. Preserve spans/model flags; remove the known Summary context output: suffix. No context overlap with native test. |
| ComVE subtask A | Two native statements, select the more sensible | Added `comve` via official-file repackaging: 9,966 train, 997 validation, 999 test after QC. 10,000 raw train, 997 validation, 1,000 test; five identical-choice train pairs and 5/3 train-evaluation pair overlaps need exclusion. Gold marks nonsense; invert for sensible selection. Original README CC BY-SA 4.0. |
| AAAC02 | Procedural valid-conclusion selection | Conditional: 16,000 native train. No model paraphrasing, unlike AAAC01. Not native MC; verify non-entailment of each derived distractor, and keep the conclusion out of the premises input. |
| GLUCOSE | Causal decisions | Conditional: 65,522 native train annotations, positive only. No random-swap negatives. Explore native causal-dimension classification with quality filtering and story grouping. Original license is CC BY-NC 4.0, contradicting HF card. |

[Screening evidence and native-example notes](../../../docs/jev/audits/grounded-reasoning-screen.json). Prefer direct usable HF sources; repackage the official ComVE CSV join and any future verified conversion that warrants it.

Published FaithDial, RAGTruth and ComVE to Jev `full` in [v2026.10.15](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/17e817ed907bdb790aac2a0d1128b017d3329817): **48,419 additional training rows including augmentations** (18,838 / 18,155 / 11,426). Existing shards and evaluation are unchanged; `default` gains these at the next canonical rebuild.

## Batch 18: cloze, reference resolution and table arithmetic (2026-10-15)

Processed the human-curated queue. Native choices only; no generated distractors. All five views passed Jev smoke builds and sampled source-gold comparisons; two examples per view were reviewed. See the [source/recast audit](../../../docs/jev/audits/cloze-batch-recast.json).

| Source | Added views | Retained train / validation / test | Source and quality checks |
|---|---|---|---|
| Children's Book Test | `cbt/ne`, `cbt/cn` | NE: 108,500 / 1,996 / 2,486; CN: 120,767 / 2,000 / 2,500 | Direct cam-cst/cbt Parquet. Preserve all ten candidates; omit blank/duplicate choices. Native books are partitioned by the source; checked context groups are disjoint across splits. BookTest remains a related larger follow-up, not part of this batch. |
| ReCAM | `recam` | 6,403 / 1,656 / — | Direct holylovenia/recam, verified byte-decoded rows against official train/dev JSON. Reserve shared articles and original unlabeled-test articles using the reproducible scripts/audit_recam.py manifest. No public test labels; original dataset license unspecified. |
| KnowRef-60K | `knowref60k` | 29,838 / 15,953 / 3,061 | Direct coref-data/knowref_60k_raw. Five human annotators according to the paper; keep native agreement strengths 4/5, exclude error rows, preserve neither and annotation strength. Reserve original sentences across name swaps. Some terse Reddit contexts remain ambiguous. |
| TabMWP | `tabmwp` | 5,317 / 1,830 / 1,992 | tasksource/tabmwp mirrors the original option-bearing JSON questions with full text tables. Reserve all native held-out tables, including free-answer questions; solutions stay in metadata. Dataset CC BY-NC-SA 4.0, distinct from MIT code. |

`preserve_choices=True` is a small opt-in on the existing MC template; ordinary text tasks keep their existing defaults. ReCAM's original test is protected but never treated as labeled data. No new task type or loader.

Published all five views to Jev `full` in [v2026.10.15.2](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/263e8c75c6a90c8bb43e2c8993b7f911e1ee5722): **112,154 additional training decisions including augmentations**. Existing shards and evaluation are unchanged; `default` gains these at the next canonical rebuild.

## Spatial-DISE (2026-10-16)

Added `spatial-dise/mcq`: native rotation, combination and hidden-face questions, via the [reproducible archive join](../../../scripts/repackage_dataset/spatial_dise.py). Retained 2,410 train / 793 validation / 1,065 test rows after reserving generator families across splits and the separate benchmark. Original composite pixels and native choices are preserved. Projection is deferred for missing viewing cues, and folding for a sampled ambiguous answer set; both have structured reasons. [Coarse image/recast audit](../../../docs/jev/audits/spatial-dise-recast.json); CC BY-NC-SA 4.0 plus original asset terms.
