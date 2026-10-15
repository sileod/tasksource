# Data changelog

Changes that affect published data, newest first: one entry per publish, `## <dataset> <version>`, with a few
concise bullets on what changed in the data. `tasksource-jev-typed-decisions` releases are tagged on the Hub
(`load_dataset("tasksource/tasksource-jev-typed-decisions", revision="v2026.10.13")`).

## tasksource-jev-typed-decisions v2026.10.15.1

- `vision`: +150,000 training rows (479,826 → 629,826) from procedural visual puzzles, 30,000 each: TRON, TRACE, Mentis Oculi paper folding, SPHINX and VisGym. Existing rows and evaluation splits are unchanged.

## tasksource-jev-typed-decisions v2026.10.15

- Append FaithDial original-response grounding, RAGTruth evidence-grounding judgments and ComVE commonsense statement choices to `full`, using human supervision and checked native labels.
- Preserve existing training shards and evaluation data. `default` includes these sources at the next canonical rebuild.
- Retain annotation provenance and source license evidence; exclude generic/uncooperative FaithDial cases, low-quality RAGTruth outputs and invalid or conflicting ComVE pairs.

## tasksource 2026-10-15.5

- New `max_examples` task annotation caps training rows for valid but low-signal sources; every consumer (Jev `full`, recasts) respects it.
- Capped at 1,000 training rows: the robust_nli stress tests and the median-split pragmeval scales (EmoBank dominance, Squinky, persuasiveness).
- Parked: the six gen_debiased_nli variants (generated hypotheses, model-filtered labels), pragmeval/mrda (context-free meeting dialogue acts) and the EmoBank arousal/valence median splits.

## tasksource 2026-10-15.4

- Add FaithDial original-response grounding and RAGTruth human-span-derived grounding decisions. Keep annotation provenance separate from inputs, reserve native dialogue/source holdouts, and omit ambiguous or low-quality cases.
- Add ComVE native two-statement commonsense choices; invert the source's nonsense index to select the more sensible statement. Exclude duplicate choices, repeated pairs and conflicting judgments.
- AAAC and GLUCOSE remain deferred: their proposed MC conversions need verified negatives rather than arbitrary distractors.

## comve 2026-10-15

- Publish official human-written subtask-A statement pairs as a data-only mirror, with native splits, original answer indices, per-file hashes and exclusion reasons.
- Preserve CC BY-SA 4.0 licensing and link the original release and reproducible packaging script.

## tasksource 2026-10-15.3

- Add `sbic/offensiveness` and `sbic/intent` from the audited aggregate, with 11,692 and 8,647 accepted training judgments respectively. Preserve native uncertainty categories and labeling details in metadata.
- Keep SBIC_Disagreement unchanged. Group-target supervision is withheld; its original annotations remain available in metadata.
- The source mirror is published; these text tasks enter Jev `full`/`default` only through the canonical rebuild, not a partial append.

## sbic 2026-10-15

- Aggregate SBIC crowd judgments once per post and worker, preserving native categorical ratings and all labeling provenance in metadata.
- Require at least three valid workers and 75% agreement per label. Keep only posts with a reliable offensiveness or intent judgment; missing answers never count as negatives.
- Preserve native splits, assigning duplicate posts to their highest-priority source holdout. Exclude posts conflicting with the existing SBIC_Disagreement partitions. Retain original group-target annotations in metadata, with their correct mapping documented; the group-target task is withheld after a questionable unanimous negative in the source audit.

## tasksource 2026-10-15.2

- Add `trip/cloze` and `trip/order`: two native story choices with human physical-plausibility labels, using the official Hugging Face Parquet conversion directly.
- Preserve the original 799/322/351 and 2,330/393/408 train/validation/test partitions. Story groups do not overlap across splits, including between views.
- Record the original license as unspecified. These tasks enter the next canonical Jev text build; they have not been published to `full`/`default` yet.

## tasksource 2026-10-15

- Add `quality`, `clear/readability`, `esconv/strategy`, `prosocial-dialog/safety` and `gap` to the English catalog. Preserve full reading passages, native options, raw reading-ease scores and source partitions.
- Keep earlier conversation context without future turns or annotation rationales; shared ProsocialDialog opening contexts favor evaluation splits. Preserve native marked pronoun spans and record human/model input provenance and original licenses.
- MedNLI remains access-restricted; LegalBench is excluded by request. These catalog additions are queued for the next canonical text build, not yet published to the Jev `full`/`default` configs.

## esconv 2026-10-15

- Preserve the official train/validation/test dialogue partitions and every native supporter-strategy label in a data-only mirror.
- Keep consecutive replies with different strategies separate. Classify the current reply using prior context only; retain conversation identity and source provenance. Original CC BY-NC 4.0 terms apply.

## tasksource-jev-typed-decisions v2026.10.13.7

- `vision`: +170,008 training rows (309,818 → 479,826) from larger source mirrors, up to 30,000 rows per family: BAPPS, DocLayNet, LVIS and COCO panoptic regions (+29,000 each), SPair-71k (+25,008), Super-CLEVR (+29,000 across six views). Existing rows and evaluation splits are unchanged.

## visgym-mcq 2026-10-14.2

- Consolidate VisGym into one balanced release: 100,000 training QAs, exactly 25,000 each for mazes, equation repair, stick rotation and block legality; 25 validation QAs per family.
- Deterministically shuffle the combined dataset, retaining verified native choices, original PNGs, family/geometry metadata and disjoint image groups. Replace the separate family registrations with `visgym/mcq`.
- Delete the superseded standalone equation, rotation and block Hub repositories after consolidation.

## zero-shot-label-nli 2026-10-14

- Rebuilt from the current catalog: 305 classification tasks (from 152), 1.84M training rows; `default` and `triplet` configs as before.
- Per task, the majority label is at most 3× the others, and tasks with only true/false/yes/no labels are capped at 1,000 training rows.
- Left out: tasks with NLI or annotator-agreement labels, or any bare-number label (e.g. HelpSteer scales). Adds `license` and `license_use` columns.

## visgym-equation-mcq 2026-10-14

- Add a reviewed fresh procedural pilot for one-match equation repair. Every offered equation is arithmetically true, and exactly one is reachable with one match transfer.
- Preserve generation seeds, geometry, Apache-2.0 source notices and native/independent candidate checks. Target a separate 100,000-QA verified training build; exclude cross-split image copies.

## visgym-rotation-mcq 2026-10-14

- Add a reviewed fresh procedural pilot for stick-orientation questions with nearby 15/30-degree alternatives, checked with the native orientation reward.
- Preserve generation seeds, geometry, Apache-2.0 source notices and native/independent candidate checks. Target a separate 100,000-QA verified training build; exclude cross-split image copies.

## visgym-block-mcq 2026-10-14

- Add a reviewed fresh procedural pilot for sliding-block move legality. Distractors collide with another block; only the current board is displayed.
- Preserve generation seeds, geometry, Apache-2.0 source notices and native/independent candidate checks. Target a separate 100,000-QA verified training build; exclude cross-split image copies.

## sphinx-mcq 2026-10-14

- Add fresh static questions from four reviewed SPHINX families: pie ordering, transformation identification, transformation result selection and geometric counting; target 100,000 distinct training QAs.
- Keep native same-scene distractors and image panel identities; use nearby count alternatives. Preserve source metadata, generation seeds, MIT code and CC BY 4.0 icon notices.

## visgym-mcq 2026-10-14

- Add 100,000 distinct fresh VisGym 2D maze training QAs and 100 validation QAs at three sizes. Select the unique shortest-path next move from four cardinal directions.
- Preserve maze geometry and seed provenance; exclude tied or trivial states and cross-split image copies. Scope excludes photographic and simulator assets.

## tron-mcq 2026-10-14.2

- Keep all 100,000 training QAs, original PNGs, choices and gold indices. Clarify numeric questions requesting two decimal places as choosing the closest offered value; some source geometry figures display rounded measurements while their gold uses hidden precision.
- Preserve original questions in metadata and record the repair against the pinned 100k release.

## trace-mcq 2026-10-14.2

- Retain the native release and add fresh procedural training instances to reach 100,000 verified QAs; native validation remains separate.
- Use independent generation seeds and same-task native training answers as distractors. Exclude validation image copies by decoded pixels as well as repeated image/question/gold combinations.
- Remove three additional standalone JSON-output directives from the MC questions while preserving source content, including chart labels named “Json”. Record fresh-generation seeds, rendering versions and resume checkpoints.

## mentis-oculi-paper-fold 2026-10-14

- Expand fresh procedural paper-fold training to 100,000 distinct, geometrically verified QAs, retaining native five-panel images and choices.
- Preserve separate held-out seed namespaces, remove image overlap and repeated questions, and record all chunk seeds and exclusions. Original benchmark rows are not ingested.

## trace-mcq 2026-10-14

- Expand selection to the complete eligible native training release (64,000 source rows) and native validation partition (2,000 source rows), without duplication to reach an artificial 100,000-row target.
- Keep original image bytes, same-task verifier-checked distractors, source partitions, repaired prompt contracts and asset attribution. Retained counts and exclusions are recorded in provenance.

## tron-mcq 2026-10-14

- Expand the five reviewed procedural environments to 100,000 distinct, verifier-checked training QAs.
- Keep native PNGs and research-use terms. Remove repeated image/question/gold combinations and held-out image copies; record resumable generation chunks and actual seeds.

## tasksource-jev-typed-decisions v2026.10.13.6

- `vision`: +191,169 training rows (118,649 → 309,818), up to 30,000 rows per source family: NLVR2, Visual7W, TallyQA, MapQA, A-OKVQA, RICO widgets, CLEVR, SPair-71k, GameQA, Valen game, AI2D, ScienceQA, TQA. Families without more unseen rows are unchanged; existing rows and evaluation splits are unchanged.

## mentis-oculi-paper-fold 2026-10-13

- Add a fresh procedural pilot: 20 training and 5 validation paper-fold puzzles with five native image panels. Published benchmark examples are not ingested.
- Recheck final unfolded option geometry for exactly one matching answer. Record generation seeds, coordinates, source revision and Apache-2.0 terms.

## trace-mcq 2026-10-13

- Add 5,000 training and 100 validation visual multiple-choice questions from TRACE's native release, preserving original image bytes and partitions.
- Distractors come from training answers for the same task and answer type, checked with the original scorer. Remove JSON output demonstrations, clarify image-label choices, and restore the missing toggle-grid transition rule from source code.
- Exclude all native validation image copies from training. Preserve CC BY 4.0 attribution and upstream asset notices.

## tron-mcq 2026-10-13

- Add a bounded static visual MC release from five seeded TRON environments: aquarium, angle bisector, bar-chart aggregate, before/after and overlapping-shape count.
- Keep native text choices where available; otherwise use verifier-rejected answers from other generations of the same environment. Reject ambiguous choices and unsupported answer formats.
- Preserve generated PNGs, record seeds and verifier results, and exclude held-out image copies from training. Mark TRON's original research-use terms explicitly.

## tasksource-jev-typed-decisions v2026.10.13.5

- Add native single-answer MemeQA and four English MET-Meme classification views. Training is capped at 20,000 rows per source family; existing training and evaluation shards are preserved.
- Image-only memes remain usable with an empty text state. Answer metadata stays out of inputs. MemeQA retains its native meme splits; MET-Meme retains its Hub partitions with held-out image copies and conflicting duplicate-image labels removed. MemeQA image terms are marked research-use; MET-Meme's license is explicitly unspecified.

## memeqa 2026-10-13

- New reproducible image mirror: 3,000 train, 1,244 validation and 1,194 test native single-answer MCQs, with original image bytes and meme-level partitions.
- Excludes native multiselect questions, empty/invalid choices and duplicate options; no Tasksource-generated answers or distractors. Annotation provenance records human verification of automatically constructed source distractors; image research-use terms remain separate from the repository's MIT license.

## t2i-safety 2026-10-13

- New generated-image toxicity mirror: 30,781 train and 366 test judgments, with native safe/unsafe labels and an explicit six-category policy. Original image bytes and split membership are preserved; image groups are shuffled with seed 0.
- Generation prompts stay out of inputs. Demographic prediction, privacy-only/mixed labels and unreviewed real images are outside this view. Registered for the next canonical vision build with a 20,000-training-row family cap.

## uk-election 2026-10-13

- New native target-sentiment mirror: 9,911 train and 2,675 test judgments, preserving the official tweet partitions and human-added targets. “Does not apply” and one invalid span are excluded.
- Training view is parked after a coarse review found unsupported topic sentiment. Annotation license is CC BY 4.0; tweet-content rights remain separate.

## gameqa 2026-10-13

- Image groups shuffled within each native split using seed 0; all questions, labels and encoded image bytes are preserved. Existing Jev rows are unchanged.

## tasksource-jev-typed-decisions v2026.10.13.4

- `source_index` (`split:row` of the original dataset row) on every row, to trace label errors back to their source.
- Stable row order across rebuilds: shuffled by `example_id`, so row-level annotations stay valid.
- Removed after manual review of random rows: `dair-ai/emotion`, CrowdFlower `text_emotion` and IndicGLUE ACTSA (many wrong labels). `app_reviews` drops reviews under four words.
- DiscoSense states include the discourse connective; PropSegmEnt question fixed; PROST removed (test-only).
- New: CEBaB, FABSA, TAB, NewsMTSC, SentiHood target/aspect sentiment and MAMS-ATSA.
- Restores 57 sources missing from v2026.10.13 (MAUD, multidogo, WikiQA, OLID, TabFact and others), now built with the rest instead of appended.

## tasksource-jev-typed-decisions v2026.10.13.3

- `vision`: GameQA and Valen Game grow to 20,000 training rows each, retaining the original 1,000-row pilots (106,085 vision training rows total). All new Valen rows are English; its preserved bilingual pilot leaves 19,493 English and 507 Chinese rows.
- New rows are shuffled together with seed 0; original shards and evaluation splits are unchanged. GameQA question identities now include the native record to avoid collisions.

## tasksource-jev-typed-decisions v2026.10.13.1

- `vision`: game decisions, 1,000 training rows each from [GameQA](https://huggingface.co/datasets/OpenMOSS-Team/GameQA-140K) multiple-choice questions and [Valen](https://huggingface.co/datasets/Valen-Team/Valen-Eval-Game) optimal-action judgments (68,085 vision training rows). Existing rows and evaluation splits are unchanged.

## tasksource-jev-typed-decisions v2026.10.13

- 18 procedural sources, including `record_eligibility`; procedural rows are label-balanced (at most 60% per answer, yes/no at most a third of questions).
- Label balance: per source and question, the majority label is at most three times the rarest; single-label questions are dropped.
- `default` mix weighs sources by option count, so binary decisions are 15% of rows.
- Known issue: 58 sources from earlier appends are missing (MAUD, multidogo, WikiQA, OLID, TabFact and others).

## procedural-typed-decisions 2026-10-13

- New task `record_eligibility`; `partial_observation_calibration` recalibrated so levels go from easy (L0) to hard (L4).
- Answers balanced: at most 60% per answer within a question, and yes/no questions at most a third of each task.


## tasksource-jev-typed-decisions: earlier releases

Untagged, before 2026-10-13.

- Text sources added: WebInstruct (MC and binary); web-page decisions (Mind2Web, WebLINX, WebSRC); SEntFiN, WANDS and SciRepEval Search; legal, biomedical and structured text (MAUD, ACORD, CUAD, Evidence Inference, TabFact, WikiSQL, MultiDoGO and others); dialogue and evidence (QuAC, MultiWOZ, SLURP, QASPER and others); stance, argument and discourse (FNC-1, UKPConvArg, DiscoGeM, PubMed RCT); linguistic annotations (OLID, iSarcasm, MAGPIE, SemCor, STREUSLE and others). Spider aggregation was later withdrawn.
- 2026-10-09: CLadder removed (evaluation-only). Missing-item, WikiHow order and QuaRel questions fixed; QuaRel uses its answer texts as options.
- 2026-10-09, `vision`: answer-style suffixes removed from 18,353 Cauldron questions; component-level license review in `metadata.licenses.license_review`. FigureQA rows are flagged: its terms prohibit redistribution.
- 2026-10-08: `vision` config, a multimodal pilot of about 66k training decisions from 30 sources, with `license` and `license_use` columns.
- `filtered-full`: a filtered, training-only snapshot of an earlier release; its exclusions are recorded as reusable evidence in `quality/filtered-full-v1/`.
