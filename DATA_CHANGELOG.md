# Data changelog

Changes that affect published data, newest first: one entry per publish, `## <dataset> <version>`, with a few
concise bullets on what changed in the data. `tasksource-jev-typed-decisions` releases are tagged on the Hub
(`load_dataset("tasksource/tasksource-jev-typed-decisions", revision="v2026.10.13")`).

## tasksource/procedural-typed-decisions v2026.10.20

- Regenerated from the reasoning-core Jev generators: same configs, questions and levels. Five configs keep identical states (`event_triggered_execution`, `goal_directed_planning`, `graph_navigation`, `instruction_execution`, `record_eligibility`); the others have revised renderings and sampling, and `arithmetic` asks graded probabilities only when they are not close to one half. `provenance.json` records the generator commit.

## tasksource/procedural-extraction v2026.10.20

- New: four procedural extraction generators (person records, plan details in force, conditional lists, log queries), 10,000 training and 500 validation and test rows each, at levels 0 to 4.

## tasksource-jev-typed-decisions v2026.10.20.5

- `extraction`: add four procedural sources (`procedural/...`, from [procedural-extraction](https://huggingface.co/datasets/tasksource/procedural-extraction)): person records with near-miss names and confusers, plan details in force after proposals and undos, conditional order lists after updates, and log queries, at difficulty levels 0 to 4 with exact spans and nulls. 10,000 training and 500 validation and test rows each; 80 sources, 383,770 training, 27,499 validation and 29,545 test rows.

## tasksource-jev-typed-decisions v2026.10.20.4

- `extraction`: add Molweni and FriendsQA (questions over multi-party dialogues), MultiDoc2Dial grounding spans, MAVEN event triggers, SLURP slots and EBM-NLP participant, intervention and outcome spans; 76 sources, 343,770 training, 25,499 validation and 27,545 test rows.

## tasksource-jev-typed-decisions v2026.10.20.3

- `extraction`: add VUA metaphor-related words, FCE learner errors and QAConv (emails, court and interview transcripts), all marked non-commercial; 68 sources, 299,640 training, 23,524 validation and 25,633 test rows.
- In entity and span-tagging sources, texts with no span at all are subsampled to at most 20% of each split (e.g. Linnaeus 10,000 -> 1,933 training rows, SkillSpan, BC2GM, NCBI-disease). Question-answering unanswerables are unchanged.

## tasksource-jev-typed-decisions v2026.10.20.2

- `extraction`: add 32 sources (65 in total), bringing it to 294,154 training, 22,559 validation and 24,633 test rows (1,015,056 training fields).
- Extractive QA: QuAC with dialogue history, CUAD contract clauses in windows of about 4,000 characters, TyDi QA GoldP English, Qasper, PolicyQA, SubjQA (books, electronics, grocery, movies, TripAdvisor), COVID-QA, TimeQA, TAT-QA text spans and ROPES. Answers are verbatim spans of the context; native unanswerables stay null.
- Span tagging: CrossNER (politics, science, music, literature, AI), SkillSpan skills and knowledge, FabNER, BUSTER, BioRED entities, LitBank, MedMentions ST21pv, Chia, Linnaeus, SciERC entities, toxic spans, Inspec keyphrases and SemEval-2014 laptop aspect terms.

## tasksource-jev-typed-decisions v2026.10.20.1

- `extraction`: add 18 sources, bringing it to 209,452 training, 14,065 validation and 13,842 test rows (719,489 training fields).
- Extractive QA: DuoRC (SelfRC and ParaphraseRC, verbatim answers and native unanswerables), ConcurrentQA (Wikipedia and email), FairytaleQA (explicit answers).
- Named entities: BC5CDR, BC2GM, AnatEM, TweetNER7, Broad Twitter Corpus, MIT Movie and Restaurant, CoNLL04, WIESP2022 astronomy, FiNER-ORD, StackOverflow-NER, LegalLensNER, GUM and English SemEval opinion targets. Twitter @-mentions are one entity.

## tasksource-jev-typed-decisions v2026.10.20

- New `extraction` config (Clairic-style): a text with several typed fields (`text` or `list`), each answered by character spans of the text or null. 123,023 training, 7,316 validation and 5,413 test rows; 402,082 training fields.
- Extractive QA (SQuAD 2.0, AdversarialQA, Natural Questions, HotpotQA, Quoref, MuSiQue with unanswerable contrast questions, DROP single-span) groups the questions about one context; named entities (CoNLL-2003, OntoNotes, WNUT-17, Few-NERD, NCBI-disease, JNLPBA, acronyms, FiNER-139 XBRL amounts) ask one list field per entity type.
- Every span is checked against the text. Each source is capped at 10,000 training and 500 rows per evaluation split; evaluation texts seen in training and splits without released labels are left out. NewsQA is excluded pending its redistribution terms. Existing configs are unchanged.

## tasksource-jev-typed-decisions v2026.10.17.2

- `full`: add SOUL review-conditioned opinion comprehension and English CLEF 2024 CheckThat check-worthiness classification. Use the native human labels, explicit questions and readable criteria. Published 35,901 new rows (9,310 SOUL; 26,591 CheckThat), bringing `full` to 2,550,012 training rows.
- SOUL keeps justifications in metadata; exclude review-text overlap across splits and contradictory duplicate decisions. CheckThat excludes unlabeled test placeholders and model-derived aliases, preserving native dev-test evaluation separately.
- Mark unresolved original-review and CheckThat data terms explicitly. Existing shards and evaluation stay unchanged; `default` receives the tasks at its next full rebuild.

## tasksource-jev-typed-decisions v2026.10.17.1

- `full`: add EMS-MCQA’s native single-answer emergency-medical practice questions, preserving all options, including five- and six-choice questions.
- Exclude multi-answer records, malformed or duplicate options and positional references that cannot survive option shuffling. Retain native source splits; exclude collection-wide aliases from loading into the task.
- Keep source website, certification/category and rationale in metadata, separate from model inputs. `default` receives the task at its next full rebuild. Published 14,111 additional training rows, bringing `full` to 2,514,111 rows.

## tasksource-jev-typed-decisions v2026.10.17

- `vision`: add ShapeWorld object-box grounding, capped at 20k training rows from the 100k-QA procedural mirror. Native scene models verify every unique target and spatial relation. Published 20,000 training rows, bringing vision to 722,314 training rows.
- Preserve existing training and evaluation shards and exclude images already present in published evaluation. Keep generator seeds, geometry and original MIT license evidence.

## tasksource/shapeworld-grounding v2026.10.17

- Add 100,000 procedurally verified training grounding questions, plus 1,000 validation and 1,000 test questions, from pinned ShapeWorld scenes.
- Use native object boxes as choices; spatial queries distinguish repeated color/shape descriptions. Save scene models, generation seeds and geometry in metadata.
- Generated image hashes are disjoint across splits. Source code and generated geometric assets use the original MIT terms.

## tasksource-jev-typed-decisions v2026.10.16.2

- `vision`: add Flickr30k Entities phrase grounding, capped at 20k training rows from the 100k-QA mirror. Full captions identify the queried phrase; options are native annotated regions. Published 20,000 new rows, bringing vision training to 702,314 rows.
- Preserve existing training and evaluation shards; exclude images already present in published evaluation. Record original noncommercial image terms.

## tasksource/flickr30k-grounding v2026.10.16

- Add 100,000 human-labeled training phrase-to-box decisions, 11,232 validation questions and 11,560 test questions, grouped by unchanged Flickr30k image.
- Preserve native image partitions; require one native target box and distinct annotated-region choices. Exclude the publisher’s known unrelated captions and multi-box/no-box targets.
- Record original noncommercial research/education and Flickr image terms, source revisions and conversion hashes.

## tasksource-jev-typed-decisions v2026.10.16.1

- `vision`: add RefCOCO, RefCOCO+ and RefCOCOg object-box decisions, sharing the existing 20k training-row family budget. The source mirror contains 100k training QAs. Published 20,000 new rows (6,665 / 6,650 / 6,685 respectively), bringing vision training to 682,314 rows.
- Native object boxes provide the choices; images, grounding geometry, source identities and original license evidence remain available.
- Append training rows only, excluding images already in published evaluation. Existing training and evaluation shards remain unchanged.

## tasksource/refcoco-grounding v2026.10.16

- Add 100,000 training, 24,527 validation and 30,903 test referring-expression decisions from RefCOCO, RefCOCO+ and RefCOCOg.
- Share unchanged COCO2014 images across the three views. Choices are native object boxes; no synthetic distractors.
- Reserve held-out photographs across variants by both image ID and image hash. Preserve original split names in metadata; native test subsets share the canonical test split.
- Record original per-image Flickr licenses and source revisions. Noncommercial and unresolved terms are retained explicitly.

## tasksource-jev-typed-decisions v2026.10.16

- `vision`: add `spatial-dise/mcq` with native spatial-reasoning choices and original composite images. Training excludes generator families reserved by validation, test or the separate Spatial-DISE benchmark.
- Added 2,410 training decisions; `vision` now has 662,314 train rows. Existing shards and evaluation are unchanged.
- Retain rotation, combination and hidden-face questions. Projection is omitted for missing viewing cues; folding for a sampled ambiguous answer set. Source terms are CC BY-NC-SA 4.0 with original third-party asset rights retained.

## tasksource/spatial-dise v2026.10.16.1

- Remove cube-folding questions after a sampled hard example had multiple visually compatible answers. The approved mirror retains 2,410 train, 793 validation and 1,065 test questions.

## tasksource 2026-10-16

- Add `spatial-dise/mcq` through a reproducible image-archive mirror: 2,410 train, 793 validation and 1,065 test questions. Native answer panels remain identifiable after option permutation; no fabricated choices.

## tasksource-jev-typed-decisions v2026.10.15.5

- `vision`: removed the grid-cell grounding rows (rico-widget/grid7: 15,180; spair71k/grid7: 30,000); the coordinate-option views of v2026.10.15.4 replace them. Other rows and shards are unchanged; vision train has 659,904 rows.

## tasksource-jev-typed-decisions v2026.10.15.4

- `vision`: coordinate grounding as multiple choice. RICO widgets and Mind2Web elements: pick the target's box among the screen's other element boxes, as 0–1000 `[x1, y1, x2, y2]`. SPair-71k: pick the corresponding point among the pair's other keypoints, as 0–1000 `(x, y)`.
- `vision`: Mind2Web training actions (next action type, element choice, box choice); its test splits stay out.
- 75,258 training rows added: rico-widget/box 30,000, spair71k/point 30,000, and 5,086 each for Mind2Web box, element and action.
- Grid-cell and axis-bin grounding views no longer enter Jev builds; the rows already published leave at the next vision rebuild.

## tasksource 2026-10-15.12

- New tasks `procedural-choices/reasoning_core` (54 tasks) and `procedural-choices/reasoning_gym` (68 tasks): verified procedural answers among options their scorer rejects, without near-correct options.
- New visual tasks `rico-widget/box`, `mind2web/box` and `spair71k/point`: native candidate boxes or keypoints as 0–1000 coordinate options.

## tasksource-jev-typed-decisions v2026.10.15.3

- Rebuild of `default` and `full`. Rows keep their published `id` and `example_id` when their source example is unchanged, also where the content changed.
- Removed sources with unreliable or redundant labels: gen_debiased_nli, hlgd, EVALution, blog_authorship/job, app_reviews, pragmeval mrda and EmoBank arousal/valence, brainteasers/SP, cos_e/v1.0, crowdflower nuclear sentiment.
- At most 1,000 training rows from robust_nli, pragmeval scales, logical-fallacy, path-naturalness, crowdflower political media, recast_verbcorner and ReSQ; 10,000 from KnowRef-60K.
- Readable option labels (e.g. "answerable from the passage", "e2 causes e1", TREC answer types) and task questions for MultiRC, ethics/justice, ethics/virtue and dnd_style_intents.
- `default` now also draws on the sources appended to `full` since the last rebuild.

## zero-shot-label-nli 2026-10-15

- Removed the parked sources listed above, and tasks whose labels are bare yes/no or true/false answers to a task question.
- Readable label hypotheses ("This example is answerable from the passage.") for the relabeled tasks; rare civil_comments attributes removed.

## tasksource 2026-10-15.11

- civil_comments threat, sexual_explicit, obscene, identity_attack and severe_toxicity parked: few positives, flagged by keyword. toxicity, insult and the rater-share tasks remain.
- Task questions for MultiRC, ethics/justice, ethics/virtue and dnd_style_intents; "Receive quest" label spelling.

## tasksource 2026-10-15.10

- KnowRef-60K capped at 10,000 training rows: some sentences stay ambiguous despite 4/5 annotator agreement.

## tasksource 2026-10-15.9

- Parked cos_e/v1.0 (its questions are CommonsenseQA's, with fewer choices) and crowdflower/sentiment_nuclear_power (171 mostly neutral tweets, some mislabeled).
- Capped at 1,000 training rows: crowdflower political-media audience, bias and message (subjective labels), recast_verbcorner (templated) and ReSQ (questions about objects the story omits).

## tasksource 2026-10-15.8

- Self-describing labels for yes/no tasks with a fixed question, e.g. "plausible answer" (MC-TACO), "answerable from the passage" (SQuAD v2, QuAC, CUAD), "same entity" (PreCo), "causal relation" (causal-news), "correct replacement" (ReCoRD), and wiki_qa, wikiqa, avicenna, ambient, I2D2, winowhy, conceptrules_v2, regset, PARARULE-Plus, missing-item-prediction.
- zero-shot-label-nli leaves out tasks whose labels are bare yes/no or true/false answers to a task question.

## tasksource-jev-typed-decisions v2026.10.15.2

- Append CBT named-entity/common-noun cloze, ReCAM, high-agreement KnowRef-60K and native text-table TabMWP decisions to `full`, with all native choices retained.
- Exclude malformed choices, error/low-agreement labels and shared source holdouts, including ReCAM's unlabeled-test articles. Add source/license provenance and keep solutions out of model inputs.
- Preserve existing training shards and evaluation data. `default` gains these sources at the next canonical rebuild.

## tasksource 2026-10-15.7

- Readable labels where codes or bare words hid the judgment: TREC answer types (with a question), SemEval-2010 relations ("e2 causes e1"), FewRel ("the relation holds"), ESCI query relevance, defeasible-nli ("strengthens the hypothesis", with premise and hypothesis marked), com2sense ("makes sense"), and negative/positive, not spam, not hate for IMDb, Rotten Tomatoes, SMS spam and hate_speech18.

## tasksource 2026-10-15.6

- Parked after checking flagged examples against the raw sources: hlgd (loose story groupings), EVALution lexical relations (wrong source relations), blog_authorship/job (industry not inferable) and app_reviews (arbitrary ratings on short reviews).
- logical-fallacy capped at 1,000 training rows (noisy labels).
- brainteasers/SP removed: the mirror's sentence-puzzle files were a copy of the word puzzles.
- path-naturalness-prediction capped at 1,000 training rows. sen-making/1 and path-naturalness-prediction carry their question as the input instead of an empty one.

## tasksource 2026-10-15.5

- Add CBT named-entity/common-noun cloze, ReCAM abstract-word cloze, KnowRef-60K pronoun resolution and native TabMWP text-table multiple choice.
- Preserve all native choices, including CBT's ten and ReCAM's five. Filter CBT's malformed candidates and KnowRef's error/low-agreement labels (require 4/5); retain annotation provenance separately from prompts.
- Reserve shared passages/tables for source holdouts, including ReCAM's original unlabeled test articles. Keep available native partitions and do not fabricate new splits.

## tabmwp 2026-10-15

- Publish native multiple-choice questions with complete textual tables, original answer options and solutions kept in metadata. Retain 5,317 train, 1,830 validation and 1,992 test questions after table-group protection.
- Preserve original CC BY-NC-SA 4.0 dataset terms, native source IDs, file hashes and reproducible conversion/exclusion records.

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
