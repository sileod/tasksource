# Data changelog

Changes that affect published data, newest first: one entry per publish, `## <dataset> <version>`, with a few
concise bullets on what changed in the data. `tasksource-jev-typed-decisions` releases are tagged on the Hub
(`load_dataset("tasksource/tasksource-jev-typed-decisions", revision="v2026.10.13")`).

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
