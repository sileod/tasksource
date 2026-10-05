# JEV filtering: per-task review

This is a review of screening evidence, not an approved exclusion list. No JEV
Hub configuration or example was changed by this review. Jev is a weak supporting
signal; DeepSeek V4 Flash is a fallible screener. Agreement between them does not
establish a label error. Flag rates below measure model disagreement, not data quality.

All 1,593,564 eligible examples have screening verdicts across 599 source tasks.
The first manual pass reviewed 38 examples from 10 priority tasks, selecting up to
one example per available verdict (ok, wrong, ambiguous, malformed). This deliberately
stratified sample cannot estimate precision or recall. Previously inspected examples
supply additional evidence where identified below. Other tasks remain unreviewed.

| Task | Flagged / screened | Initial recommendation |
| --- | ---: | --- |
| discosense | 4,310 / 8,479 (50.8%) | Review continuation-task instructions before judging flags. Bad distractors alone do not invalidate a usable question. No bulk exclusion. |
| cloth | 1,106 / 12,477 (8.9%) | Audit omitted passage context. Some isolated cloze sentences are underdetermined; grammatical sentences can still be usable. Do not automatically replace labels. |
| ReSQ | 235 / 2,049 (11.5%) | Prioritize source/context review. Missing entities and unsupported spatial relationships occur in flagged and passed examples. |
| IntentGrasp/all | 826 / 14,041 (5.9%) | Split review by intent domain. User intents, conversational intents and scientific edits use different taxonomies. Close label boundaries are not clear errors. |
| oasst2_pairwise_rlhf_reward | 244 / 10,606 (2.3%) | Preserve subjective preference examples unless independently demonstrably broken. Odd replies remain valid comparison candidates. |
| hh-rlhf/helpful-rejection-sampled | 294 / 5,926 (5.0%) | Preserve intentionally weak, truncated or spammy rejected responses. Those properties can explain the preference label. |
| seahorse_summarization_evaluation | 1,391 / 18,213 (7.6%) | Adjudicate by the exact evaluation criterion, separately by criterion and language. A bad summary is an input to evaluate, not necessarily a bad example. |
| ConTRoL-nli | 1,657 / 6,922 (23.9%) | Audit pairing/labels and preserve NLI conventions. Clear mismatches exist, but difficult entailment judgments need individual evidence. |
| qasc | 1,365 / 5,897 (23.1%) | Review specific option ambiguity and question completeness. Sentence-completion formats are valid; disputed scientific answers need domain review. |
| swag/regular | 1,697 / 12,497 (13.6%) | Preserve the continuation task. The preferred ending need not be logically entailed. Inspect actual corruption separately. |

## Concrete evidence

- **CLOTH:** `323e6fd9e3de05d5` asks which school club the speaker joins,
  with art/music/Chinese/history as options and no surrounding context.
  `c1e6ad7507f344ab` proposes changing a contrastive connector to a temporal
  connector without the preceding passage. That is not sufficient evidence to relabel it.
- **ReSQ:** `bfa005afd66a3fd1` asks about a book absent from the supplied story.
  Passed example `80106276a1d78de1` assumes a red jacket touches a yellow cap;
  the description only says both are worn. This shows screening misses as well as flags.
- **IntentGrasp:** `cabd66a5bc237740` disputes wish versus consolation for a
  supportive utterance. Both are plausible without the taxonomy's annotation rules.
  Earlier example `a6d8b0a56eae6811` asks for a payment extension; the broad
  payment-not-on-time option is the only matching supplied category.
- **HH preferences:** `3c5d70bce7e52e4f` has an incomplete rejected reply and a
  complete chosen clarification. Calling the whole example malformed removes
  precisely the behavior the preference task teaches.
- **OASST preferences:** earlier example `a2e71dba9f90a298` ends with a question
  about 1+1 and compares two valid answers. The screen incorrectly called them
  answers rather than replies, despite that being exactly what the task requires.
- **Seahorse:** `63d8a750d7f8df3d` asks about unnecessary repetition, but the
  flag discusses factual relevance/corruption. `0daadf9cf8674309` asks about main
  ideas, while the flag discusses repetition; its reason also says No is correct
  despite a No gold label and a wrong verdict. These are unreliable exclusion signals.
- **ConTRoL:** `963267ed5e885ab7` pairs a student-clubs passage with an unrelated
  student-loan question and an entailment gold. This is a concrete pairing/label
  problem worth tracing back to the source. In `db822b1996aa088d`, the gold
  contradiction is not established by a passage listing four musical themes;
  the screening objection alone still does not establish the correct convention.
- **QASC:** `a263c357d2c61cb4` contains near-duplicate greenhouse-gas options
  and several plausible environmental outcomes. That merits option-level review,
  rather than interpreting every model disagreement as a wrong gold answer.
- **DiscoSense / SWAG:** garbled distractors and multiple conceivable continuations
  can trigger flags even when the gold remains defensible. Their high disagreement
  rates should prompt task-instruction review before applying any row-level filter.

## Reproduce the review packs

```bash
PYTHONPATH=.:src python scripts/review_jev_filtering.py \
  --tasks discosense cloth ReSQ IntentGrasp/all \
  oasst2_pairwise_rlhf_reward hh-rlhf/helpful-rejection-sampled \
  seahorse_summarization_evaluation ConTRoL-nli qasc swag/regular
```

The script makes no API calls. It reads cached screen verdicts, exports all task
counts to `build/jev-task-filter-review/per-task.csv`, and writes complete selected
examples to `examples.jsonl`. Missing example text is recovered by exact direct-row
ID from the local v11 build shards. `sampling.json` records sampling scope and
missing source texts. It creates no removal CSV and changes no training data.

Before exclusions, record a reviewed policy per task: what the question measures,
which contextual inputs are required, valid source labeling conventions, and what
specific evidence warrants removal. Heterogeneous tasks need finer review groups
(e.g. Seahorse criterion/language and IntentGrasp domain). Each proposed exclusion
should cite its source example ID and concrete evidence; subjective or unresolved
cases stay. A stronger model can assist selected unresolved cases after this review,
but cannot turn disagreement alone into an exclusion decision.

The existing two-model confirmation rule and capture-recapture estimates are not
sufficient adjudication evidence. In particular, dependent, weak detectors and small
samples do not support treating estimated overlaps as measured label-error rates.
