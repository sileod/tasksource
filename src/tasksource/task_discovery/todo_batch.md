# Human-curated dataset queue

This is the user-approved queue, in batch order. Keep it separate from machine-discovered candidates. Discovery may suggest additions for review; it must not rewrite or silently promote entries here. Update statuses only after checking code and publication.

## Batch 1 — implemented and published

15 sources, 28 task views; Git `140072e`; Jev Hub commit `9d0d1c7dd5f88e1295f2f2bfb954200c26c08e10`, added to `default` and `full`.

MAUD (six views), ACORD, Evidence Inference, DDI, Hallmarks of Cancer, Patent Classification (abstract), CUAD answerability, TabFact, SQuAD v2 answerability, MultiDoGO intent (six domains), InfoTabS, WikiSQL (aggregation/column/operator), EDOS, SemEval 2014 ABSA (laptops/restaurants).

## Batch 2 — first views published; remaining projections queued

| Source | Requested views | Status / remaining checks |
|---|---|---|
| FEVEROUS | Evidence-conditioned supports/refutes/NEI | Pending faithful evidence/table join and native split verification; heavy preparation may need a mirror |
| HateXplain | Majority classification; optional votes | Majority view published; existing vote view retained |
| QuAC | Answerability, yes/no/neither, follow-up | Answerability published; yes/no and follow-up deferred after ambiguous native labels in coarse review |
| SLURP | Intent; possible slot tagging | Text-only mirror and 60-class scenario/action intent published; slots pending |
| MultiWOZ 2.2 | Intent, dialogue act, categorical slots | Dialogue act and categorical slots published; intent deferred after stale active-intent labels in coarse review; single-act rows only |
| Schema-Guided Dialogue | Intent and categorical slot values | Intent published; categorical-slot projection deferred after ambiguous economical → moderate state in native dialogue 1_00000, turn 12 |
| HotpotQA | Native supporting-fact membership | Published; explicit native sentence identities |
| MCScript2.0 | Native two-choice comprehension | Original XML mirror and native MC view published; license still unspecified |

Eleven views across batches 2–3 were added to Jev `default` and `full` in [Hub commit f2529310](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/commit/f2529310ce099771d5d6911e94ba24f4b1ce3ec4): 10,272 train, 1,060 validation and 795 test rows. One exact exported training row per task was inspected, alongside source/recast samples; [review findings](../../../docs/jev/dialogue-expansion-audit.json) preserve the model identifier, sample IDs, revisions and limitations. Existing Parquet shards were preserved.

## Batch 3 — three views published; remaining sources queued

| Source | Requested views | Status / remaining checks |
|---|---|---|
| UKPConvArg1Strict | More convincing argument, native pair preferences | Pending original topic CSV conversion and source-group split policy |
| VAST | Topic-conditioned pro/con/neutral | Pending native CSV, label ontology and split checks |
| FNC-1 | Headline/body stance | Pending original CSV join; inspected Hub copy strips source text |
| DiscoGeM | Discourse relation / annotator votes | Pending original partitions and vote verification |
| UKP Sentential Argument Mining | Topic-conditioned pro/con/no argument | Deferred: official archive requires authentication; sentence reconstruction and separate text rights must be verified |
| CoQA | Native exact yes/no with prior dialogue | Published to Jev default/full |
| QASPER | Native scientific yes/no; possible evidence selection | Binary consensus view published; all cited evidence must occur in supplied text; answer style is not used as a target |
| TOPv2 | Root intent; possible slot types | Root intent published to Jev default/full |
| PERSPECTRUM | Claim/perspective supports/opposes | Pending original perspective join and claim splits |
| PubMed 200k RCT | Sentence rhetorical role | Pending source preserving PMID/abstract identity and native partitions |

## Batch 4 — queued after the earlier batches

| Source | Requested views | Checks before integration |
|---|---|---|
| MAVEN | Sentence + marked trigger → event type | Prioritize; verify trigger spans, native document splits and ontology |
| EmpatheticDialogues | Speaker situation → emotion | Verify situation labels; group by conversation |
| Spider | SQL aggregation/structure; native column selection | Prioritize; join original schemas without exposing gold SQL in inputs |
| OLID | Offensive, targeted, target type | Preserve hierarchical eligibility and native available labels |
| iSarcasmEval | Author-intended sarcasm | Verify original English train CSV and terms |
| SemCor | WordNet sense-definition selection | Repackage pinned sense inventory; verify spans and native alternatives |
| MAGPIE | Idiomatic/literal use | Verify native filtered train/dev/test; likely mirror |
| PreCo | Mention-pair coreference | Verify native mentions/clusters; do not infer negatives from missing annotation |
| ReDial | Movie-conditioned human judgments | Join native movie annotations; preserve dialogue splits |
| MAVEN-Arg | Event-conditioned argument role | Verify trigger/argument alignment and document splits; likely mirror |
| SemEval 2023 Task 3 | Annotated persuasion techniques | Verify span/multilabel projection and original article partitions |
| STREUSLE | Marked expression → supersense | Verify multiword spans, inventory and native splits |
| SST-5 | Five-level sentence sentiment | Check existing coverage and faithful sentence-level native partitions |

## Selection rules

Use genuine labeled training data, existing templates and modest declarative mappings. Preserve native choices, complete evidence and source identities; do not fabricate distractors. Keep evaluation-only sources out of training. Prefer directly usable Hugging Face sources; archive joins, loader repairs and expensive transformations justify a reproducible data-only mirror. Record source terms even when unspecified. Reject or defer sources whose supervision cannot be mapped faithfully, recording the reason here.
