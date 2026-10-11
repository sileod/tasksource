# Human-curated dataset queue

This is the user-approved queue, in batch order. Keep it separate from machine-discovered candidates. Discovery may suggest additions for review; it must not rewrite or silently promote entries here. Update statuses only after checking code and publication.

## Batch 1 — implemented and published

15 sources, 28 task views; Git `140072e`; Jev Hub commit `9d0d1c7dd5f88e1295f2f2bfb954200c26c08e10`, added to `default` and `full`.

MAUD (six views), ACORD, Evidence Inference, DDI, Hallmarks of Cancer, Patent Classification (abstract), CUAD answerability, TabFact, SQuAD v2 answerability, MultiDoGO intent (six domains), InfoTabS, WikiSQL (aggregation/column/operator), EDOS, SemEval 2014 ABSA (laptops/restaurants).

## Batch 2 — integration in progress

| Source | Requested views | Status / remaining checks |
|---|---|---|
| FEVEROUS | Evidence-conditioned supports/refutes/NEI | Pending faithful evidence/table join and native split verification; heavy preparation may need a mirror |
| HateXplain | Majority classification; optional votes | Majority view implemented locally; existing vote view retained |
| QuAC | Answerability, yes/no/neither, follow-up | Answerability implemented locally; yes/no and follow-up deferred after ambiguous native labels in coarse review |
| SLURP | Intent; possible slot tagging | Text-only mirror published; intent integration in progress; slots pending |
| MultiWOZ 2.2 | Intent, dialogue act, categorical slots | Dialogue act and categorical slots implemented locally; intent deferred after stale active-intent labels in coarse review; single-act rows only |
| Schema-Guided Dialogue | Intent and categorical slot values | Intent implemented locally; schema-grounded slot projection pending |
| HotpotQA | Native supporting-fact membership | Implemented locally; explicit native sentence identities |
| MCScript2.0 | Native two-choice comprehension | Original XML mirror published; integration in progress; license still unspecified |

These views have not yet been added to the Jev training export.

## Batch 3 — queued; a few initial mappings in progress

| Source | Requested views | Status / remaining checks |
|---|---|---|
| UKPConvArg1Strict | More convincing argument, native pair preferences | Pending original topic CSV conversion and source-group split policy |
| VAST | Topic-conditioned pro/con/neutral | Pending native CSV, label ontology and split checks |
| FNC-1 | Headline/body stance | Pending original CSV join; inspected Hub copy strips source text |
| DiscoGeM | Discourse relation / annotator votes | Pending original partitions and vote verification |
| UKP Sentential Argument Mining | Topic-conditioned pro/con/no argument | Pending original source and split verification |
| CoQA | Native exact yes/no with prior dialogue | Implemented locally; publication pending |
| QASPER | Native scientific yes/no; possible evidence selection | Binary consensus mapping in progress; text evidence required; answer style is not used as a target |
| TOPv2 | Root intent; possible slot types | Root intent implemented locally; publication pending |
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
