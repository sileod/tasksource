# Legal, biomedical and structured-text training views

These annotations use the existing classification and multiple-choice templates. Every source revision is pinned; native evaluation partitions are retained, and missing splits are not synthesized.

| Source | Task views | Preparation |
|---|---|---|
| MAUD | Six question-specific merger-agreement classifications | Native CSV; only question/subquestion combinations with unambiguous label ontologies |
| ACORD | Attorney-rated clause relevance, ordered 0–4 | Mirror joins explicit qrels to queries and clauses; unjudged pairs are not negatives |
| Evidence Inference | Intervention/comparator/outcome effect direction | Mirror joins labels and evidence by prompt **and** article; majority verified labels, excluding ties and known bad prompts |
| DDI | Drug-pair interaction type | Native sentence and explicitly named drug pair |
| Hallmarks of Cancer | Ten question-conditioned binary labels per sentence | Preserves multilabel annotations and source PMID identity |
| Patent Classification | Abstract → patent section | Native abstract configuration; full patents are omitted to avoid unnecessary very long inputs |
| CUAD | Native extraction-question answerability | Complete contract and native answer-span presence |
| TabFact / InfoTabS | Table-supported statement classification | Complete source tables and captions/titles |
| SQuAD v2 | Native question answerability | Complete passage and native answer-span presence |
| MultiDoGO | Intent, separately for six domains | Current utterance with earlier annotated dialogue text only; no future turns or intent labels in context |
| ReCoRD | Binary classification of every native entity candidate | All matching annotated aliases are positive; hidden-label test is omitted |
| WikiSQL | Aggregation, selected column, condition operator | Schema and question; conditions additionally specify column/value, never gold SQL |
| EDOS | Native binary sexism annotation | Official CSV mirrored with native partitions; category/vector annotations stay in metadata |
| SemEval 2014 ABSA | Laptop / restaurant aspect sentiment | Native aspect spans verified against sentence text; conflict labels retained |

The conversion entry point is `python -m scripts.repackage_dataset`; ACORD, EDOS and Evidence Inference builders live in `scripts/repackage_dataset/specialized_text.py`. Mirrors use `sentence1`, `sentence2`, `labels` and JSON-string `metadata`, with conversion/source hashes and split policies in `provenance.json`.

Licenses remain source-specific. DDI is non-commercial; InfoTabS and MultiDoGO record original publisher terms even where Hub metadata is incomplete. Patent Classification and ABSA remain unspecified where no dataset license was verified. Evidence Inference's annotation/code license does not clear the rights to all article text, so its complete text rows remain unspecified. No source is silently relabeled under Tasksource's code license.

For Jev publication, retain the full evidence and exclude examples exceeding the existing request-size budget instead of truncating around a known answer. Per-row source identities and native partitions accompany incremental text exports in a metadata sidecar.

Structural audit counts and scope: [specialized-text-audit.json](audits/specialized-text-audit.json). MAUD, patents, TabFact and WikiSQL exclude identical inputs with conflicting targets and retain only one native copy, with heldouts taking precedence. ABSA removes training sentences duplicated in its test set. MultiDoGO retains multi-intent utterances as context but excludes them as single-intent targets. Native split membership is otherwise unchanged.
