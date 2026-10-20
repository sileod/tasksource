# English span-structure source audit — 2026-10-20

English only. No Hub writes, uploads or publication. `DATA_CHANGELOG.md` was not edited.
All active tasks were actually loaded through `load_task`; every generated QA answer and every
assembled extraction span was checked verbatim. The checked-in [audit JSON](span-structure.json)
contains per-split field counts, null shares, provenance, licenses and two complete example rows
per admitted or locally converted collection. The conversion-only ids below are proposed names,
**not registered catalog tasks**.

## Verdicts

| Request | Verdict and relevance | Active task ids / proposed pending ids | License |
|---|---|---|---|
| 1. English UD syntax | Added EWT and GUM: syntactic argument/modifier subtrees add structural decisions beyond NER. Direct, data-only Hub upload; no mirror needed. | `ud/ewt`, `ud/gum` | EWT CC BY-SA 4.0; GUM CC BY-NC-SA 4.0, non-commercial |
| 2. QA-SRL | Prepared locally; skipped active registration because Bank 2.0 Hub uploads are script-only with no usable Parquet conversion. `luheng/qa_srl` is older, lacks validator judgments, and does not satisfy the requested validation filter. Human predicate-argument sets are relevant. | pending `qasrl/orig` | Unspecified data scope; MIT repository software license does not clear the underlying texts |
| 3. BioScope | Prepared public abstracts/papers locally; skipped active registration because the BigBio conversion branch contains no data files and the legacy loader needs archive extraction and URL repair. Clinical text is masked and needs restricted originals: excluded. Expert cue/scope boundaries add negation and speculation. | pending `bioscope/abstracts`, `bioscope/papers` | CC BY 2.0 according to BigBio card; underlying article rights remain applicable |
| 3. SFU Review | Skipped: publisher download requires name/institution/email; no verified usable English Hub copy with native train split and cue-to-scope identity found. Spanish SFU is excluded. The accessible auxiliary BIO reformat drops cue identities and does not establish original train partitions. No trustworthy conversion can be completed from that view. | none | Unspecified; no data grant established from accessible publisher page |
| 3. Sherlock / ConanDoyle-neg | Prepared locally; skipped active registration because available Hub views lack reliable native split provenance or cue identities. The 235-row `joey234` upload matches the official test set's negated-sentence count, not the 3,644-sentence training split. Pinned gold-column source retains all negation instances and official story partitions. | pending `sherlock/scopes` | Unspecified annotation license; original organizers say stories are public domain and annotations freely available |
| 4. Coreference | Added PreCo's mention-set view, reusing the catalog's existing original Hub dataset. At least two distinct mentions, earliest anchor unique in displayed document. LitBank coreference is not needed for this first collection; its existing entity view is unchanged. | `preco/mentions` | Research purposes, non-commercial, as already recorded by the existing source review |
| 5. Relations | Added original authors' Re-DocRED upload: human-revised document-level target entity sets. Only `train_revised`, `dev_revised`, `test_revised`; no distant split. | `redocred/entities` | MIT author card for annotations; Wikipedia source text retains its terms |
| 6. Chunking | Added existing CoNLL-2000 token task to the extraction NER path, including noun/verb phrases and the other native chunk types. Pinned the actual Parquet conversion commit and preserved train/test. | `conll2000` | Unspecified; no concrete corpus license found. Missing licenses are allowed by implementing-sources.md; the Apache loader license is not treated as a data grant. |

## Source pins and evidence

- UD: [`universal-dependencies/universal_dependencies@c3fed78df6b9f8cd27ed24cc7142e9eccf1e6703`](https://huggingface.co/datasets/universal-dependencies/universal_dependencies/tree/c3fed78df6b9f8cd27ed24cc7142e9eccf1e6703), configs `en_ewt`, `en_gum`. Human-corrected basic dependency trees. Original licenses: [EWT](https://github.com/UniversalDependencies/UD_English-EWT/blob/4a4d77f599ea53cc405f85d0cec4b2f14f81d42b/LICENSE.txt), [GUM](https://github.com/UniversalDependencies/UD_English-GUM/blob/1fe635509c649e376dfb449d528424ab78f4eaee/LICENSE.txt).
- PreCo: [`coref-data/preco_raw@02c8ffedb9d0b49168809a572ff553c2f6f9f7ca`](https://huggingface.co/datasets/coref-data/preco_raw/tree/02c8ffedb9d0b49168809a572ff553c2f6f9f7ca). Human mention clusters; [publisher](https://preschool-lab.github.io/PreCo/). Native train 36,620 documents, development 500; no released labeled test. No synthesized holdouts for this view.
- Re-DocRED: [`tonytan48/Re-DocRED@e0ab3489edfe72c968261bffed5243b6fefddd22`](https://huggingface.co/datasets/tonytan48/Re-DocRED/tree/e0ab3489edfe72c968261bffed5243b6fefddd22). Author describes human revision of 3,053/500/500 train/dev/test documents and MIT terms. Model proposals in the original revision workflow do not become labels without human review.
- CoNLL-2000: [`eriktks/conll2000@bda188850926d9a842f700ffce57c5232b794ce6`](https://huggingface.co/datasets/eriktks/conll2000/tree/bda188850926d9a842f700ffce57c5232b794ce6). Hub API's actual `refs/convert/parquet` commit. Procedural chunks from human Penn Treebank parses; native WSJ sections 15–18 train (8,937 sentences) and 20 test (2,013).
- QA-SRL: [Bank 2.0 format](https://github.com/uwnlp/qasrl-bank/blob/29034398c26e419d530a2ec1e81994ca4a4d4e42/FORMAT.md), [original archive](https://qasrl.org/data/qasrl-v2.tar), SHA256 `caba7a2d18584d6210b9a176ef6b4808d96f3d48194083dcb20db03697793e07`. `biu-nlp/qa_srl2018@4911b977f03d95b8e327136623898403ca325d33` and `kleinay/qa_srl@5b94fa5e146412a5d7e645245f5b151bacee8e2b` are loader-only; `biu-nlp/qa_srl2020@92ff90949cd65739b3a362f59bc7ac15b0851579` is gold evaluation-only and cannot supply training.
- BioScope: [BigBio source loader and annotation evidence](https://huggingface.co/datasets/bigbio/bioscope/blob/43ba162359f01d6436590ba2a014421b6ad09621/bioscope.py); its conversion commit `12f670cc752011d9797e343947d8694824a2e28e` contains no Parquet data. [Working original archive URL](https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/bioscope.zip), SHA256 `b20b525506f783ab9c120548ac4022ab587bf7cc419beaf759f7253f0cf0b1d1`. BigBio assigns the public corpus to train only; no evaluation split is invented.
- SFU: [original download form](https://www.sfu.ca/~mtaboada/download/downloadNegation.html); [corpus overview](https://www.sfu.ca/~mtaboada/SFU_Review_Corpus). No form submission or identity information sent.
- Sherlock: [organizers' availability statement](https://www.aclweb.org/portal/content/sem-shared-task-2012-resolving-scope-and-focus-negation), [human annotation paper](https://aclanthology.org/L12-1077/), [pinned native-column copy](https://github.com/mosharafhossain/negation-cue/tree/03cb052efdb205efdf403223a71764237a353f6e/data/cd-sco). Original train is Baskervilles, dev is Wisteria, test is Cardboard/Circle. SHA256s: merged train/dev `ea966b062893cdc18d8d74b1c635f50db623f3f6e9d605fa4e3e262272901e64`, test `cb71fad09c44b8d4d7147cd7275fd025583d1c8d6f1458f039565c685c85b996`. The converter verifies 3,644/787/1,089 native sentences before output. Rejected Hub copies: `joey234/conandoyle_cue_scope@8efa0bb5b3e5e41d2b15f85c354d5c52f7997f0c` (235 rows), `MisterStino/negation-scope-cue-given@08142c28570c0265ff149b9f8e12037b56487bb0` (scopes without recoverable cue identity); [auxiliary BIO reformat](https://github.com/jerbarnes/multitask_negation_for_targeted_sentiment/tree/2f3eb5184abb53cbc957a3fac36ddfb6f922ed6e/data/auxiliary_tasks/en) also collapses separate negations.

## Mapping and safeguards

UD base relations map to six fixed fields:

| UD relation (including subtypes) | Readable field |
|---|---|
| `nsubj`, including `nsubj:pass` | subject |
| `obj` | object |
| `iobj` | indirect object |
| `obl`, including `obl:agent` | oblique modifier |
| `ccomp` | clausal complement |
| `xcomp` | open clausal complement |

One eligible unique-surface head per sentence (prefer VERB, then earliest token). All six fields
are lists, allowing multiple dependents. Each answer includes the dependent's entire basic-tree
subtree, including punctuation if the tree attaches it there. Native sentence text is aligned
exactly, with whitespace gaps only. If any subtree for a field is non-contiguous, the **entire
field** is omitted and counted; remaining valid dependents are never presented as a complete
set and rejected positives never become null. Missing relations are earned nulls. No enhanced
or empty-node edges are used. Native `dev` becomes tasksource `validation` without repartitioning.

PreCo tokens are joined with one space, matching the existing catalog's representation. Mentions
are sorted by document position, including the anchor itself; singleton clusters and ambiguous
anchors are omitted. Re-DocRED uses the same token-offset helper. Relation direction is native
`h -> t`, readable relation names reuse `tasks.DOCRED_RELATIONS`, targets are aggregated per
`(h, relation)` and represented by their earliest annotated mention. Anchor text must occur
once and identify only one source entity. No unannotated entity/relation pair becomes a null.

QA-SRL retains `orig` only. At least two distinct human workers must judge a question valid,
and valid judgments must form a strict majority. Model answer judgments are excluded; worker
round suffixes are normalized before counting. One modal **complete real validator reading**
is retained, with deterministic tie-breaking; conflicting readings are never unioned into an
invented set. Predicate character positions disambiguate repeated verbs. Lists are answer sets.

BioScope walks XML text/tails once, recording nested cue and scope offsets, preserving cue refs
and discontinuous segments. Sherlock keeps each gold triplet separately, locates sub-token
cue affixes and scope complements exactly and merges only adjacent whitespace-separated scope
segments. Native no-cue sentences give earned nulls; unlocated positive fields are counted and
omitted. SFU has no accepted rows or invented examples.

Every new collection shares a task prefix. The active QA views use `metadata.field` and
`metadata.group`, `_span_records`, `qa_rows` and `LIST_QA`; chunks reuse `NER`/`ner_rows`.
UD shares one collection budget across EWT/GUM. No source download/setup is in task preprocessing.

## Verification

- Requested test command: `PYTHONPATH=.:src python -m pytest -q tests/test_extraction.py tests/test_task_catalogs.py tests/test_catalog_naming.py tests/test_licenses.py` — **82 passed**.
- Actual `load_task` for every active task, all splits; exact-span validation with `max_mismatch=0` for new QA views. Separate full extraction audit validates **every** output span, including local conversions. Reproduce with `scripts/audit_span_structure.py`.
- Builder smoke: `PYTHONPATH=.:src python scripts/build_jev_extraction.py --output build/span-structure/smoke --tasks ud/ewt ud/gum preco/mentions redocred/entities conll2000 --max-rows 300 --max-rows-eval 50` — **1,500 train, 200 validation, 200 test**. Each active task contributes 300 training rows; each available evaluation split contributes 50.
- Existing builder deduplication and train/evaluation text-overlap removal apply. Detailed smoke counts are in the audit JSON. Native all-empty chunk sentences are omitted (3 train, 1 test).
- Repackage outputs remain under ignored `build/span-structure/pending-*`; no mirror is registered or uploaded. Each generated dataset card links the original distribution, the **specific** conversion script and linked provenance with source hashes/settings. Scripts: [`span_structure.py`](../../../scripts/repackage_dataset/span_structure.py), [`sherlock_spans.py`](../../../scripts/repackage_dataset/sherlock_spans.py).
- Git index/worktree creation in the supplied checkout is blocked by a read-only shared Git directory. The requested commit is prepared on branch `span-structure` in `/tmp/span-structure-review`; the working-file edits also remain in the supplied checkout.

## Split counts and null shares

Counts below are **grouped extraction rows before builder deduplication, overlap filtering and caps**.
The machine-readable audit also gives loaded QA record counts (`input_rows`) and field counts.
A row can contain several fields; `null share` is the share of **fields**, not rows.

| Collection | Status | Train rows / null share | Validation rows / null share | Test rows / null share |
|---|---|---:|---:|---:|
| `ud/ewt` | added | 10,007 / 35.5% | 1,423 / 36.9% | 1,380 / 36.4% |
| `ud/gum` | added | 9,418 / about 36% | 1,342 / about 36% | 1,267 / about 36% |
| `preco/mentions` | added | 43,242 / 0.00% | 607 / 0.00% | — |
| `redocred/entities` | added | 7,618 / 0.00% | 1,434 / 0.00% | 1,431 / 0.00% |
| `conll2000` | added | 8,934 / 46.81% | — | 2,012 / 47.12% |
| `qasrl/orig` | local only | 49,777 / 0.00% | 9,717 / 0.00% | 11,254 / 0.00% |
| `bioscope/abstracts` | local only | 11,871 / 71.28% | — | — |
| `bioscope/papers` | local only | 2,671 / 70.21% | — | — |
| `sherlock/scopes` | local only | 3,642 / 62.67% | 787 / 67.50% | 1,089 / 64.23% |
| SFU Review | skipped; no accepted rows | — | — | — |

UD asks the relations the head has plus one it lacks (one earned null per row).

Original-sentence/doc counts differ from grouped extraction rows: UD omits sentences with no
eligible unique head, PreCo omits singleton/ambiguous clusters and chunks documents at eight
fields, Re-DocRED chunks each document at eight relation fields. BioScope can split sentences
with more than eight fields. All QA-SRL questions have positive validated spans; no speculative
nulls are generated. Both BioScope cue fields are asked on no-cue sentences, explaining its
high earned-null share. Sherlock retains native no-negation sentences.

## Dropped annotations

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| EWT: no eligible unambiguous head (sentences) | 2,537 | 578 | 697 |
| EWT: non-contiguous subtrees / omitted fields | 91 / 91 | 5 / 5 | 9 / 9 |
| GUM: no eligible unambiguous head (sentences) | 1,896 | 233 | 197 |
| GUM: non-contiguous subtrees / omitted fields | 125 / 124 | 26 / 26 | 20 / 19 |
| PreCo: singleton clusters | 1,997,261 | 25,995 | — |
| PreCo: ambiguous-anchor clusters | 194,960 | 2,674 | — |
| Re-DocRED: ambiguous-anchor relation fields | 15,250 | 2,958 | 2,855 |
| QA-SRL local: questions failing validation policy | 5,501 | 909 | 1,168 |
| Sherlock local: unlocated cue fields (scope and aggregate-cue fields omitted as needed) | 6 | 0 | 0 |

UD had no failed native-text alignments in these snapshots. BioScope public XML had no missing
scope references. QA-SRL had no overlapping selected readings. These are annotation/filter
counts, separate from builder duplicate/overlap removal. No dropped positive becomes null.

## Two example rows per collection

The compact examples below preserve the complete state, instructions, all field answers and
character offsets. `[]` means null. The linked JSON retains the full extraction schema including
`null`, source, split, types and metadata. Pending examples demonstrate local conversions only;
they are not active catalog tasks. SFU has no accepted rows to show.

### `ud/ewt`

Example 1:

```json
{
  "state": "A least your going.!",
  "fields": [
    {
      "name": "subject",
      "instruction": "Every subject of 'going', including each dependent's full subtree, as written.",
      "values": [
        {
          "text": "you",
          "start": 8,
          "end": 11
        }
      ]
    },
    {
      "name": "object",
      "instruction": "Every object of 'going', including each dependent's full subtree, as written.",
      "values": []
    },
    {
      "name": "indirect_object",
      "instruction": "Every indirect object of 'going', including each dependent's full subtree, as written.",
      "values": []
    },
    {
      "name": "oblique_modifier",
      "instruction": "Every oblique modifier of 'going', including each dependent's full subtree, as written.",
      "values": [
        {
          "text": "A least",
          "start": 0,
          "end": 7
        }
      ]
    },
    {
      "name": "clausal_complement",
      "instruction": "Every clausal complement of 'going', including each dependent's full subtree, as written.",
      "values": []
    },
    {
      "name": "open_clausal_complement",
      "instruction": "Every open clausal complement of 'going', including each dependent's full subtree, as written.",
      "values": []
    }
  ]
}
```

Example 2:

```json
{
  "state": "I got so excited!!!!",
  "fields": [
    {
      "name": "subject",
      "instruction": "Every subject of 'got', including each dependent's full subtree, as written.",
      "values": [
        {
          "text": "I",
          "start": 0,
          "end": 1
        }
      ]
    },
    {
      "name": "object",
      "instruction": "Every object of 'got', including each dependent's full subtree, as written.",
      "values": []
    },
    {
      "name": "indirect_object",
      "instruction": "Every indirect object of 'got', including each dependent's full subtree, as written.",
      "values": []
    },
    {
      "name": "oblique_modifier",
      "instruction": "Every oblique modifier of 'got', including each dependent's full subtree, as written.",
      "values": []
    },
    {
      "name": "clausal_complement",
      "instruction": "Every clausal complement of 'got', including each dependent's full subtree, as written.",
      "values": []
    },
    {
      "name": "open_clausal_complement",
      "instruction": "Every open clausal complement of 'got', including each dependent's full subtree, as written.",
      "values": [
        {
          "text": "so excited!!!!",
          "start": 6,
          "end": 20
        }
      ]
    }
  ]
}
```

### `ud/gum`

Example 1:

```json
{
  "state": "\"What are you doing?",
  "fields": [
    {
      "name": "subject",
      "instruction": "Every subject of 'doing', including each dependent's full subtree, as written.",
      "values": [
        {
          "text": "you",
          "start": 10,
          "end": 13
        }
      ]
    },
    {
      "name": "object",
      "instruction": "Every object of 'doing', including each dependent's full subtree, as written.",
      "values": [
        {
          "text": "What",
          "start": 1,
          "end": 5
        }
      ]
    },
    {
      "name": "indirect_object",
      "instruction": "Every indirect object of 'doing', including each dependent's full subtree, as written.",
      "values": []
    },
    {
      "name": "oblique_modifier",
      "instruction": "Every oblique modifier of 'doing', including each dependent's full subtree, as written.",
      "values": []
    },
    {
      "name": "clausal_complement",
      "instruction": "Every clausal complement of 'doing', including each dependent's full subtree, as written.",
      "values": []
    },
    {
      "name": "open_clausal_complement",
      "instruction": "Every open clausal complement of 'doing', including each dependent's full subtree, as written.",
      "values": []
    }
  ]
}
```

Example 2:

```json
{
  "state": "And now we did yoga!",
  "fields": [
    {
      "name": "subject",
      "instruction": "Every subject of 'did', including each dependent's full subtree, as written.",
      "values": [
        {
          "text": "we",
          "start": 8,
          "end": 10
        }
      ]
    },
    {
      "name": "object",
      "instruction": "Every object of 'did', including each dependent's full subtree, as written.",
      "values": [
        {
          "text": "yoga",
          "start": 15,
          "end": 19
        }
      ]
    },
    {
      "name": "indirect_object",
      "instruction": "Every indirect object of 'did', including each dependent's full subtree, as written.",
      "values": []
    },
    {
      "name": "oblique_modifier",
      "instruction": "Every oblique modifier of 'did', including each dependent's full subtree, as written.",
      "values": []
    },
    {
      "name": "clausal_complement",
      "instruction": "Every clausal complement of 'did', including each dependent's full subtree, as written.",
      "values": []
    },
    {
      "name": "open_clausal_complement",
      "instruction": "Every open clausal complement of 'did', including each dependent's full subtree, as written.",
      "values": []
    }
  ]
}
```

### `preco/mentions`

Example 1:

```json
{
  "state": "I 'm a Chinese girl . My name is Li Ying . I 'm twelve . I 'm in No . 1 Middle school . Chen Hong is my friend . She is nine . Mr Wu is my Chinese teacher and Miss Zhao is my English teacher .",
  "fields": [
    {
      "name": "entity_2",
      "instruction": "Every mention of the entity first mentioned as 'Chen Hong', including that first mention.",
      "values": [
        {
          "text": "Chen Hong",
          "start": 88,
          "end": 97
        },
        {
          "text": "my friend",
          "start": 101,
          "end": 110
        },
        {
          "text": "She",
          "start": 113,
          "end": 116
        }
      ]
    },
    {
      "name": "entity_4",
      "instruction": "Every mention of the entity first mentioned as 'Mr Wu', including that first mention.",
      "values": [
        {
          "text": "Mr Wu",
          "start": 127,
          "end": 132
        },
        {
          "text": "my Chinese teacher",
          "start": 136,
          "end": 154
        }
      ]
    },
    {
      "name": "entity_5",
      "instruction": "Every mention of the entity first mentioned as 'Miss Zhao', including that first mention.",
      "values": [
        {
          "text": "Miss Zhao",
          "start": 159,
          "end": 168
        },
        {
          "text": "my English teacher",
          "start": 172,
          "end": 190
        }
      ]
    }
  ]
}
```

Example 2:

```json
{
  "state": "Hello ! I`m Tony Green . My telephone number is five five seven four one zero two , and my school ID card number is two six five nine . This is my English book . It 's blue . I like it very much .",
  "fields": [
    {
      "name": "entity_0",
      "instruction": "Every mention of the entity first mentioned as 'I`m', including that first mention.",
      "values": [
        {
          "text": "I`m",
          "start": 8,
          "end": 11
        },
        {
          "text": "Tony Green",
          "start": 12,
          "end": 22
        },
        {
          "text": "My",
          "start": 25,
          "end": 27
        },
        {
          "text": "my",
          "start": 88,
          "end": 90
        },
        {
          "text": "my",
          "start": 144,
          "end": 146
        },
        {
          "text": "I",
          "start": 175,
          "end": 176
        }
      ]
    },
    {
      "name": "entity_1",
      "instruction": "Every mention of the entity first mentioned as 'My telephone number', including that first mention.",
      "values": [
        {
          "text": "My telephone number",
          "start": 25,
          "end": 44
        },
        {
          "text": "five five seven four one zero two",
          "start": 48,
          "end": 81
        }
      ]
    },
    {
      "name": "entity_2",
      "instruction": "Every mention of the entity first mentioned as 'my school ID card number', including that first mention.",
      "values": [
        {
          "text": "my school ID card number",
          "start": 88,
          "end": 112
        },
        {
          "text": "two six five nine",
          "start": 116,
          "end": 133
        }
      ]
    },
    {
      "name": "entity_3",
      "instruction": "Every mention of the entity first mentioned as 'This', including that first mention.",
      "values": [
        {
          "text": "This",
          "start": 136,
          "end": 140
        },
        {
          "text": "my English book",
          "start": 144,
          "end": 159
        },
        {
          "text": "It",
          "start": 162,
          "end": 164
        },
        {
          "text": "it",
          "start": 182,
          "end": 184
        }
      ]
    }
  ]
}
```

### `redocred/entities`

Example 1:

```json
{
  "state": "\" Pass It On \" is a song by The Coral , which was taken as the second single from their 2003 album , Magic and Medicine . \" Pass It On \" is The Coral 's highest charting single to date , reaching number five in the UK Singles Chart . \" Pass It On \" was not only released on CD and vinyl but also on DVD which contains 2 b - sides and a 3-minute documentary about the band . The song is similar in structure to The Beatles song \" You Like Me Too Much \" . It appeared in an episode of Scrubs , \" My Turf War \" and is featured in the theatrical trailer for Ken Loach 's film Looking for Eric .",
  "fields": [
    {
      "name": "entity_3_P175",
      "instruction": "Every entity linked from 'Magic and Medicine' by the relation 'performer'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "The Coral",
          "start": 28,
          "end": 37
        }
      ]
    },
    {
      "name": "entity_3_P527",
      "instruction": "Every entity linked from 'Magic and Medicine' by the relation 'has part(s)'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "Pass It On",
          "start": 2,
          "end": 12
        }
      ]
    },
    {
      "name": "entity_3_P577",
      "instruction": "Every entity linked from 'Magic and Medicine' by the relation 'publication date'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "2003",
          "start": 88,
          "end": 92
        }
      ]
    },
    {
      "name": "entity_9_P800",
      "instruction": "Every entity linked from 'Beatles' by the relation 'notable work'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "You Like Me Too Much",
          "start": 429,
          "end": 449
        }
      ]
    },
    {
      "name": "entity_10_P175",
      "instruction": "Every entity linked from 'You Like Me Too Much' by the relation 'performer'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "Beatles",
          "start": 414,
          "end": 421
        }
      ]
    },
    {
      "name": "entity_11_P527",
      "instruction": "Every entity linked from 'Scrubs' by the relation 'has part(s)'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "My Turf War",
          "start": 494,
          "end": 505
        }
      ]
    },
    {
      "name": "entity_12_P179",
      "instruction": "Every entity linked from 'My Turf War' by the relation 'part of the series'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "Scrubs",
          "start": 483,
          "end": 489
        }
      ]
    },
    {
      "name": "entity_12_P361",
      "instruction": "Every entity linked from 'My Turf War' by the relation 'part of'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "Scrubs",
          "start": 483,
          "end": 489
        }
      ]
    }
  ]
}
```

Example 2:

```json
{
  "state": "Oleg Ivanovich Dahl (; 25 May 1941 , Moscow – 3 March 1981 , Kiev ) was a Soviet actor . He acted in films , from classics of drama to fairy tales and adventures . His most popular works included Zhenya , Zhenechka and Katyusha ( 1967 ) , Chronicles of a Dive Bomber ( 1967 ) , An Old , Old Tale ( 1970 ) , King Lear ( 1969 ) , On Thursday and Never Again ( 1977 ) , September Vacation ( 1979 ) . Oleg Dahl played his last cinema role in Uninvited Friend by Leonid Maryagin in 1981 . He worked in Sovremennik Theatre ( 1963 - 1971 , 1973 - 1975 ) and in Malaia Bronnaia Theatre ( 1975 - 1978 ) .",
  "fields": [
    {
      "name": "entity_0_P19",
      "instruction": "Every entity linked from 'Oleg Ivanovich Dahl' by the relation 'place of birth'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "Moscow",
          "start": 37,
          "end": 43
        }
      ]
    },
    {
      "name": "entity_0_P20",
      "instruction": "Every entity linked from 'Oleg Ivanovich Dahl' by the relation 'place of death'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "Kiev",
          "start": 61,
          "end": 65
        }
      ]
    },
    {
      "name": "entity_0_P27",
      "instruction": "Every entity linked from 'Oleg Ivanovich Dahl' by the relation 'country of citizenship'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "Soviet",
          "start": 74,
          "end": 80
        }
      ]
    },
    {
      "name": "entity_0_P569",
      "instruction": "Every entity linked from 'Oleg Ivanovich Dahl' by the relation 'date of birth'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "25 May 1941",
          "start": 23,
          "end": 34
        }
      ]
    },
    {
      "name": "entity_0_P570",
      "instruction": "Every entity linked from 'Oleg Ivanovich Dahl' by the relation 'date of death'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "3 March 1981",
          "start": 46,
          "end": 58
        },
        {
          "text": "1981",
          "start": 477,
          "end": 481
        }
      ]
    },
    {
      "name": "entity_2_P131",
      "instruction": "Every entity linked from 'Moscow' by the relation 'located in the administrative territorial entity'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "Soviet",
          "start": 74,
          "end": 80
        }
      ]
    },
    {
      "name": "entity_2_P17",
      "instruction": "Every entity linked from 'Moscow' by the relation 'country'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "Soviet",
          "start": 74,
          "end": 80
        }
      ]
    },
    {
      "name": "entity_6_P161",
      "instruction": "Every entity linked from 'Zhenya , Zhenechka and Katyusha' by the relation 'cast member'. Return the first mention of each target entity in document order.",
      "values": [
        {
          "text": "Oleg Ivanovich Dahl",
          "start": 0,
          "end": 19
        }
      ]
    }
  ]
}
```

### `conll2000`

Example 1:

```json
{
  "state": "First , a pop quiz .",
  "fields": [
    {
      "name": "adjective_phrase",
      "instruction": "Every adjective phrase chunk in the text, as written.",
      "values": []
    },
    {
      "name": "adverb_phrase",
      "instruction": "Every adverb phrase chunk in the text, as written.",
      "values": [
        {
          "text": "First",
          "start": 0,
          "end": 5
        }
      ]
    },
    {
      "name": "noun_phrase",
      "instruction": "Every noun phrase chunk in the text, as written.",
      "values": [
        {
          "text": "a pop quiz",
          "start": 8,
          "end": 18
        }
      ]
    },
    {
      "name": "subordinate_clause",
      "instruction": "Every subordinate clause chunk in the text, as written.",
      "values": []
    }
  ]
}
```

Example 2:

```json
{
  "state": "First , generalize .",
  "fields": [
    {
      "name": "adverb_phrase",
      "instruction": "Every adverb phrase chunk in the text, as written.",
      "values": [
        {
          "text": "First",
          "start": 0,
          "end": 5
        }
      ]
    },
    {
      "name": "prepositional_phrase",
      "instruction": "Every prepositional phrase chunk in the text, as written.",
      "values": []
    },
    {
      "name": "particle",
      "instruction": "Every particle chunk in the text, as written.",
      "values": []
    },
    {
      "name": "verb_phrase",
      "instruction": "Every verb phrase chunk in the text, as written.",
      "values": [
        {
          "text": "generalize",
          "start": 8,
          "end": 18
        }
      ]
    }
  ]
}
```

### `qasrl/orig` (local only)

Example 1:

```json
{
  "state": "How did life begin ?",
  "fields": [
    {
      "name": "predicate_1_argument_0",
      "instruction": "What did something do? For predicate 'did' at characters [4, 7) (zero-based, end-exclusive). Return every answer span.",
      "values": [
        {
          "text": "begin",
          "start": 13,
          "end": 18
        }
      ]
    },
    {
      "name": "predicate_1_argument_1",
      "instruction": "What did something? For predicate 'did' at characters [4, 7) (zero-based, end-exclusive). Return every answer span.",
      "values": [
        {
          "text": "life",
          "start": 8,
          "end": 12
        }
      ]
    },
    {
      "name": "predicate_3_argument_0",
      "instruction": "What began? For predicate 'begin' at characters [13, 18) (zero-based, end-exclusive). Return every answer span.",
      "values": [
        {
          "text": "life",
          "start": 8,
          "end": 12
        }
      ]
    }
  ]
}
```

Example 2:

```json
{
  "state": "It focuses on CFCs .",
  "fields": [
    {
      "name": "predicate_1_argument_0",
      "instruction": "What does something focus on? For predicate 'focuses' at characters [3, 10) (zero-based, end-exclusive). Return every answer span.",
      "values": [
        {
          "text": "on CFCs",
          "start": 11,
          "end": 18
        }
      ]
    },
    {
      "name": "predicate_1_argument_1",
      "instruction": "What focuses? For predicate 'focuses' at characters [3, 10) (zero-based, end-exclusive). Return every answer span.",
      "values": [
        {
          "text": "It",
          "start": 0,
          "end": 2
        }
      ]
    }
  ]
}
```

### `bioscope/abstracts` (local only)

Example 1:

```json
{
  "state": "Succinate alone had no effect.",
  "fields": [
    {
      "name": "negation_cues",
      "instruction": "Every negation cue in the text, as written.",
      "values": [
        {
          "text": "no",
          "start": 20,
          "end": 22
        }
      ]
    },
    {
      "name": "speculation_cues",
      "instruction": "Every speculation cue in the text, as written.",
      "values": []
    },
    {
      "name": "negation_scope_0",
      "instruction": "The full scope of the negation cue 'no' at [20, 22) (zero-based characters, end-exclusive). Return every scope segment.",
      "values": [
        {
          "text": "no effect",
          "start": 20,
          "end": 29
        }
      ]
    }
  ]
}
```

Example 2:

```json
{
  "state": "it had no effect on IL-10 secretion.",
  "fields": [
    {
      "name": "negation_cues",
      "instruction": "Every negation cue in the text, as written.",
      "values": [
        {
          "text": "no",
          "start": 7,
          "end": 9
        }
      ]
    },
    {
      "name": "speculation_cues",
      "instruction": "Every speculation cue in the text, as written.",
      "values": []
    },
    {
      "name": "negation_scope_0",
      "instruction": "The full scope of the negation cue 'no' at [7, 9) (zero-based characters, end-exclusive). Return every scope segment.",
      "values": [
        {
          "text": "no effect on IL-10 secretion",
          "start": 7,
          "end": 35
        }
      ]
    }
  ]
}
```

### `bioscope/papers` (local only)

Example 1:

```json
{
  "state": "Possible Signaling Role for Adenosine",
  "fields": [
    {
      "name": "negation_cues",
      "instruction": "Every negation cue in the text, as written.",
      "values": []
    },
    {
      "name": "speculation_cues",
      "instruction": "Every speculation cue in the text, as written.",
      "values": [
        {
          "text": "Possible",
          "start": 0,
          "end": 8
        }
      ]
    },
    {
      "name": "speculation_scope_0",
      "instruction": "The full scope of the speculation cue 'Possible' at [0, 8) (zero-based characters, end-exclusive). Return every scope segment.",
      "values": [
        {
          "text": "Possible Signaling Role for Adenosine",
          "start": 0,
          "end": 37
        }
      ]
    }
  ]
}
```

Example 2:

```json
{
  "state": "So, roX1 and roX2 might be partially redundant.",
  "fields": [
    {
      "name": "negation_cues",
      "instruction": "Every negation cue in the text, as written.",
      "values": []
    },
    {
      "name": "speculation_cues",
      "instruction": "Every speculation cue in the text, as written.",
      "values": [
        {
          "text": "might",
          "start": 18,
          "end": 23
        }
      ]
    },
    {
      "name": "speculation_scope_0",
      "instruction": "The full scope of the speculation cue 'might' at [18, 23) (zero-based characters, end-exclusive). Return every scope segment.",
      "values": [
        {
          "text": "might be partially redundant",
          "start": 18,
          "end": 46
        }
      ]
    }
  ]
}
```

### `sherlock/scopes` (local only)

Example 1:

```json
{
  "state": "I will not tell . ''",
  "fields": [
    {
      "name": "negation_scope_0",
      "instruction": "The scope of the negation cue 'not' at [7, 10) (zero-based characters, end-exclusive). Return every scope segment, excluding the cue itself.",
      "values": [
        {
          "text": "I will",
          "start": 0,
          "end": 6
        },
        {
          "text": "tell",
          "start": 11,
          "end": 15
        }
      ]
    },
    {
      "name": "negation_cues",
      "instruction": "Every negation cue in the text, as written.",
      "values": [
        {
          "text": "not",
          "start": 7,
          "end": 10
        }
      ]
    }
  ]
}
```

Example 2:

```json
{
  "state": "That was imprudent .",
  "fields": [
    {
      "name": "negation_scope_0",
      "instruction": "The scope of the negation cue 'im' at [9, 11) (zero-based characters, end-exclusive). Return every scope segment, excluding the cue itself.",
      "values": [
        {
          "text": "That was",
          "start": 0,
          "end": 8
        },
        {
          "text": "prudent",
          "start": 11,
          "end": 18
        }
      ]
    },
    {
      "name": "negation_cues",
      "instruction": "Every negation cue in the text, as written.",
      "values": [
        {
          "text": "im",
          "start": 9,
          "end": 11
        }
      ]
    }
  ]
}
```
