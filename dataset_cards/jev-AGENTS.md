# AGENTS.md: using tasksource-jev-typed-decisions

Notes for coding agents (and people) who load, filter, or train on this dataset.

## What a row is

One row is one **decision**: a `state` (the text), a `question`, and a
`target` over `options`.

| `kind` | `options` | `target` | loss that fits |
|---|---|---|---|
| `choice` | unordered answers, order shuffled per row | distribution over `options`, sums to 1 | cross-entropy against the distribution |
| `score` | ordered levels, lowest first | distribution over levels, sums to 1; its mean is the expected level | cross-entropy, or a distance on the expected level |
| `noul` | empty | `[p]`, the probability that the answer to the yes/no question is yes | binary cross-entropy with a soft target |

- Targets are often soft: annotator votes, or mean ratings split between the two
  nearest levels. Don't argmax them unless you want hard labels.
- Never assume the gold answer is the first option: `choice` options are permuted per row.
- The same ordinal task appears as both `choice` and `score` (a per-row hash decides).
  Treat `kind` as part of the request, not as a property of the source.

## Rows are flat; groups link them

Related decisions (several questions about one text, the variants of one source
example, the questions about one packed state) are separate rows sharing a
`group_id`. Groups are stored contiguously, and the whole group is always in one split.

To build multi-question requests, group by `group_id` **and** `state`: some
variants reformat the text, so a group can hold more than one state. Within one
`(group_id, state)` pair, `question_id`s are unique.

```python
from itertools import groupby
from datasets import load_dataset

ds = load_dataset("tasksource/tasksource-jev-typed-decisions", split="train", streaming=True)

def requests(rows):
    for (group, state), members in groupby(rows, key=lambda r: (r["group_id"], r["state"])):
        members = list(members)
        yield {
            "state": state,
            "questions": {m["question_id"]: {"type": m["kind"], "instructions": m["question"],
                                             **({"criteria": m["options"]} if m["options"] else {})}
                          for m in members},
            "targets": {m["question_id"]: m["target"] for m in members},
            "source": members[0]["source"],
        }
```

`groupby` works on the stored order because groups are contiguous. Shuffle
requests, not rows, if you want to keep them together.

## Columns you will filter on

- `source`: the tasksource task a row comes from (`multilingual/...` for non-English,
  `procedural-typed-decisions/...` for generated tasks). `sources.yaml` maps each
  source to its Hub dataset, revision, original dataset and licenses.
- `license_use`: `commercial`, `non-commercial` or `unspecified`, the most restrictive
  license found on the source's Hub cards and in Data Provenance Initiative annotations.
  `license` lists them. Best effort, not legal advice.
- `variant`: `direct` (the source example as is), `label_verification` (a yes/no check of
  one label), `criteria_permutation`, `instruction_paraphrase`, `paired_text_format`
  (surface variations), `packed_derived` (questions over 2-4 packed items).
- `split`: the source's own split (`train`, `dev`, `test`); it matches the Hub split.

```python
ds = ds.filter(lambda use: use == "commercial", input_columns="license_use")
direct = ds.filter(lambda v: v == "direct", input_columns="variant")  # e.g. a cleaner eval
```

## Evaluation caveats

- Validation and test rows whose text appears in train were removed.
- BIG-bench, MMLU and BLiMP are excluded, so they are clean for evaluation. GLUE,
  SuperGLUE, HellaSwag, PIQA and many other public benchmarks are **in** the training
  data (their train splits), so don't report zero-shot results on them after training here.
- Eval splits include variants and packed questions over the same examples; for a
  per-example metric, keep `variant == "direct"`.

## Rebuilding

The dataset is built by `scripts/build_jev_dataset.py` in
[tasksource](https://github.com/sileod/tasksource); `build-manifest.json` and
`build-report.jsonl` record the settings, code commit and source revisions of this release.
