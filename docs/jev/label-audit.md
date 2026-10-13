# Label audit

Every published source example is screened for wrong gold labels by an LLM, and doubtful ones are
confirmed by a second, independent model before anything is removed.

## Steps

1. **Screen** (`scripts/jev_error_pass.py`). DeepSeek V4 Flash (Albert) reads several examples of one
   source per prompt and returns `ok`, `wrong`, `ambiguous` or `malformed` for each gold answer, judged
   against the dataset's own conventions. One direct row per `example_id` is checked, `default` first.
   Constructed sources (procedural, synthetic logic) are skipped: their labels are right by design.
   Soft or uncertain targets are skipped too.
2. **Confirm** (`--confirm`). Batched screening over-flags hard examples, so each flagged example is
   re-checked alone: DeepSeek reasons step by step after examples of the same source, and Jev scores
   the gold. An example is bad only when both reject it (DeepSeek picks another option and Jev gives
   the gold under 0.3).
3. **Review and apply**. A source with more than 8% bad examples disagrees systematically (a labeling
   convention, or hard data); it is reviewed by reading random rows and fixed, parked or kept, rather
   than losing its flagged examples. The remaining bad examples go to `build/jev-error-pass/bad-examples.csv`;
   once reviewed, copying them to `src/tasksource/metadata/jev_bad_examples.csv` makes the build drop them.

`scripts/jev_second_detector.py` scores a fixed sample of 30 examples per source with Jev. With the
DeepSeek screen, two independent detectors give a capture-recapture estimate of the wrong labels both
miss (`capture-recapture.csv`).

## After adding sources

Verdicts are keyed by `example_id`, a hash of the decision content, so a rerun screens only examples
not seen before: new sources and rows whose text or labels changed. After a publish:

```bash
set -a; . ~/.jev_synth.env; set +a
PYTHONPATH=.:src python scripts/jev_error_pass.py             # screen unseen examples
PYTHONPATH=.:src python scripts/jev_error_pass.py --confirm   # confirm flags, write bad-examples.csv
```

Both resume where they stopped. Outputs are in `build/jev-error-pass/`: `verdicts.jsonl` (one verdict per
example, with the row when flagged) and `per-source-summary.csv` (flag rates per source).
`source_index` on published rows points a flagged example back to its original dataset row.

## First pass (2026-10-13)

1.59M examples screened: 94.8% ok, 4.0% wrong, 0.6% malformed, 0.5% ambiguous. Of 543 sources with at
least 200 examples, 212 are under 2% flagged, 104 over 8% and 14 over 20%. Commonsense (8.9%), intent
(6.8%) and sentiment (6.7%) have the highest flag rates, safety the lowest (1.0%).

Flags are not errors: reading random rows of the most flagged sources found real label noise in some
(`dair-ai/emotion`, CrowdFlower `text_emotion`, IndicGLUE ACTSA, since removed), missing input in others
(DiscoSense lacked its discourse connective, since restored), and defensible labels in the rest. A [per-task review](audits/jev-filter-review.md) of the first screening
records the manual decisions behind these readings.
On the 17,822-example Jev sample, DeepSeek flagged 906, Jev 2,997, and both 530.
