# Reusable exclusion evidence

This resource records selection evidence from the historical `filtered-full`
configuration. It is independent of the ongoing DeepSeek audit. No new model
calls, relabeling, approved removal list or training-data change is involved.

Compare `full/train` at `d2ab1d12be4463fb7ac1f877ed2193b3bb59642a` with
`filtered-full/train` at `84008c8a8f4640a2b8a3b222cb62f23ae5fa6e73` by their
original unique decision `id`. Frozen source files are checksum-verified. Every
selected ID must exist in the source and selection counts must reconcile.

| Observed exclusion category | Rows |
| --- | ---: |
| Non-commercial or unspecified license policy | 1,025,115 |
| Explicit benchmark-source holdout policy | 122,480 |
| Unattributed remaining exclusions | 46,927 |
| All excluded decisions | 1,194,522 |

License and benchmark attribution reproduces the published rules and reconciles
their exact counts. These are training policies, not source-label defects.
The policy categories are mutually exclusive: benchmark holdout takes precedence
over licensing. Both rules apply to 68,655 rows; this precedence reproduces the
report's attribution without counting those rows twice.
The remaining 46,927 rows combine 23,138 reported GLM rejections, 7 unresolved
API errors and 23,782 overlap, positional-option, budget and conflicting-target
exclusions. The published release supplies aggregate counts, not the per-row
GLM or other remaining verdicts. Their individual attribution cannot be inferred
uniquely from selection membership. None is labeled a verified error here.

## Files

- `excluded.parquet`: every excluded decision's original ID, source, group,
  question ID, variant, split, license-use policy, reconstructed category and
  candidate decision key where available.
- `candidates.parquet`: the 46,927 unattributed exclusions, with complete original
  rows and canonical decision keys. This is a review resource, not a blacklist.
- `manifest.json`: pinned revisions, fingerprints, category counts, per-source
  counts, reconstructed benchmark rules and explicit attribution limitations.

The Hub directory is `quality/filtered-full-v1/`. These are sidecar resources;
existing dataset configurations and Parquet shards remain unchanged.

## Reuse across recasts

`tasksource.quality.decision_key` accepts published typed decisions and canonical
Tasksource `recast='jev'` rows. The SHA256 identity includes the task, exact input,
question, decision kind and semantic gold/distribution. Neutral choice order is
canonicalized; score order and positional option references remain significant.
Corrected gold or context, different questions, and different label vocabularies
do not silently inherit an earlier flag. IDs based on row position are insufficient
when sampling or preprocessing changes.

```python
from huggingface_hub import hf_hub_download
from tasksource.quality import ExclusionIndex, decision_key

path = hf_hub_download(
    'tasksource/tasksource-jev-typed-decisions',
    'quality/filtered-full-v1/candidates.parquet', repo_type='dataset',
)
evidence = ExclusionIndex.from_parquet(path)
matches = evidence.match(canonical_decision)
# Matches indicate an unattributed historical exclusion, not an approved defect.
```

For another instruct/chat rendering, attach `decision_key(canonical_decision)`
before formatting and carry that identity with the unchanged example; query it
with `evidence.lookup(key)`. Recompute it if context, task or gold changes. If an
old recast did not preserve that identity and changes question wording or input
formatting, an explicit adapter or source mapping is needed. There is no fuzzy
matching or automatic group-wide propagation. A rejected derived question does
not establish that every question in its group is bad.

## Reproduction

```bash
PYTHONPATH=.:src python scripts/extract_jev_exclusions.py
```

The script exports exact set membership, reproduces policy categories and validates
source hashes and selection coverage. Per-row reviewer reasons and adjudication
can later be attached to these identities without recasting all training data.
The ongoing audit's model flags still require review before becoming an approved
cross-recast removal resource.
