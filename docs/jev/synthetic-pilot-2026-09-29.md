# Synthetic Jev pilot review, 2026-09-29

The audited synthetic run samples all 36 domains and 24 skills without targeting
evaluation labels. A deterministic 4,000-spec sample from the revised config has
5,944 questions: 45.6% choice, 39.2% noul, and 15.3% score. Difficulty,
ambiguity, and distractor weights favor clearer examples while retaining every
level. The published Tasksource catalog separately includes AG News, emotion,
Banking77, and SuperGLUE CB source tasks; this synthetic run is broad decision
practice, not a replacement for those task families. Hold out direct source
families when measuring transfer to their benchmarks.

Live pilot artifact: `.synthetic_runs/wide_simple_prompt_pilot_24/` (ignored by
Git). Albert generated 24 states; 23 passed deterministic validation and 19
passed the stricter critic. The retained 19 states contain 22 real Jev 1.13
decisions across 15 domains and 15 skills. The critic used Albert DeepSeek
with a stricter prompt, while Jev labeling and the independent GPT-4.1-mini
audit used OpenRouter. OpenRouter usage is subject to the account's pricing and
limits. All 22 audit answers were complete;
the independent auditor disagreed on three and flagged one confident
disagreement. The flagged case concerned three inverter units. The state
explicitly defines a clock offset over five minutes as a mismatch, so Jev's
answer including Unit 12 appears correct and the auditor's omission appears
wrong. Audit flags are diagnostic, not ground truth.

The critic rejected a choice question whose SLA wording left both a preliminary
report and a final report defensible, and a deployment question where waiting
and inspecting logs were both valid actions. Two retained choice disagreements
still involve plausible alternative actions or ownership, so manual review
remains necessary before claiming clean supervision. The default export keeps
audited disagreements and preserves provenance; use the audit fields for
curation rather than treating the critic or auditor as an oracle.

The Albert provider accepts an optional `ALBERT_API_KEY_2` environment variable
in the audited and 4,000-state configs. When present, generation and critic
requests rotate across distinct keys; the raw generation records store only a
zero-based credential slot, never a secret. The second key passed model
preflight. A 32-state generation comparison took 69 seconds with two keys
(16 requests per slot) and 89 seconds with one key. Neither short run showed
a rate-limit error, so this suggests a speed benefit but does not establish
independent sustained quotas.
