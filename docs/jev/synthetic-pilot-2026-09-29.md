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

## Broad interpretive text understanding

The audited and 4,000-state configurations add low-weight general reading skills
through the ordinary domain sampler. The current 14 skills cover topic, emotion,
communicative intent, claim support, stance, document purpose, main point,
implicit concern, intended audience, argument role, stakeholder perspective,
social implication, evidence strength, and message tone. They share generation,
validation, critic, Jev annotation, and independent audit with other skills.
Arithmetic, chronology, entity lookup, and literal retrieval are left to the
procedural segment. There are no benchmark labels or benchmark-specific paths.
Prompt files have stable names (`generate.txt`, `critic.txt`, and
`teacher_audit.txt`) rather than version suffixes.

A deterministic 4,000-spec draw yielded 5,890 questions, including 606 general
reading questions (10.3%). All 36 domains appeared. Each of the 14 general
skills appeared in at least 19 domains. The overall mix still includes existing
skills such as toxicity, sentiment, and groundedness. Difficulty levels 1–5
remain available; the general segment intentionally contains both simple text
classification and questions requiring several cues.

The first natural live pilot, `.synthetic_runs/natural_general_pilot_115/`, used
an earlier eight-skill candidate mix. It yielded 96 retained states and 123
Jev decisions from 115 generated states. All 18 retained general questions had
Jev/auditor agreement, but manual review found an entity-type question that
classified an issue rather than an entity. That, plus overlap with the
procedural generators, prompted the current interpretive mix.

The current natural live pilot is
`.synthetic_runs/interpretive_general_pilot_120/` (ignored by Git). Its 120
Albert generations produced 117 validated states, 94 critic-approved and
retained states, and 119 Jev decisions. Sixteen retained questions used nine
of the general reading skills. The independent auditor answered 118 of 119
questions, disagreed with Jev on 12 overall and one general question, flagged
34 questions for review, and found no confident disagreements; 93 of 94 states
passed audit. Manual review of the 16 general questions found useful variety,
but also drift: two `claim_support` questions asked for a main concern, an
`audience_inference` question became ticket routing, an `implicit_concern`
question guessed a customer's feelings without the customer's own words, and a
`practical_implication` question duplicated policy application. The last skill
has been replaced by `social_implication`, and the generation and critic prompts
now explicitly reject those drifts. Jev/auditor agreement did not catch them.

That current pilot took 582 seconds end to end with one Albert key and pilot
critic/auditor limits of 120 requests per minute: about 13,900 retained states
or 17,700 decisions per day if that rate holds. The earlier 115-state pilot
took 481 seconds with two Albert keys: about 17,200 retained states or 22,100
decisions per day by the same extrapolation. These are short-run estimates,
not sustained throughput guarantees; the production configs use 40 critic and
auditor requests per minute, and quota, retries, cost, and longer-run quality
may change throughput. Auditor agreement is diagnostic, not ground truth.

A focused quality probe, `.synthetic_runs/interpretive_skill_quality_probe_16/`,
sampled four specs each for claim support, intended audience, implicit concern,
and social implication from the ordinary 4,000-spec draw. Fourteen of 16 states
passed the original validator and critic, yielding 21 decisions. Manual review
found an ambiguous intended audience, a negotiation implication with several
plausible readings, and a question that projected today's idle crew into a
client visit tomorrow. The auditor flagged the first two but agreed on the
third. The critic now requires a per-question, exact state quote and explicit
skill, support, and unique-answer checks. On the same 15 validated states this
stricter DeepSeek check rejected the ambiguous negotiation question, although
it still missed the unwarranted projection about tomorrow. This is a concrete
remaining quality risk. A larger manual sample is needed before treating an
unattended 4,000-state run as clean training data.
