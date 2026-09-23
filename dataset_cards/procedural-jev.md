---
pretty_name: procedural-jev
language:
- en
license: apache-2.0
task_categories:
- text-classification
tags:
- tasksource
- jev
- system-one
- procedural
- synthetic
- multi-question
---

# procedural-jev

Procedurally generated decision problems. Each row is one structured state
(JSON) with **several typed questions over that same state**, following the
Jev / System One request shape: `choice` (pick one criterion), `noul` (a
number in [0, 1]; a probability or a yes/no), and `score` (an ordered rubric).
Every answer is computed exactly from the state by rules that the state
itself spells out, so the labels are noise-free.

This is an independent dataset. It is not an official TypeSafe Jev dataset and
is not produced by or affiliated with TypeSafe or OpenJev.

## Configs

| config | questions |
|---|---|
| `entity_belief_tracking` | `world_location` (choice), `agent_belief_location` (choice), `belief_matches_world` (noul) |
| `event_state_reconstruction` | `current_owner` (choice), `is_open` (noul), `current_severity` (score) |
| `evidence_sufficiency` | `claim_supported` (noul), `has_conflict` (noul), `strongest_support_origin` (choice) |
| `multi_view_adjudication` | `intent` (choice), `is_urgent` (noul), `workflow_impact` (score) |
| `partial_observation_calibration` | `incident_real` (noul, exact Bayesian posterior) |
| `policy_applicability` | `access_allowed` (noul), `governing_policy` (choice), `review_risk` (score) |
| `state_perturbation` | `material_change` (noul), `changed_dimension` (choice), `risk_direction` (score) |

## Schema

| field | meaning |
|---|---|
| `id` | `task:split:index` |
| `level` | Difficulty level (0–4); larger levels add events, records, sensors, or distractors. |
| `state` | The state, as a JSON string. |
| `questions` | JSON object of named System One questions (`type`, `instructions`, `criteria`). |
| `answers` | JSON object of reference answers, in the System One `answers` shape. |
| one column per question | Flat label, for browsing and filtering: a `ClassLabel` for choice, score, and yes/no noul questions; a float for graded noul (`incident_real`). |

States are unique within a split, and validation/test states never occur in
train.

## Use

As a multi-question Jev request, send `{"state": json.loads(row["state"]),
"questions": json.loads(row["questions"])}` and compare with `row["answers"]`.
The same rows are included, grouped by state, in
[`tasksource/tasksource-jev`](https://huggingface.co/datasets/tasksource/tasksource-jev).

## Reproduction

Generation is deterministic (row `i` of a split is seeded by `task:split:i`).
From a [tasksource](https://github.com/sileod/tasksource) checkout:

```bash
PYTHONPATH=.:src python scripts/build_procedural_jev.py --output build/procedural-jev --upload
```

Generators live in `src/tasksource/jev/procedural/`.
