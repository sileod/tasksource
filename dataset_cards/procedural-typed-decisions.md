---
pretty_name: procedural-typed-decisions
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
configs:
- config_name: arithmetic
  data_files:
  - split: train
    path: arithmetic/train-*.parquet
  - split: validation
    path: arithmetic/validation-*.parquet
  - split: test
    path: arithmetic/test-*.parquet
- config_name: entity_belief_tracking
  data_files:
  - split: train
    path: entity_belief_tracking/train-*.parquet
  - split: validation
    path: entity_belief_tracking/validation-*.parquet
  - split: test
    path: entity_belief_tracking/test-*.parquet
- config_name: event_state_reconstruction
  data_files:
  - split: train
    path: event_state_reconstruction/train-*.parquet
  - split: validation
    path: event_state_reconstruction/validation-*.parquet
  - split: test
    path: event_state_reconstruction/test-*.parquet
- config_name: evidence_sufficiency
  data_files:
  - split: train
    path: evidence_sufficiency/train-*.parquet
  - split: validation
    path: evidence_sufficiency/validation-*.parquet
  - split: test
    path: evidence_sufficiency/test-*.parquet
- config_name: multi_view_adjudication
  data_files:
  - split: train
    path: multi_view_adjudication/train-*.parquet
  - split: validation
    path: multi_view_adjudication/validation-*.parquet
  - split: test
    path: multi_view_adjudication/test-*.parquet
- config_name: needle_retrieval
  data_files:
  - split: train
    path: needle_retrieval/train-*.parquet
  - split: validation
    path: needle_retrieval/validation-*.parquet
  - split: test
    path: needle_retrieval/test-*.parquet
- config_name: partial_observation_calibration
  data_files:
  - split: train
    path: partial_observation_calibration/train-*.parquet
  - split: validation
    path: partial_observation_calibration/validation-*.parquet
  - split: test
    path: partial_observation_calibration/test-*.parquet
- config_name: policy_applicability
  data_files:
  - split: train
    path: policy_applicability/train-*.parquet
  - split: validation
    path: policy_applicability/validation-*.parquet
  - split: test
    path: policy_applicability/test-*.parquet
- config_name: record_aggregation
  data_files:
  - split: train
    path: record_aggregation/train-*.parquet
  - split: validation
    path: record_aggregation/validation-*.parquet
  - split: test
    path: record_aggregation/test-*.parquet
- config_name: state_perturbation
  data_files:
  - split: train
    path: state_perturbation/train-*.parquet
  - split: validation
    path: state_perturbation/validation-*.parquet
  - split: test
    path: state_perturbation/test-*.parquet
- config_name: table_lookup
  data_files:
  - split: train
    path: table_lookup/train-*.parquet
  - split: validation
    path: table_lookup/validation-*.parquet
  - split: test
    path: table_lookup/test-*.parquet
---

# procedural-typed-decisions

Procedurally generated decision problems. Each row is one structured state
(JSON, or a table, CSV, key=value lines, or prose for the arithmetic,
retrieval, and aggregation configs) with **several typed questions over that same state**, following the
Jev / System One request shape: `choice` (pick one criterion), `noul` (a
number in [0, 1]; a probability or a yes/no), and `score` (an ordered rubric).
Every answer is computed exactly from the state by rules that the state
itself spells out, so the labels are noise-free.

This is an independent dataset. It is not an official TypeSafe Jev dataset and
is not produced by or affiliated with TypeSafe or OpenJev.

## Configs

| config | questions |
|---|---|
| `arithmetic` | An order with a discount/shipping rule, an account ledger, or a schedule; each state asks 2–5 of: `amount_due` / `final_balance` / `finish_time` (choice among the result and typical slips), `within_budget`, `went_negative`, `done_by_deadline` (noul), `random_line_bulk`, `random_is_deposit`, `random_is_long` (noul, exact probability k/n), `budget_use`, `net_change` (score, descriptive levels), `lines_above`, `withdrawal_count`, `starts_before_noon` (score), `largest_line`, `lowest_day`, `longest_task` (choice) |
| `entity_belief_tracking` | `world_location` (choice), `agent_belief_location` (choice), `belief_matches_world` (noul) |
| `event_state_reconstruction` | `current_owner` (choice), `is_open` (noul), `current_severity` (score) |
| `evidence_sufficiency` | `claim_supported` (noul), `has_conflict` (noul), `strongest_support_origin` (choice) |
| `multi_view_adjudication` | `intent` (choice), `is_urgent` (noul), `workflow_impact` (score) |
| `needle_retrieval` | `value_of_id` (choice), `id_has_value` (noul), `id_listed` (noul); up to ~300 records whose ids differ from the target by one or two digits |
| `partial_observation_calibration` | `incident_real` (noul, exact Bayesian posterior) |
| `policy_applicability` | `access_allowed` (noul), `governing_policy` (choice), `review_risk` (score) |
| `record_aggregation` | `count_in_category` (score), `largest_quantity` (choice), `any_out_of_stock` (noul), `total_above` (noul) |
| `state_perturbation` | `material_change` (noul), `changed_dimension` (choice), `risk_direction` (score) |
| `table_lookup` | `find_person` (choice, two-condition filter), `manager_of` (choice, join), `started_before` (noul), `count_matching` (score) |

## Schema

| field | meaning |
|---|---|
| `id` | `task:split:index` |
| `level` | Difficulty level (0–4); larger levels add events, records, sensors, or distractors. |
| `state` | The state: a JSON string, or rendered text for the retrieval and aggregation configs. |
| `questions` | JSON object of named System One questions (`type`, `instructions`, `criteria`). |
| `answers` | JSON object of reference answers, in the System One `answers` shape. |
| one column per question | Flat label, for browsing and filtering: a `ClassLabel` for choice, score, and yes/no noul questions; a float for graded noul (`incident_real`, `random_*`); the option text for open numeric choices (`amount_due`, `final_balance`, `finish_time`). Null when the state does not ask that question (`arithmetic` only). |

States are unique within a split, and validation/test states never occur in
train.

## Use

As a multi-question Jev request, send `{"state": row["state"], "questions":
json.loads(row["questions"])}` (parsing the state first when it is JSON) and
compare with `row["answers"]`.
The same rows are included, grouped by state, in
[`tasksource/tasksource-jev-typed-decisions`](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions).

## Reproduction

Generation is deterministic (row `i` of a split is seeded by `task:split:i`).
From a [tasksource](https://github.com/sileod/tasksource) checkout:

```bash
PYTHONPATH=.:src python scripts/build_procedural_jev.py --output build/procedural-typed-decisions --upload
```

Generators live in `src/tasksource/jev/procedural/`.
