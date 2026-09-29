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
- config_name: all
  default: true
  data_files:
  - split: train
    path: all/train-*.parquet
  - split: validation
    path: all/validation-*.parquet
  - split: test
    path: all/test-*.parquet
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
- config_name: policy_under_uncertainty
  data_files:
  - split: train
    path: policy_under_uncertainty/train-*.parquet
  - split: validation
    path: policy_under_uncertainty/validation-*.parquet
  - split: test
    path: policy_under_uncertainty/test-*.parquet
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
- config_name: taxonomy_routing
  data_files:
  - split: train
    path: taxonomy_routing/train-*.parquet
  - split: validation
    path: taxonomy_routing/validation-*.parquet
  - split: test
    path: taxonomy_routing/test-*.parquet
---

# procedural-typed-decisions

Procedurally generated decision problems. Each row is one structured state
(JSON, or a table, CSV, key=value lines, or prose for the arithmetic,
retrieval, and aggregation configs) with **several typed questions over that same state**, following the
Jev / System One request shape: `choice` (pick one criterion), `noul` (a
number in [0, 1]; a probability or a yes/no), and `score` (an ordered rubric).
Every answer is computed exactly from the state by rules that the state
itself spells out, so the labels are noise-free. Several configs vary the number of options
(4 to 60), to balance the binary and 4–6-option questions that dominate the rest of Jev.

This is an independent dataset. It is not an official TypeSafe Jev dataset and
is not produced by or affiliated with TypeSafe or OpenJev.

## Configs

| config | questions |
|---|---|
| `all` (default) | Every config below in one table, with a `task` column and the shared fields only (no flat label columns); the first 1,000 train rows cycle through levels and tasks, the rest is shuffled |
| `arithmetic` | An order with a discount/shipping rule, an account ledger, or a schedule; each state asks 2–5 of: `amount_due` / `final_balance` / `finish_time` (choice among the result and typical slips), `within_budget`, `went_negative`, `done_by_deadline` (noul), `random_line_bulk`, `random_is_deposit`, `random_is_long` (noul, exact probability k/n), `budget_use`, `net_change` (score, descriptive levels), `lines_above`, `withdrawal_count`, `starts_before_noon` (score), `largest_line`, `lowest_day`, `longest_task` (choice) |
| `entity_belief_tracking` | `world_location` (choice), `agent_belief_location` (choice), `belief_matches_world` (noul), from level 2 `nested_belief_location` (choice: where A thinks B believes an object is); 4 to 16 locations |
| `event_state_reconstruction` | `current_owner` (choice), `is_open` (noul), `current_severity` (score); the log is shuffled from level 2 and has voided entries from level 3 |
| `evidence_sufficiency` | `claim_supported` (noul), `has_conflict` (noul), `strongest_support_origin` (choice); retractions from level 2, mirrored (non-independent) origins from level 3, validity by collection day at level 4 |
| `multi_view_adjudication` | `intent` (choice), `is_urgent` (noul), `workflow_impact` (score); near-threshold signals from level 2, auth failures counted from login events from level 3, deadlines as clock times at level 4 |
| `needle_retrieval` | `value_of_id` (choice), `id_has_value` (noul), `id_listed` (noul); up to ~300 records whose ids differ from the target by one or two digits, and from level 2 a chain of one to three id reissues to follow; 6 to 20 options |
| `partial_observation_calibration` | `incident_real` (noul, exact Bayesian posterior); 1 to 6 sensors |
| `policy_applicability` | `access_allowed` (noul), `governing_policy` (choice), `review_risk` (score); 2 one-constraint policies at level 0, about 12 policies of up to 5 constraints, many of them near misses, at level 4 |
| `policy_under_uncertainty` | `access_allowed` (noul), `governing_policy` (choice), `requester_role` (choice); exact posteriors over a role known through history counts and reports of stated reliability |
| `record_aggregation` | `count_in_category` (score), `largest_quantity` (choice), `any_out_of_stock` (noul), `total_above` (noul), from level 2 `count_filtered` (score, quantity and stock filters) |
| `state_perturbation` | `material_change` (noul), `changed_dimension` (choice), `risk_direction` (score); 1 to 8 records with up to 4 simultaneous changes whose risk effects can offset, and look-alike non-material fields |
| `table_lookup` | `find_person` (choice, two-condition filter, through the manager from level 2 and with a start-year condition from level 3; 6 to 40 options, capped by the table), `manager_of` (choice, join), `started_before` (noul), `count_matching` (score) |
| `taxonomy_routing` | `route` (choice among the 4–60 categories of a routing guide drawn fresh per state; many rules share a condition with the right one), `belongs_to` (noul), `conditions_met` (score, 0–4) |

## Schema

| field | meaning |
|---|---|
| `id` | `task:split:index` |
| `level` | Difficulty level (0–4), calibrated against Jev (see below). |
| `state` | The state: a JSON string, or rendered text for the retrieval and aggregation configs. |
| `questions` | JSON object of named System One questions (`type`, `instructions`, `criteria`). |
| `answers` | JSON object of reference answers, in the System One `answers` shape. |
| one column per question | Flat label, for browsing and filtering: a `ClassLabel` for choice, score, and yes/no noul questions; a float for graded noul (`incident_real`, `random_*`); the option text for open numeric choices (`amount_due`, `final_balance`, `finish_time`). Null when the state does not ask that question (`arithmetic`, and level-dependent questions). |

States are unique within a split, and validation/test states never occur in
train. In each config, the first 1,000 train rows cycle through the levels
(easiest first) for browsing; the rest of the split is shuffled.

## Difficulty by level

Level 0 is meant to be easy for a strong decision model and level 4 hard. The
table gives Jev's chance-adjusted accuracy, kappa = (accuracy − chance) /
(1 − chance), on 40 fresh states per level (every question of each state;
`typesafe/jev-1.13-20260917`, September 2026). 1 is perfect, 0 is chance.

| config | level 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `arithmetic` | 0.83 | 0.64 | 0.59 | 0.54 | 0.59 |
| `entity_belief_tracking` | 0.91 | 0.93 | 0.78 | 0.74 | 0.66 |
| `event_state_reconstruction` | 0.97 | 0.99 | 0.92 | 0.73 | 0.60 |
| `evidence_sufficiency` | 0.99 | 0.97 | 0.78 | 0.72 | 0.78 |
| `multi_view_adjudication` | 0.82 | 0.87 | 0.82 | 0.73 | 0.74 |
| `needle_retrieval` | 1.00 | 0.99 | 0.78 | 0.83 | 0.36 |
| `partial_observation_calibration` | 0.73 | 0.37 | 0.60 | 0.18 | 0.23 |
| `policy_applicability` | 0.73 | 0.60 | 0.53 | 0.64 | 0.35 |
| `policy_under_uncertainty` | 0.35 | 0.42 | 0.38 | 0.57 | 0.39 |
| `record_aggregation` | 0.99 | 0.96 | 0.84 | 0.75 | 0.70 |
| `state_perturbation` | 0.95 | 0.89 | 0.44 | 0.63 | 0.50 |
| `table_lookup` | 1.00 | 0.98 | 0.97 | 0.90 | 0.87 |
| `taxonomy_routing` | 1.00 | 0.98 | 0.92 | 0.78 | 0.64 |

Probability answers are scored above by their rounding to yes/no; Jev's mean
absolute error on the exact probability grows from 0.16 (level 0) to 0.29
(level 4) on `incident_real`, and stays around 0.33 on the posterior
`access_allowed` of `policy_under_uncertainty`. `policy_under_uncertainty` and
`partial_observation_calibration` (exact posteriors) are hard from level 0 on;
`table_lookup` and `multi_view_adjudication` remain the easiest at level 4.

A second model, `upstage/solar-decide` (10 states per level; it takes at most
26 options, so the longest lists are left out), shows the same easy-to-hard
slope on most configs, e.g. 0.95 → 0.53 on `event_state_reconstruction`,
1.00 → 0.48 on `evidence_sufficiency`, 0.88 → 0.42 on `policy_applicability`;
`arithmetic`, `table_lookup`, and `state_perturbation` stay easy for it (about
0.8–0.9 at every level). Rerun with `scripts/calibrate_procedural_levels.py`
(`--model` for another model of the OpenRouter decisions API).

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

## Citation

Generated with [tasksource](https://github.com/sileod/tasksource); please cite:

```bibtex
@inproceedings{sileo-2024-tasksource,
    title = "tasksource: A Large Collection of {NLP} tasks with a Structured Dataset Preprocessing Framework",
    author = "Sileo, Damien",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1361",
    pages = "15655--15684",
}
```
