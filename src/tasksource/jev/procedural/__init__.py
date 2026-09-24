"""Procedural Jev tasks: one generated state, several typed questions over it.

Each module exposes ``generate(rng, level)`` returning a ``Problem`` whose
answers are derived from the state by the rules the state itself states.
``scripts/build_procedural_jev.py`` publishes them as ``tasksource/procedural-typed-decisions``,
one config per task.
"""

import json

from . import (
    entity_belief_tracking, event_state_reconstruction, evidence_sufficiency,
    multi_view_adjudication, needle_retrieval, partial_observation_calibration,
    policy_applicability, record_aggregation, state_perturbation, table_lookup,
)

TASKS = {
    module.__name__.rsplit(".", 1)[-1]: module
    for module in (
        entity_belief_tracking, event_state_reconstruction, evidence_sufficiency,
        multi_view_adjudication, needle_retrieval, partial_observation_calibration,
        policy_applicability, record_aggregation, state_perturbation, table_lookup,
    )
}

REPO_ID = "tasksource/procedural-typed-decisions"
SOURCE_PREFIX = "procedural-typed-decisions/"


def jev_rows(example, source, split, index):
    """One published-dataset row as grouped Jev decisions sharing its state.

    Rows follow the tasksource-jev training schema; the id's first three
    ``:`` fields name the group, so every question over a state stays together.
    """
    questions = json.loads(example["questions"])
    answers = json.loads(example["answers"])
    rows = []
    for qid, spec in questions.items():
        answer = answers[qid]
        if spec["type"] == "noul":
            options, target = [], [float(answer["noul"])]
        elif spec["type"] == "score":
            options = list(spec["criteria"])
            target = [float(i == answer["score"]) for i in range(len(options))]
        else:
            options = list(spec["criteria"])
            target = [float(answer["probabilities"][option]) for option in options]
        rows.append({
            "id": f"{source.replace('/', '-')}:{split}:{index}:{qid}",
            "kind": spec["type"],
            "options": options,
            "target": target,
            "state": example["state"],
            "question": spec["instructions"],
            "source": source,
            "variant": "direct",
            "split": split,
        })
    return rows
