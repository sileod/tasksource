"""Procedural Jev tasks: one generated state, several typed questions over it.

The generators are reasoning-core's Jev tasks (``reasoning_core/tasks/jev/``, https://github.com/sileod/reasoning-core);
``TASKS`` maps each published config to its task, and ``TASKS[name].generate(rng, level)`` returns one state with
every question asked of it, as a ``Problem`` in the published answer schema. ``scripts/build_procedural_jev.py``
publishes them as ``tasksource/procedural-typed-decisions``, one config per task. reasoning-core is needed only
to generate, not to read the published data.
"""

import json
from dataclasses import dataclass

REPO_ID = "tasksource/procedural-typed-decisions"
SOURCE_PREFIX = "procedural-typed-decisions/"
# published config -> reasoning-core task (two were renamed when ported)
GENERATORS = {
    "arithmetic": "practical_arithmetic", "entity_belief_tracking": "entity_belief_tracking",
    "event_state_reconstruction": "event_state_reconstruction", "event_triggered_execution": "event_triggered_execution",
    "evidence_sufficiency": "claim_support", "goal_directed_planning": "goal_directed_planning",
    "graph_navigation": "graph_navigation", "instruction_execution": "instruction_execution",
    "multi_view_adjudication": "multi_view_adjudication", "needle_retrieval": "needle_retrieval",
    "partial_observation_calibration": "partial_observation_calibration",
    "policy_applicability": "policy_applicability", "policy_under_uncertainty": "policy_under_uncertainty",
    "record_aggregation": "record_aggregation", "record_eligibility": "record_eligibility",
    "state_perturbation": "state_perturbation", "table_lookup": "table_lookup", "taxonomy_routing": "taxonomy_routing",
}


@dataclass
class Problem:
    state: dict | str  # str for states rendered as tables, CSV, or prose
    questions: dict    # {qid: {type, instructions, criteria}}
    answers: dict      # {qid: {type, noul} | {type, score, legend, probabilities, confidence} | {type, choice, ...}}


def _answer(q):
    """A labelled reasoning-core question (jev_request) as a published answer."""
    target = q.get("target")
    if q["type"] == "noul":
        return {"type": "noul", "noul": float(target["true"] if target else q["label"])}
    if q["type"] == "score":
        n = len(q["criteria"])
        return {"type": "score", "score": float(q["label"]),
                "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                "probabilities": {str(i): float(target[str(i)] if target else i == q["label"]) for i in range(n)},
                "confidence": 1.0}
    return {"type": "choice", "choice": q["label"], "confidence": 1.0,
            "probabilities": {o: float(target[o] if target else o == q["label"]) for o in q["criteria"]}}


class Generator:
    """One reasoning-core task behind the ``generate(rng, level)`` interface of the build scripts."""

    def __init__(self, name):
        self.name = name

    def generate(self, rng, level):
        from reasoning_core import get_task
        from reasoning_core.decision import jev_request
        request = jev_request(get_task(self.name), rng.randrange(2 ** 32), level)
        questions = {qid: {k: q[k] for k in ("type", "instructions", "criteria") if k in q}
                     for qid, q in request["questions"].items()}
        return Problem(request["state"], questions, {qid: _answer(q) for qid, q in request["questions"].items()})


TASKS = {config: Generator(name) for config, name in GENERATORS.items()}


def reasoning_core_provenance():
    """The reasoning-core commit and task versions that generate the data."""
    import subprocess
    from pathlib import Path
    import reasoning_core
    from reasoning_core import get_task
    root = Path(reasoning_core.__file__).resolve().parents[1]
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no", "reasoning_core") or git(
            "status", "--porcelain", "reasoning_core/tasks/jev"):
        raise RuntimeError(f"reasoning-core checkout {root} has uncommitted changes; commit them first")
    return {"generator": "https://github.com/sileod/reasoning-core", "commit": git("rev-parse", "HEAD"),
            "tasks": {config: {"task": name, "task_version": get_task(name).task_version}
                      for config, name in GENERATORS.items()}}


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
