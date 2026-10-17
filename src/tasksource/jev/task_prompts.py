"""Task-specific prompts for Jev decisions, tagged by style.

Jev phrases most tasks with a few generic instructions ("Choose the criterion that best describes
the state.") over generic fields (``text_A``/``text_B``). Evaluation suites usually state each task's
own question and name fields naturally ("Classify the relationship between the premise and
hypothesis." / ``Premise: ...``). ``task_prompts.json`` (next to this module) holds, per Jev source,
natural names for its state fields and reviewed task-specific instructions in several styles, so a
training run can select styles and layouts and compare them in controlled experiments:

- ``directive``: an imperative, task-specific instruction ("Classify the stance of the post toward the topic.")
- ``question``: a direct question ("Does the post support the topic?")
- ``explained``: one or two sentences that also say what the decision rests on
- ``terse``: a short cue ("Stance of the perspective toward the claim:")
- ``conversational``: how a user would ask an assistant ("Can you tell me whether this review is positive?")
- ``annotation``: how a labeling guideline phrases it ("Label the premise–hypothesis pair with its inference relation.")
- ``grounded``: a question that names the state fields ("According to the passage, is the answer to the question yes?")
- ``system``: a system prompt for an automated pipeline ("You are a support-ticket router. Assign the intent of the message.")

Layouts: ``inline`` puts the content after the instruction with an empty state, as benchmarks often
ask single-text decisions; ``fields`` keeps it in a JSON state keyed by the natural field names.
Sources with a fixed label set also get one definition per label; :func:`describe_options` renders
options as ``label: definition``. Only the wording changes: targets and option order are untouched.
The sidecar is generated and reviewed by ``scripts/build_jev_task_prompts.py``.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

PROMPTS_PATH = Path(__file__).with_name("task_prompts.json")
STYLES = ("directive", "question", "explained", "terse", "conversational", "annotation", "grounded", "system")
LAYOUTS = ("inline", "fields")


@lru_cache(maxsize=None)
def load_task_prompts(path=PROMPTS_PATH):
    """``{source: {"fields": {state_field: natural_name}, "text_name": str | None, "styles": {style: [str]},
    "labels": {label: definition}}}`` (``labels`` only for fixed label sets)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["tasks"]


def state_fields(state, names):
    """Split a ``name: value`` state into ``[(name, value)]`` over the given field names, in order.

    Returns ``None`` when the state does not start with one of the fields (e.g. free text)."""
    parts = re.split(r"^(%s):[ ]?" % "|".join(map(re.escape, names)), state, flags=re.M)
    if len(parts) < 3 or parts[0].strip():
        return None
    return [(name, value.strip()) for name, value in zip(parts[1::2], parts[2::2])]


def task_style(state, source, choice, layout="inline", styles=STYLES, prompts=None):
    """Render one decision of ``source`` with a task-specific instruction.

    ``choice`` in [0, 1) picks one instruction among those of the selected ``styles``; ``layout`` is
    ``inline`` or ``fields`` (see the module docstring). Returns ``(state, question)``, or ``None`` when
    the source has no reviewed instruction in these styles or its state does not match its fields.

    Apply it to single-question rows: on a state shared by several questions, the task instruction
    would replace questions it does not describe."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    spec = (load_task_prompts() if prompts is None else prompts).get(source)
    instructions = [s for style in styles for s in (spec or {}).get("styles", {}).get(style, ())]
    if not instructions:
        return None
    if spec["fields"]:
        named = state_fields(state, list(spec["fields"]))
        if named is None:
            return None
        named = [(spec["fields"][name], value) for name, value in named]
    else:
        named = [(spec["text_name"], state.strip())]
    named = [(name, value) for name, value in named if value]
    instruction = instructions[min(int(choice * len(instructions)), len(instructions) - 1)]
    if not named:
        return "", instruction
    if layout == "inline":
        body = named[0][1] if not spec["fields"] else "\n".join(f"{name}: {value}" for name, value in named)
        return "", f"{instruction}\n{body}"
    return json.dumps(dict(named), ensure_ascii=False), instruction


def describe_options(options, source, prompts=None):
    """``label: definition`` for each option when ``source`` defines all of them, else the options unchanged."""
    labels = (load_task_prompts() if prompts is None else prompts).get(source, {}).get("labels") or {}
    if not all(option in labels for option in options):
        return list(options)
    return [f"{option}: {labels[option]}" for option in options]
