"""Relative sampling weights for builders that mix many tasks (Jev, instruct, label-nli).

A weight scales a task family's share of the mix; the default is 1. They are
judgment calls, not measurements: templated or probing data teaches little per
extra row, while human-written adversarial and long-document data stays varied.
Patterns are regular expressions searched in the task id (``multilingual/`` ids
included); the first match wins.
"""

import re

WEIGHTS = [
    # templated or synthetic probes: a few thousand rows show the pattern
    (r"^(linguisticprobing|robust_nli|gen_debiased_nli)", 0.25),
    (r"^babi_nli", 0.5),
    # long documents and summaries: varied inputs, few alternatives elsewhere
    (r"^(doc-nli|ConTRoL-nli|mctest-nli|summarize_from_feedback|seahorse_summarization_evaluation)", 5),
    # human-written adversarial or hard NLI and sentiment
    (r"^(anli/|WANLI|dynasent/|FOL-nli|(multilingual/)?xnli)", 3),
    # preference pairs
    (r"(_dpo|dpo_pairs)", 3),
]


def task_weight(task_id):
    for pattern, weight in WEIGHTS:
        if re.search(pattern, task_id):
            return weight
    return 1
