"""Relative sampling weights for builders that mix many tasks (Jev, instruct, label-nli).

Every task weighs 1 except a few groups: templated probes teach little per
extra row, while hard human-written NLI, long documents and preference pairs
stay varied. A builder scales a task's rows (or its family's share) by the
weight. Patterns are regular expressions searched in the task id, first match
wins; keep the list short.
"""

import re

WEIGHTS = {
    r"^UNLI$": 8,  # 55k human probability judgements: the best calibration source
    r"(linguisticprobing|robust_nli|gen_debiased_nli)": 0.1,
    r"universal_dependencies": 0.2,
    r"(label_nli|dpo|dataset_train_nli)": 5,
    r"(anli|WANLI|dynasent/|xnli|FOL-nli)": 3,
    r"(doc-nli|ConTRoL|mctest-nli|summ)": 5,
}


def task_weight(task_id):
    for pattern, weight in WEIGHTS.items():
        if re.search(pattern, task_id):
            return weight
    return 1
