"""Deterministic, auditable subrecasts for the published Jev corpus."""

import hashlib

from datasets import Dataset, concatenate_datasets

from .jev_prompt_augmentations import instruction_variants, paired_state_variants


def stable_fraction(identifier, salt):
    """Map an identifier and namespace to a stable value in ``[0, 1)``."""
    digest = hashlib.sha256(f"{salt}:{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def augment_jev_internal(
    dataset,
    noul_rate=0.05,
    score_rate=0.0,
    permutation_rate=0.05,
    prompt_rate=0.05,
    paired_format_rate=0.05,
):
    """Add conservative low-frequency variants while retaining every direct row.

    The function is deterministic and idempotent. Score augmentation is off by
    default because it is valid only for explicitly ordered criteria.
    """
    if not len(dataset) or "variant" not in dataset.column_names:
        return dataset

    augmented = []
    existing_ids = set(dataset["id"])
    for row in dataset:
        if row["variant"] != "direct":
            continue
        options = row["options"]
        correct = max(range(len(row["target"])), key=row["target"].__getitem__)

        noul_id = row["id"] + ":noul-label-verification"
        if noul_id not in existing_ids and stable_fraction(row["id"], "noul") < noul_rate:
            propose_correct = stable_fraction(row["id"], "noul-answer") < 0.5
            if propose_correct or len(options) == 1:
                proposed = correct
            else:
                offset = 1 + int(
                    stable_fraction(row["id"], "noul-option") * (len(options) - 1)
                )
                proposed = (correct + offset) % len(options)
            augmented.append({
                **row,
                "id": noul_id,
                "kind": "noul",
                "options": [],
                "target": [float(proposed == correct)],
                "question": f'Is "{options[proposed]}" the correct label for this example?',
                "variant": "label_verification",
            })

        score_id = row["id"] + ":score-ordered-rubric"
        if (score_id not in existing_ids and len(options) >= 2
                and stable_fraction(row["id"], "score") < score_rate):
            augmented.append({
                **row,
                "id": score_id,
                "kind": "score",
                "question": (
                    "Score the example using the ordered rubric in options; "
                    "the target distribution identifies the correct rubric level."
                ),
                "variant": "ordered_rubric",
            })

        permutation_id = row["id"] + ":choice-criteria-permutation"
        if (permutation_id not in existing_ids and len(options) >= 2
                and stable_fraction(row["id"], "permutation") < permutation_rate):
            order = sorted(
                range(len(options)),
                key=lambda index: stable_fraction(row["id"], f"permutation-{index}"),
            )
            if order == list(range(len(options))):
                order = order[1:] + order[:1]
            augmented.append({
                **row,
                "id": permutation_id,
                "options": [options[index] for index in order],
                "target": [row["target"][index] for index in order],
                "variant": "criteria_permutation",
            })

        prompt_id = row["id"] + ":choice-instruction-paraphrase"
        if (prompt_id not in existing_ids
                and stable_fraction(row["id"], "prompt") < prompt_rate):
            variants = instruction_variants(row["question"], options)
            index = int(stable_fraction(row["id"], "prompt-variant") * len(variants))
            augmented.append({
                **row,
                "id": prompt_id,
                "question": variants[min(index, len(variants) - 1)],
                "variant": "instruction_paraphrase",
            })

        state_id = row["id"] + ":choice-paired-text-format"
        state_variants = paired_state_variants(row["state"])
        if (state_id not in existing_ids and state_variants
                and stable_fraction(row["id"], "paired-state") < paired_format_rate):
            index = int(
                stable_fraction(row["id"], "paired-state-variant")
                * len(state_variants)
            )
            augmented.append({
                **row,
                "id": state_id,
                "state": state_variants[min(index, len(state_variants) - 1)],
                "variant": "paired_text_format",
            })

    if not augmented:
        return dataset
    additions = Dataset.from_list(augmented, features=dataset.features)
    return concatenate_datasets([dataset, additions])
