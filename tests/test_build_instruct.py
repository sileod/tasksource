from datasets import Dataset

from scripts.build_instruct_dataset import interleave, preference_pairs


def rows():
    return Dataset.from_dict({
        "inputs": ["a", "b", "c", "d"], "targets": ["yes.", "B.", "O", "no."],
        "options": [["yes.", "no."], ["A.", "B.", "C."], [], ["no."]], "task": ["t1", "t1", "tok", "t2"],
    })


def test_interleave_round_robins_tasks():
    assert interleave(rows())["task"] == ["t1", "tok", "t2", "t1"]


def test_pairs_reject_another_offered_answer():
    pairs = preference_pairs(rows())
    assert pairs["prompt"] == ["a", "b"]  # no options, or no other option: no pair
    assert pairs["rejected"][0] == "no." and pairs["rejected"][1] in {"A.", "C."}
    assert preference_pairs(rows())["rejected"] == pairs["rejected"]
