import random

from tasksource.jev.synthetic import workflows as wf


def test_one_many_option_slot_and_no_rule_skills():
    rng = random.Random(0)
    for _ in range(50):
        assert sum(s["range"] == wf.MANY_RANGE for s in wf.sample_slots(rng, 6, many=True)) == 1
    assert not set(wf.SKILLS) & wf.RULE_SKILLS


def test_valid_workflow_checks_types_counts_and_skills():
    slots = [{"type": "noul", "range": wf.CHOICE_RANGE}, {"type": "choice", "range": wf.MANY_RANGE}]
    skills = ["sentiment", "incident_routing"]
    good = {"questions": [{"skill": "sentiment", "type": "noul", "question": "Is the writer upset?", "options": []},
                          {"skill": "incident_routing", "type": "choice", "question": "Which queue?",
                           "options": [f"queue {i}" for i in range(15)]}]}
    assert wf.valid_workflow(good, slots, skills)
    bad = {"questions": [good["questions"][0], {**good["questions"][1], "options": ["a", "b"]}]}
    assert not wf.valid_workflow(bad, slots, skills)
    assert not wf.valid_workflow({"questions": [good["questions"][0], {**good["questions"][1], "skill": "urgency"}]}, slots, skills)


def test_fits_tolerates_neighbouring_score_levels_only():
    clear = {"kind": "clear", "index": 2, "ordinal": True}
    assert wf.fits(clear, [0, 0.1, 0.3, 0.6, 0])
    assert not wf.fits(clear, [0, 0, 0.1, 0.1, 0.8])
    choice = {"kind": "clear", "index": 0, "ordinal": False}
    assert wf.fits(choice, [0.7, 0.3]) and not wf.fits(choice, [0.3, 0.7])
    assert wf.fits({"p_yes": 0.8}, [0.6]) and not wf.fits({"p_yes": 0.8}, [0.2])


def test_confident_disagreement():
    assert wf.confident_disagreement("choice", [0.9, 0.1], [0.1, 0.9])
    assert not wf.confident_disagreement("choice", [0.6, 0.4], [0.1, 0.9])
    assert not wf.confident_disagreement("score", [0.9, 0.1, 0], [0.1, 0.9, 0])
    assert wf.confident_disagreement("noul", [0.95], [0.05])


def test_targets_are_reproducible():
    question = {"type": "choice", "options": ["a", "b", "c"]}
    assert wf.sample_target(wf.stable_rng(1, "x"), question) == wf.sample_target(wf.stable_rng(1, "x"), question)
