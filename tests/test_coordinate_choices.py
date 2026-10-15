"""Coordinate options: gold plus native distractors, as 0–1000 integers."""
from tasksource.grounding import coordinate_choices, coordinate_text


def test_coordinate_text():
    assert coordinate_text([0.1, 0.2, 0.30049, 1.0]) == '[100, 200, 300, 1000]'
    assert coordinate_text((0.5, 0.0004)) == '(500, 0)'


def test_choices_keep_gold_and_drop_near_or_identical_distractors():
    gold = [0.1, 0.1, 0.2, 0.2]
    others = [[0.6, 0.6, 0.7, 0.7], [0.1001, 0.1, 0.2, 0.2], [0.11, 0.1, 0.21, 0.2], [0.3, 0.3, 0.4, 0.4]]
    choices, label = coordinate_choices(gold, others, key='row-1')
    assert choices[label] == '[100, 100, 200, 200]'
    assert sorted(choices) == ['[100, 100, 200, 200]', '[300, 300, 400, 400]', '[600, 600, 700, 700]']
    assert coordinate_choices(gold, others, key='row-1') == (choices, label)  # stable per key


def test_choices_cap_and_no_distractor():
    points = [(i / 20, 0.5) for i in range(20)]
    choices, label = coordinate_choices((0.99, 0.99), points, key='k', max_options=8)
    assert len(choices) == 8 and choices[label] == '(990, 990)'
    assert coordinate_choices((0.5, 0.5), [(0.51, 0.5)], key='k') is None
