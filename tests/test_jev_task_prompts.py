import json

import pytest

from tasksource.jev.task_prompts import describe_options, state_fields, task_style

PROMPTS = {
    "pairs": {"fields": {"text_A": "Premise", "text_B": "Hypothesis"}, "text_name": None,
              "styles": {"question": ["Does the premise entail the hypothesis?"],
                         "directive": ["Classify the relation of the hypothesis to the premise."]}},
    "single": {"fields": {}, "text_name": "Review", "styles": {"terse": ["Sentiment of the review:"]},
               "labels": {"positive": "The reviewer liked it.", "negative": "The reviewer disliked it."}},
}
PAIR = "text_A: A man sleeps.\nIt is night.\ntext_B: A man is awake."


def test_state_fields_keeps_multiline_values():
    assert state_fields(PAIR, ["text_A", "text_B"]) == [("text_A", "A man sleeps.\nIt is night."), ("text_B", "A man is awake.")]
    assert state_fields("free text", ["text_A"]) is None


def test_inline_moves_content_after_instruction():
    state, question = task_style(PAIR, "pairs", 0.0, styles=["question"], prompts=PROMPTS)
    assert state == ""
    assert question == "Does the premise entail the hypothesis?\nPremise: A man sleeps.\nIt is night.\nHypothesis: A man is awake."


def test_fields_layout_and_choice_over_selected_styles():
    state, question = task_style(PAIR, "pairs", 0.99, layout="fields", prompts=PROMPTS)
    assert json.loads(state) == {"Premise": "A man sleeps.\nIt is night.", "Hypothesis": "A man is awake."}
    assert question == "Does the premise entail the hypothesis?"  # STYLES order: directive, question, ...
    assert task_style(PAIR, "pairs", 0.0, layout="fields", prompts=PROMPTS)[1] == "Classify the relation of the hypothesis to the premise."


def test_unnamed_text():
    assert task_style("Great food.", "single", 0.5, prompts=PROMPTS) == ("", "Sentiment of the review:\nGreat food.")
    assert task_style("Great food.", "single", 0.5, layout="fields", prompts=PROMPTS) == ('{"Review": "Great food."}', "Sentiment of the review:")


def test_missing_style_source_or_unmatched_state():
    assert task_style("Great food.", "single", 0.0, styles=["directive"], prompts=PROMPTS) is None
    assert task_style(PAIR, "missing", 0.0, prompts=PROMPTS) is None
    assert task_style("no fields here", "pairs", 0.0, prompts=PROMPTS) is None
    with pytest.raises(ValueError):
        task_style(PAIR, "pairs", 0.0, layout="json", prompts=PROMPTS)


def test_describe_options_needs_every_label():
    assert describe_options(["negative", "positive"], "single", prompts=PROMPTS) == ["negative: The reviewer disliked it.", "positive: The reviewer liked it."]
    assert describe_options(["negative", "mixed"], "single", prompts=PROMPTS) == ["negative", "mixed"]
    assert describe_options(["yes"], "pairs", prompts=PROMPTS) == ["yes"]
