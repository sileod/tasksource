"""Hand-authored, meaning-preserving instruction variants for Jev decisions.

Keep this module deliberately small. Variants may change only the wording of
the decision request; they must never add facts, uncertainty, or label meaning.
"""

CLASSIFICATION_INSTRUCTION = "Choose the criterion that best describes the state."
MULTIPLE_CHOICE_INSTRUCTION = "Choose the criterion that best answers the question."
TOKEN_INSTRUCTION = "Choose the criterion that best labels the target token."

GENERIC_CLASSIFICATION = (
    "Select the label that best applies to the state.",
    "Choose the most appropriate category for the state.",
    "Which of the supplied criteria best matches the state?",
)

GENERIC_MULTIPLE_CHOICE = (
    "Select the option that best answers the question.",
    "Choose the most appropriate answer from the supplied options.",
    "Which supplied option best answers the question?",
)

# Label-group-specific wording is used only when the complete normalized label
# set matches. These are manually reviewed semantic equivalences.
LABEL_GROUP_VARIANTS = {
    frozenset({"entailment", "neutral", "contradiction"}): (
        "Classify the relationship between the first and second texts.",
        "Choose the natural-language inference relation that best applies.",
    ),
    frozenset({"entailment", "contradiction"}): (
        "Decide whether the first text entails or contradicts the second.",
    ),
    frozenset({"negative", "neutral", "positive"}): (
        "Classify the sentiment expressed in the state.",
        "Choose the sentiment label that best applies.",
    ),
    frozenset({"negative", "positive"}): (
        "Classify the state as negative or positive.",
    ),
}

# Checkpointed shards may contain the earlier field-specific wording. These
# exact rewrites let the public view vary field labels without a mismatch.
PAIR_QUESTION_REWRITES = {
    "Classify the relationship between text_A and text_B.":
        "Classify the relationship between the first and second texts.",
    "Decide whether text_A entails or contradicts text_B.":
        "Decide whether the first text entails or contradicts the second.",
}


def instruction_variants(instruction, options):
    """Return vetted alternatives, most specific first, without duplicates."""
    normalized = frozenset(str(option).strip().casefold() for option in options)
    specific = LABEL_GROUP_VARIANTS.get(normalized, ())
    if instruction == MULTIPLE_CHOICE_INSTRUCTION:
        generic = GENERIC_MULTIPLE_CHOICE
    else:
        generic = GENERIC_CLASSIFICATION
    return tuple(dict.fromkeys((*specific, *generic)))


def paired_state_variants(state):
    """Return neutral formatting variants for canonical paired-text states."""
    if not state.startswith("text_A: ") or "\ntext_B: " not in state:
        return ()
    text_a, text_b = state[len("text_A: "):].split("\ntext_B: ", 1)
    return (
        f"First text:\n{text_a}\n\nSecond text:\n{text_b}",
        f"Passage A:\n{text_a}\n\nPassage B:\n{text_b}",
        f"A: {text_a}\nB: {text_b}",
    )


def published_pair_style(state, question, fraction):
    """Choose one vetted paired-field format for the public training view."""
    question = PAIR_QUESTION_REWRITES.get(question, question)
    variants = paired_state_variants(state)
    if not variants or "text_A" in question or "text_B" in question:
        return state, question
    formats = (state, *variants)
    return formats[min(int(fraction * len(formats)), len(formats) - 1)], question


def published_question_style(question, options, state, fraction):
    """Use reviewed equivalent requests in the public view without extra rows."""
    if question not in (CLASSIFICATION_INSTRUCTION, MULTIPLE_CHOICE_INSTRUCTION):
        return question
    candidates = instruction_variants(question, options)
    labels = frozenset(str(option).strip().casefold() for option in options)
    paired = state.startswith(("text_A: ", "First text:\n", "Passage A:\n", "A: "))
    if question == CLASSIFICATION_INSTRUCTION and not paired and labels in {
        frozenset({"entailment", "neutral", "contradiction"}),
        frozenset({"entailment", "contradiction"}),
    }:
        candidates = GENERIC_CLASSIFICATION
    styles = (question, *candidates)
    return styles[min(int(fraction * len(styles)), len(styles) - 1)]
