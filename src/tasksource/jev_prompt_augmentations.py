"""Hand-authored, meaning-preserving instruction variants for Jev recasts.

Keep this module deliberately small. Variants may change only the wording of
the decision request; they must never add facts, uncertainty, or label meaning.
"""

CLASSIFICATION_INSTRUCTION = "Choose the criterion that best describes the state."
MULTIPLE_CHOICE_INSTRUCTION = "Choose the criterion that best answers the question."

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
        "Classify the relationship between text_A and text_B.",
        "Choose the natural-language inference relation that best applies.",
    ),
    frozenset({"entailment", "contradiction"}): (
        "Decide whether text_A entails or contradicts text_B.",
    ),
    frozenset({"negative", "neutral", "positive"}): (
        "Classify the sentiment expressed in the state.",
        "Choose the sentiment label that best applies.",
    ),
    frozenset({"negative", "positive"}): (
        "Classify the state as negative or positive.",
    ),
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
