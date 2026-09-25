"""Default Jev questions for common label sets.

Most Tasksource annotations have no ``question``, so their Jev rows fell back
to a generic "choose the criterion" instruction, unlike real Jev requests.
A label set that says what is being judged gets a default question here; an
annotation's own ``question`` always wins, and other tasks stay generic.
"""

NLI3 = "Does text_A entail text_B, contradict it, or neither?"
NLI2 = "Does text_A entail text_B?"
SENTIMENT = "What sentiment does the text express?"

# (sorted lowercased label names) -> (question for one text, question for text_A/text_B pairs)
DEFAULT_QUESTIONS = {
    ("contradiction", "entailment", "neutral"): (None, NLI3),
    ("entailed", "not-entailed"): (None, NLI2),
    ("entailment", "not_entailment"): (None, NLI2),
    ("entailment", "not-entailment"): (None, NLI2),
    ("entailment", "non-entailment"): (None, NLI2),
    ("entailment", "neutral"): (None, NLI2),
    ("not_paraphrase", "paraphrase"): (None, "Is text_B a paraphrase of text_A?"),
    ("different", "same"): (None, "Do text_A and text_B mean the same thing?"),
    ("different meaning", "same meaning"): (None, "Do text_A and text_B mean the same thing?"),
    ("negative", "neutral", "positive"): (SENTIMENT, None),
    ("negative", "positive"): (SENTIMENT, None),
    ("neg", "pos"): (SENTIMENT, None),
    ("1 star", "2 stars", "3 stars", "4 stars", "5 stars"): ("How many stars does the review give?", None),
    ("not toxic", "toxic"): ("Is the text toxic?", None),
    ("not offensive", "offensive"): ("Is the text offensive?", None),
}


def default_question(criteria, paired):
    """The default question for these criteria, or ``None``."""
    single, pair = DEFAULT_QUESTIONS.get(tuple(sorted(str(c).strip().lower() for c in criteria)), (None, None))
    return pair if paired else single
