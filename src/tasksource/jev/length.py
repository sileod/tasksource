"""Length budgets measured on complete rendered Jev requests."""

import json


def render_request(state, questions):
    """Render a multi-question System One request as the text to be measured.

    ``questions`` are training rows with ``question_id``, ``kind``,
    ``question``, and ``options``.
    """
    rendered = {}
    for row in questions:
        question = {"type": row["kind"], "instructions": row["question"]}
        if row["options"]:
            question["criteria"] = {option: None for option in row["options"]}
        rendered[row["question_id"]] = question
    return json.dumps({"state": state, "questions": rendered}, ensure_ascii=False)


class LengthBudget:
    """Whether a rendered request fits ``max_tokens``.

    With a tokenizer (a Hugging Face name or any object with ``encode``) the
    count is exact up to the fixed ``overhead`` reserved for chat/wire
    framing. Without one, the count is the UTF-8 byte length plus that
    overhead. This is deliberately conservative: byte-level BPE and
    byte-fallback SentencePiece tokens each cover at least one byte, so the
    byte count upper-bounds their token count (roughly 4x for English).
    """

    def __init__(self, max_tokens=4096, tokenizer=None, overhead=64):
        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.overhead = overhead

    def count(self, text):
        if self.tokenizer is None:
            return len(text.encode("utf-8")) + self.overhead
        return len(self.tokenizer.encode(text, add_special_tokens=False)) + self.overhead

    def fits(self, state, questions):
        return self.count(render_request(state, questions)) <= self.max_tokens
