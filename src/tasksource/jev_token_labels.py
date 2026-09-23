"""Readable criterion names for conservative Jev token-task adapters."""

import re


MAX_TOKEN_CRITERIA = 32
MAX_JEV_TOKENS_PER_SEQUENCE = 2


TOKEN_LABEL_NAMES = {
    # Universal POS tags.
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary verb",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
    # Common entity types.
    "PER": "person",
    "PERSON": "person",
    "ORG": "organization",
    "LOC": "location",
    "GPE": "geopolitical entity",
    "MISC": "miscellaneous",
    "DATE": "date",
    "TIME": "time",
    "MONEY": "money",
    "PERCENT": "percentage",
    "DISEASE": "disease",
    # Universal Dependencies relations commonly found in Tasksource.
    "acl": "clausal modifier of a noun",
    "advcl": "adverbial clause modifier",
    "advmod": "adverbial modifier",
    "amod": "adjectival modifier",
    "appos": "appositional modifier",
    "aux": "auxiliary",
    "case": "case marking",
    "cc": "coordinating conjunction",
    "ccomp": "clausal complement",
    "clf": "classifier",
    "compound": "compound",
    "conj": "conjunct",
    "cop": "copula",
    "csubj": "clausal subject",
    "dep": "unspecified dependency",
    "det": "determiner",
    "discourse": "discourse element",
    "dislocated": "dislocated element",
    "expl": "expletive",
    "fixed": "fixed expression",
    "flat": "flat multiword expression",
    "goeswith": "split word",
    "iobj": "indirect object",
    "list": "list item",
    "mark": "subordinating marker",
    "nmod": "nominal modifier",
    "nsubj": "nominal subject",
    "nummod": "numeric modifier",
    "obj": "object",
    "obl": "oblique nominal",
    "orphan": "orphan",
    "parataxis": "parataxis",
    "punct": "punctuation",
    "reparandum": "overridden disfluency",
    "root": "sentence root",
    "vocative": "vocative",
    "xcomp": "open clausal complement",
}

_BIO_PREFIXES = {
    "B": "beginning of",
    "I": "inside",
    "E": "end of",
    "L": "last token of",
    "S": "single-token",
    "U": "single-token",
}
_BAD_LABELS = {"", "_", "unknown", "unk"}


def readable_token_labels(names):
    """Return whether an ontology is semantic enough to expose at runtime."""
    names = [str(name).strip() for name in names]
    lowered = [name.lower() for name in names]
    return (
        2 <= len(names) <= MAX_TOKEN_CRITERIA
        and len(set(names)) == len(names)
        and not all(name.isdigit() for name in names)
        and not any(re.fullmatch(r"(?:label|class)[_-]?\d+", name) for name in lowered)
        and sum(name not in _BAD_LABELS for name in lowered) / len(names) > 0.8
    )


def _readable_entity_type(value):
    pieces = value.split("-")
    expanded = [
        TOKEN_LABEL_NAMES.get(piece, TOKEN_LABEL_NAMES.get(piece.upper(), piece.replace("_", " ").lower()))
        for piece in pieces
    ]
    separator = " / " if pieces[0].lower() in {
        "person", "organization", "location", "product", "building", "event",
    } and len(pieces) > 1 else " "
    return separator.join(expanded)


def normalize_token_label(name):
    """Expand a compact token label without discarding BIO/BILOU semantics."""
    name = str(name).strip()
    if name == "O":
        return "outside any named entity"
    if name in TOKEN_LABEL_NAMES:
        return TOKEN_LABEL_NAMES[name]

    match = re.fullmatch(r"([BIESUL])-(.+)", name, flags=re.IGNORECASE)
    if match:
        prefix, entity_type = match.groups()
        entity = _readable_entity_type(entity_type)
        # Chunk tags are phrases, while NER tags denote entities.
        suffix = "phrase" if entity_type.upper() in {"NP", "VP", "PP", "ADJP", "ADVP", "SBAR"} else "entity"
        phrase_names = {
            "NP": "noun", "VP": "verb", "PP": "prepositional",
            "ADJP": "adjective", "ADVP": "adverb", "SBAR": "subordinate clause",
        }
        if suffix == "phrase":
            entity = phrase_names.get(entity_type.upper(), entity)
        if prefix.upper() in {"S", "U"}:
            return f"a single-token {entity} {suffix}"
        article = "an" if entity[0].lower() in "aeiou" else "a"
        return f"{_BIO_PREFIXES[prefix.upper()]} {article} {entity} {suffix}"

    # UD subtypes retain their useful qualifier.
    if ":" in name:
        base, subtype = name.split(":", 1)
        if base in TOKEN_LABEL_NAMES:
            return f"{TOKEN_LABEL_NAMES[base]} ({subtype.replace('_', ' ')})"
    return TOKEN_LABEL_NAMES.get(name.upper(), name.replace("_", " "))
