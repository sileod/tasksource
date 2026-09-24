"""Answer constructors shared by the procedural Jev generators."""

import json
import math
from dataclasses import dataclass


@dataclass
class Problem:
    state: dict | str  # str for states rendered as tables, CSV, or prose
    questions: dict
    answers: dict
    data: dict = None  # underlying records, for tests; never published


# Bounded vocabularies: choice answers must recur across splits, since each
# question is also published as a flat ClassLabel column built from train.
CITIES = ["Oslo", "Lima", "Accra", "Hanoi", "Quito", "Porto", "Dakar", "Perth", "Tunis", "Osaka",
          "Bergen", "Cusco", "Lagos", "Hue", "Cuenca", "Braga", "Thies", "Darwin", "Sfax", "Kobe"]
PEOPLE = ["Ada", "Bruno", "Chloe", "Dmitri", "Elif", "Farah", "Goran", "Hana", "Ines", "Jonas",
          "Kenji", "Lena", "Malik", "Nora", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tariq",
          "Uma", "Viktor", "Wen", "Ximena", "Yusuf", "Zoe", "Amir", "Bea", "Carlos", "Dara",
          "Emeka", "Freya", "Gil", "Hugo", "Ivy", "Jae", "Kofi", "Lucia", "Mei", "Nils"]
COLORS = ["red", "blue", "green", "black", "white", "gray", "orange", "purple", "yellow", "brown"]
OBJECTS = ["lamp", "chair", "kettle", "drill", "scarf", "clock", "vase", "rope", "tent", "mug"]
STYLES = ("json", "table", "csv", "lines")


def phrase(rng, templates, **fields):
    """One of several equivalent wordings, so questions are not a single template."""
    return rng.choice(templates).format(**fields)


def render_records(records, style):
    """A list of flat records as JSON, a Markdown table, CSV, or key=value lines."""
    columns = list(records[0])
    if style == "json":
        return json.dumps(records, ensure_ascii=False)
    if style == "table":
        rows = [" | ".join(str(r[c]) for c in columns) for r in records]
        return "\n".join([" | ".join(columns), " | ".join("---" for _ in columns), *rows])
    if style == "csv":
        return "\n".join([",".join(columns), *(",".join(str(r[c]) for c in columns) for r in records)])
    return "\n".join("; ".join(f"{c}={r[c]}" for c in columns) for r in records)


def sround(x, rng):
    """Stochastic rounding, so fractional difficulty steps change sizes on average."""
    low = math.floor(x)
    return low + (rng.random() < x - low)


def choice_answer(choice, options):
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": {option: float(option == choice) for option in options},
        "confidence": 1.0,
    }


def noul_answer(value):
    return {"type": "noul", "noul": float(value)}


def score_answer(index, criteria):
    return {
        "type": "score",
        "score": float(index),
        "legend": {str(i): value for i, value in enumerate(criteria)},
        "probabilities": {str(i): float(i == index) for i in range(len(criteria))},
        "confidence": 1.0,
    }
