"""Answer constructors shared by the procedural Jev generators."""

import math
from dataclasses import dataclass


@dataclass
class Problem:
    state: dict
    questions: dict
    answers: dict


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
