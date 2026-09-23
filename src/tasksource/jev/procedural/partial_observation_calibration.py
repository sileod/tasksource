"""Exact posterior probability of a hidden binary state from independent noisy sensors."""

from fractions import Fraction

from ._common import Problem, sround

RATES = [
    (Fraction(4, 5), Fraction(1, 5)),
    (Fraction(3, 4), Fraction(1, 4)),
    (Fraction(2, 3), Fraction(1, 3)),
    (Fraction(9, 10), Fraction(2, 5)),
]
PRIORS = [Fraction(1, 5), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(4, 5)]


def _text(x):
    return f"{x.numerator}/{x.denominator}"


def generate(rng, level=0):
    n_sensors = sround(2 + 0.55 * level, rng)
    prior = rng.choice(PRIORS)
    yes, no = prior, 1 - prior
    sensors = []
    for i in range(max(1, n_sensors)):
        tpr, fpr = rng.choice(RATES)
        positive = rng.choice([True, False])
        sensors.append({
            "id": f"S{i+1}",
            "observed": "positive" if positive else "negative",
            "p_positive_given_incident": _text(tpr),
            "p_positive_given_no_incident": _text(fpr),
        })
        yes *= tpr if positive else 1 - tpr
        no *= fpr if positive else 1 - fpr
    posterior = yes / (yes + no)
    state = {
        "prior_probability_incident": _text(prior),
        "assumption": "Sensor observations are conditionally independent given whether the incident is real.",
        "sensors": sensors,
    }
    questions = {
        "incident_real": {
            "type": "noul",
            "instructions": "What is the posterior probability that the incident is real after conditioning on every sensor observation?",
        }
    }
    answers = {"incident_real": {"type": "noul", "noul": round(float(posterior), 6)}}
    return Problem(state, questions, answers)
