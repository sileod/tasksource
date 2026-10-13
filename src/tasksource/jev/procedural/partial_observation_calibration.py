"""Exact posterior probability of a hidden binary state from independent noisy sensors."""

from fractions import Fraction

from ._common import Problem, sround

# Small-denominator fractions: enough distinct states at every level, still exact by hand.
FRACTIONS = sorted({Fraction(n, d) for d in range(2, 13) for n in range(1, d)})
PRIORS = [x for x in FRACTIONS if Fraction(1, 10) <= x <= Fraction(9, 10)]
TPRS = [x for x in FRACTIONS if Fraction(3, 5) <= x <= Fraction(19, 20)]
FPRS = [x for x in FRACTIONS if Fraction(1, 20) <= x <= Fraction(2, 5)]
N_SENSORS = [1, 2, 3, 4, 6]


def _text(x):
    return f"{x.numerator}/{x.denominator}"


def generate(rng, level=0):
    n_sensors = N_SENSORS[level]
    prior = rng.choice(PRIORS)
    yes, no = prior, 1 - prior
    sensors = []
    for i in range(max(1, n_sensors)):
        tpr, fpr = rng.choice(TPRS), rng.choice(FPRS)
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
