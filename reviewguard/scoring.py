import numpy as np
from .config import Config

NAMES = {
    "D": "Semantic duplication",
    "C": "Deception classifier",
    "L": "Language patterns",
    "R": "Rating mismatch",
    "E": "Extreme sentiment",
}


def score_signals(
    signals: dict, config: Config | None = None
) -> tuple[float, str, dict]:
    """Renormalize available weights; return continuous score and four bands."""
    config = config or Config()
    active = {k: v for k, v in signals.items() if v is not None}
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in active.values()):
        raise ValueError("Signals must be finite values in [0, 1].")
    total = sum(config.weights[k] for k in active)
    if total <= 0:
        raise ValueError("At least one available signal needs positive weight.")
    contributions = {
        k: (100 * config.weights[k] * signals[k] / total if k in active else 0.0)
        for k in config.weights
    }
    score = float(np.clip(sum(contributions.values()), 0, 100))
    band = (
        "Low"
        if score <= 30
        else "Moderate" if score <= 60 else "High" if score <= 80 else "Very High"
    )
    return score, band, contributions


def rating_mismatch(compound: float, rating: float | None) -> float | None:
    if rating is None or not np.isfinite(rating):
        return None
    return float(abs(compound - (rating - 3) / 2) / 2)


def duplication_signal(semantic, lexical, count=1):
    evidence = max(semantic, lexical)
    return (
        float(np.clip(0.9 * (evidence - 0.60) / 0.35 + 0.1 * min(count / 3, 1), 0, 1))
        if evidence > 0.60
        else 0.0
    )
