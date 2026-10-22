"""Probability checks shared by decision-model validation tools."""

import math


def distribution(values, expected=None) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values or (expected is None and len(values) < 2):
        raise ValueError("expected a nonempty option probability distribution")
    if expected is not None and len(values) != expected:
        raise ValueError(f"expected {expected} probabilities, got {len(values)}")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("probabilities must be finite numbers in [0, 1]")
    if not math.isclose(sum(values), 1., abs_tol=1e-5):
        raise ValueError("probabilities must sum to one")
    return list(values)
