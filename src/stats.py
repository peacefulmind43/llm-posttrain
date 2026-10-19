"""Confidence intervals for accuracy."""

from math import sqrt


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Unlike the normal approximation p ± z·sqrt(p(1-p)/n), it stays inside
    [0, 1] and does not collapse to zero width when p is 0 or 1.
    """
    if n == 0:
        return (0.0, 1.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
