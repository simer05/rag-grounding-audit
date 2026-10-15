"""Confidence intervals and paired tests."""
import math

from scipy.stats import binomtest


def wilson(k, n, z=1.96):
    """Rate with a 95% Wilson score interval: (p, low, high)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a, b):
    """Exact McNemar test on paired booleans. Returns (a_only, b_only, p_value)."""
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only
    p = binomtest(a_only, n, 0.5).pvalue if n else 1.0
    return a_only, b_only, p
