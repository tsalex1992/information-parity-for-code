"""Minimal statistical helpers for the main-paper aggregation pipeline."""

import math

import numpy as np
from scipy import stats

MIN_CORRELATION_N = 4


def fisher_pooled_summary(
    rs,
    ns,
    *,
    min_n: int = MIN_CORRELATION_N,
) -> tuple[float, float, int] | None:
    """Return a descriptive pooled Fisher-z summary across model-level effects."""
    valid_pairs = []
    for r, n in zip(rs, ns):
        try:
            r_val = float(r)
            n_val = int(n)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(r_val) or n_val < min_n:
            continue
        valid_pairs.append((float(np.clip(r_val, -0.999999, 0.999999)), n_val))

    if len(valid_pairs) < 2:
        return None

    zs = [np.arctanh(r) for r, _ in valid_pairs]
    weights = [max(n - 3, 1) for _, n in valid_pairs]
    total_weight = float(sum(weights))
    if total_weight <= 0:
        return None

    z_mean = float(np.average(zs, weights=weights))
    se = 1.0 / math.sqrt(total_weight)
    z_test = z_mean / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_test)))
    if not math.isfinite(p_val):
        return None
    return float(np.tanh(z_mean)), float(p_val), len(valid_pairs)
