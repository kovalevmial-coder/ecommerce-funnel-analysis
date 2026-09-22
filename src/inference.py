"""Confirmatory statistics for the e-commerce funnel analysis (Stage 04).

This module is the pure-statistics layer of the project: every function is
self-contained and dataframe-free (NumPy/SciPy only), so the confirmatory
notebook can reason about the maths without a data-processing dependency.
Each formula is pinned against statsmodels/scipy in tests/test_inference.py
with honest, per-assertion relative tolerances.
"""
from __future__ import annotations


def wilson_ci(success: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson (score) confidence interval for a binomial proportion.

    Returns ``(ci_low, ci_high)`` for the observed proportion ``success / n``
    at standard-normal critical value ``z`` (1.96 ~ two-sided 95%). The score
    method is preferred over Wald: Wald's normal approximation degenerates to
    a point exactly when ``success == n``, can invert outside (0, 1) for
    proportions near the edges, and is badly sized for small ``n``. Wilson
    keeps coverage near nominal for small and edge proportions, and by
    construction confines the interval to (0, 1), touching a boundary only
    when the data itself is at one (``success == 0`` / ``success == n``) —
    that is honest, not a special case.

    Args:
        success: number of successful trials; must satisfy 0 <= success <= n.
        n: total trials; must be positive.
        z: standard-normal critical value (1.96 for a two-sided 95% CI).

    Returns:
        ``(ci_low, ci_high)`` floats with ``ci_low <= success / n <= ci_high``
        and both endpoints within [0, 1] by construction.

    Raises:
        ValueError: if ``n <= 0``, or ``success`` falls outside ``[0, n]``.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got n={n}")
    if not 0 <= success <= n:
        raise ValueError(
            f"0 <= success <= n required, got success={success}, n={n}"
        )
    p = success / n
    # Wilson's closed form: re-centre the observed proportion by z^2/(2n) and
    # widen the radius by the z^2/(4n^2) variance term, normalised by
    # (1 + z^2/n). Written as a ratio of non-negative terms the boundary
    # behaviour is transparent; the min/max clamp only ever activates at the
    # data boundaries themselves (success == 0 / success == n).
    centre = p + (z**2) / (2 * n)
    radius = z * ((p * (1 - p) / n) + (z**2) / (4 * n**2)) ** 0.5
    normalise = 1 + (z**2) / n
    ci_low = max(0.0, (centre - radius) / normalise)
    ci_high = min(1.0, (centre + radius) / normalise)
    return ci_low, ci_high