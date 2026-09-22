"""Confirmatory statistics for the e-commerce funnel analysis (Stage 04).

This module is the pure-statistics layer of the project: every function is
self-contained and dataframe-free (NumPy/SciPy only), so the confirmatory
notebook can reason about the maths without a data-processing dependency.
Each formula is pinned against statsmodels/scipy in tests/test_inference.py
with honest, per-assertion relative tolerances.
"""
from __future__ import annotations

import math

from scipy import stats


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


def two_prop_ztest(
    a_events: int, a_n: int, b_events: int, b_n: int
) -> dict[str, float | bool | None]:
    """Pooled two-proportion z-test with a Wald difference confidence interval.

    Tests H0: p_a == p_b against H1: p_a != p_b (two-sided) by computing the
    z statistic under the pooled null rate ``(a_events + b_events) /
    (a_n + b_n)`` — the same model statsmodels ``proportions_ztest`` uses by
    default (prop_var=False). The ``ci_low``/``ci_high`` pair is the Wald
    interval of the observed difference ``p_a - p_b`` at the teaching z=1.96,
    built on the unpooled large-sample standard error
    ``sqrt(p_a(1-p_a)/a_n + p_b(1-p_b)/b_n)``.

    (controlling note) Plan file docs/superpowers/plans/04-inference.md and
    the task brief refer to a ``two_sided=True`` keyword for the statsmodels
    cross-check ``confint_proportions_2indep``; the actual statsmodels
    signature has no such parameter — it is two-sided by default and resolves
    alpha to z=1.95996... This module intentionally pins z=1.96 (teaching
    scale of the plan), so the Wald CI is cross-checked at rel=1e-3, not
    1e-6. This is a documented parametrisation diff, not a code bug.
    """
    if a_n <= 0 or b_n <= 0:
        raise ValueError(f"n must be > 0, got a_n={a_n}, b_n={b_n}")
    if not (0 <= a_events <= a_n and 0 <= b_events <= b_n):
        raise ValueError(
            f"0 <= events <= n required, got a_events={a_events}, a_n={a_n}, "
            f"b_events={b_events}, b_n={b_n}"
        )

    p_a = a_events / a_n
    p_b = b_events / b_n
    diff = p_a - p_b
    pooled = (a_events + b_events) / (a_n + b_n)

    # Sparse guard (rule: min expected count < 5): the normal approximation is
    # unreliable when too few events; report rates descriptively, no stats.
    min_expected = min(
        a_n * p_a, a_n * (1 - p_a), b_n * p_b, b_n * (1 - p_b)
    )
    sparse = min_expected < 5

    if sparse:
        return {
            "p_a": float(p_a),
            "p_b": float(p_b),
            "diff": float(diff),
            "z": None,
            "p_value": None,
            "ci_low": None,
            "ci_high": None,
            "pooled": float(pooled),
            "sparse": True,
            "n_ok": False,
        }

    # Pooled-variance z under H0: the null rate dominates both arms' sampling
    # error, so the standard error uses pooled, not the observed rates.
    se_pooled = math.sqrt(
        pooled * (1 - pooled) * (1 / a_n + 1 / b_n)
    )
    z = diff / se_pooled
    p_value = 2.0 * stats.norm.sf(abs(z))
    # Wald interval of the difference on the unpooled SE; z=1.96 is the plan's
    # teaching scale (see parametrisation note in the docstring).
    se_wald = math.sqrt(
        p_a * (1 - p_a) / a_n + p_b * (1 - p_b) / b_n
    )
    return {
        "p_a": float(p_a),
        "p_b": float(p_b),
        "diff": float(diff),
        "z": float(z),
        "p_value": float(p_value),
        "ci_low": float(diff - 1.96 * se_wald),
        "ci_high": float(diff + 1.96 * se_wald),
        "pooled": float(pooled),
        "sparse": False,
        "n_ok": True,
    }
