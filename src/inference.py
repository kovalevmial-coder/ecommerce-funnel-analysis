"""Confirmatory statistics for the e-commerce funnel analysis (Stage 04).

This module is the pure-statistics layer of the project: every function is
self-contained and dataframe-free (NumPy/SciPy only), so the confirmatory
notebook can reason about the maths without a data-processing dependency.
``chi2_independence`` still *accepts* a polars/pandas-style frame by
duck-typing its ``.to_numpy()`` method — no dataframe library is imported
here, keeping the module pure-stat. Each formula is pinned against
statsmodels/scipy in tests/test_inference.py with honest, per-assertion
relative tolerances.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
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


def chi2_independence(table: object) -> dict[str, float | int | bool]:
    """Chi-square test of independence on a contingency table.

    Wraps ``scipy.stats.chi2_contingency`` with its default arguments. For a
    2x2 table scipy applies Yates' continuity correction to the chi2
    statistic and p_value (``correction=True``); the expected counts that
    feed ``expected_min`` come from marginal totals alone and are identical
    under ``correction=True/False``, so ``expected_min`` is simply the
    minimum cell of the expected table returned by the same call.

    ``expected_min`` is Cochran's rule-of-thumb floor (every expected cell
    >= 5); ``valid = expected_min >= 5`` is a *guard flag* for the caller's
    routing decision (chi-square vs exact/pairwise), NOT an error signal —
    the chi2/p_value statistics are still reported when ``valid=False``.

    Input normalization (fail-fast):
      - a polars/pandas-style frame is accepted via duck-typed
        ``.to_numpy()`` (the confirmatory notebook builds its contingency
        from a polars groupby);
      - a ragged list-of-lists raises ``ValueError`` (NumPy >= 1.24 raises
        instead of silently building an object array);
      - any non-2-D result raises ``ValueError`` rather than being
        broadcast into a wrong-shaped table.

    Args:
        table: 2-D contingency table as list-of-lists, ndarray, or a frame
            exposing ``to_numpy()`` (polars DataFrame).

    Returns:
        Dict with ``chi2`` (float), ``dof`` (int), ``p_value`` (float),
        ``expected_min`` (float), ``valid`` (bool: ``expected_min >= 5``).

    Raises:
        ValueError: if the input cannot normalize to a 2-D numeric array
            (ragged rows, 1-D/3-D shape, non-array-like input).
    """
    # Duck-typed frame support: polars/pandas expose to_numpy(), ndarray does
    # not — so the pure-stat module never imports a dataframe library.
    if hasattr(table, "to_numpy"):
        table = table.to_numpy()
    try:
        arr = np.asarray(table)
    except ValueError as exc:
        # Ragged rows: NumPy raises rather than broadcasting into a shape.
        raise ValueError(
            f"contingency table must be rectangular: {exc}"
        ) from exc
    if arr.ndim != 2:
        # A 1-D (or 3-D) array is not a contingency table; fail fast.
        raise ValueError(
            f"contingency table must be 2-D, got ndim={arr.ndim}"
        )
    chi2, p_value, dof, expected = stats.chi2_contingency(arr)
    expected_min = float(expected.min())
    return {
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_value),
        "expected_min": expected_min,
        "valid": expected_min >= 5,
    }


def bonferroni(p_values: Sequence[float]) -> list[float]:
    """Bonferroni family-wise-error correction: ``min(p * k, 1)`` per value.

    ``k = len(p_values)`` counts every entry as one tested hypothesis,
    including NaN placeholders (a missing p-value still consumed a test in
    the multiple-comparison family). A NaN input yields a NaN output —
    the slot is preserved rather than crashing or being silently dropped —
    and an empty input yields an empty list. The ``1.0`` cap keeps the
    result inside the probability range when ``p * k > 1``.

    Args:
        p_values: raw p-values (NaN allowed as a missing-value marker).

    Returns:
        Corrected p-values as plain Python floats, same length/order as
        input; empty list for empty input.
    """
    k = len(p_values)
    corrected = []
    for p in p_values:
        p = float(p)
        # math.isnan rather than `p != p`: explicit and readable.
        corrected.append(math.nan if math.isnan(p) else min(p * k, 1.0))
    return corrected


def fdr_bh(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction (step-up); output = BH-adjusted p.

    For the p-values sorted ascending, the adjusted value at rank ``i`` is
    ``q_i = min(1, min_{j >= i} k * p_j / j)`` — a running minimum taken
    from the largest rank downward. That backward pass enforces
    monotonicity (a smaller original p can never receive a larger adjusted
    value), which is what collapses tied p-values onto one common q and
    what ``statsmodels.stats.multitest.multipletests(method="fdr_bh")``
    reproduces at rel=1e-6 (pinned in tests/test_inference.py).

    ``k = len(p_values)`` follows the same hypothesis-count convention as
    :func:`bonferroni`: NaN slots take no rank but still count toward the
    family size. A NaN input yields a NaN output at its original position
    without corrupting the corrected values of the remaining hypotheses;
    an empty input yields an empty list.

    Args:
        p_values: raw p-values (NaN allowed as a missing-value marker).

    Returns:
        BH-adjusted p-values as plain Python floats, same length/order as
        input; empty list for empty input.
    """
    values = [float(p) for p in p_values]
    if not values:
        return []
    k = len(values)
    # Rank only non-NaN values (ascending); NaN slots keep their positions.
    ranked = sorted(
        (i for i, p in enumerate(values) if not math.isnan(p)),
        key=values.__getitem__,
    )
    adjusted = [math.nan] * len(values)
    # Step-up from the largest rank: q = min(raw BH ratio, previous q, 1.0).
    running = 1.0
    for rank in range(len(ranked), 0, -1):
        idx = ranked[rank - 1]
        raw = values[idx] * k / rank
        running = min(raw, running, 1.0)
        adjusted[idx] = running
    return adjusted


# Sample-size / power helpers shared validation: every probability argument
# must be a valid open interval (0, 1) value, and the implied H1 rate p1 must
# stay inside (0, 1) too — otherwise the pooled/unpooled variances under the
# square roots can go negative and the closed form loses its finite meaning.
def _validate_two_prop_args(base_rate: float, mde: float, alpha: float) -> float:
    """Validate shared args and return the H1 proportion ``p1 = base_rate + mde``.

    Raises:
        ValueError: if ``base_rate``, ``alpha``, or the implied ``p1`` fall
            outside (0, 1), or ``mde == 0`` (a zero effect cannot be sized).
    """
    if not 0 < base_rate < 1:
        raise ValueError(f"base_rate must be in (0, 1), got base_rate={base_rate}")
    if mde == 0:
        raise ValueError(f"mde must be non-zero, got mde={mde}")
    p1 = base_rate + mde
    if not 0 < p1 < 1:
        # Guard for the positive-finite post-condition: with p1 outside (0,1)
        # the variance term p1(1-p1) is negative (or zero) and the closed form
        # produces NaN/negative/garbage instead of a meaningful sample size.
        raise ValueError(
            f"p1 = base_rate + mde must be in (0, 1), got p1={p1}"
        )
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got alpha={alpha}")
    return p1


def required_n_per_group(
    base_rate: float, mde: float, alpha: float, power: float
) -> float:
    """Sample size per group for a two-proportion test (closed form).

    Sizes the normal-approximation test of H0: p == base_rate against a
    two-sided alternative at level ``alpha``, designed to reach ``power`` for
    the effect p1 = base_rate + mde. The sign of ``mde`` picks the H1
    direction p1 = base_rate + mde; magnitude sizes n. Assumptions, all
    documented and pinned in tests/test_inference.py:
      - normal approximation (whole design is normal-theory);
      - balanced design, ratio 1:1 (n per group);
      - no continuity correction;
      - two-sided alpha (z_{1-alpha/2}) and one-sided power tail (z_power).

    Formula (pooled H0 variance, e.g. Fleiss/standard two-proportion setup in
    the normal-approximation family):
      n = [ z_{1-alpha/2} * sqrt(2*pbar*(1-pbar)) + z_power
            * sqrt(p0*(1-p0) + p1*(1-p1)) ]^2 / mde^2
    with pbar = (p0 + p1)/2 the pooled H0 rate under the balanced design.

    Cross-checked against ``statsmodels.stats.power.NormalIndPower`` (Cohen's
    h/arcsine effect size) at rel=0.01: the closed form sizes on the raw
    pooled proportion difference while statsmodels sizes on the arcsine
    family, so the two normal-approximations differ by a small structural
    factor on tiny effects — a formula-family difference, not a bug, absorbed
    by the documented 1% band.

    Args:
        base_rate: baseline conversion rate p0 in (0, 1).
        mde: minimum detectable effect size, p1 - p0, non-zero.
        alpha: type-I error probability, in (0, 1).
        power: type-II complement 1-beta, the target power, in (0, 1).

    Returns:
        n per group as a positive finite float (round up for a plan).

    Raises:
        ValueError: on invalid base_rate/mde/alpha/power or if the implied
            p1 leaves (0, 1).
    """
    p1 = _validate_two_prop_args(base_rate, mde, alpha)
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got power={power}")
    pbar = (base_rate + p1) / 2
    # z_{1-alpha/2}: two-sided rejection at level alpha; z_power: z_{1-beta}.
    z_crit = stats.norm.isf(alpha / 2)
    z_power = stats.norm.isf(1 - power)
    se_h0 = math.sqrt(2 * pbar * (1 - pbar))
    se_h1 = math.sqrt(base_rate * (1 - base_rate) + p1 * (1 - p1))
    n = ((z_crit * se_h0 + z_power * se_h1) / mde) ** 2
    if not math.isfinite(n):
        # Post-condition guard: a finite effect can only aim at a positive,
        # finite n; overflow (absurdly tiny mde) fails fast rather than
        # silently returning inf.
        raise ValueError(f"sample size not finite for mde={mde}")
    return float(n)


def achieved_power(base_rate: float, mde: float, n: int, alpha: float) -> float:
    """Power achieved for a given per-group sample size (inverse formula).

    Inverts :func:`required_n_per_group`: given ``n`` per group instead of a
    target power, return the power of the same normal-approximation,
    balanced, two-sided test of the effect p1 = base_rate + mde. From the
    sizing equation, n = ((z_crit*se_h0 + z_beta*se_h1)/mde)^2, solving for
    z_beta and passing it through the normal CDF — the "normal CDF
    difference" form:
      power = Phi( (|mde|*sqrt(n) - z_{1-alpha/2}*se_h0) / se_h1 )
    where se_h0 and se_h1 are the pooled-/unpooled standard errors of the
    difference. |mde| appears under the H1 shift because both symmetric
    tails of the two-sided test can reject; p1 = base_rate + mde (signed)
    still feeds the variance. Assumptions mirror required_n_per_group.

    Args:
        base_rate: baseline conversion rate p0 in (0, 1).
        mde: effect size p1 - p0, non-zero.
        n: subjects per group; must be an integer value >= 1 (integral
            floats/numpy ints accepted; bools are rejected so True cannot
            silently mean n=1).
        alpha: type-I error probability, in (0, 1).

    Returns:
        Achieved power as a float in [0, 1].

    Raises:
        ValueError: on invalid base_rate/mde/n/alpha or if p1 leaves (0, 1).
    """
    if isinstance(n, bool) or not isinstance(n, (int, float, np.integer)):
        raise ValueError(f"n must be an integer >= 1, got n={n!r}")
    n_val = int(n)
    if n_val != n or n_val < 1:
        raise ValueError(f"n must be an integer >= 1, got n={n}")
    p1 = _validate_two_prop_args(base_rate, mde, alpha)
    pbar = (base_rate + p1) / 2
    z_crit = stats.norm.isf(alpha / 2)
    se_h0 = math.sqrt(2 * pbar * (1 - pbar))
    se_h1 = math.sqrt(base_rate * (1 - base_rate) + p1 * (1 - p1))
    z_power = (abs(mde) * math.sqrt(n_val) - z_crit * se_h0) / se_h1
    return float(stats.norm.cdf(z_power))
