"""Tests for the confirmatory statistics in src/inference.py.

Wilson (score) confidence interval pins three distinct things:
1. The teaching default z=1.96 produces the true Wilson endpoints on boundary
   and interior points (rel=1e-6 against hand-computed constants).
2. The formula matches statsmodels ``proportion_confint(method="wilson")`` at
   rel=1e-6 when both use the same z — this pins the algebra with no
   z-rounding confound (statsmodels' two-sided alpha=0.05 resolves to
   z=1.95996..., not the Stage-04 teaching constant 1.96).
3. The guards raise ValueError on impossible inputs.

Chi-square + multiple-comparison corrections pin:
4. chi2_independence reproduces scipy.stats.chi2_contingency exactly
   (identity, including the 2x2 Yates correction defaults) and exposes the
   expected-table minimum behind the sparse guard (valid = expected_min >= 5).
5. The polars DataFrame input path yields the same numbers as list/ndarray.
6. bonferroni/fdr_bh match hand values and statsmodels multipletests at
   rel=1e-6 (fdr_bh including tied and extreme p-values), and are NaN-safe:
   a NaN p-value propagates to a NaN output position without crashing.

Plan-artifact correction (documented per controller ruling): the published
known values in docs/superpowers/plans/04-inference.md Task 1 Step 1 are NOT
Wilson endpoints. (success=0, n=100) -> 0.0362 and (100,100) -> 0.9638 are
Clopper-Pearson endpoints (statsmodels ``proportion_confint(method="beta")``
returns 0.0362167 / 0.9637833), and (50,100) -> (0.4038, 0.5962) is a 4-decimal
rounding of the true endpoints. All three published anchors fail rel=1e-6. The
constants below are the true Wilson endpoints at z=1.96, verified against
statsmodels with the identical z (diff 0.0 for every endpoint).

Brief-example correction (task-3): the task brief's sparse example
[[10,0],[0,10]] actually has expected_min = 5.0 (row/col totals 10/10/20 give
expected cells of exactly 5), so it is the boundary for valid, not a sparse
counterexample. The genuinely sparse test uses [[20,0],[0,2]]
(expected_min ~ 0.18); [[10,0],[0,10]] pins the valid=True boundary instead.

Sample size / power (task-4) pins:
7. required_n_per_group matches statsmodels NormalIndPower().solve_power on a
   Cohen's-h effect size at rel=0.01 — a deliberately loose, documented
   tolerance: our closed form sizes on the raw pooled proportion difference
   while statsmodels sizes on the arcsine (Cohen's h) family, so the two
   normal-approximations differ by a small structural factor on tiny effects.
   Sanity: larger mde -> smaller n; n positive finite.
8. achieved_power is the inverse of required_n_per_group: at
   n = round(required_n(...)) it recovers the target power within a loose
   band (rel=5%, absorbing the integer rounding of n).
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportion_effectsize,
    proportions_ztest,
)

from src.inference import (
    achieved_power,
    bonferroni,
    chi2_independence,
    fdr_bh,
    required_n_per_group,
    two_prop_ztest,
    wilson_ci,
)

# True Wilson endpoints at the pinned teaching default z=1.96, computed from
# the closed form and cross-checked against statsmodels with the same z.
_KNOWN_AT_Z196 = {
    (0, 100): (0.0, 0.036994807476001905),
    (100, 100): (0.963005192523998, 0.9999999999999998),
    (50, 100): (0.4038298285901472, 0.5961701714098528),
}


def test_wilson_ci_known_values_at_default_z() -> None:
    """The default z=1.96 must reproduce the true Wilson endpoints."""
    for (success, n), (want_low, want_high) in _KNOWN_AT_Z196.items():
        low, high = wilson_ci(success, n)
        assert low == pytest.approx(want_low, rel=1e-6)
        assert high == pytest.approx(want_high, rel=1e-6)


def test_wilson_ci_matches_statsmodels_at_same_z() -> None:
    """The Wilson algebra must match statsmodels exactly at edge AND interior points.

    statsmodels' default alpha=0.05 maps to z=1.95996..., not the 1.96 teaching
    default, so the *same* z constant is passed here to remove the only source
    of endpoint difference; the formula itself is then pinned at rel=1e-6. The
    interior-tolerance note from ``test_funnel_rates`` about z-rounding does
    not apply: a regression in the formula can no longer hide behind a z
    mismatch.
    """
    z = float(stats.norm.isf(0.025))
    for success, n in [(0, 100), (1, 100), (50, 100), (100, 200), (200, 200)]:
        got_low, got_high = wilson_ci(success, n, z=z)
        ref_low, ref_high = proportion_confint(success, n, method="wilson")
        assert got_low == pytest.approx(float(ref_low), rel=1e-6)
        assert got_high == pytest.approx(float(ref_high), rel=1e-6)


def test_wilson_ci_raises_on_non_positive_n() -> None:
    """n=0 or negative n is a caller bug; fail fast and loud."""
    with pytest.raises(ValueError, match="n"):
        wilson_ci(0, 0)
    with pytest.raises(ValueError, match="n"):
        wilson_ci(5, -1)


def test_wilson_ci_raises_when_success_outside_range() -> None:
    """success outside [0, n] would produce a meaningless proportion."""
    with pytest.raises(ValueError, match="success"):
        wilson_ci(-1, 100)
    with pytest.raises(ValueError, match="success"):
        wilson_ci(101, 100)


def test_two_prop_ztest_normal_path_matches_statsmodels() -> None:
    """The pooled z-statistic and two-sided p-value must match statsmodels exactly.

    statsmodels ``proportions_ztest`` uses pooled H0 variance by default
    (prop_var=False), which is the same model as ``two_prop_ztest``, so the
    comparison is apples-to-apples at rel=1e-6. The Wald difference CI is
    cross-checked at the looser rel=1e-3: statsmodels resolves alpha=0.05 to
    z=1.95996... via ``norm.isf``, while this module pins the teaching
    constant z=1.96 — a documented parametrisation difference, not a code bug.
    """
    a_events, a_n, b_events, b_n = 50, 1000, 40, 1000
    res = two_prop_ztest(a_events, a_n, b_events, b_n)
    # Point estimates and the pooled H0 rate are deterministic fractions.
    assert res["p_a"] == pytest.approx(50 / 1000, rel=1e-9)
    assert res["p_b"] == pytest.approx(40 / 1000, rel=1e-9)
    assert res["diff"] == pytest.approx(0.01, rel=1e-9)
    assert res["pooled"] == pytest.approx((50 + 40) / 2000, rel=1e-9)

    ref_z, ref_p = proportions_ztest(
        [a_events, b_events], [a_n, b_n], alternative="two-sided"
    )
    assert res["z"] == pytest.approx(float(ref_z), rel=1e-6)
    assert res["p_value"] == pytest.approx(float(ref_p), rel=1e-6)

    ref_low, ref_high = confint_proportions_2indep(
        a_events, a_n, b_events, b_n, method="wald", compare="diff"
    )
    assert res["ci_low"] == pytest.approx(float(ref_low), rel=1e-3)
    assert res["ci_high"] == pytest.approx(float(ref_high), rel=1e-3)

    assert res["sparse"] is False
    assert res["n_ok"] is True
    # The result dict must carry plain floats, not numpy scalars.
    # np.float64 subclasses float, so isinstance would pass a leaked numpy
    # scalar; the contract promises plain Python floats (strict type check).
    assert type(res["z"]) is float


def test_two_prop_ztest_normal_path_has_non_none_stats() -> None:
    """Normal path: z/p/CI are computed (sparse guard must not misfire)."""
    res = two_prop_ztest(50, 1000, 40, 1000)
    assert res["z"] is not None
    assert res["p_value"] is not None
    assert res["ci_low"] is not None
    assert res["ci_high"] is not None


def test_two_prop_ztest_sparse_returns_description_only() -> None:
    """Sparse cells (min expected count < 5) must not produce a z/CI.

    a_events=2, a_n=100 gives min(n_a*p_a, n_a*(1-p_a), ...) = 2*0.02 = 0.04 < 5,
    so the normal approximation is unreliable: return rates descriptively and
    set z/p/CI to None, flagging ``sparse=True`` and ``n_ok=False``.
    """
    res = two_prop_ztest(2, 100, 50, 1000)
    assert res["sparse"] is True
    assert res["n_ok"] is False
    assert res["z"] is None
    assert res["p_value"] is None
    assert res["ci_low"] is None
    assert res["ci_high"] is None
    # Descriptive rates are still reported.
    assert res["p_a"] == pytest.approx(0.02, rel=1e-9)
    assert res["p_b"] == pytest.approx(0.05, rel=1e-9)
    assert res["diff"] == pytest.approx(-0.03, rel=1e-9)
    assert res["pooled"] == pytest.approx((2 + 50) / 1100, rel=1e-9)
    assert isinstance(res["p_a"], float)


def test_two_prop_ztest_raises_on_non_positive_n() -> None:
    """n=0 on either arm is a caller bug; fail fast and loud."""
    with pytest.raises(ValueError, match="n"):
        two_prop_ztest(50, 0, 40, 1000)
    with pytest.raises(ValueError, match="n"):
        two_prop_ztest(50, 1000, 40, 0)


def test_two_prop_ztest_raises_when_events_outside_range() -> None:
    """events outside [0,1] on either arm would produce a meaningless rate."""
    with pytest.raises(ValueError, match="events"):
        two_prop_ztest(-1, 1000, 40, 1000)
    with pytest.raises(ValueError, match="events"):
        two_prop_ztest(1001, 1000, 40, 1000)
    with pytest.raises(ValueError, match="events"):
        two_prop_ztest(50, 1000, 1001, 1000)


# --- chi2_independence -------------------------------------------------------
# Identity cross-check: the wrapper must be a transparent pass-through to
# scipy.stats.chi2_contingency with its default arguments (correction=True for
# 2x2), so calling scipy on the same table pins chi2/dof/p_value exactly.

_CHI2_TABLE_2X2 = [[50, 30], [20, 40]]


def test_chi2_independence_matches_scipy_identity() -> None:
    """chi2/dof/p_value/expected_min must equal scipy on the same table.

    Identity (rel ~0) is the strongest possible pin: no parametrisation note
    applies because both sides call scipy.stats.chi2_contingency with defaults
    (Yates correction applies to 2x2 chi2, not to the expected table — the
    expected counts are computed identically under correction=True/False).
    """
    got = chi2_independence(_CHI2_TABLE_2X2)
    chi2_ref, p_ref, dof_ref, expected_ref = stats.chi2_contingency(
        _CHI2_TABLE_2X2
    )
    assert got["chi2"] == pytest.approx(float(chi2_ref), rel=1e-12)
    assert got["dof"] == dof_ref
    assert got["p_value"] == pytest.approx(float(p_ref), rel=1e-12)
    # expected_min is the min of the scipy-returned expected table (the
    # ruling: expected counts are correction-independent, same call).
    assert got["expected_min"] == pytest.approx(
        float(expected_ref.min()), rel=1e-12
    )
    assert got["valid"] is True
    # Plain Python types, not numpy scalars (np.float64 subclasses float).
    assert type(got["chi2"]) is float
    assert type(got["dof"]) is int
    assert type(got["p_value"]) is float
    assert type(got["expected_min"]) is float
    assert type(got["valid"]) is bool


def test_chi2_independence_polars_input_path() -> None:
    """A polars DataFrame (the notebook_04 contingency shape) must normalize
    to the same result as the equivalent list-of-lists input.

    The confirmatory notebook builds a contingency table from a polars
    groupby; this pins the `.to_numpy()` conversion path inside the wrapper.
    """
    df = pl.DataFrame(
        {
            "converted": [10, 15, 12, 8],
            "not_converted": [20, 25, 38, 52],
        }
    )
    from_polars = chi2_independence(df)
    from_list = chi2_independence(df.to_numpy().tolist())
    for key in ("chi2", "dof", "p_value", "expected_min", "valid"):
        assert from_polars[key] == from_list[key], key
    chi2_ref, p_ref, dof_ref, expected_ref = stats.chi2_contingency(
        df.to_numpy()
    )
    assert from_polars["chi2"] == pytest.approx(float(chi2_ref), rel=1e-12)
    assert from_polars["p_value"] == pytest.approx(float(p_ref), rel=1e-12)
    assert from_polars["expected_min"] == pytest.approx(
        float(expected_ref.min()), rel=1e-12
    )


def test_chi2_independence_sparse_expected_min_below_five() -> None:
    """A genuinely sparse table (expected cell < 5) must set valid=False.

    [[20,0],[0,2]] has row/col totals 20/2/22 -> expected min = 2*2/22 ~
    0.18 < 5, so the chi-square approximation is unreliable downstream.
    chi2 and p_value are still reported (valid is a guard flag for the
    notebook's routing decision, not an error signal). The brief's example
    [[10,0],[0,10]] is numerically NOT sparse (expected_min = 5.0 exactly,
    see boundary test below) and is corrected in the module docstring.
    """
    table = [[20, 0], [0, 2]]
    got = chi2_independence(table)
    chi2_ref, p_ref, _dof_ref, expected_ref = stats.chi2_contingency(table)
    assert got["expected_min"] == pytest.approx(
        float(expected_ref.min()), rel=1e-12
    )
    assert got["expected_min"] < 5
    assert got["valid"] is False
    # Stats are still computed; valid=False routes the caller, it does not
    # suppress the estimate.
    assert got["chi2"] == pytest.approx(float(chi2_ref), rel=1e-12)
    assert got["p_value"] == pytest.approx(float(p_ref), rel=1e-12)


def test_chi2_independence_expected_min_boundary_is_valid() -> None:
    """expected_min exactly 5.0 must be valid=True (inclusive >= rule).

    This pins both the inclusive boundary condition and documents the
    brief-example correction: [[10,0],[0,10]] has expected cells all 5.0
    (10*10/20), not 0 as the brief parenthetical claimed.
    """
    got = chi2_independence([[10, 0], [0, 10]])
    assert got["expected_min"] == pytest.approx(5.0, rel=1e-12)
    assert got["valid"] is True


def test_chi2_independence_raises_on_ragged_or_non_2d() -> None:
    """Ragged rows or 1-D input would silently misparse as a contingency;
    fail fast instead of broadcasting into a wrong-shaped table."""
    with pytest.raises(ValueError):
        chi2_independence([[1, 2], [3]])
    with pytest.raises(ValueError):
        chi2_independence([1, 2, 3])


# --- bonferroni --------------------------------------------------------------

def test_bonferroni_hand_values_and_cap() -> None:
    """min(p*k, 1) with k=3 must reproduce the hand-computed product,
    and p*k>1 must be capped at 1.0 (p-values are probabilities)."""
    assert bonferroni([0.01, 0.02, 0.03]) == pytest.approx(
        [0.03, 0.06, 0.09], rel=1e-12
    )
    # k=2: 0.6*2=1.2 and 0.7*2=1.4 both exceed 1 -> capped.
    capped = bonferroni([0.6, 0.7])
    assert capped == [1.0, 1.0]
    assert all(type(v) is float for v in capped)


def test_bonferroni_empty_and_nan_safe() -> None:
    """Empty input -> empty output; a NaN p-value maps to NaN (not a crash
    and not a silent drop) while neighbours still get their correction."""
    assert bonferroni([]) == []
    got = bonferroni([0.01, float("nan"), 0.03])
    assert got[0] == pytest.approx(0.03, rel=1e-12)
    assert math.isnan(got[1])
    assert got[2] == pytest.approx(0.09, rel=1e-12)


# --- fdr_bh ------------------------------------------------------------------

# Hand-set lists: ties (two equal low p-values) and an extreme p among large
# values exercise BH's rank/monotonicity steps that a plain p*m list would miss.
_BH_LISTS = (
    [0.01, 0.01, 0.03, 0.05],   # tie at the lowest rank
    [0.0001, 0.4, 0.5, 0.6],    # extreme p among large ps
    [0.2, 0.02, 0.3, 0.05, 0.01],  # mixed, no ties
)


def test_fdr_bh_matches_statsmodels_multipletests() -> None:
    """Benjamini-Hochberg must equal statsmodels method='fdr_bh' at rel=1e-6
    on lists containing ties and an extreme p-value."""
    for p_values in _BH_LISTS:
        got = fdr_bh(p_values)
        ref = multipletests(p_values, method="fdr_bh")[1]
        for g, r in zip(got, ref):
            assert g == pytest.approx(float(r), rel=1e-6)
        assert all(type(v) is float for v in got)


def test_fdr_bh_tie_behavior_and_extreme_p() -> None:
    """Tied low p-values must receive the same corrected value (BH's rank
    monotonicity pulls the first tied value down to the second's), and an
    extreme p must stay extreme after correction."""
    tied = fdr_bh([0.01, 0.01, 0.03, 0.05])
    # statsmodels reference for this exact list: [0.02, 0.02, 0.04, 0.05].
    assert tied[0] == pytest.approx(tied[1], rel=1e-12)
    assert tied[0] == pytest.approx(0.02, rel=1e-6)

    extreme = fdr_bh([0.0001, 0.4, 0.5, 0.6])
    # statsmodels reference: [0.0004, 0.6, 0.6, 0.6].
    assert extreme[0] == pytest.approx(0.0004, rel=1e-6)
    assert extreme[0] < min(extreme[1:])


def test_fdr_bh_empty_and_nan_safe() -> None:
    """Empty input -> empty output; NaN passes through to its own slot
    without corrupting the corrected values of the remaining hypotheses."""
    assert fdr_bh([]) == []
    got = fdr_bh([0.01, float("nan"), 0.03])
    assert got[0] == pytest.approx(0.03, rel=1e-6)
    assert math.isnan(got[1])
    assert got[2] == pytest.approx(0.045, rel=1e-6)
    assert all(type(v) is float for v in got if not math.isnan(v))


# --- required_n_per_group / achieved_power -----------------------------------
# Cross-check anchor: p0=0.02, mde=0.005 -> p1=0.025 (the Stage-04 example),
# alpha=0.05 two-sided, power=0.8. The rel=0.01 tolerance is documented in the
# module header: closed form (raw pooled proportion difference) vs statsmodels
# NormalIndPower (Cohen's h / arcsine family) differ structurally by a small
# factor on tiny effects — that is a formula-family difference, not a bug.

def test_required_n_per_group_matches_statsmodels_solve_power() -> None:
    """Closed-form n/group must match statsmodels solve_power at rel=0.01.

    statsmodels sizes on Cohen's h (arcsine effect size); this module sizes on
    the raw pooled-H0 proportion difference. Both are normal-approximation
    formulas but not algebraically identical, hence the documented 1% band
    rather than a tight pin.
    """
    n_ours = required_n_per_group(0.02, 0.005, 0.05, 0.8)
    n_sm = NormalIndPower().solve_power(
        effect_size=proportion_effectsize(0.02, 0.025),
        alpha=0.05,
        power=0.8,
        ratio=1.0,
        alternative="two-sided",
    )
    assert n_ours == pytest.approx(float(n_sm), rel=0.01)
    # Contract: a positive finite plain float.
    assert n_ours > 0
    assert math.isfinite(n_ours)
    assert type(n_ours) is float


def test_required_n_per_group_mde_monotonicity() -> None:
    """A larger detectable effect needs a smaller sample: n(mde=0.005) > n(0.010).

    Monotonicity is a structural sanity check on the 1/mde^2 scaling: it would
    catch a sign or squared-term regression that a single cross-check point
    could miss.
    """
    n_small_mde = required_n_per_group(0.02, 0.005, 0.05, 0.8)
    n_large_mde = required_n_per_group(0.02, 0.010, 0.05, 0.8)
    assert n_small_mde > n_large_mde > 0
    assert math.isfinite(n_large_mde)


def test_required_n_per_group_negative_mde_is_accepted() -> None:
    """mde sign only picks the H1 direction (p1 = p0 + mde); magnitude sizes n.

    A negative mde (detecting a decrease) must produce a valid positive n —
    the guards reject only mde == 0, not the direction.
    """
    n_down = required_n_per_group(0.05, -0.01, 0.05, 0.8)
    assert n_down > 0
    assert math.isfinite(n_down)
    assert type(n_down) is float


def test_required_n_per_group_guards() -> None:
    """Invalid probability/rate arguments must raise ValueError, fail fast.

    Guards: 0 < base_rate < 1, mde != 0 (and p1 stays inside (0,1)),
    0 < alpha < 1, 0 < power < 1.
    """
    with pytest.raises(ValueError, match="base_rate"):
        required_n_per_group(0.0, 0.005, 0.05, 0.8)
    with pytest.raises(ValueError, match="base_rate"):
        required_n_per_group(1.0, 0.005, 0.05, 0.8)
    with pytest.raises(ValueError, match="mde"):
        required_n_per_group(0.02, 0.0, 0.05, 0.8)
    # p1 = base_rate + mde must remain a valid probability.
    with pytest.raises(ValueError, match="p1"):
        required_n_per_group(0.99, 0.05, 0.05, 0.8)
    with pytest.raises(ValueError, match="alpha"):
        required_n_per_group(0.02, 0.005, 0.0, 0.8)
    with pytest.raises(ValueError, match="alpha"):
        required_n_per_group(0.02, 0.005, 1.0, 0.8)
    with pytest.raises(ValueError, match="power"):
        required_n_per_group(0.02, 0.005, 0.05, 0.0)
    with pytest.raises(ValueError, match="power"):
        required_n_per_group(0.02, 0.005, 0.05, 1.0)


def test_achieved_power_round_trip_with_required_n() -> None:
    """achieved_power at n = round(required_n(...)) recovers the target power.

    The round-trip pins the two functions as mutual inverses. The band is
    loose (rel=5%) because rounding n to a whole subject shifts power by a
    small discrete step — that rounding, not a formula mismatch, is the
    expected residual.
    """
    target = 0.8
    n = round(required_n_per_group(0.02, 0.005, 0.05, target))
    got = achieved_power(0.02, 0.005, n, 0.05)
    assert got == pytest.approx(target, rel=0.05)
    assert 0.0 <= got <= 1.0
    assert type(got) is float


def test_achieved_power_monotonicity_in_n() -> None:
    """Power must increase with n: more data, more chance to detect the effect."""
    p0, mde, alpha = 0.02, 0.005, 0.05
    n_star = round(required_n_per_group(p0, mde, alpha, 0.8))
    power_small = achieved_power(p0, mde, n_star // 2, alpha)
    power_large = achieved_power(p0, mde, 2 * n_star, alpha)
    assert 0.0 < power_small < power_large <= 1.0


def test_achieved_power_guards() -> None:
    """n must be an integer >= 1; probability args mirror required_n guards."""
    with pytest.raises(ValueError, match="n"):
        achieved_power(0.02, 0.005, 0, 0.05)
    with pytest.raises(ValueError, match="n"):
        achieved_power(0.02, 0.005, -100, 0.05)
    with pytest.raises(ValueError, match="n"):
        achieved_power(0.02, 0.005, 10.5, 0.05)
    with pytest.raises(ValueError, match="base_rate"):
        achieved_power(0.0, 0.005, 100, 0.05)
    with pytest.raises(ValueError, match="mde"):
        achieved_power(0.02, 0.0, 100, 0.05)
    with pytest.raises(ValueError, match="alpha"):
        achieved_power(0.02, 0.005, 100, 0.0)
