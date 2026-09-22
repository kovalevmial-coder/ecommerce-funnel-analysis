"""Tests for the Wilson (score) confidence interval in src/inference.py.

The file pins three distinct things:
1. The teaching default z=1.96 produces the true Wilson endpoints on boundary
   and interior points (rel=1e-6 against hand-computed constants).
2. The formula matches statsmodels ``proportion_confint(method="wilson")`` at
   rel=1e-6 when both use the same z — this pins the algebra with no
   z-rounding confound (statsmodels' two-sided alpha=0.05 resolves to
   z=1.95996..., not the Stage-04 teaching constant 1.96).
3. The guards raise ValueError on impossible inputs.

Plan-artifact correction (documented per controller ruling): the published
known values in docs/superpowers/plans/04-inference.md Task 1 Step 1 are NOT
Wilson endpoints. (success=0, n=100) -> 0.0362 and (100,100) -> 0.9638 are
Clopper-Pearson endpoints (statsmodels ``proportion_confint(method="beta")``
returns 0.0362167 / 0.9637833), and (50,100) -> (0.4038, 0.5962) is a 4-decimal
rounding of the true endpoints. All three published anchors fail rel=1e-6. The
constants below are the true Wilson endpoints at z=1.96, verified against
statsmodels with the identical z (diff 0.0 for every endpoint).
"""
from __future__ import annotations

import pytest
from scipy import stats
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
)

from src.inference import two_prop_ztest, wilson_ci

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
    assert isinstance(res["z"], float)


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
    """events outside [0, n] on either arm would produce a meaningless rate."""
    with pytest.raises(ValueError, match="events"):
        two_prop_ztest(-1, 1000, 40, 1000)
    with pytest.raises(ValueError, match="events"):
        two_prop_ztest(1001, 1000, 40, 1000)
    with pytest.raises(ValueError, match="events"):
        two_prop_ztest(50, 1000, 1001, 1000)
