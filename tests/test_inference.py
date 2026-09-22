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
from statsmodels.stats.proportion import proportion_confint

from src.inference import wilson_ci

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