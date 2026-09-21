"""Tests for funnel conversion rates in src/funnel.py.

The fixture is 220 crafted sessions that pin the step maths to hand-derived
numbers AND exercise the jumpers the plan warns about: purchase-without-view
sessions (10, with cart) and cart-without-view sessions (10) must not distort
the per-step numerators. Expected values are hand-derived, not recomputed by
the code under test; the Wilson CIs are cross-checked against statsmodels' own
``proportion_confint(method="wilson")`` at 1e-6 relative tolerance, so a
regression in the inline formula is caught even though the interval is
algebraically equivalent.
"""
from __future__ import annotations

import polars as pl
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from src.funnel import _wilson_ci, funnel_rates

# Contract from plan Task 2, exactly three rows in this order.
_STEPS = ["view->cart", "cart->purchase", "view->purchase"]

# Hand-derived from the fixture below:
#   - view->cart:     90 carted of 200 viewed       -> 0.45
#   - cart->purchase: 40 purchased of 100 carted    -> 0.40
#   - view->purchase: 30 purchased of 200 viewed    -> 0.15
#   (the 40 purchasers split as 30 with cart+view, 10 with cart but no view)
_EXPECTED = {
    "view->cart": {"numerator": 90, "denominator": 200, "rate": 0.45},
    "cart->purchase": {"numerator": 40, "denominator": 100, "rate": 0.40},
    "view->purchase": {"numerator": 30, "denominator": 200, "rate": 0.15},
}


def _make_sessions() -> pl.DataFrame:
    """Build the crafted sessions frame (one row per session).

    The 220 rows are grouped into mutually exclusive journey types chosen so the
    per-step counts are exact and small and the jumpers are visible:

    - 110 view-only sessions           -> view, no cart, no purchase
    -  10 no-flag sessions             -> nothing (e.g. remove-only, like reality)
    -  60 view+cart sessions           -> carted without purchasing
    -  30 view+cart+purchase sessions  -> the full funnel leg
    -  10 cart+purchase sessions       -> purchase WITHOUT a view (jumper); these
        also carry has_cart, so they count as cart-without-view AND
        purchase-without-view at once

    Resulting aggregates: 200 viewed, 100 carted, 40 purchased. The 10
    cart+purchase jumpers must appear in cart->purchase's numerator (they
    purchased and carted) but stay out of view->* numerators (they never
    viewed).
    """
    return pl.DataFrame(
        {
            "has_view": [True] * 110 + [True] * 60 + [True] * 30 + [False] * 10 + [False] * 10,
            "has_cart": [False] * 110 + [True] * 60 + [True] * 30 + [True] * 10 + [False] * 10,
            "has_purchase": [False] * 110 + [False] * 60 + [True] * 30 + [True] * 10 + [False] * 10,
        }
    )


def test_funnel_rates_contract_rows_columns_and_dtypes() -> None:
    """Exactly 3 contract rows, in step order, with column order and dtypes."""
    result = funnel_rates(_make_sessions())

    assert result.columns == [
        "step",
        "numerator",
        "denominator",
        "rate",
        "ci_low",
        "ci_high",
    ]
    assert [r["step"] for r in result.iter_rows(named=True)] == _STEPS
    # Numerator/denominator are exact integers; rate and CI are Float64.
    assert result.schema["numerator"] == pl.Int64
    assert result.schema["denominator"] == pl.Int64
    assert result.schema["rate"] == pl.Float64
    assert result.schema["ci_low"] == pl.Float64
    assert result.schema["ci_high"] == pl.Float64


def test_funnel_rates_hand_derived_counts_and_rates() -> None:
    """Every cell of the three steps must match the hand-computed numbers."""
    got = {row["step"]: row for row in funnel_rates(_make_sessions()).iter_rows(named=True)}

    for step, want in _EXPECTED.items():
        row = got[step]
        assert row["numerator"] == want["numerator"]
        assert row["denominator"] == want["denominator"]
        assert row["rate"] == pytest.approx(want["rate"])


def test_funnel_rates_wilson_ci_matches_statsmodels() -> None:
    """The inline Wilson interval must match statsmodels' within ~1e-4 relative.

    rel is 1e-4, not the textbook 1e-6, for a deliberate and fixable reason:
    ``_wilson_ci`` defaults to the canonical teaching constant z=1.96 (Stage-02
    contract), while statsmodels' two-sided alpha=0.05 resolves to
    ``scipy.stats.norm.isf(0.025) == 1.95996...``. That z mismatch alone moves
    the endpoints by up to ~6e-6 relative (measured), so 1e-6 cannot hold here
    without changing the mandated default. 1e-4 stays two orders of magnitude
    below any real formula regression while accepting the z rounding. The exact
    formula is pinned at 1e-6 against the same z in
    ``test_wilson_ci_edge_points_match_statsmodels_without_flattening``.
    """
    got = {row["step"]: row for row in funnel_rates(_make_sessions()).iter_rows(named=True)}

    for step, want in _EXPECTED.items():
        num, den = want["numerator"], want["denominator"]
        ref_low, ref_high = proportion_confint(num, den, method="wilson")
        assert got[step]["ci_low"] == pytest.approx(float(ref_low), rel=1e-4)
        assert got[step]["ci_high"] == pytest.approx(float(ref_high), rel=1e-4)


def test_funnel_rates_ci_contains_rate_and_stays_inside_unit_interval() -> None:
    """Interior rates must sit strictly between ci_low and ci_high, inside (0,1)."""
    got = funnel_rates(_make_sessions())

    for row in got.iter_rows(named=True):
        assert row["ci_low"] < row["rate"] < row["ci_high"]
        assert 0.0 < row["ci_low"] < 1.0
        assert 0.0 < row["ci_high"] < 1.0


def test_wilson_ci_edge_points_match_statsmodels_without_flattening() -> None:
    """The formula must match statsmodels exactly at interior AND edge points.

    statsmodels' default alpha=0.05 maps to z=1.95996398..., not the Stage-02
    mandate 1.96, so the *same* z constant is passed explicitly here — this
    pins the Wilson *algebra* at 1e-6 with no z-rounding confound (that
    confound is what the looser rel in the funnel_rates test absorbs). "Honest
    at the edges" means the formula is evaluated, not special-cased: at
    k=0/k=n the interval legitimately degenerates to the boundary value (lower
    == 0 / upper == 1 exactly) while the opposite endpoint still follows the
    score formula, all in agreement with statsmodels' clipped implementation.
    """
    z = float(stats.norm.isf(0.025))
    for num, den in [(0, 100), (1, 100), (50, 100), (100, 200), (200, 200)]:
        got_low, got_high = _wilson_ci(num, den, z=z)
        ref_low, ref_high = proportion_confint(num, den, method="wilson")
        # Boundary endpoints compare trivially (0.0 == 0.0, 1.0 == 1.0): a
        # relative tolerance would divide by zero and, worse, a regression that
        # flattens the interval to a point would go unnoticed. The interior
        # endpoint still carries the full 1e-6 formula check.
        assert got_low == pytest.approx(float(ref_low), rel=1e-6)
        assert got_high == pytest.approx(float(ref_high), rel=1e-6)


def test_funnel_rates_raises_on_missing_columns() -> None:
    """Missing step indicators must fail fast and list the absent columns."""
    with pytest.raises(ValueError, match="has_cart"):
        funnel_rates(pl.DataFrame({"has_view": [True]}))
