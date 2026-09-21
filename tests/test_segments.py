"""Tests for segment labels and the per-segment funnel in src/segments.py.

Two fixtures. The first is a small ``session_features``-shaped frame (the raw
columns ``hour, dayofweek, month, session_number, median_price`` plus the
funnel booleans the stage carries downstream); every label is hand-derived
from the raw column: price tier from the band edges in ``PRICE_TIERS``, visit
kind from ``session_number``, weekend from the polars dayofweek convention
(0=Mon..6=Sun), daypart from ``hour`` and ``month_label`` from the documented
dataset-scope year rule. The second (``_structured_sessions``) is a 20-row
frame whose per-segment funnel arithmetic is small enough to verify by hand;
``segment_funnel`` runs through the internal ``assign_segments`` path here
(the fixture carries no label columns), exactly as it does on real
``sessions.parquet``.
"""
from __future__ import annotations

import polars as pl
import pytest
from statsmodels.stats.proportion import proportion_confint

from src.segments import DAYPARTS, PRICE_TIERS, assign_segments, segment_funnel

# raw columns ``assign_segments`` reads. Everything else in the contract (the
# ``has_*``/``net_cart`` funnel booleans) is ignored by it but must survive a
# round-trip untouched.
_RAW_COLUMNS = ["hour", "dayofweek", "month", "session_number", "median_price"]
_DUMMY_COLUMNS = {
    "has_view": True,
    "has_cart": True,
    "has_purchase": True,
    "net_cart": True,
}
_NEW_COLUMNS = ["price_tier", "visit_kind", "is_weekend", "daypart", "month_label"]

# (hour, daypart) pairs for the four band interiors.
_DAYPART_HOURS = [(2, "night"), (8, "morning"), (14, "afternoon"), (20, "evening")]


def _raw_sessions() -> pl.DataFrame:
    """Build the crafted raw fixture, one row per label assertion.

    Row design (kept hand-derivable): one row per boundary value for the price
    tiers, plus hour 2/8/14/20 for the four dayparts, ``session_number`` 1 vs 2
    for new/returning, ``dayofweek`` 4 vs 5 for weekday/weekend, a null
    ``median_price`` for the explicit NA tier, and months 10/1 to pin the
    year rule in both directions. The ``median_price`` values are exact floats
    so the tier expectations are unambiguous at the band edges.
    """
    rows = [
        # price-tier boundaries: budget <5, mid [5,30), premium [30,150),
        # luxury [150, inf). 0.0 and a negative both land in budget (r<5).
        (8, 0, 10, 1, 4.9),
        (8, 0, 10, 1, 5.0),
        (8, 0, 10, 1, 29.99),
        (8, 0, 10, 1, 30.0),
        (8, 0, 10, 1, 149.99),
        (8, 0, 10, 1, 150.0),
        (8, 0, 10, 1, 0.0),
        (8, 0, 10, 1, -79.37),
        (8, 0, 10, 1, None),
    ]
    df = pl.DataFrame(
        rows,
        schema={
            "hour": pl.Int64,
            "dayofweek": pl.Int64,
            "month": pl.Int64,
            "session_number": pl.Int64,
            "median_price": pl.Float64,
        },
        orient="row",
    )
    # Funnel booleans are contract passengers here — assign_segments ignores
    # them but the round-trip tests assert they survive unscathed.
    for col, val in _DUMMY_COLUMNS.items():
        df = df.with_columns(pl.lit(val).alias(col))
    return df


def _synthetic_sessions() -> pl.DataFrame:
    """Build the label-completeness fixture with a null-priced session baked in."""
    raw = pl.DataFrame(
        {
            "user_session": ["s1", "s2", "s3", "s4", "s5", "s6", "s7"],
            "user_id": [1, 1, 2, 2, 3, 4, 5],
            "month": [10, 11, 12, 1, 2, 10, 1],
            "dayofweek": [4, 5, 0, 2, 6, 3, 1],
            "hour": [2, 8, 14, 20, 6, 12, 18],
            "session_number": [1, 2, 1, 2, 1, 1, 1],
            "n_view": [0, 0, 0, 0, 0, 0, 0],
            "n_cart": [0, 0, 0, 0, 0, 0, 0],
            "n_remove": [0, 0, 0, 0, 0, 0, 0],
            "n_purchase": [0, 0, 0, 0, 0, 0, 0],
            "has_view": [True, True, True, True, True, True, True],
            "has_cart": [True, True, True, True, True, True, True],
            "has_purchase": [True, True, True, True, True, True, True],
            "net_cart": [True, True, True, True, True, True, True],
            "median_price": [None, 4.9, 7.0, 30.0, 150.0, 149.99, 5.0],
            "n_products": [0, 0, 0, 0, 0, 0, 0],
        }
    )
    # one month=10 session + one month=1 session -> the year rule must label
    # both 2019-10 and 2020-01.
    return raw


def test_price_tier_boundaries_land_in_documented_bands() -> None:
    """Every band edge value must land in exactly the documented tier."""
    result = assign_segments(_raw_sessions())
    got = result["price_tier"].to_list()
    expected = [
        "budget",
        "mid",
        "mid",
        "premium",
        "premium",
        "luxury",
        "budget",
        "budget",
        "NA",
    ]
    assert got == expected


def test_null_median_price_explicit_na() -> None:
    """A session without priced events gets 'NA', not fabricated into a band."""
    result = assign_segments(_raw_sessions())
    null_row = result.filter(pl.col("median_price").is_null())
    assert null_row["price_tier"].to_list() == ["NA"]


def test_exhaustive_tier_mapping_over_real_price_scope() -> None:
    """Every real price in the dataset scope maps into exactly one tier.

    Property-style sweep: walk the actual observed price continuum
    ([-80, 328], covering all four bands) at a fine step. No value may fall
    through ``otherwise`` (a gap) — assert it does not by checking each sweep
    value lands in the tier the documented bands prescribe.
    """
    prices = list(range(-80, 329))  # -80..328 (ints cover the band edges)
    sweep = pl.DataFrame(
        {"hour": 0, "dayofweek": 0, "month": 10, "session_number": 1, "median_price": prices}
    )
    result = assign_segments(sweep)
    tiers = result["price_tier"].to_list()
    # Recompute what each integer price *should* be from the documented bands.
    for price, tier in zip(prices, tiers):
        if price < 5:
            assert tier == "budget"
        elif price < 30:
            assert tier == "mid"
        elif price < 150:
            assert tier == "premium"
        else:
            assert tier == "luxury"


def test_tier_chain_stays_in_lockstep_with_constants() -> None:
    """The when-chain must agree with PRICE_TIERS, not just with hand-literal copies.

    The boundary tests above re-encode the bands by hand; this one derives the
    expectation from the constants themselves, so a drift between the chain and
    PRICE_TIERS is caught even if both are edited consistently by hand.
    """
    # Sample every band interior plus both extremes (-100..400 at step 0.1).
    prices = [p / 10 for p in range(-1000, 4001)]
    expected = {}
    for price in prices:
        for name, (band_lo, band_hi) in PRICE_TIERS.items():
            if band_lo <= price < band_hi:
                expected[price] = name
                break

    sweep = pl.DataFrame(
        {"hour": 0, "dayofweek": 0, "month": 10, "session_number": 1, "median_price": prices}
    )
    for price, tier in zip(prices, assign_segments(sweep)["price_tier"].to_list()):
        assert tier == expected[price], f"chain {tier} != PRICE_TIERS {expected[price]} at {price}"


def test_visit_kind_daypart_and_weekend_labels() -> None:
    """new/returning, weekday/weekend and daypart must come from their raw cols."""
    result = assign_segments(_synthetic_sessions())

    # session_number 1 -> new, 2 -> returning (indices 0/1 in the fixture).
    assert result["visit_kind"].to_list()[:2] == ["new", "returning"]

    # dayofweek 4 -> not weekend, 5 -> weekend (polars 0=Mon..6=Sun).
    assert result["is_weekend"].to_list()[:2] == [False, True]

    # hours 2/8/14/20 -> night/morning/afternoon/evening.
    assert result["daypart"].to_list()[:4] == ["night", "morning", "afternoon", "evening"]


def test_month_label_year_rule() -> None:
    """month {10,11,12} -> 2019, {1,2} -> 2020; anything else is ambiguous."""
    result = assign_segments(_synthetic_sessions())
    months = [(row["month"], row["month_label"]) for row in result.iter_rows(named=True)]
    by_month = dict(months)
    assert by_month[10] == "2019-10"
    assert by_month[11] == "2019-11"
    assert by_month[12] == "2019-12"
    assert by_month[1] == "2020-01"
    assert by_month[2] == "2020-02"


def test_month_outside_dataset_scope_raises() -> None:
    """A month outside {1,2,10,11,12} cannot be mapped to a year -- fail loudly."""
    out_of_scope = pl.DataFrame(
        {
            "hour": [8],
            "dayofweek": [1],
            "month": [7],
            "session_number": [1],
            "median_price": [10.0],
        }
    )
    with pytest.raises(ValueError, match="month"):
        assign_segments(out_of_scope)


def test_label_derivation_leaves_raw_columns_untouched() -> None:
    """assign_segments must not mutate the raw columns it segments on."""
    raw = _synthetic_sessions()
    before = raw.clone()
    result = assign_segments(raw)

    for col in _RAW_COLUMNS:
        assert pl.Series(result[col]).equals(pl.Series(before[col]))  # NaN-wise equal
        assert result[col].dtype == before[col].dtype


def test_new_label_columns_and_contract_preservation() -> None:
    """The output must add exactly the five label columns, in documented order."""
    raw = _synthetic_sessions()
    result = assign_segments(raw)

    assert result.columns[:-5] == raw.columns
    assert result.columns[-5:] == _NEW_COLUMNS
    assert {col: dtype for col, dtype in result.schema.items() if col in _NEW_COLUMNS} == {
        "price_tier": pl.Utf8,
        "visit_kind": pl.Utf8,
        "is_weekend": pl.Boolean,
        "daypart": pl.Utf8,
        "month_label": pl.Utf8,
    }


def test_assign_segments_raises_on_missing_columns() -> None:
    """Missing raw input must fail fast and list the absent columns."""
    incomplete = _raw_sessions().drop("median_price")
    with pytest.raises(ValueError, match="median_price"):
        assign_segments(incomplete)


def test_band_edges_match_declared_constants() -> None:
    """The tier band edges and daypart edges are exactly as documented."""
    assert PRICE_TIERS == {
        "budget": (-float("inf"), 5.0),
        "mid": (5.0, 30.0),
        "premium": (30.0, 150.0),
        "luxury": (150.0, float("inf")),
    }
    assert DAYPARTS == {
        "night": (0, 6),
        "morning": (6, 12),
        "afternoon": (12, 18),
        "evening": (18, 24),
    }


def test_hour_24h_range_is_exhaustive_across_dayparts() -> None:
    """Every real hour 0..23 must land in exactly one daypart."""
    sweep = pl.DataFrame(
        {"hour": list(range(24)), "dayofweek": 0, "month": 10, "session_number": 1, "median_price": 1.0}
    )
    result = assign_segments(sweep)
    dayparts = result["daypart"].to_list()
    expected = {
        "night": list(range(0, 6)),
        "morning": list(range(6, 12)),
        "afternoon": list(range(12, 18)),
        "evening": list(range(18, 24)),
    }
    for part, hours in expected.items():
        assert [h for h, p in zip(range(24), dayparts) if p == part] == hours


def _structured_sessions() -> pl.DataFrame:
    """Build a 20-row frame whose per-segment funnel counts are hand-derivable.

    Row design (journey as has_view/has_cart/has_purchase, in that order):

    - ``new`` (session_number 1): 5 view-only budget, 1 view-only luxury,
      2 view+cart budget, 1 view+cart luxury, 1 view+cart+purchase budget,
      1 view+cart+purchase luxury, 1 cart+purchase budget jumper (no view --
      proves the per-segment boolean-AND numerators behave exactly like the
      Stage-02 aggregate step).
    - ``returning`` (session_number 2): 4 view-only luxury, 2 view+cart
      luxury, 2 view+cart+purchase luxury.

    ``median_price`` 4.9 vs 150.0 lands strictly inside budget/luxury; the raw
    ``hour``/``dayofweek``/``month`` are constant (``assign_segments`` needs
    them but they do not affect the two families under test). Armchair sums
    from this layout:

    - new:        viewed 11, carted 6, purchased 3  -> 5/11, 3/6, 2/11
    - returning:  viewed  8, carted 4, purchased 2  -> 4/8,  2/4,  2/8
    - budget:     viewed  8, carted 4, purchased 2  -> 3/8,  2/4,  1/8
    - luxury:     viewed 11, carted 6, purchased 3  -> 6/11, 3/6,  3/11
    with segment sizes 12 new, 8 returning, 9 budget (5 view-only + 2
    view+cart + 1 full funnel + 1 jumper), 11 luxury (3 new + 8 returning).
    """
    rows = [
        # session_number, median_price, has_view, has_cart, has_purchase
        (1, 4.9, True, False, False),   # new, budget, view-only
        (1, 4.9, True, False, False),   # new, budget, view-only
        (1, 4.9, True, False, False),   # new, budget, view-only
        (1, 4.9, True, False, False),   # new, budget, view-only
        (1, 4.9, True, False, False),   # new, budget, view-only
        (1, 150.0, True, False, False),  # new, luxury, view-only
        (1, 4.9, True, True, False),    # new, budget, view+cart
        (1, 4.9, True, True, False),    # new, budget, view+cart
        (1, 150.0, True, True, False),  # new, luxury, view+cart
        (1, 4.9, True, True, True),     # new, budget, full funnel
        (1, 150.0, True, True, True),   # new, luxury, full funnel
        (1, 4.9, False, True, True),    # new, budget, cart+purchase jumper (no view)
        (2, 150.0, True, False, False),  # returning, luxury, view-only
        (2, 150.0, True, False, False),  # returning, luxury, view-only
        (2, 150.0, True, False, False),  # returning, luxury, view-only
        (2, 150.0, True, False, False),  # returning, luxury, view-only
        (2, 150.0, True, True, False),   # returning, luxury, view+cart
        (2, 150.0, True, True, False),   # returning, luxury, view+cart
        (2, 150.0, True, True, True),    # returning, luxury, full funnel
        (2, 150.0, True, True, True),    # returning, luxury, full funnel
    ]
    return pl.DataFrame(
        rows,
        schema={
            "session_number": pl.Int64,
            "median_price": pl.Float64,
            "has_view": pl.Boolean,
            "has_cart": pl.Boolean,
            "has_purchase": pl.Boolean,
        },
        orient="row",
        # Constant raw columns assign_segments requires (daypart/hour/week
        # labels are irrelevant here — kept constant so they never add a
        # second segment value to the visit_kind/price_tier families).
    ).with_columns(
        [
            pl.lit(8, dtype=pl.Int64).alias("hour"),
            pl.lit(1, dtype=pl.Int64).alias("dayofweek"),
            pl.lit(10, dtype=pl.Int64).alias("month"),
        ]
    )


def _assert_segment_rows(
    result: pl.DataFrame,
    expected: dict[str, dict[str, tuple[int, int]]],
    n_sessions: dict[str, int],
) -> None:
    """Assert exact numerator/denominator/rate/CI rows and per-segment sizes."""
    assert result["n_sessions"].to_list() == [n for seg in expected for n in [n_sessions[seg]] * 3]
    for row in result.iter_rows(named=True):
        num, den = expected[row["segment"]][row["step"]]
        assert row["numerator"] == num
        assert row["denominator"] == den
        assert row["rate"] == pytest.approx(num / den)
        # Wilson CI cross-checked against statsmodels, rel 1e-4 for the same
        # documented z=1.96-vs-normal-quantile rounding as test_funnel_rates.
        ref_low, ref_high = proportion_confint(num, den, method="wilson")
        assert row["ci_low"] == pytest.approx(float(ref_low), rel=1e-4)
        assert row["ci_high"] == pytest.approx(float(ref_high), rel=1e-4)


def test_segment_funnel_visit_kind_exact_rows() -> None:
    """new/returning rows must match the hand-derived counts, rates and sizes."""
    result = segment_funnel(_structured_sessions(), "visit_kind")

    assert result.columns == [
        "step",
        "numerator",
        "denominator",
        "rate",
        "ci_low",
        "ci_high",
        "segment",
        "n_sessions",
    ]
    # Lexicographic segment order (new < returning), _FUNNEL_STEPS order inside.
    assert result["segment"].to_list() == ["new", "new", "new", "returning", "returning", "returning"]
    assert [r["step"] for r in result.iter_rows(named=True)] == [
        "view->cart", "cart->purchase", "view->purchase",
        "view->cart", "cart->purchase", "view->purchase",
    ]
    assert result.schema["segment"] == pl.Utf8
    assert result.schema["n_sessions"] == pl.Int64
    assert result.schema["rate"] == pl.Float64

    _assert_segment_rows(
        result,
        expected={
            "new": {"view->cart": (5, 11), "cart->purchase": (3, 6), "view->purchase": (2, 11)},
            "returning": {"view->cart": (4, 8), "cart->purchase": (2, 4), "view->purchase": (2, 8)},
        },
        n_sessions={"new": 12, "returning": 8},
    )


def test_segment_funnel_price_tier_exact_rows() -> None:
    """budget/luxury rows must match the hand-derived counts, rates and sizes."""
    result = segment_funnel(_structured_sessions(), "price_tier")

    assert result["segment"].to_list() == ["budget", "budget", "budget", "luxury", "luxury", "luxury"]

    _assert_segment_rows(
        result,
        expected={
            "budget": {"view->cart": (3, 8), "cart->purchase": (2, 4), "view->purchase": (1, 8)},
            "luxury": {"view->cart": (6, 11), "cart->purchase": (3, 6), "view->purchase": (3, 11)},
        },
        n_sessions={"budget": 9, "luxury": 11},
    )


def test_segment_funnel_na_price_tier_is_reported_not_dropped() -> None:
    """A null-median session keeps its NA segment row with computed rates.

    The NA price tier is a real segment of the frame: its funnel rates are
    computed exactly like every other segment's (price-independent steps) and
    its size is reported, so a consumer can exclude it from price-tier
    comparisons with full transparency. Choosing has_cart=True here keeps the
    cart->purchase denominator >= 1 (no zero-denominator edge).
    """
    df = pl.DataFrame(
        {
            "session_number": [2, 1],
            "median_price": [None, 4.9],
            # NA row: view+cart no purchase; budget row: full funnel.
            "has_view": [True, True],
            "has_cart": [True, True],
            "has_purchase": [False, True],
            "hour": [8, 8],
            "dayofweek": [1, 1],
            "month": [10, 10],
        }
    )
    result = segment_funnel(df, "price_tier")

    assert "NA" in result["segment"].to_list()
    assert result["n_sessions"].to_list() == [1, 1, 1, 1, 1, 1]
    _assert_segment_rows(
        result,
        expected={
            "NA": {"view->cart": (1, 1), "cart->purchase": (0, 1), "view->purchase": (0, 1)},
            "budget": {"view->cart": (1, 1), "cart->purchase": (1, 1), "view->purchase": (1, 1)},
        },
        n_sessions={"NA": 1, "budget": 1},
    )


@pytest.mark.parametrize(
    "by", ["price_tier", "visit_kind", "is_weekend", "daypart", "month_label"]
)
def test_segment_funnel_accepts_every_documented_family(by: str) -> None:
    """Each documented family is accepted and yields (segment x 3) tidy rows.

    The expected segment count is computed via assign_segments (its own
    contract is pinned by the label tests above), so this test only guards the
    segment_funnel contract: rows = segments x 3 steps, in documented shape.
    """
    sessions = _structured_sessions()
    result = segment_funnel(sessions, by)

    n_segments = assign_segments(sessions)[by].n_unique()
    assert result.height == n_segments * 3
    assert result.columns == [
        "step",
        "numerator",
        "denominator",
        "rate",
        "ci_low",
        "ci_high",
        "segment",
        "n_sessions",
    ]
    # segment stays Utf8 even for the boolean is_weekend family (contract).
    assert result.schema["segment"] == pl.Utf8
    assert result.schema["n_sessions"] == pl.Int64


def test_segment_funnel_rejects_unknown_family() -> None:
    """A typo'd family must fail loudly instead of segmenting on an arbitrary column."""
    with pytest.raises(ValueError, match="weekend"):
        segment_funnel(_structured_sessions(), "weekend")
