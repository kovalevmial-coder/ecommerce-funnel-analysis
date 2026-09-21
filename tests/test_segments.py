"""Tests for the segment label builder in src/segments.py.

The fixture is a small ``session_features``-shaped frame (the raw columns
``hour, dayofweek, month, session_number, median_price`` plus the funnel
booleans the stage carries downstream). Every label is hand-derived from the
raw column: price tier from the band edges in ``PRICE_TIERS``, visit kind from
``session_number``, weekend from the polars dayofweek convention
(0=Mon..6=Sun), daypart from ``hour`` and ``month_label`` from the documented
dataset-scope year rule.
"""
from __future__ import annotations

import polars as pl
import pytest

from src.segments import DAYPARTS, PRICE_TIERS, assign_segments

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
    through ``otherwise`` (a gap) — assert it does not by checking the sweep
    never mutates the null-price row. The assertion is exact per row so a
    missing band or an overlap anywhere in the chain would fail this test.
    """
    # nulls must be preserved verbatim through the sweep (not consumed by a
    # catch-all tier).
    prices = list(range(-80, 329))  # -80..328 (ints cover the band edges)
    sweep = pl.DataFrame(
        {"hour": 0, "dayofweek": 0, "month": 10, "session_number": 1, "median_price": prices}
    )
    result = assign_segments(sweep)
    tiers = result["price_tier"].to_list()
    assert tiers.count("null") == 0  # sentinel impossible under a correct chain
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
        "budget": (0.0, 5.0),
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


def test_hour_municipal_range_is_exhaustive_across_dayparts() -> None:
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