"""Tests for the session-level funnel feature builder in src/funnel.py.

The fixture is 19 event rows spanning 6 crafted sessions (>=5 required by the
plan) plus one null-session row. Expected values are hand-derived, not computed
from the code under test: medians of exact float32-representable prices, session
numbers ordered by *first* event time, and daypart/weekday/month values pinned to
the session's *first* event.
"""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from src.funnel import session_features

# Output column contract (plan Task 1), in exact order.
_CONTRACT_COLUMNS = [
    "user_session",
    "user_id",
    "month",
    "dayofweek",
    "hour",
    "session_number",
    "n_view",
    "n_cart",
    "n_remove",
    "n_purchase",
    "has_view",
    "has_cart",
    "has_purchase",
    "net_cart",
    "median_price",
    "n_products",
]

_CONTRACT_DTYPES = {
    "user_session": pl.Utf8,
    "user_id": pl.Int64,
    "month": pl.Int64,
    "dayofweek": pl.Int64,
    "hour": pl.Int64,
    "session_number": pl.Int64,
    "n_view": pl.Int64,
    "n_cart": pl.Int64,
    "n_remove": pl.Int64,
    "n_purchase": pl.Int64,
    "has_view": pl.Boolean,
    "has_cart": pl.Boolean,
    "has_purchase": pl.Boolean,
    "net_cart": pl.Boolean,
    "median_price": pl.Float64,
    "n_products": pl.Int64,
}


def _make_events() -> pl.DataFrame:
    """Build the crafted events fixture.

    Deliberate design choices behind the rows:

    - s1 spans two dayparts (08:15 and 20:05) and two hours -> pins that the
      session time columns come from the *first* event, not last/mode.
    - s2's last event (Nov 08) is *after* s3's first event (Nov 07): sorts by
      max/mean event time would mis-order them, so the session numbers for user
      101 (s1=1, s2=2, s3=3) prove ordering by first event time.
    - s3 has all-null prices -> median_price must stay null (sessions without
      priced events), simulating the rule that null medians are not filled.
    - s5 (cart 2 > remove 1) and s6 (cart 2 == remove 2) pin net_cart's strict
      n_cart > n_remove semantics in both directions; s5 and s6 are also the
      multi-product sessions (n_products 2 each, multi-price medians).
    """
    rows = [
        # (event_time, event_type, product_id, price, user_id, user_session)
        # s1 (user 101): first event 2019-11-01 08:15 (Fri, dayofweek 4)
        ("2019-11-01 08:15:00", "view", 1, 10.0, 101, "s1"),
        ("2019-11-01 08:40:00", "cart", 1, 10.0, 101, "s1"),
        ("2019-11-01 09:00:00", "purchase", 1, 10.0, 101, "s1"),
        ("2019-11-01 20:05:00", "view", 1, 10.0, 101, "s1"),
        # s2 (user 101): first event 2019-11-03 18:00 (Sun, dayofweek 6); last
        # event Nov 08 is deliberately later than s3's start
        ("2019-11-03 18:00:00", "view", 2, 5.0, 101, "s2"),
        ("2019-11-03 18:30:00", "cart", 2, 5.0, 101, "s2"),
        ("2019-11-08 09:00:00", "remove_from_cart", 2, 5.0, 101, "s2"),
        # s3 (user 101): first event 2019-11-07 12:00 (Thu, dayofweek 3);
        # all prices null -> null median
        ("2019-11-07 12:00:00", "view", 5, None, 101, "s3"),
        ("2019-11-07 12:20:00", "purchase", 5, None, 101, "s3"),
        # s4 (user 202): one cart event 2019-11-10 09:00 (Sun, dayofweek 6)
        ("2019-11-10 09:00:00", "cart", 7, 30.0, 202, "s4"),
        # s5 (user 303): cart 2 > remove 1 -> net_cart True; 2 products
        ("2019-12-05 14:00:00", "cart", 10, 55.5, 303, "s5"),
        ("2019-12-05 14:05:00", "remove_from_cart", 10, 55.5, 303, "s5"),
        ("2019-12-05 14:10:00", "cart", 11, 60.0, 303, "s5"),
        ("2019-12-05 14:15:00", "purchase", 11, 60.0, 303, "s5"),
        # s6 (user 404): cart 2 == remove 2 -> net_cart False; 2 products
        ("2019-12-25 10:00:00", "cart", 20, 20.0, 404, "s6"),
        ("2019-12-25 10:15:00", "remove_from_cart", 20, 20.0, 404, "s6"),
        ("2019-12-25 10:30:00", "cart", 21, 25.0, 404, "s6"),
        ("2019-12-25 10:45:00", "remove_from_cart", 21, 25.0, 404, "s6"),
        # null session: belongs to no session, must be dropped from the output
        ("2019-11-15 00:00:00", "view", 50, 1.0, 999, None),
    ]
    return pl.DataFrame(
        rows,
        schema=[
            "event_time",
            "event_type",
            "product_id",
            "price",
            "user_id",
            "user_session",
        ],
        orient="row",
    ).with_columns(
        # Mirror the real loader's output: timezone-aware UTC microsecond
        # datetimes and float32 prices (src/data_io raw schema).
        pl.col("event_time").str.to_datetime(time_zone="UTC", time_unit="us"),
        pl.col("price").cast(pl.Float32),
    )


def test_session_features_contract_columns_and_dtypes() -> None:
    """The output frame must have exactly the contract columns, in order."""
    result = session_features(_make_events())

    assert result.columns == _CONTRACT_COLUMNS
    assert result.schema == _CONTRACT_DTYPES


def test_session_features_hand_derived_row_values() -> None:
    """Every output cell must match the hand-computed expectations."""
    result = session_features(_make_events())
    got = {row["user_session"]: row for row in result.iter_rows(named=True)}

    # 6 sessions survive; the null-session row is absent by construction.
    assert set(got) == {"s1", "s2", "s3", "s4", "s5", "s6"}

    # Hand-derived expectations. median_price None for s3 (no priced events);
    # medians for s5/s6 are float32-exact averages (57.75, 22.5).
    expected = {
        "s1": {"user_session": "s1",  # cart>remove; view+purchase; pin hour=8 (first event, not 20)
            "user_id": 101, "month": 11, "dayofweek": 4, "hour": 8,
            "session_number": 1, "n_view": 2, "n_cart": 1, "n_remove": 0,
            "n_purchase": 1, "has_view": True, "has_cart": True,
            "has_purchase": True, "net_cart": True, "median_price": 10.0,
            "n_products": 1,
        },
        "s2": {"user_session": "s2",  # cart==remove -> net_cart False despite has_cart True
            "user_id": 101, "month": 11, "dayofweek": 6, "hour": 18,
            "session_number": 2, "n_view": 1, "n_cart": 1, "n_remove": 1,
            "n_purchase": 0, "has_view": True, "has_cart": True,
            "has_purchase": False, "net_cart": False, "median_price": 5.0,
            "n_products": 1,
        },
        "s3": {"user_session": "s3",  # null median; purchase without cart
            "user_id": 101, "month": 11, "dayofweek": 3, "hour": 12,
            "session_number": 3, "n_view": 1, "n_cart": 0, "n_remove": 0,
            "n_purchase": 1, "has_view": True, "has_cart": False,
            "has_purchase": True, "net_cart": False, "median_price": None,
            "n_products": 1,
        },
        "s4": {"user_session": "s4",  # cart without view
            "user_id": 202, "month": 11, "dayofweek": 6, "hour": 9,
            "session_number": 1, "n_view": 0, "n_cart": 1, "n_remove": 0,
            "n_purchase": 0, "has_view": False, "has_cart": True,
            "has_purchase": False, "net_cart": True, "median_price": 30.0,
            "n_products": 1,
        },
        "s5": {"user_session": "s5",  # cart 2 > remove 1 -> net_cart True; multi-product, multi-price
            "user_id": 303, "month": 12, "dayofweek": 3, "hour": 14,
            "session_number": 1, "n_view": 0, "n_cart": 2, "n_remove": 1,
            "n_purchase": 1, "has_view": False, "has_cart": True,
            "has_purchase": True, "net_cart": True, "median_price": 57.75,
            "n_products": 2,
        },
        "s6": {"user_session": "s6",  # cart 2 == remove 2 -> net_cart False; multi-product
            "user_id": 404, "month": 12, "dayofweek": 2, "hour": 10,
            "session_number": 1, "n_view": 0, "n_cart": 2, "n_remove": 2,
            "n_purchase": 0, "has_view": False, "has_cart": True,
            "has_purchase": False, "net_cart": False, "median_price": 22.5,
            "n_products": 2,
        },
    }

    for session, want in expected.items():
        assert got[session] == pytest.approx(want)


def test_session_features_tied_first_event_gets_ordinal_numbers() -> None:
    """Two sessions of one user with identical first event times must rank 1,2.

    Guards an accidental switch to dense ranking (which would give both 1);
    ordinal ranking guarantees distinct contiguous visit indices no matter how
    the tie resolves (documented as arbitrary input order).
    """
    events = pl.DataFrame(
        {
            "event_time": [datetime(2019, 11, 1, 8, 0, tzinfo=timezone.utc)] * 4,
            "event_type": ["view"] * 4,
            "user_id": [1, 1, 1, 1],
            "user_session": ["a", "a", "b", "b"],
            "price": [1.0] * 4,
            "product_id": [1] * 4,
        }
    ).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))

    result = session_features(events)
    assert sorted(result["session_number"].to_list()) == [1, 2]
    assert result["session_number"].n_unique() == 2


def test_session_features_drops_null_session_rows() -> None:
    """Rows with a null user_session cannot belong to a session and are dropped."""
    result = session_features(_make_events())

    assert result.height == 6
    # The null-session row was the only event of user 999.
    assert 999 not in result["user_id"].to_list()


def test_session_features_raises_on_missing_columns() -> None:
    """Missing required inputs must fail fast and list the absent columns."""
    events = _make_events().drop(["price", "product_id"])

    with pytest.raises(ValueError, match="price"):
        session_features(events)

    try:
        session_features(events)
    except ValueError as err:
        assert "price" in str(err)
        assert "product_id" in str(err)