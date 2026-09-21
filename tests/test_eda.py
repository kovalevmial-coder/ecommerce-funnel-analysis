"""Tests for the data-integrity gate in src/eda.py.

The synthetic event frame is crafted so every returned key of
``integrity_report`` takes a non-trivial value: sessions that skip funnel steps,
a session id reused across two users, missing values encoded BOTH as empty
strings (the raw CSV convention) and as real nulls, and zero-price rows.
"""
from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from src.eda import integrity_report


@pytest.fixture
def events() -> pl.DataFrame:
    """A 12-event synthetic frame covering every integrity-report branch."""
    def t(day: int, hour: int = 0) -> datetime:
        return datetime(2019, 10, day, hour, 0, 0, tzinfo=timezone.utc)

    return pl.DataFrame(
        {
            "event_time": [t(1), t(1, 1), t(2), t(2, 1), t(3), t(3, 1), t(4), t(4, 1), t(5), t(5, 1), t(5, 2), t(6)],
            "event_type": [
                "view", "cart",                       # sess_a: full view->cart
                "view", "purchase",                   # sess_b: purchase WITHOUT cart
                "cart",                               # sess_c: cart WITHOUT view
                "view",                               # sess_d: only-view session
                "view", "remove_from_cart",           # sess_e: view + remove, not only-view
                "view", "cart", "purchase",           # sess_shared: SAME id, user 6 then user 7
                "view",                               # null session row
            ],
            "product_id": [1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 6, 7],
            "category_id": [101, 101, 102, 102, 103, 104, 105, 105, 106, 106, 106, 107],
            "category_code": [
                "", "electronics.phone",   # row 1 missing via empty string
                "", "",                    # sess_b missing category
                "furniture.chair",
                "apparel",
                "", "",                    # sess_e missing category
                "electronics.phone", "electronics.phone", "electronics.phone",
                None,                      # row 12 missing via real null
            ],
            "brand": [
                "", "Apple",      # row 1 missing
                "IKEA", "IKEA",
                "",               # sess_c missing brand
                "Nike",
                "Adidas", "Adidas",
                "Apple", "Apple", "Apple",
                "",               # row 12 missing brand
            ],
            "price": [0.0, 999.0, 49.9, 49.9, 15.0, 5.0, 0.0, 0.0, 700.0, 700.0, 700.0, 5.0],
            "user_id": [1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, 8],
            "user_session": [
                "sess_a", "sess_a",
                "sess_b", "sess_b",
                "sess_c",
                "sess_d",
                "sess_e", "sess_e",
                "sess_shared", "sess_shared", "sess_shared",
                None,
            ],
        }
    )


def _expected_full_report() -> dict:
    """The exact report the fixture above must produce."""
    return {
        "rows": 12,
        "distinct_users": 8,
        "distinct_sessions": 6,            # null-session rows are not sessions
        "null_session_rows": 1,
        "duplicate_sessions": 1,           # sess_shared reused by user 6 and user 7
        "event_type_counts": {"cart": 3, "purchase": 2, "remove_from_cart": 1, "view": 6},
        "brand_missing_pct": 25.0,         # rows 1, 5, 12 (3/12)
        "category_code_missing_pct": 50.0, # rows 1, 3, 4, 7, 8, 12 (6/12)
        "price_min": 0.0,
        "price_max": 999.0,
        "price_zero_share": 0.25,          # rows 1, 7, 8 (3/12)
        "zero_price_purchases": 0,
        "purchase_without_cart_sessions": 2,  # sess_b/user2 and sess_shared/user7
        "cart_without_view_sessions": 1,      # sess_c/user3
        "only_view_sessions": 1,              # sess_d/user4 (sess_e has remove_from_cart)
        "exact_duplicate_rows": 0,
    }


def test_integrity_report_full(events: pl.DataFrame) -> None:
    """The gate must return the exact expected stats for the crafted fixture."""
    assert integrity_report(events) == _expected_full_report()


def test_integrity_report_detects_exact_duplicate_rows() -> None:
    """A row repeated verbatim must be counted by the duplicate-row check."""
    dup = pl.DataFrame(
        {
            "event_time": [
                datetime(2019, 10, 1, 0, 0, 0, tzinfo=timezone.utc),
                datetime(2019, 10, 1, 0, 0, 0, tzinfo=timezone.utc),
            ],
            "event_type": ["view", "view"],
            "product_id": [9, 9],
            "category_id": [109, 109],
            "category_code": ["electronics.phone", "electronics.phone"],
            "brand": ["Apple", "Apple"],
            "price": [12.5, 12.5],
            "user_id": [9, 9],
            "user_session": ["sess_dup", "sess_dup"],
        }
    )
    report = integrity_report(dup)
    assert report["rows"] == 2
    assert report["exact_duplicate_rows"] == 1
    # The one session contains only view events, so it is an only-view session.
    assert report["only_view_sessions"] == 1
    assert report["event_type_counts"] == {"view": 2}