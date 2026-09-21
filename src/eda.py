"""Data-integrity gate for raw event frames.

``integrity_report`` is the first check any downstream analysis must pass
(golden standard 1: data integrity before conclusions). It answers three
questions that govern everything later: is the data volume what we expect, how
do we define a session, and how "clean" are the funnel-step columns.
"""
from __future__ import annotations

import polars as pl


def _missing_pct(events: pl.DataFrame, col: str) -> float:
    """Percent of rows whose value is missing, counting "" as missing to 1 dp.

    The raw CSVs encode missing cells as empty strings with no NULL typecast at
    load time, so a plain null_count() would under-report by roughly 0.
    """
    n = events.height
    missing = events.filter(pl.col(col).is_null() | (pl.col(col).eq(""))).height
    return round(100.0 * missing / n, 1)


def integrity_report(events: pl.DataFrame) -> dict:
    """Compute the data-integrity statistics for an event frame.

    Returns a dict with these keys:

    - ``rows``: total event rows.
    - ``distinct_users``: distinct ``user_id`` values.
    - ``distinct_sessions``: distinct non-null ``user_session`` values.
    - ``null_session_rows``: rows with a null ``user_session`` (not a real
      session and therefore excluded from all session-level counts).
    - ``duplicate_sessions``: session ids mapped to more than one ``user_id``
      (session-id reuse across users — grouping by ``user_session`` alone would
      wrongly merge those journeys).
    - ``event_type_counts``: row counts per ``event_type``.
    - ``brand_missing_pct`` / ``category_code_missing_pct``: percent of rows
      missing the value (empty string treated as missing), 0..100.
    - ``price_min`` / ``price_max`` / ``price_zero_share``: price range and the
      share of rows priced at exactly 0.0.
    - ``zero_price_purchases``: purchase rows with price 0.0 — any count > 0
      means free products appear in the funnel and should be reported as such.
    - ``purchase_without_cart_sessions``: sessions reaching purchase with no
      cart event (19% on the real data — a funnel diagnostic, not a drop).
    - ``cart_without_view_sessions``: sessions with a cart but no view event.
    - ``only_view_sessions``: sessions whose events are all ``view``.
    - ``exact_duplicate_rows``: verbatim-duplicated rows; 1.1M on the real
      data, so always computed rather than assumed zero.

    Session-level counts group by ``(user_session, user_id)``: a session id is
    only meaningful per user (see ``duplicate_sessions``), and null-session
    rows are dropped because they cannot be assigned to a session.
    """
    n = events.height

    # Row-level volume and identity.
    distinct_sessions = events["user_session"].drop_nulls().n_unique()
    n_null_sessions = events["user_session"].null_count()
    exact_duplicate_rows = n - events.unique().height

    # Per-type event counts; group_by().len() emits columns (event_type, len).
    event_type_counts = {
        row["event_type"]: row["len"]
        for row in events.group_by("event_type").len().iter_rows(named=True)
    }

    # Session-level funnel flags over the effective (session, user) unit.
    sess_events = (
        events.filter(pl.col("user_session").is_not_null())
        .group_by("user_session", "user_id")
        .agg(
            pl.col("event_type").eq("purchase").any().alias("has_purchase"),
            pl.col("event_type").eq("cart").any().alias("has_cart"),
            pl.col("event_type").eq("view").any().alias("has_view"),
            pl.col("event_type").ne("view").any().alias("has_non_view"),
        )
    )

    # Sessions whose event-id is reused by a second user are the one genuine
    # "duplicate session" in this schema; they flag the unit-of-analysis risk.
    duplicate_sessions = (
        events.filter(pl.col("user_session").is_not_null())
        .group_by("user_session")
        .agg(pl.col("user_id").n_unique())
        .filter(pl.col("user_id") > 1)
        .height
    )

    return {
        "rows": n,
        "distinct_users": events["user_id"].n_unique(),
        "distinct_sessions": distinct_sessions,
        "null_session_rows": n_null_sessions,
        "duplicate_sessions": duplicate_sessions,
        "event_type_counts": event_type_counts,
        "brand_missing_pct": _missing_pct(events, "brand"),
        "category_code_missing_pct": _missing_pct(events, "category_code"),
        "price_min": float(events["price"].min()),
        "price_max": float(events["price"].max()),
        "price_zero_share": round(events.filter(pl.col("price").eq(0.0)).height / n, 4),
        "zero_price_purchases": events.filter(
            pl.col("event_type").eq("purchase") & pl.col("price").eq(0.0)
        ).height,
        "purchase_without_cart_sessions": sess_events.filter(
            pl.col("has_purchase") & ~pl.col("has_cart")
        ).height,
        "cart_without_view_sessions": sess_events.filter(
            pl.col("has_cart") & ~pl.col("has_view")
        ).height,
        "only_view_sessions": sess_events.filter(
            pl.col("has_view") & ~pl.col("has_non_view")
        ).height,
        "exact_duplicate_rows": exact_duplicate_rows,
    }