"""Session-level funnel features and funnel conversion rates (Stage 02).

``session_features`` collapses the raw event stream into one row per
``user_session`` so later stages can segment and test the view->cart->purchase
funnel on a clean unit of analysis. Implementing the units correctly here
listens to the Stage-01 integrity findings: a session id is only meaningful in
company of its ``user_id``, and rows without a session id are not a session.
"""
from __future__ import annotations

import polars as pl

# Columns the events frame must expose for the pipeline to be computable.
# ``product_id`` is required beyond the five funnel booleans because
# ``n_products`` counts distinct products per session; without it the contract
# is unfulfillable.
_REQUIRED_COLUMNS = [
    "event_time",
    "event_type",
    "user_id",
    "user_session",
    "price",
    "product_id",
]

# Output columns in the exact contractual order (plan Task 1, lines 24-25).
# ``first_event_time`` is an internal intermediate, never part of the contract.
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


def session_features(events: pl.DataFrame) -> pl.DataFrame:
    """Build one row of funnel features per user_session.

    The plan contract asks for one row per ``user_session``, so the group key
    here is ``user_session`` alone. The honest unit of analysis is the
    ``(user_session, user_id)`` pair: 272 real session ids are reused across
    different users (integrity scan), so the row for such an id merges two
    journeys and keeps the ``user_id`` of the session's *earliest* event (the
    first user to open the id; ties resolve arbitrarily to one of the tied
    users). Rows with a null ``user_session`` cannot belong to any session and
    are dropped. At ~4.5M sessions this contamination is <0.01% and is
    deliberate; future consumers must know the row for a shared id is polluted.

    Per-session attributes are derived from the session's *first* event:
    ``month`` (1-12), ``dayofweek`` (0=Mon..6=Sun, python convention — polars
    ``dt.weekday`` is 1=Mon, hence the -1) and ``hour`` reflect the session
    start time; assigning instead from the last or modal event would mislabel
    a session that spans dayparts, and Stage 03's segmenter reads these raw
    columns directly.

    ``n_*`` are per-event-type counts; ``has_*`` are reached-step indicators
    (>=1 event of that type). ``net_cart`` is the conservative robustness
    "cart survived the session" flag: True iff ``n_cart > n_remove``, i.e. at
    least one cart event remains after cancelling ``remove_from_cart``
    one-for-one. The primary funnel deliberately uses ``has_cart`` (gross, not
    net), so ``net_cart`` never replaces it — it exists only to quantify how
    much ``remove_from_cart`` erodes the funnel in the robustness stage.

    ``session_number`` is the 1-based visit index of the session for its
    ``user_id``, ordered by the session's first ``event_time`` — the "new vs
    returning visit" segmentation input. Ties (two sessions whose earliest
    event times are exactly equal) are ranked in arbitrary input order; such
    sessions are observationally indistinguishable at microsecond precision,
    so any tie order is valid.

    ``median_price`` is the median of the session's non-null ``price`` values,
    cast to Float64 (the raw events store price as Float32). A session whose
    events all carry a null price has a null median — it is intentionally NOT
    filled, because filling would fabricate a priced journey.

    Args:
        events: frame with at least ``event_time`` (datetime), ``event_type``,
            ``user_id``, ``user_session``, ``price`` and ``product_id``.

    Returns:
        Frame with the contract columns, one row per non-null ``user_session``,
        sorted by (``user_id``, ``session_number``) for a deterministic order.

    Raises:
        ValueError: if any required column is missing, listing the absent ones
            (fail-fast: better a loud error than silently wrong aggregations).
    """
    missing = [col for col in _REQUIRED_COLUMNS if col not in events.columns]
    if missing:
        raise ValueError(
            f"events is missing required columns: {missing}; "
            f"got {events.columns}"
        )

    sess = (
        events.filter(pl.col("user_session").is_not_null())
        .group_by("user_session")
        .agg(
            # user of the session's earliest event: per-group sort of user_id by
            # event_time then first — the only user for ~all sessions, and the
            # documented tie-break for the 272 shared ids.
            pl.col("user_id").sort_by(pl.col("event_time")).first().alias("user_id"),
            # Internal anchor for session_number ordering and the time columns.
            pl.col("event_time").min().alias("first_event_time"),
            pl.col("event_type").eq("view").sum().cast(pl.Int64).alias("n_view"),
            pl.col("event_type").eq("cart").sum().cast(pl.Int64).alias("n_cart"),
            pl.col("event_type")
            .eq("remove_from_cart")
            .sum()
            .cast(pl.Int64)
            .alias("n_remove"),
            pl.col("event_type").eq("purchase").sum().cast(pl.Int64).alias("n_purchase"),
            pl.col("event_type").eq("view").any().alias("has_view"),
            pl.col("event_type").eq("cart").any().alias("has_cart"),
            pl.col("event_type").eq("purchase").any().alias("has_purchase"),
            # median over non-null prices; all-null session stays null (not
            # filled). Cast Float64: float32 input must not leak its dtype.
            pl.col("price").median().cast(pl.Float64).alias("median_price"),
            pl.col("product_id").n_unique().cast(pl.Int64).alias("n_products"),
        )
        .with_columns(
            # Session start time drips into the raw columns Stage 03 segments on.
            pl.col("first_event_time").dt.month().cast(pl.Int64).alias("month"),
            (pl.col("first_event_time").dt.weekday() - 1).cast(pl.Int64).alias("dayofweek"),
            pl.col("first_event_time").dt.hour().cast(pl.Int64).alias("hour"),
        )
        .with_columns(
            # 1-based visit index per user by first event time; ordinal rank is
            # contiguous even with ties (which resolve to arbitrary order).
            pl.col("first_event_time")
            .rank("ordinal")
            .over("user_id")
            .cast(pl.Int64)
            .alias("session_number"),
        )
        .with_columns(
            # Strict one-for-one cancellation: net_cart <=> a cart survives.
            (pl.col("n_cart") > pl.col("n_remove")).alias("net_cart"),
        )
        .select(_CONTRACT_COLUMNS)
        .sort(["user_id", "session_number"])
    )
    return sess


# Stages of the funnel, in the order every consumer (tests, notebooks, README)
# must see them. Each entry is (step label, converting-step indicator,
# basis-step indicator): the numerator is the boolean AND of the two (a session
# that reached the later step after the earlier one), the denominator the count
# of the basis step alone.
_FUNNEL_STEPS = [
    ("view->cart", "has_cart", "has_view"),
    ("cart->purchase", "has_purchase", "has_cart"),
    ("view->purchase", "has_purchase", "has_view"),
]


def _wilson_ci(num: int, den: int, z: float = 1.96) -> tuple[float, float]:
    """Score (Wilson) 95% confidence interval for a binomial proportion.

    Returns ``(ci_low, ci_high)`` for the observed proportion ``num / den``,
    using the Wilson score interval with critical value ``z`` (1.96 ~ 95%).
    The score method is preferred over the Wald interval here: Wald's normal
    approximation collapses to a point exactly when ``num == den``, can invert
    outside (0, 1) for proportions near the edges, and is badly sized for small
    ``den``. Wilson keeps coverage near nominal for small and edge proportions,
    and by construction confines the interval to (0, 1) for ``0 < num < den``
    (equal-weight: both endpoints fall strictly inside). It degrades to the
    exact boundary value (0 or 1) only when the data itself is at a boundary
    (``num == 0`` / ``num == den``) — that is honest, not a special case.

    This helper is the Stage-02 inline placeholder.

    # TODO(stage-04): replace this local implementation with
    # src/inference.wilson_ci and delete the marker.

    Args:
        num: successful trials (non-negative integer).
        den: total trials; must satisfy 1 <= num <= den.
        z: standard-normal critical value (1.96 for a two-sided 95% CI).

    Returns:
        ``(ci_low, ci_high)`` floats, with ``ci_low <= num / den <= ci_high``
        and both endpoints within [0, 1] by construction.
    """
    if den < 1 or num > den:
        raise ValueError(f"wilson_ci requires 0 <= num <= den, got num={num}, den={den}")
    p = num / den
    # Wilson's closed form: re-centre the sample proportion towards z^2/(2n)
    # and widen the radius by the z^2/(4n^2) variance term, all normalised by
    # (1 + z^2/n). Rewriting the textbook equation this way makes the boundary
    # behaviour transparent (it stays a ratio of non-negative terms).
    centre = p + (z**2) / (2 * den)
    radius = z * ((p * (1 - p) / den) + (z**2) / (4 * den**2)) ** 0.5
    normalise = 1 + (z**2) / den
    ci_low = max(0.0, (centre - radius) / normalise)
    ci_high = min(1.0, (centre + radius) / normalise)
    return ci_low, ci_high


def funnel_rates(sessions: pl.DataFrame) -> pl.DataFrame:
    """Compute the view->cart->purchase conversion funnel with Wilson CIs.

    One rate per step, each derived from boolean aggregates over the session
    features (one row per session output by ``session_features``):

    - ``view->cart``: carted-and-viewed / viewed. A viewer who carted anywhere
      in any of their sessions is a converted viewer; cart events without a
      prior view in the same session are genuine converts of the view step (the
      view may predate the cart) and must count both numerator and denominator.
    - ``cart->purchase``: purchased-and-carted / carted.
    - ``view->purchase``: purchased-and-viewed / viewed. The numerator counts
      *any* purchaser who saw a product, regardless of whether they carted —
      i.e. it deliberately includes "jumpers" who bought without adding to the
      cart (direct-purchase journeys). Excluding them would overstate the drop
      caused by the cart step; the 10 direct purchasers in the real data are a
      small but real segment this step must not sweep under the rug.

    Confidence intervals come from the Wilson score method (see ``_wilson_ci``),
    which stays honest for small step counts where the Wald normal
    approximation would be point-esque or out of range.

    Args:
        sessions: frame with boolean ``has_view``, ``has_cart`` and
            ``has_purchase`` columns (as produced by ``session_features``).

    Returns:
        Exactly three rows (``view->cart``, ``cart->purchase``,
        ``view->purchase``) with columns ``step``, ``numerator``,
        ``denominator``, ``rate`` (num/den), ``ci_low``, ``ci_high``.

    Raises:
        ValueError: if any of the three ``has_*`` step columns is missing,
            listing the absent ones (fail-fast before silently empty counts).
    """
    required = ("has_view", "has_cart", "has_purchase")
    missing = [col for col in required if col not in sessions.columns]
    if missing:
        raise ValueError(
            f"sessions is missing required columns: {missing}; "
            f"got {sessions.columns}"
        )

    rows = []
    for step, conv_col, base_col in _FUNNEL_STEPS:
        # Co-occurrence via boolean AND: a session converted the step iff it
        # reached *both* the earlier (basis) and the later (converting) stage
        # in any order. Summing only ``conv_col`` would count jumpers who never
        # reached the basis at all (e.g. purchasers who never viewed).
        numerator = int((sessions[conv_col] & sessions[base_col]).sum())
        denominator = int(sessions[base_col].sum())
        rate = numerator / denominator
        ci_low, ci_high = _wilson_ci(numerator, denominator)
        rows.append(
            {
                "step": step,
                "numerator": numerator,
                "denominator": denominator,
                "rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "step": pl.Utf8,
            "numerator": pl.Int64,
            "denominator": pl.Int64,
            "rate": pl.Float64,
            "ci_low": pl.Float64,
            "ci_high": pl.Float64,
        },
    )
