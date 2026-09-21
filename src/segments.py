"""Segment definitions and the session segment-label builder (Stage 03).

``assign_segments`` is the single owner of every categorical label in the
pipeline: it derives ``price_tier``, ``visit_kind``, ``is_weekend``,
``daypart`` and ``month_label`` from the raw ``session_features`` columns and
appends them to the frame. No label derivation happens anywhere else, so a
segmenting consumer only ever needs to call this function and read the new
columns.

``segment_funnel`` pivots the Stage-02 funnel rate computation into a tidy
per-segment frame by delegating the per-step maths to ``_funnel_from_flags``
(the same single source of truth ``funnel_rates`` uses), guaranteeing the
segmented and aggregate views of the same data always agree by construction.
"""
from __future__ import annotations

import polars as pl

from src.funnel import _funnel_from_flags

# Fixed merchandising bands for the price tier, in EUR. These are a company
# decision for cosmetics retail (affordable/bestseller/mid-range/prestige price
# points), NOT data-driven quantiles: tier boundaries must stay stable across
# time and catalog changes so business comparisons are a policy choice, not a
# moving target. Bands are [low, high): budget <5, mid [5,30), premium
# [30,150), luxury [150, inf).
PRICE_TIERS: dict[str, tuple[float, float]] = {
    "budget": (-float("inf"), 5.0),  # negatives and 0.0 are budget (r < 5)
    "mid": (5.0, 30.0),
    "premium": (30.0, 150.0),
    "luxury": (150.0, float("inf")),
}

# Dayparts as half-open [start, end) hour bands spanning a full day.
DAYPARTS: dict[str, tuple[int, int]] = {
    "night": (0, 6),
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 24),
}

# Raw columns assign_segments reads. The funnel booleans (has_*, net_cart)
# also arrive on session_features rows but are not inputs to segmentation.
_REQUIRED_RAW_COLUMNS = ["hour", "dayofweek", "month", "session_number", "median_price"]

# Order in which the derived labels are appended to the output frame.
_NEW_COLUMNS = ["price_tier", "visit_kind", "is_weekend", "daypart", "month_label"]


def assign_segments(sessions: pl.DataFrame) -> pl.DataFrame:
    """Append the five categorical segment labels to a session_features frame.

    Each label derives exclusively from the raw columns ``hour, dayofweek,
    month, session_number`` and ``median_price``; those columns are returned
    untouched and in their original order, with the labels appended after them.

    Label semantics:
    - ``price_tier``: ``median_price`` mapped onto the merchandising bands in
      ``PRICE_TIERS``. A null median price (a session whose events carry no
      prices at all) becomes ``"NA"`` explicitly rather than a fabricated band:
      the session stays in the funnel for price-independent steps but is
      excluded from price-tier comparisons. Reported, never silently dropped.
    - ``visit_kind``: ``"new"`` iff ``session_number == 1``, else
      ``"returning"`` (session_number is the 1-based per-user visit index from
      ``session_features``).
    - ``is_weekend``: True iff ``dayofweek`` is 5 or 6 (polars convention
      0=Mon..6=Sun, inherited unchanged from ``session_features``).
    - ``daypart``: ``hour`` mapped onto the ``DAYPARTS`` half-open bands.
    - ``month_label``: ``"{year}-{month:02d}"`` where the year follows the
      documented dataset-scope rule — this dataset spans exactly Oct 2019-Feb
      2020, so months 10-12 map to 2019 and months 1-2 to 2020; any other
      month is ambiguous and raises ValueError.

    Args:
        sessions: frame exposing at least ``hour``, ``dayofweek``, ``month``,
            ``session_number`` and ``median_price`` (the ``session_features``
            contract).

    Returns:
        A new frame with the original columns unchanged (same order) followed
        by ``price_tier:str``, ``visit_kind:str``, ``is_weekend:bool``,
        ``daypart:str``, ``month_label:str``. The input frame is not mutated.

    Raises:
        ValueError: if any raw input column is missing (listed), or if a
            ``month`` value is outside {1, 2, 10, 11, 12} so its year cannot be
            decided under the dataset-scope rule.
    """
    missing = [col for col in _REQUIRED_RAW_COLUMNS if col not in sessions.columns]
    if missing:
        raise ValueError(
            f"sessions is missing required columns: {missing}; "
            f"got {sessions.columns}"
        )

    # Guard the year scope before deriving any label: an ambiguous month would
    # otherwise produce a wrong month_label silently. The union of unique
    # values is enough — pick it eagerly so the ValueError names the offending
    # value rather than just the column.
    bad_months = [int(m) for m in sessions["month"].unique().to_list() if int(m) not in (1, 2, 10, 11, 12)]
    if bad_months:
        raise ValueError(
            f"month values {bad_months} are outside the documented dataset "
            "scope (Oct 2019-Feb 2020); the year rule cannot map them."
        )

    tier = (
        pl.when(pl.col("median_price").is_null())
        .then(pl.lit("NA"))
        # Exhaustive single chain over the PRICE_TIERS bands: no gaps (a price
        # below 5, incl. 0.0/negatives, is budget) and no overlaps (each
        # threshold belongs to exactly one tier). Kept in lockstep with the
        # declared constants; the property test sweeps the whole price scope.
        .when(pl.col("median_price") < 5.0)
        .then(pl.lit("budget"))
        .when(pl.col("median_price") < 30.0)
        .then(pl.lit("mid"))
        .when(pl.col("median_price") < 150.0)
        .then(pl.lit("premium"))
        .otherwise(pl.lit("luxury"))
        .alias("price_tier")
    )

    visit_kind = pl.when(pl.col("session_number") == 1).then(pl.lit("new")).otherwise(pl.lit("returning")).alias("visit_kind")

    is_weekend = pl.col("dayofweek").is_in([5, 6]).alias("is_weekend")

    # Single exhaustive chain over DAYPARTS in hour order: the chain is total
    # (every hour value maps to a part), so there is no silent gap. The 0..23
    # contract itself is enforced upstream — ``hour`` is extracted from a real
    # datetime by ``session_features`` — not re-checked here.
    daypart = (
        pl.when(pl.col("hour") < 6)
        .then(pl.lit("night"))
        .when(pl.col("hour") < 12)
        .then(pl.lit("morning"))
        .when(pl.col("hour") < 18)
        .then(pl.lit("afternoon"))
        .otherwise(pl.lit("evening"))
        .alias("daypart")
    )

    # Year comes from the documented dataset-scope rule: months 10-12 are the
    # tail of 2019, months 1-2 the head of 2020 (kept as a vectorized
    # when/otherwise so the 4.5M-row real frame is never pushed through
    # Python per-row). The raw month is left untouched in the output frame.
    year = (pl.when(pl.col("month") >= 10).then(pl.lit(2019)).otherwise(pl.lit(2020)))
    month_label = (
        pl.concat_str([year.cast(pl.Utf8), pl.lit("-"), pl.col("month").cast(pl.Utf8).str.zfill(2)], separator="")
        .alias("month_label")
    )

    return sessions.with_columns(
        # Appending is enough: polars explicitly refuses to overwrite a
        # pre-existing column with the same name, which double-guards the
        # "labels only from raw columns, raw columns untouched" rule.
        [tier, visit_kind, is_weekend, daypart, month_label]
    )


# The only ``by`` families segment_funnel is allowed to segment on. Pinning the
# set makes a typo a loud error instead of silently segmenting on some other
# column the frame happens to carry.
_SEGMENT_FAMILIES = frozenset(
    {"price_tier", "visit_kind", "is_weekend", "daypart", "month_label"}
)


def segment_funnel(sessions: pl.DataFrame, by: str) -> pl.DataFrame:
    """Compute the funnel rates separately within each value of a segment column.

    One rate per (segment, step) for the three ``_FUNNEL_STEPS`` transitions,
    conditioned within each segment's sessions. The per-step math is delegated
    to ``_funnel_from_flags`` — the same single source of truth ``funnel_rates``
    uses — so a segment's rates are *exactly* what ``funnel_rates`` would
    return for that segment's rows in isolation; the segmented and aggregate
    views of the same data cannot disagree.

    ``n_sessions`` is the full size of the segment (every session that carries
    the segment label), which is the practical-significance input for Stage 04.
    It must NOT be read as a step denominator: each step's denominator is the
    step-specific subset of the segment (e.g. ``cart->purchase`` divides only
    the segment's carted sessions), exactly as in ``funnel_rates``.

    The ``"NA"`` price tier (sessions with no priced events) is a real segment
    of this frame: its rates are computed like every other segment's and its
    size is reported, because its sessions still participate in the
    price-independent funnel steps. Consumers of confirmatory price-tier
    comparisons exclude it with an explicit documented filter — it is never
    silently dropped here.

    Args:
        sessions: frame with the boolean ``has_view``/``has_cart``/
            ``has_purchase`` columns, plus either the ``by`` column already
            present or the raw columns (``hour, dayofweek, month,
            session_number, median_price``) so ``assign_segments`` can derive
            it. A frame carrying partial label columns is not supported (the
            repository treats ``assign_segments`` as the single label owner,
            and polars fails loudly on a duplicated column name).
        by: one of ``price_tier``, ``visit_kind``, ``is_weekend``, ``daypart``,
            ``month_label``.

    Returns:
        Tidy frame with ``funnel_rates``' columns (``step``, ``numerator``,
        ``denominator``, ``rate``, ``ci_low``, ``ci_high``) plus ``segment:str``
        and ``n_sessions:int``. One segment carries its three steps in
        ``_FUNNEL_STEPS`` order; segments are ordered by ascending label
        (strings lexicographically, booleans False-then-True), which keeps the
        frame deterministic for the fixed 5-value families.

    Raises:
        ValueError: if ``by`` is not one of the documented families, or (via
            ``_funnel_from_flags``/``assign_segments``) if the frame lacks the
            required flag columns or raw columns.
    """
    if by not in _SEGMENT_FAMILIES:
        raise ValueError(
            f"by={by!r} is not a segment family; expected one of "
            f"{sorted(_SEGMENT_FAMILIES)}"
        )

    # assign_segments is the single owner of the label columns. If the input
    # already carries ``by`` the label is trusted as-is (the caller is assumed
    # to have produced it via assign_segments); otherwise the owner is called
    # here so segment_funnel never invents a label derivation of its own.
    labelled = assign_segments(sessions) if by not in sessions.columns else sessions

    # partition_by is a single pass over the frame (one partition per segment
    # value) with partitions kept in input order; the explicit re-sort below is
    # what makes the output order a contract rather than an accident.
    parts: list[tuple[object, pl.DataFrame]] = []
    for sub in labelled.partition_by(by):
        segment_label = sub[by][0]
        funnel = _funnel_from_flags(sub).with_columns(
            pl.lit(segment_label).alias("segment"),
            # Height of the partition == number of sessions in the segment;
            # cf. docstring: the funnel denominators come from _funnel_from_flags
            # (step-specific subsets), never from this column.
            pl.lit(sub.height, dtype=pl.Int64).alias("n_sessions"),
        )
        parts.append((segment_label, funnel))

    parts.sort(key=lambda part: part[0])
    return pl.concat([funnel for _, funnel in parts])
