"""Lazy, schema-typed loading of raw e-commerce event CSVs.

The raw monthly files (2019-Oct.csv ... 2020-Feb.csv) are 200+ MB each, so we
scan them lazily, apply a fixed schema at the boundary, and never materialize
more than the streamed pipeline requires.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import polars as pl

# Fixed dtypes for every raw CSV. `event_time` stays a string here because the
# files literally end the timestamp in " UTC"; the conversion to a timezone-aware
# Datetime happens in load_events_lazy.
_RAW_SCHEMA = {
    "event_time": pl.Utf8,
    "event_type": pl.Utf8,
    "product_id": pl.Int64,
    "category_id": pl.Int64,
    "category_code": pl.Utf8,
    "brand": pl.Utf8,
    "price": pl.Float32,
    "user_id": pl.Int64,
    "user_session": pl.Utf8,
}


def _month_sort_key(path: Path) -> datetime:
    """Extract a chronological key from a ``%Y-%b.csv`` filename.

    Month names sort lexicographically (Nov < Oct in ASCII), so a plain
    ``sorted()`` would not order the files by time. Parsing the prefix gives a
    real chronological key; a malformed filename fails loudly rather than being
    silently misplaced.
    """
    return datetime.strptime(path.name, "%Y-%b.csv")


def event_files(raw_dir: Path) -> list[Path]:
    """Return the raw monthly event CSVs in chronological order.

    Args:
        raw_dir: directory holding the raw monthly CSV exports.

    Returns:
        List of every ``*.csv`` inside ``raw_dir``, ordered by their
        ``YYYY-Mon`` filename prefix.
    """
    return sorted(raw_dir.glob("*.csv"), key=_month_sort_key)


def load_events_lazy(paths: Sequence[Path]) -> pl.LazyFrame:
    """Lazily concatenate raw event CSVs into one schema-typed LazyFrame.

    Each file is scanned with a fixed schema and a streaming-friendly low-memory
    setting; nothing is read into memory until the caller collects. The trailing
    " UTC" suffix on ``event_time`` is stripped before parsing so polars can
    produce a timezone-aware UTC timestamp.

    Args:
        paths: raw monthly CSV paths, typically from ``event_files``.

    Returns:
        A lazy frame with ``event_time`` as ``pl.Datetime(us, UTC)`` and the
        remaining columns typed per ``_RAW_SCHEMA``.
    """
    frames = [
        pl.scan_csv(path, schema=_RAW_SCHEMA, low_memory=True) for path in paths
    ]
    return pl.concat(frames, how="vertical").with_columns(
        pl.col("event_time")
        .str.strip_suffix(" UTC")
        .str.to_datetime(time_zone="UTC", time_unit="us")
    )


def write_events_parquet(raw_dir: Path, out_file: Path) -> None:
    """Stream all raw events into a single parquet file.

    Uses polars' streaming sink so the full frame is never held in memory.
    ``sink_parquet`` replaces an existing ``out_file`` (drop semantics); dtypes
    are inherited from ``load_events_lazy`` (fixed widths, so the artifact is
    deterministic).

    Args:
        raw_dir: directory holding the raw monthly CSV exports.
        out_file: destination parquet path. Overwritten if present.
    """
    load_events_lazy(event_files(raw_dir)).sink_parquet(out_file)