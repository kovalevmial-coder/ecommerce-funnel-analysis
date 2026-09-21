"""Tests for the raw-event loader in src/data_io.py.

The two tiny CSVs emulate two monthly e-commerce event files: they carry the
exact raw header (event_time strings literally end in " UTC") so the loader is
proven against the real on-disk format, not a sanitized copy.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from src.data_io import event_files, load_events_lazy, write_events_parquet

# Header is copied verbatim from the real 2019-Oct.csv source files.
_RAW_HEADER = (
    "event_time,event_type,product_id,category_id,category_code,brand,price,"
    "user_id,user_session"
)


def _write_month_csv(path: Path, rows: list[tuple]) -> None:
    """Write one raw-format monthly CSV into the given path."""
    path.write_text(_RAW_HEADER + "\n" + "\n".join(",".join(str(v) for v in r) for r in rows) + "\n")


@pytest.fixture
def raw_dir(tmp_path) -> Path:
    """Build a two-month raw dataset (4 + 2 events) in a scratch directory."""
    _write_month_csv(
        tmp_path / "2019-Oct.csv",
        [
            ("2019-10-01 00:00:03 UTC", "view", 100, 200, "electronics.phone", "Apple", 999.0, 42, "sess1"),
            ("2019-10-02 12:34:56 UTC", "cart", 100, 200, "electronics.phone", "Apple", 999.0, 42, "sess1"),
            ("2019-10-03 08:00:00 UTC", "view", 101, 201, "furniture", "IKEA", 49.9, 7, "sess2"),
            ("2019-10-31 23:59:59 UTC", "remove_from_purchase", 101, 201, "", "", 49.9, 7, "sess2"),
        ],
    )
    _write_month_csv(
        tmp_path / "2019-Nov.csv",
        [
            ("2019-11-15 10:00:00 UTC", "view", 102, 202, "apparel", "Nike", 59.99, 3, "sess3"),
            ("2019-11-30 21:00:00 UTC", "cart", 102, 202, "apparel", "Nike", 59.99, 3, "sess3"),
        ],
    )
    return tmp_path


def test_event_files_returns_sorted_monthly_csvs(raw_dir) -> None:
    """Glob must return the raw CSVs in sorted (chronological) order."""
    paths = event_files(raw_dir)
    assert paths == [raw_dir / "2019-Oct.csv", raw_dir / "2019-Nov.csv"]


def test_load_events_lazy_concats_and_parses_time(raw_dir) -> None:
    """Lazy load must concatenate months and parse event_time to UTC datetime."""
    df = load_events_lazy([raw_dir / "2019-Oct.csv", raw_dir / "2019-Nov.csv"]).collect()

    assert df.height == 6
    assert df.schema["event_time"] == pl.Datetime(time_unit="us", time_zone="UTC")
    assert df.schema["price"] == pl.Float32
    assert df.schema["product_id"] == pl.Int64
    assert df.schema["user_id"] == pl.Int64

    # Known raw string round-trips to its exact UTC instant (Oct 2019, " UTC" suffix removed).
    assert df["event_time"].item(0) == datetime(2019, 10, 1, 0, 0, 3, tzinfo=timezone.utc)
    assert df["event_time"].item(2) == datetime(2019, 10, 3, 8, 0, 0, tzinfo=timezone.utc)

    # Rows survive concatenation across both months, in file order.
    assert df["event_type"].to_list() == ["view", "cart", "view", "remove_from_purchase", "view", "cart"]


def test_write_events_parquet_round_trips(raw_dir) -> None:
    """sink_parquet must emit a file that reads back with same rows and datetime type."""
    out_file = raw_dir / "events.parquet"
    write_events_parquet(raw_dir, out_file)

    assert out_file.exists()

    back = pl.read_parquet(out_file)
    assert back.height == 6
    assert back.schema["event_time"] == pl.Datetime(time_unit="us", time_zone="UTC")

    # Streaming sinks do not guarantee row order, so check the known round-trip
    # instant by presence rather than by row index.
    assert datetime(2019, 10, 1, 0, 0, 3, tzinfo=timezone.utc) in back["event_time"].to_list()
    assert back["price"].dtype == pl.Float32
    assert back["product_id"].dtype == pl.Int64