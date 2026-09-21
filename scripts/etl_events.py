"""ETL: raw event CSVs -> data/processed/events.parquet + integrity gate.

Idempotent by design (re-running overwrites the parquet via sink_parquet's drop
semantics). The full raw frame is never held in memory: the parquet is written
with a streaming sink first, then the integrity report is computed off that
single file rather than the 2.3 GB CSV set.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

# Scripts run as `python scripts/etl_events.py` get `scripts/` on sys.path, not
# the repo root; the pyproject pythonpath setting only applies to pytest. Make
# the src/ package importable regardless of the invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_io import event_files, load_events_lazy, write_events_parquet
from src.eda import integrity_report

_REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = _REPO_ROOT / "data"
OUT_FILE = RAW_DIR / "processed" / "events.parquet"


def main() -> None:
    t0 = time.perf_counter()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_events_parquet(RAW_DIR, OUT_FILE)

    # Read the freshly written artifact (one fast file) for both the sink
    # completeness check and the integrity report.
    back = pl.scan_parquet(OUT_FILE).collect(engine="streaming")
    report = integrity_report(back)

    elapsed = time.perf_counter() - t0
    print("=== integrity report ===")
    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"parquet rows: {back.height}")
    print(f"elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()