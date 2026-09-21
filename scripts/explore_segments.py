"""EXPLORATORY (not production): per-segment funnel tables for notebook 03.

Descriptive only — no statistics and no assertions here (confirmatory tests
live in stages 04/05). Reads the staging ``data/processed/sessions.parquet``,
attaches the five segment labels via ``assign_segments``, runs
``segment_funnel`` for each documented ``by`` family, writes one tidy frame
per family to ``data/processed/segment_<by>.csv`` (the gitignored inputs in
BS-V6X-0001/P) and prints the per-segment size table so notebook 03 can
sanity-check molecule counts against what it derives itself.

Note on ``n_sessions``: it is the segment's full size; a step's denominator
inside the same frame is the step-specific subset and may be smaller — see
``segment_funnel``'s docstring. The ``"NA"`` price tier (sessions without a
priced event) is reported here and excluded only in stage-04/05 confirmatory
comparisons, explicitly and by documented filter.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

# Scripts run as `python scripts/explore_segments.py` get `scripts/` on
# sys.path, not the repo root; the pyproject pythonpath setting only applies to
# pytest. Make the src/ package importable regardless of the invocation dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segments import _SEGMENT_FAMILIES, assign_segments, segment_funnel

_REPO_ROOT = Path(__file__).resolve().parents[1]
SESSIONS_FILE = _REPO_ROOT / "data" / "processed" / "sessions.parquet"
OUT_DIR = _REPO_ROOT / "data" / "processed"


def _print_size_table(funnel: pl.DataFrame, by: str) -> None:
    """Render segment sizes as a markdown table for notebook sanity checks."""
    sizes = funnel.select("segment", "n_sessions").unique().sort("segment")
    print(f"\n=== {by} segment sizes (n_sessions) ===")
    for row in sizes.iter_rows(named=True):
        print(f"| {row['segment']} | {row['n_sessions']} |")


def main() -> None:
    t0 = time.perf_counter()
    if not SESSIONS_FILE.exists():
        raise FileNotFoundError(
            f"{SESSIONS_FILE} not found; run `scripts/etl_pipeline.py` first"
        )

    sessions = pl.read_parquet(SESSIONS_FILE)
    # Attach the labels once, then feed the narrow (by, has_*) projection to
    # segment_funnel per family — avoids five redundant full redecompositions
    # of the 4.5M-row frame while keeping the in-memory working set small.
    labelled = assign_segments(sessions)
    del sessions

    for by in sorted(_SEGMENT_FAMILIES):
        funnel = segment_funnel(
            labelled.select([by, "has_view", "has_cart", "has_purchase"]), by
        )
        out_path = OUT_DIR / f"segment_{by}.csv"
        funnel.write_csv(out_path)
        _print_size_table(funnel, by)
        print(f"wrote {out_path} ({funnel.height} rows)")

    elapsed = time.perf_counter() - t0
    print(f"\nelapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()