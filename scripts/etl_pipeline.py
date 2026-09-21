"""ETL: raw events -> session features -> sessions.parquet + funnel report.

Single-pass pipeline: load the events parquet produced by etl_events.py, build
one row per session via ``session_features``, stream that to
``data/processed/sessions.parquet`` (sink_parquet overwrites, so re-running is
idempotent), then report the view->cart->purchase funnel with Wilson CIs and
the two jump diagnostics consumed by notebook 02: direct-purchase sessions
(purchased but never carted) and cart-no-view sessions (carted but never
viewed).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

# Scripts run as `python scripts/etl_pipeline.py` get `scripts/` on sys.path,
# not the repo root; the pyproject pythonpath setting only applies to pytest.
# Make the src/ package importable regardless of the invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.funnel import funnel_rates, session_features

_REPO_ROOT = Path(__file__).resolve().parents[1]
EVENTS_FILE = _REPO_ROOT / "data" / "processed" / "events.parquet"
SESSIONS_FILE = _REPO_ROOT / "data" / "processed" / "sessions.parquet"


def _compose_sessions_table(funnel: pl.DataFrame) -> str:
    """Render the funnel frame as a markdown table, readers can paste it into
    notebook 02 / README unchanged."""
    header = "| step | numerator | denominator | rate | ci_low | ci_high |"
    sep = "|---|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in funnel.iter_rows(named=True):
        lines.append(
            f"| {row['step']} | {row['numerator']} | {row['denominator']} "
            f"| {row['rate']:.5f} | {row['ci_low']:.5f} | {row['ci_high']:.5f} |"
        )
    return "\n".join(lines)


def main() -> None:
    t0 = time.perf_counter()

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"{EVENTS_FILE} not found; run `scripts/etl_events.py` first"
        )

    # The raw event frame lives on disk; materialise it once (20.7M rows) so
    # session_features gets the full frame in one group_by, then stream the
    # collapsed sessions back to parquet — only the 4.5M-row result is ever
    # the working set afterwards.
    events = pl.scan_parquet(EVENTS_FILE).collect(engine="streaming")
    sessions = session_features(events)
    # Free the input frame before the sink holds the output frame in memory.
    del events
    sessions.lazy().sink_parquet(SESSIONS_FILE)

    funnel = funnel_rates(sessions)
    n_direct_purchase = int((sessions["has_purchase"] & ~sessions["has_cart"]).sum())
    n_cart_no_view = int((sessions["has_cart"] & ~sessions["has_view"]).sum())

    elapsed = time.perf_counter() - t0
    print("=== funnel table ===")
    print(_compose_sessions_table(funnel))
    print("=== diagnostics ===")
    print(f"direct-purchase sessions (has_purchase=1 & has_cart=0): {n_direct_purchase}")
    print(f"cart-no-view sessions (has_cart=1 & has_view=0): {n_cart_no_view}")
    print(f"sessions rows: {sessions.height}")
    print(f"elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()