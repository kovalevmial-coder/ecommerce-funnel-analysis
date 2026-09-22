"""Build and execute notebook 01: data integrity + descriptive EDA.

Driven by the Stage 05 pre-registration plan: the notebook fixes the
*descriptive* facts (volume, units, missingness, event mix, price feature)
that later confirmatory stages depend on, so the price-tier bands are pinned
from this notebook's price view BEFORE any segment-conversion outcome exists.
All analysis lives in ``src``; this script only declares the cells and hands
them to the shared runner (``scripts.build_notebook``), so the executed
``.ipynb`` is guaranteed reproducible and committed with its outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The build script runs as `python notebooks/build_01_data_integrity.py`, so
# the first sys.path entry is notebooks/ and the repo root is NOT importable;
# insert it explicitly (same pattern as scripts/etl_pipeline.py) before any
# `scripts.*` import.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_notebook import build_and_execute  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# --- Cell sources -----------------------------------------------------------
# Cells are declared here as plain strings and handed to the runner; the
# runner converts each to nbformat v4 cells and executes them against the
# pinned `ecommerce-funnel` kernel, failing loudly on any cell error.

_C1_STAGE_PURPOSE = """\
## Data integrity and descriptive EDA (Stage 05 gate)

**Why this notebook exists.** Before any confirmatory work, the project must
fix the *descriptive* facts everything else stands on: how many rows, users
and sessions the dataset really has, how missing the raw columns are, what
the event mix looks like, and what the price feature looks like. The
price-tier bands used for the segmentation hypotheses are pinned from this
notebook's price view — **before** any segment-conversion outcome exists.
Nothing here is a hypothesis test; that is the design of a pre-registration
gate (see `notes/01_data_integrity.md` and `notes/05_pre_registration.md`,
RU learning notes, and `reports/pre_registration.md` for the fixed
decisions).

**Unit of analysis.** Every session-level metric in this project uses the
pair **`(user_session, user_id)`** as its unit — not the bare
`user_session`, and not the `user_id`. Why:

- *Not `user_id`*: a user can open many visits; collapsing to users would
  destroy exactly the visit structure the funnel describes. The funnel and
  the segments operate on session-level reached-step indicators
  (see `notes/00_scope_and_plan.md`).
- *Not bare `user_session` either*: the integrity scan found **272 session
  ids reused across different users** in this dataset. Grouping by
  `user_session` alone would silently merge two distinct journeys into one
  row. `integrity_report` reports that count precisely because it measures
  this collapsion risk; the affected share is <0.01% of sessions, so the
  single-key `src.funnel.session_features` pipeline is a documented,
  deliberate approximation.

Session-level numbers in this notebook therefore group by the
`(user_session, user_id)` pair when they come from `integrity_report`, and
the session-diagnostics section below states its own grouping key.
"""

_C2_SETUP = """\
import os
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    \"\"\"Resolve the repo root from the kernel's CWD (cwd-independent cell).

    The build runner launches the kernel from the repo root, but a reader who
    re-executes the .ipynb from another directory must resolve the same data.
    Walk up until a directory contains both src/ and the events artifact; the
    is_file() guards below fail loudly instead of silently loading wrong data.
    \"\"\"
    probe = Path(os.getcwd()).resolve()
    for _ in range(8):
        if (probe / "src").is_dir() and (
            probe / "data" / "processed" / "events.parquet"
        ).is_file():
            return probe
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    raise RuntimeError(
        "repo root not found from CWD="
        + str(Path(os.getcwd()))
        + "; expected a dir with src/ and data/processed/events.parquet"
    )


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

EVENTS_FILE = REPO_ROOT / "data" / "processed" / "events.parquet"
SESSIONS_FILE = REPO_ROOT / "data" / "processed" / "sessions.parquet"

# Fail loudly if a processed artifact is missing — an empty EDA would
# otherwise sail through and silently change every downstream number.
for _f in (EVENTS_FILE, SESSIONS_FILE):
    if not _f.is_file():
        raise FileNotFoundError(f"missing processed artifact: {_f}")

# NOTE: src.data_io only loads the RAW monthly CSVs (the Stage-01 ETL side);
# it has no processed-parquet loader, so this notebook reads the artifacts
# with polars and feeds them to the src functions that implement the analysis.
import polars as pl  # noqa: E402
import src.plots  # noqa: E402,F401  (applies the shared matplotlib style)
from matplotlib import pyplot as plt  # noqa: E402
from src.eda import integrity_report  # noqa: E402

print(f"repo root:   {REPO_ROOT}")
print(f"events file: {EVENTS_FILE}")
print(f"sessions:    {SESSIONS_FILE}")
"""

_C3_LOAD = """\
# events is 20.7M rows; integrity_report needs the eager frame (it calls
# .unique()/.group_by()/...), so materialise it here once and reuse it.
events = pl.read_parquet(EVENTS_FILE)
report = integrity_report(events)
print(f"loaded {events.height:,} event rows")
"""

_C4_TABLE = """\
# Render the report dict as markdown tables. Building them from the dict —
# not from hand-written literals — guarantees the notebook can never drift
# from src.eda.integrity_report.
from IPython.display import Markdown, display

def _int(v: int) -> str:
    return f"{v:,}"

def _pct(v: float) -> str:
    return f"{100.0 * v:.2f}%"

_sections = [
    ("**Volume and identity**", [
        ("rows (events)", _int(report["rows"])),
        ("distinct users", _int(report["distinct_users"])),
        ("distinct sessions", _int(report["distinct_sessions"])),
        ("rows without a session id", _int(report["null_session_rows"])),
        ("session ids reused across users", _int(report["duplicate_sessions"])),
        ("exact duplicate rows", _int(report["exact_duplicate_rows"])),
    ]),
    ("**Missing values (empty string counts as missing)**", [
        ("brand", f'{report["brand_missing_pct"]}%'),
        ("category_code", f'{report["category_code_missing_pct"]}%'),
    ]),
    ("**Price**", [
        ("price min (EUR)", f"{report['price_min']:.2f}"),
        ("price max (EUR)", f"{report['price_max']:.2f}"),
        ("rows priced exactly 0.0", _pct(report["price_zero_share"])),
        ("purchase rows priced 0.0", _int(report["zero_price_purchases"])),
    ]),
    ("**Session-level diagnostics (grouped by (user_session, user_id))**", [
        ("purchase but no cart", _int(report["purchase_without_cart_sessions"])),
        ("cart but no view", _int(report["cart_without_view_sessions"])),
        ("view-only sessions", _int(report["only_view_sessions"])),
    ]),
    ("**Event-type row counts**", [
        (k, _int(v)) for k, v in sorted(
            report["event_type_counts"].items(), key=lambda kv: -kv[1]
        )
    ]),
]

_lines = []
for _title, _pairs in _sections:
    _lines.append(_title)
    _lines.append("| metric | value |")
    _lines.append("|---|---:|")
    for _k, _v in _pairs:
        _lines.append(f"| {_k} | {_v} |")
    _lines.append("")

display(Markdown("\\n".join(_lines)))
"""

_C5_EVENT_MIX = """\
# Event-type mix: raw row counts per type, ordered by volume. This is the
# EVENT view (rows), explicitly NOT the session funnel of notebook 02.
_counts = dict(sorted(report["event_type_counts"].items(), key=lambda kv: -kv[1]))
_labels = list(_counts.keys())
_values = [float(v) for v in _counts.values()]
_palette = src.plots._PALETTE  # reuse the project palette for look consistency
_colors = [_palette[i % len(_palette)] for i in range(len(_labels))]

_fig, _ax = plt.subplots()
_ax.bar(range(len(_labels)), _values, color=_colors, edgecolor="white", zorder=3)
for _i, (_lab, _v) in enumerate(zip(_labels, _values)):
    _ax.text(_i, _v, f"{_v / 1e6:.2f}M",
             ha="center", va="bottom", fontsize=10, zorder=5)
_ax.set_xticks(range(len(_labels)))
_ax.set_xticklabels(_labels, rotation=20, ha="right")
_ax.set_ylabel("event rows")
_ax.set_ylim(0, max(_values) * 1.12)
_ax.set_title("event-type mix (rows, descriptive)")
_ax.text(0.01, 0.95, f"{int(sum(_values)):,} rows total",
         transform=_ax.transAxes, fontsize=9, va="top")

# Resolve the destination via the project convention, anchored to the repo
# root so the executed notebook is independent of the kernel's CWD.
_mix_out = REPO_ROOT / src.plots.assets_path("integrity_events_mix")
_fig.savefig(_mix_out, dpi=150)
plt.close(_fig)
print(f"saved {_mix_out}")
"""

_C6_EVENT_MIX_MD = """\
**What the mix shows.** `view` dominates at ~9.66M rows — roughly 1.7x the
cart volume. `remove_from_cart` (~3.98M) sits **below** cart (~5.77M), about
0.69 remove events per cart event, and `purchase` (~1.29M) is the smallest
of the four. This is the raw *event* mix, not the *session* funnel: the
funnel (view → cart → purchase across ~4.5M sessions) is computed in
notebook 02. Read here: browsing overwhelms purchasing, and the non-trivial
remove volume is real shopping behaviour — which is exactly why the
robustness `net_cart` column (cart minus remove) exists in `src.funnel`.
"""

_C7_SESSION_DIAGNOSTICS = """\
# Re-derive the two funnel diagnostics from the persisted session frame
# (sessions.parquet IS the src.funnel.session_features output) instead of
# echoing integrity_report's numbers, so the notebook shows the actual
# reached-step shares on the funnel's own unit. The masks use the same
# boolean semantics as src.funnel._funnel_from_flags.
sessions = pl.read_parquet(SESSIONS_FILE)
n_sessions = sessions.height
n_purchase_sessions = int(sessions["has_purchase"].sum())
n_cart_sessions = int(sessions["has_cart"].sum())

direct_purchase = int((sessions["has_purchase"] & ~sessions["has_cart"]).sum())
cart_no_view = int((sessions["has_cart"] & ~sessions["has_view"]).sum())

print(f"sessions rows: {n_sessions:,}")
print(
    "purchase-without-cart sessions: "
    f"{direct_purchase:,}  = {direct_purchase / n_purchase_sessions:.2%} "
    f"of {n_purchase_sessions:,} purchase sessions"
)
print(
    "cart-without-view sessions:     "
    f"{cart_no_view:,}  = {cart_no_view / n_cart_sessions:.2%} "
    f"of {n_cart_sessions:,} cart sessions"
)
"""

_C8_SESSION_MD = """\
**What the two diagnostics mean for the funnel.** The funnel treats steps as
*reached-step indicators* — a session converts a step iff its `has_*` flags
satisfy that step's boolean contract (`src.funnel._FUNNEL_STEPS`); event
order within a session is irrelevant.

- **Purchase without cart (≈19% of purchase sessions).** These "direct
  purchase" journeys bought without adding to a cart. They still count toward
  **view → purchase**: that step's numerator is `has_purchase & has_view`,
  which does not require `has_cart` — excluding them would overstate the drop
  caused by the cart step. They contribute nothing to view → cart or
  cart → purchase.
- **Cart without view (≈20% of cart sessions).** Sessions that carted without
  ever viewing are excluded from **view → cart**: both numerator and
  denominator of that step are conditioned on `has_view`
  (`has_cart & has_view` / `has_view`). They remain genuine cart-havers for
  **cart → purchase** and stay in that step's denominator.

Neither is data corruption — both are real user behaviour the funnel
definition must name explicitly. The absolute counts here differ from
`integrity_report` by a handful of rows because this section keys on
`user_session` alone (sessions.parquet contract) while the report groups by
`(user_session, user_id)`; the *shares* are unaffected.
"""

_C9_PRICE = """\
# Price is treated here as a FEATURE: its range, mass and decile bands feed
# the tier pin-down in reports/pre_registration.md. No conversion outcome is
# inspected in this notebook.
prices = events["price"].drop_nulls()
quantiles = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
decile_values = prices.quantile(quantiles)

_dec_lines = ["| percentile | price (EUR) |", "|---:|---:|"]
# decile_values is a list here (polars returns one for multi-quantile calls),
# not a Series — iterate it directly so the cell is version-agnostic.
for _q, _v in zip(quantiles, decile_values):
    _dec_lines.append(f"| {_q:>4.1%} | {float(_v):.2f} |")
display(Markdown("\\n".join(_dec_lines)))

# Honest histogram clipping: the raw range spans ~-79..328 but the bulk sits
# in 0..~95, driven by a price=0 (free) mass and a tiny high-price tail. An
# unclipped x-axis would flatten the data onto two bars. State the clip.
_p99 = float(prices.quantile(0.99))
_clip = prices.filter((prices >= 0.0) & (prices <= _p99))
_excluded = 1.0 - _clip.len() / prices.len()

_fig, _ax = plt.subplots()
_ax.hist(_clip.to_numpy(), bins=80, range=(0.0, _p99),
         color=src.plots._PALETTE[0], edgecolor="white")
_ax.set_xlabel("price (EUR)")
_ax.set_ylabel("rows")
_ax.set_title("price distribution, display clipped to [0, P99]")
_ax.text(0.02, 0.95,
         f"raw range {report['price_min']:.2f}..{report['price_max']:.2f} EUR; "
         f"{_excluded:.2%} of priced rows outside this view",
         transform=_ax.transAxes, fontsize=9, va="top")

_price_out = REPO_ROOT / src.plots.assets_path("integrity_price_distribution")
_fig.savefig(_price_out, dpi=150)
plt.close(_fig)
print(f"saved {_price_out}")
"""

_C10_PRICE_MD = """\
**Price is a feature view, not an outcome view.** This notebook looks at the
price *feature* only. The decile bands above are what the pre-registration
document uses to pin the tier boundaries (`<5 / [5,30) / [30,150) / 150+ EUR`)
and to check each band has ≥500 sessions before any test runs. Whether a tier
converts better is the confirmatory, pre-registered work of the later
notebooks — inspecting it here would be exactly the post-hoc tuning the
golden standards forbid.
"""

_C11_NON_SIGNIFICANCE = """\
**Nothing in this notebook is a significance claim.** This is the descriptive
stage: every number printed above is a point characteristic of a real dataset
(its volume, missingness, event mix and price feature), not an estimate
targeted at an underlying population and not a test of any hypothesis. There
are no p-values, no confidence intervals, no effect sizes and no acceptance
decisions in this file — by design. Hypothesis families, α = 0.05,
Bonferroni/FDR corrections, the sparse rule (n·p < 5) and the
practical-significance thresholds are fixed in `reports/pre_registration.md`
and executed only in the confirmatory notebooks that follow this stage.
"""

_CELIS = [
    {"kind": "markdown", "source": _C1_STAGE_PURPOSE},
    {"kind": "code", "source": _C2_SETUP},
    {"kind": "code", "source": _C3_LOAD},
    {"kind": "code", "source": _C4_TABLE},
    {"kind": "code", "source": _C5_EVENT_MIX},
    {"kind": "markdown", "source": _C6_EVENT_MIX_MD},
    {"kind": "code", "source": _C7_SESSION_DIAGNOSTICS},
    {"kind": "markdown", "source": _C8_SESSION_MD},
    {"kind": "code", "source": _C9_PRICE},
    {"kind": "markdown", "source": _C10_PRICE_MD},
    {"kind": "markdown", "source": _C11_NON_SIGNIFICANCE},
]

if __name__ == "__main__":
    # Execute against the pinned kernel; the runner raises on any cell error.
    build_and_execute(
        "01_data_integrity",
        _CELIS,
        OUT_DIR,
        title="Notebook 01 — data integrity and descriptive EDA",
    )
    print("built notebooks/01_data_integrity.ipynb")