"""Build and execute notebook 03: descriptive segmentation (feature views).

Driven by the Stage 06 plan (Task 2): this notebook shows the three
pre-registered segment families (price tier, visit kind, weekend) as pure
feature views — per-segment funnel rates, segment sizes, the price-conversion
curve, and the patterns that become candidate hypotheses. Per
reports/pre_registration.md §1 the whole notebook is DESCRIPTIVE: it states
observed rates and shapes with no significance claims; the pre-registered
comparisons (families A/B/C) are executed exclusively in notebook 04.

All analysis lives in ``src`` (``assign_segments``, ``segment_funnel``,
``plot_segment_rates``, ``plot_price_conversion_curve``, ``assets_path``); this
script only declares the cells and hands them to the shared runner
(``scripts.build_notebook``), so the executed ``.ipynb`` is reproducible and
committed with its outputs. Chart filenames are the plan's exact names
(``seg_price_tier_view_cart``, ``seg_visit_cart_purchase``,
``seg_weekend_cart_purchase``, ``price_conversion_curve``) resolved via
``src.plots.assets_path`` under the repo root.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The build script runs as `python notebooks/build_03_segmentation.py`, so the
# first sys.path entry is notebooks/ and the repo root is NOT importable; insert
# it explicitly (same pattern as scripts/etl_pipeline.py and build_02_funnel.py)
# before any `scripts.*` import.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_notebook import build_and_execute  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# --- Cell sources -----------------------------------------------------------
# Cells are declared here as plain strings and handed to the runner; the runner
# converts each to nbformat v4 cells and executes them against the pinned
# `ecommerce-funnel` kernel, failing loudly on any cell error.

_C1_INTRO = """\
**What is being segmented.** The pre-registration (`reports/pre_registration.md`
§2) approves three session segments for confirmatory comparison — **price tier**
(`budget`/`mid`/`premium`/`luxury` on merchandising bands), **visit kind**
(`new` vs `returning` by `session_number`), and **weekend vs weekday**
(`dayofweek` in {5, 6}). This notebook is the **descriptive** view of those
segments — the *feature views* the pre-registration names in §1: the overall
funnel is descriptive, and confirmatory comparisons begin only with its §3 list.

> **No inference here.** Everything in this notebook is descriptive: we state
> observed rates, sizes and shapes, with 95% Wilson CIs as sampling-uncertainty
> of the observed point estimates only. There are **no p-values, no tests, no
> significance claims**. The confirmed tests are the pre-registered comparisons
> — families A, B, C with their α, multiplicity corrections, sparse rule and
> practical-significance gate — run **exclusively in notebook 04**. The
> patterns at the end of this notebook are **candidate hypotheses to be tested
> there**; nothing is claimed as established here.
"""

# Setup mirrors notebook 02: resolve the repo root from the kernel's CWD so the
# notebook is path-independent, import src only, and pull the exact plot
# helpers this notebook needs.
_C2_SETUP = """\
import os
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    \"\"\"Resolve the repo root from the kernel's CWD (cwd-independent cell).

    The build runner launches the kernel from the repo root, but a reader who
    re-executes the .ipynb from another directory must resolve the same data.
    Walk up until a directory contains both src/ and the sessions artifact; the
    is_file() guard below fails loudly instead of silently loading wrong data.
    \"\"\"
    probe = Path(os.getcwd()).resolve()
    for _ in range(8):
        if (probe / "src").is_dir() and (
            probe / "data" / "processed" / "sessions.parquet"
        ).is_file():
            return probe
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    raise RuntimeError(
        "repo root not found from CWD="
        + str(Path(os.getcwd()))
        + "; expected a dir with src/ and data/processed/sessions.parquet"
    )


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

SESSIONS_FILE = REPO_ROOT / "data" / "processed" / "sessions.parquet"

# Fail loudly if the processed artifact is missing — an empty segments table
# would otherwise sail through and silently change every downstream number.
if not SESSIONS_FILE.is_file():
    raise FileNotFoundError(f"missing processed artifact: {SESSIONS_FILE}")

# Analysis lives in src only — the notebook renders, it does not re-implement
# segment maths. src.plots import applies the shared matplotlib style, and the
# Agg backend set there is exactly what the headless saves of the four charts
# need.
import polars as pl  # noqa: E402
from IPython.display import Image, Markdown, display  # noqa: E402
from src.segments import assign_segments, segment_funnel  # noqa: E402
import src.plots  # noqa: E402,F401  (applies project style, sets Agg backend)
from src.plots import (  # noqa: E402
    assets_path,
    plot_price_conversion_curve,
    plot_segment_rates,
)

print(f"repo root:    {REPO_ROOT}")
print(f"sessions:     {SESSIONS_FILE}")
"""

_C3_LOAD = """\
# One row per user_session, the frozen unit of the whole project. assign_segments
# is the single owner of the five label columns — the notebook never derives a
# segment label itself, so src and notebook cannot drift.
sessions = pl.read_parquet(SESSIONS_FILE)
seg = assign_segments(sessions)
del sessions  # keep only the labelled frame; the raw frame is not needed again

print(f"loaded {seg.height:,} session rows")
"""

_C4_SIZES = """\
# Segment sizes come from the segment_funnel frames themselves (n_sessions is
# the contract column for segment size) so this table is byte-consistent with
# data/processed/segment_<by>.csv written by scripts/explore_segments.py. The
# three frames are reused by the charts below — sizes and plotted rates come
# from the same objects by construction.
seg_price = segment_funnel(seg, "price_tier")
seg_visit = segment_funnel(seg, "visit_kind")
seg_weekend = segment_funnel(seg, "is_weekend")

# The "NA" price tier is counted from median_price nulls, not from the funnel
# frame: assign_segments labels a null price "NA", and with zero nulls in this
# dataset the segment produces no rows, so it must be reported explicitly rather
# than silently absent from the funnel output.
n_na_tier = int(seg["median_price"].null_count())


def _size_section(funnel: pl.DataFrame, subheader: str) -> str:
    \"\"\"Render one family's (segment, n_sessions) pairs as a markdown block.\"\"\"
    rows = funnel.select("segment", "n_sessions").unique().sort("segment")
    lines = [f"**{subheader}**", "", "| segment | n_sessions |", "|---|---:|"]
    total = 0
    for r in rows.iter_rows(named=True):
        lines.append(f"| {r['segment']} | {r['n_sessions']:,} |")
        total += r["n_sessions"]
    lines.append(f"| **total** | **{total:,}** |")
    return "\\n".join(lines)


_lines = [
    _size_section(seg_price, "price_tier"),
    f"| NA (null median_price) | **{n_na_tier:,}** |",
    "",
    _size_section(seg_visit, "visit_kind"),
    "",
    # is_weekend labels are the frame's boolean-to-string "false"/"true"; the
    # legend maps them so the business reading is unambiguous without renaming
    # the frame's own values.
    _size_section(seg_weekend, "is_weekend  (false = weekday, true = weekend)"),
]
display(Markdown("\\n".join(_lines)))
"""

_C5_SIZES_MD = """\
**Reading the sizes table.** Each family's sizes sum to all 4,535,941 sessions —
the segments partition the session table, they do not overlap.

- **price tier:** budget (≈47.4 %) and mid (≈42.8 %) together are 90 % of all
  sessions; premium is ≈8.9 %; **luxury is 42,438 sessions = 0.94 %**. The
  **"NA" tier (null `median_price`) holds 0 sessions** in this dataset — no
  session lacks a price, so the tier simply does not appear; it is reported
  explicitly rather than silently dropped. Luxury is already earmarked by the
  pre-registration's practical-significance rule (§5): at 0.94 % it is *below*
  the 1 % affected-volume threshold, so **by design** it is too small to be
  decision-relevant in any comparison it participates in. The narrative below
  respects that — a 42 k-session segment cannot carry a business verdict alone.
- **visit kind:** returning (≈63.9 %) outnumber new (≈36.1 %) by ~1.8× — a
  volume gap that matters for family B's capability, not for its direction.
- **weekend:** weekday (≈73.9 %) vs weekend (≈26.1 %).

These are feature views: they describe what the segments look like *before* any
comparison is tested (notebook 04).
"""

# Chart 1: view->cart by price tier -> assets/seg_price_tier_view_cart.png. The
# output path is resolved via src.plots.assets_path (the project convention)
# anchored to the repo root, so re-execution from any directory writes to the
# same asset. Displayed inline so the executed notebook carries the chart.
_C6_PLOT_TIER = """\
_tier_out = REPO_ROOT / assets_path("seg_price_tier_view_cart")
plot_segment_rates(seg_price, "view->cart", _tier_out)
display(Image(filename=str(_tier_out)))
print(f"saved {_tier_out}")
"""

_C7_PLOT_TIER_MD = """\
**Descriptive direction — price tier vs view→cart.** The gradient runs
monotone-decreasing across tiers: budget 23.6 % → mid 15.9 % → premium 5.8 % →
luxury 4.3 %. The cheapest tier converts at roughly **5.5×** the luxury rate
(23.6 % vs 4.3 %) on this sample — precisely the signal family A was designed
around. But this is **numbers only**: no test yet, and per the note above the
luxury leg must not be dressed up as an actionable finding (0.94 % of sessions).
Notebook 04 tests the pre-registered 4×2 χ² and the six pairwise z-tests.
"""

_C8_PLOT_VISIT = """\
_visit_out = REPO_ROOT / assets_path("seg_visit_cart_purchase")
plot_segment_rates(seg_visit, "cart->purchase", _visit_out)
display(Image(filename=str(_visit_out)))
print(f"saved {_visit_out}")
"""

_C9_PLOT_VISIT_MD = """\
**Descriptive direction — visit kind vs cart→purchase.** Returning sessions
convert slightly above new ones (13.0 % vs 12.4 %, ≈0.6 pp) — a mild positive
reading for returning visitors on this sample. Descriptive only; whether the
difference is distinguishable from noise is family B's single pre-registered
z-test in notebook 04.
"""

_C10_PLOT_WEEKEND = """\
_weekend_out = REPO_ROOT / assets_path("seg_weekend_cart_purchase")
plot_segment_rates(seg_weekend, "cart->purchase", _weekend_out)
display(Image(filename=str(_weekend_out)))
print(f"saved {_weekend_out}")
"""

_C11_PLOT_WEEKEND_MD = """\
**Descriptive direction — weekend vs cart→purchase.** Weekday sessions convert
slightly above weekend ones (13.0 % vs 12.2 %, ≈0.9 pp) — a mild *negative*
"weekend effect" on this sample. Descriptive only; family C's single z-test in
notebook 04 decides whether it is distinguishable from noise.
"""

# Chart 4: continuous price view -> assets/price_conversion_curve.png. The
# prices frame is the labelled session table (it carries median_price and the
# has_* step booleans the curve's contract needs).
_C12_PLOT_PRICE = """\
_curve_out = REPO_ROOT / assets_path("price_conversion_curve")
plot_price_conversion_curve(seg, _curve_out)
display(Image(filename=str(_curve_out)))
print(f"saved {_curve_out}")
"""

_C13_PLOT_PRICE_MD = """\
**Reading the price curve.** Sessions are binned by median price into deciles
(x = median price of each bin, EUR); both step rates carry Wilson ribbons.

- **view→cart** is broadly **price-decreasing, but not monotonic at the low
  end**: the rate is ≈22 % in the cheapest decile, peaks near ≈26 % around the
  ~2 EUR bin, then falls steadily to ≈5.7 % in the top decile (≈59 EUR) — a
  roughly 4–5× span from the low-price peak to the top decile. The dip of the
  very cheapest bin below the second is visible but not interpreted.
- **cart→purchase** is **approximately flat** across the whole price range: it
  wanders within ≈11–14 % with no systematic increasing or decreasing trend.

No significance claims: these are the observed shapes of this dataset. Whether
any slope is distinguishable from flat is not tested here.
"""

_C14_HYPOTHESES = """\
## Candidate hypotheses for notebook 04 (to be tested, not concluded here)

Patterns observed above, listed as hypotheses to be **tested** against the
pre-registered rules — nothing in this list is claimed as established.

**Family A — price tier × view→cart**
- H (descriptive signal): view→cart decreases with price tier (budget 23.6 % >
  mid 15.9 % > premium 5.8 % > luxury 4.3 %), notably a ~5.5× budget-vs-luxury
  gap. Notebook 04: 4×2 χ² + 6 pairwise z-tests (Bonferroni α* = 0.0083, FDR as
  sensitivity, sparse rule, NA tier excluded with an explicit note).

**Family B — visit kind × cart→purchase**
- H (descriptive signal): returning converts slightly above new (13.0 % vs
  12.4 %, ≈0.6 pp). Notebook 04: single two-proportion z-test, new vs returning,
  on cart sessions.

**Family C — weekend × cart→purchase**
- H (descriptive signal): weekday converts slightly above weekend (13.0 % vs
  12.2 %, ≈0.9 pp). Notebook 04: single two-proportion z-test, weekend vs
  weekday, on cart sessions.

The pre-registered decision rule (§4–§5) decides what, if anything, passes:
α = 0.05 two-sided, per-family Bonferroni, and — separately from any p-value —
the practical gate (CI excludes 0 AND |relative uplift| ≥ 10 % AND affected
volume ≥ 1 %, i.e. ≥ 45,360 sessions, with luxury below the bar by design).
"""

_CELIS = [
    {"kind": "markdown", "source": _C1_INTRO},
    {"kind": "code", "source": _C2_SETUP},
    {"kind": "code", "source": _C3_LOAD},
    {"kind": "code", "source": _C4_SIZES},
    {"kind": "markdown", "source": _C5_SIZES_MD},
    {"kind": "code", "source": _C6_PLOT_TIER},
    {"kind": "markdown", "source": _C7_PLOT_TIER_MD},
    {"kind": "code", "source": _C8_PLOT_VISIT},
    {"kind": "markdown", "source": _C9_PLOT_VISIT_MD},
    {"kind": "code", "source": _C10_PLOT_WEEKEND},
    {"kind": "markdown", "source": _C11_PLOT_WEEKEND_MD},
    {"kind": "code", "source": _C12_PLOT_PRICE},
    {"kind": "markdown", "source": _C13_PLOT_PRICE_MD},
    {"kind": "markdown", "source": _C14_HYPOTHESES},
]

if __name__ == "__main__":
    # Execute against the pinned kernel; the runner raises on any cell error.
    build_and_execute(
        "03_segmentation",
        _CELIS,
        OUT_DIR,
        title="Notebook 03 — descriptive segmentation (feature views)",
    )
    print("built notebooks/03_segmentation.ipynb")