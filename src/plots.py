"""Project-wide plotting style and the funnel chart (Stage 03).

The module-level matplotlib setup is the single place where the project's
visual look is defined — consistent figure size, fonts and palette — so every
later stage (segmentation charts, hypothesis-testing plots, notebooks, README)
inherits the same style by importing one module instead of re-declaring rcParams
per notebook. Per-plot configuration that only one figure needs stays local to
its function; nothing in the module globals encodes a single chart's layout.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

import matplotlib as mpl

# Force the headless Agg backend BEFORE pyplot is imported. This module is
# imported by tests and CI jobs that run without a display or X server; the
# default interactive backend would fail to initialize there. Agg renders to a
# file buffer, which is exactly what ``plot_funnel`` needs — every figure is
# saved, never shown.
mpl.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402  (backend must be set first)

# Shared rcParams: a small, tasteful default that keeps every exported chart
# consistent (same canvas, same font sizes, quiet spines, tight bbox on save).
# ``figure.figsize`` keeps the standard "one business chart" aspect; per-figure
# size overrides stay inside the plotting function, not here.
_RCPARAMS = {
    "figure.figsize": (8.0, 5.0),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.facecolor": "white",
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
    "savefig.bbox": "tight",
}
plt.rcParams.update(_RCPARAMS)

# Ordered categorical palette for funnel steps; stage 03+ charts reuse these
# indices so step A keeps the same colour across the whole project. Three steps
# map to a muted blue / amber / green sequence chosen for visibility on white.
_PALETTE = ["#4C72B0", "#DD8452", "#55A868"]


def assets_path(name: str) -> Path:
    """Resolve a chart's destination by the project convention.

    Every stage that exports a figure saves it through this helper so asset
    files land in one predictable, repo-root ``assets/`` directory (matching the
    README Quick Start contract) instead of scattering next to each notebook.
    Strips nothing — callers pass a bare stem, the ``.png`` suffix is applied.

    Args:
        name: asset stem (e.g. ``"funnel_check"``).

    Returns:
        ``Path("assets") / "<name>.png"``.
    """
    return Path("assets") / f"{name}.png"


# Contract columns every consumer of ``funnel_rates`` output expects.
_REQUIRED_COLUMNS = ["step", "numerator", "denominator", "rate", "ci_low", "ci_high"]


def plot_funnel(rates: pl.DataFrame, out_path: Path) -> None:
    """Render the view->cart->purchase funnel to a PNG.

    The funnel is drawn one bar per step on the *absolute session count* axis
    (the ``numerator`` column). Each bar is labelled with its absolute count and
    its conversion rate (the ``rate`` column, the step's business number), and
    carries a 95% Wilson CI error bar. The CI is stored in rate units
    (0..1) but the bar is on the count axis, so the endpoints are converted to
    count units via ``ci_* * denominator`` before drawing — otherwise the
    interval would be an invisible sliver at these scales.

    The figure is always closed at the end (no leftover open figures, no global
    state) and never shown interactively — the PNG is the only artifact.

    Args:
        rates: ``funnel_rates`` output — the three contract rows with columns
            ``step``, ``numerator``, ``denominator``, ``rate``, ``ci_low``,
            ``ci_high``.
        out_path: destination PNG path; missing parent directories are created.

    Returns:
        None (the chart is written to ``out_path``).

    Raises:
        ValueError: if any contract column is missing, listing the absent ones
            (fail-fast: a plot of silently wrong columns would mislead, not help).
    """
    missing = [col for col in _REQUIRED_COLUMNS if col not in rates.columns]
    if missing:
        raise ValueError(
            f"rates is missing required columns: {missing}; got {rates.columns}"
        )

    steps = rates["step"].to_list()
    counts = rates["numerator"].to_list()
    denominators = rates["denominator"].to_list()
    conv_rates = rates["rate"].to_list()
    ci_low = rates["ci_low"].to_list()
    ci_high = rates["ci_high"].to_list()

    fig, ax = plt.subplots()
    x = range(len(steps))
    bars = ax.bar(x, counts, width=0.55, color=_PALETTE, edgecolor="white", zorder=3)

    for xi, (count, den, lo, hi) in enumerate(zip(counts, denominators, ci_low, ci_high)):
        # The Wilson interval in count units, anchored on the bar top. Errors
        # below/above the estimate are separate so a CI pinned to a boundary
        # does not draw a negative whisker.
        lower = count - lo * den
        upper = hi * den - count
        ax.errorbar(
            xi,
            count,
            yerr=[[lower], [upper]],
            fmt="none",
            ecolor="#333333",
            capsize=3,
            elinewidth=1,
            zorder=4,
        )
        # Two-line label above each bar: the conversion rate (business number)
        # on top, the absolute session count beneath it. The 4%-of-tallest-bar
        # offset keeps the text clear of the whisker yet below the ylim ceiling.
        ax.text(
            xi,
            count + upper + 0.04 * max(counts),
            f"{conv_rates[xi]:.1%}\n{count:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            zorder=5,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(steps)
    ax.set_xlabel("Funnel step")
    ax.set_ylabel("Sessions")
    ax.set_ylim(0, max(counts) * 1.22)
    ax.set_title("view → cart → purchase conversion funnel")

    # Proxy artist so the 95% error bars are explainable in the legend.
    proxy = plt.Line2D(
        [0], [0], marker="|", linestyle="none", color="#333333",
        markersize=10, markerfacecolor="#333333", label="95% CI (Wilson)",
    )
    ax.legend(handles=[proxy], loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
