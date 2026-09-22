"""Project-wide plotting style and the stage chart builders (Stages 02-03).

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

from src.funnel import _FUNNEL_STEPS, _wilson_ci

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

# Per-step metadata from the single funnel definition (src.funnel._FUNNEL_STEPS):
# a step's colour index into _PALETTE and its (converting, basis) boolean pair.
# Plot functions look both up here so a step is never recoloured or re-derived
# in some chart while the funnel logic says something else.
_STEP_META: dict[str, tuple[str, str, str]] = {
    label: (_PALETTE[i], conv, base) for i, (label, conv, base) in enumerate(_FUNNEL_STEPS)
}


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


def plot_funnel_stages(rates: pl.DataFrame, out_path: Path) -> None:
    """Render the classic trapezoid funnel chart (one stage per funnel step).

    Companion to :func:`plot_funnel`: the bar chart shows each step's *rate*
    (the funnel read as three conditional probabilities on one axis), while the
    trapezoid chart shows *volume-shape* — absolute sessions narrowing top to
    bottom, the way business stakeholders read a funnel. Both consume the same
    ``funnel_rates`` frame, so the counts and rates on the trapezoids cannot
    drift from the numbers table they are rendered from.

    Trapezoids are drawn as ``fill_betweenx`` bands: each stage spans its own
    y-row, width-proportional to ``numerator``, labelled with the absolute
    count / basis / conversion rate in the centre and the step name at the
    left. No CI whiskers here — the per-step uncertainty is already the bar
    chart's job (``plot_funnel``) and a width-scaled CI is invisible at these
    magnitudes.

    Args:
        rates: ``funnel_rates`` output — the three contract rows with columns
            ``step``, ``numerator``, ``denominator``, ``rate``, ``ci_low``,
            ``ci_high``.
        out_path: destination PNG path; missing parent directories are created.

    Returns:
        None (the chart is written to ``out_path``).

    Raises:
        ValueError: if any contract column is missing, listing the absent ones.
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
    widths = [count / max(counts) for count in counts]  # 0..1 relative widths

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    n = len(counts)
    # Top-to-bottom row bands: the first (widest) stage sits at the top.
    for i, (step, count, den, rate, width) in enumerate(
        zip(steps, counts, denominators, conv_rates, widths)
    ):
        y_top = n - i
        y_bot = n - i - 1
        # Tapered look: a stage narrows from its own width (top) toward the next
        # stage's width (bottom); the last stage keeps its own width throughout
        # (no next stage to narrow toward).
        w_bot = widths[i + 1] if i + 1 < n else width
        ax.fill_betweenx(
            [y_bot, y_top],
            [0.5 - width / 2, 0.5 - w_bot / 2],
            [0.5 + width / 2, 0.5 + w_bot / 2],
            color=_STEP_META[step][0],
            alpha=0.85,
            zorder=2,
        )
        ax.text(0.02, (y_bot + y_top) / 2, step, ha="left", va="center", zorder=3)
        ax.text(
            0.5,
            (y_bot + y_top) / 2,
            f"{count:,} of {den:,}\n({rate:.1%})",
            ha="center",
            va="center",
            zorder=3,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.2, n + 0.2)
    ax.axis("off")
    ax.set_title("view → cart → purchase conversion funnel (sessions)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Contract columns the segment-rate chart reads: n_sessions feeds the bar labels
# and ci_low/ci_high the whiskers, so a frame missing either would silently
# produce a misleading chart.
_SEGMENT_REQUIRED_COLUMNS = ["step", "rate", "ci_low", "ci_high", "segment", "n_sessions"]


def plot_segment_rates(seg_rates: pl.DataFrame, step: str, out_path: str) -> None:
    """Render one step's conversion rate per segment as horizontal bars.

    Each segment of ``seg_rates`` becomes one horizontal bar at its ``rate``,
    sorted ascending so the highest-converting segment sits at the top. The bar
    carries the 95% CI as whiskers in rate units (``ci_low``/``ci_high`` are
    stored on the same 0..1 scale, unlike ``plot_funnel`` where the bar is on
    the count axis) and a two-line label with the rate and ``n_sessions``. The
    ``"NA"`` price tier is just another row of this frame: it is drawn when
    present and never silently dropped — Stage 04/05 decide price-tier
    exclusions, this descriptive chart reports everything it is given.

    Only rows whose ``step`` matches are drawn. Callers usually pass the whole
    ``segment_funnel`` output (all three steps, as the exploration script
    builds) and this function filters to the requested one; passing a frame
    already filtered to one step is equally fine.

    The figure is always closed at the end and never shown interactively — the
    PNG is the only artifact.

    Args:
        seg_rates: ``segment_funnel`` output — rows with ``step``, ``rate``,
            ``ci_low``, ``ci_high``, ``segment`` and ``n_sessions``.
        step: the funnel step to plot (e.g. ``"view->cart"``); an empty match
            raises rather than silently plotting nothing.
        out_path: destination PNG path (str or Path; coerced via ``Path()``,
            missing parent directories are created).

    Returns:
        None (the chart is written to ``out_path``).

    Raises:
        ValueError: if any contract column is missing (listed), or if ``step``
            matches no row of ``seg_rates`` (the available steps are listed).
    """
    missing = [col for col in _SEGMENT_REQUIRED_COLUMNS if col not in seg_rates.columns]
    if missing:
        raise ValueError(
            f"seg_rates is missing required columns: {missing}; "
            f"got {seg_rates.columns}"
        )

    one_step = seg_rates.filter(pl.col("step") == step)
    if one_step.height == 0:
        raise ValueError(
            f"step {step!r} not in seg_rates; available steps are "
            f"{sorted(seg_rates['step'].unique().to_list())}"
        )

    # Ascending by rate: barh draws y=0 at the bottom, so the best-converting
    # segment naturally ends up at the top of the chart.
    rows = one_step.sort("rate")
    segments = rows["segment"].to_list()
    rates = rows["rate"].to_list()
    ci_low = rows["ci_low"].to_list()
    ci_high = rows["ci_high"].to_list()
    n_sessions = rows["n_sessions"].to_list()
    # Step colour from the shared palette (fallback: first colour for a step the
    # funnel definition does not know — the chart still must render).
    color = _STEP_META[step][0] if step in _STEP_META else _PALETTE[0]

    fig, ax = plt.subplots()
    y = list(range(len(segments)))
    ax.barh(y, rates, height=0.55, color=color, edgecolor="white", zorder=3)

    # CI whiskers in rate units (horizontal bars take xerr). Errors are split
    # below/above the estimate so an interval pinned to 0 or 1 never draws a
    # negative whisker.
    for yi, (lo, hi) in enumerate(zip(ci_low, ci_high)):
        ax.errorbar(
            rates[yi],
            yi,
            xerr=[[rates[yi] - lo], [hi - rates[yi]]],
            fmt="none",
            ecolor="#333333",
            capsize=3,
            elinewidth=1,
            zorder=4,
        )

    # 25% headroom above the largest interval for the rate/n labels (mirrors the
    # funnel chart's 22% margin). Whiskers end at ci_high, so anchoring each
    # label there keeps it clear of the bar tip.
    xmax = max(max(ci_high), max(rates)) * 1.25
    for yi, (hi, rate, n) in enumerate(zip(ci_high, rates, n_sessions)):
        ax.text(
            hi + 0.02 * xmax,
            yi,
            f"{rate:.1%}\nn={n:,}",
            va="center",
            fontsize=10,
            zorder=5,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(segments)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Conversion rate")
    ax.set_title(f"{step} conversion rate by segment")

    # Proxy artist so the 95% whiskers are explainable in the legend.
    proxy = plt.Line2D(
        [0],
        [0],
        marker="|",
        linestyle="none",
        color="#333333",
        markersize=10,
        markerfacecolor="#333333",
        label="95% CI (Wilson)",
    )
    ax.legend(handles=[proxy], loc="lower right")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Contract columns the price-curve chart reads (the session_features shape).
_PRICE_CURVE_COLUMNS = ["median_price", "has_view", "has_cart", "has_purchase"]

# The funnel steps the continuous price view plots. view->purchase is excluded
# because the question is the price→conversion shape of the two chained
# transitions, not the jumpers-inclusive end-to-end rate.
_PRICE_CURVE_STEPS = ("view->cart", "cart->purchase")


def plot_price_conversion_curve(prices: pl.DataFrame, out_path: str) -> None:
    """Plot how conversion varies across session median price (binned, with CIs).

    Descriptive exploration of the price→conversion shape without hard business
    cuts: the x-axis is the session's ``median_price``, split into equal-count
    decile bins *derived inside this function* (the 10/20/..90 quantiles of the
    frame's own prices; ties collapse so a heavily-repeated price never creates
    an empty bin). Each bin is plotted at the median of the prices inside it —
    an honest representative point — with the view->cart and cart->purchase
    rates and their 95% Wilson CIs as translucent ribbons.

    Sessions whose ``median_price`` is null have no priceable axis position and
    are excluded explicitly (the real dataset carries none; a fully-null frame
    raises). A step with no basis sessions in a bin (e.g. a bin with no carts)
    has no definable rate — that point is skipped and the curve simply has a gap.

    The figure is always closed at the end and never shown interactively — the
    PNG is the only artifact.

    Args:
        prices: frame with float ``median_price`` and boolean ``has_view``,
            ``has_cart``, ``has_purchase`` (the ``session_features`` contract).
        out_path: destination PNG path (str or Path; coerced via ``Path()``,
            missing parent directories are created).

    Returns:
        None (the chart is written to ``out_path``).

    Raises:
        ValueError: if any contract column is missing (listed), or if no row
            carries a non-null ``median_price`` (nothing binable to plot).
    """
    missing = [col for col in _PRICE_CURVE_COLUMNS if col not in prices.columns]
    if missing:
        raise ValueError(
            f"prices is missing required columns: {missing}; got {prices.columns}"
        )

    # Unpriced sessions cannot be placed on a price scale — exclude them here,
    # explicitly, before binning (fabricating an axis position would mislead).
    priced = prices.filter(pl.col("median_price").is_not_null())
    if priced.height == 0:
        raise ValueError(
            "prices has no rows with a non-null median_price; nothing to bin "
            f"(median_price is null for all {prices.height} rows)"
        )
    binned = _binned_conversion(priced)

    fig, ax = plt.subplots()
    for step in _PRICE_CURVE_STEPS:
        color, _, _ = _STEP_META[step]
        sub = binned.filter(pl.col("step") == step).sort("x")
        xs = sub["x"].to_list()
        rates = sub["rate"].to_list()
        ci_low = sub["ci_low"].to_list()
        ci_high = sub["ci_high"].to_list()
        # NaN in a bin with an empty denominator breaks the line there, so the
        # curve shows an honest gap instead of a fabricated join.
        ax.plot(xs, rates, color=color, marker="o", markersize=4, linewidth=2, label=step, zorder=3)
        ax.fill_between(xs, ci_low, ci_high, color=color, alpha=0.2, zorder=2)

    ax.set_xlabel("Session median price (EUR)")
    ax.set_ylabel("Conversion rate")
    ax.set_ylim(bottom=0)
    ax.set_title("How conversion varies with median price")

    # Proxy artist so the translucent ribbons are explainable in the legend.
    proxy = plt.Line2D(
        [0],
        [0],
        marker="|",
        linestyle="none",
        color="#333333",
        markersize=10,
        markerfacecolor="#333333",
        label="95% CI (Wilson)",
    )
    ax.legend(handles=[*ax.get_lines(), proxy], loc="upper right")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _binned_conversion(prices: pl.DataFrame) -> pl.DataFrame:
    """Bin sessions by median-price deciles and compute per-bin funnel rates.

    Decile breaks are the 10/20/..90 quantiles of the frame's own median prices,
    deduplicated into strictly ascending cut edges (a duplicate break means a
    repeated quantile price — the merged bin simply covers a wider price span).
    Each bin is anchored at the median price of its own sessions (the x the
    plotter uses) and reports the Wilson CI of each step's rate. A step whose
    basis (denominator) is empty in a bin yields null rate/CI — the caller draws
    such a point as a gap.

    Args:
        prices: non-empty frame with float ``median_price`` and the boolean
            ``has_*`` step columns (null prices already removed by the caller).

    Returns:
        Long frame with ``step``, ``x`` (bin median price), ``rate``,
        ``ci_low``, ``ci_high`` — one row per (bin, plotted step); bins in
        price order.
    """
    quantiles = [i / 10 for i in range(1, 10)]
    # Each quantile expression defaults to the source column name; alias them
    # distinctly or the eager collect rejects the duplicated projections.
    values = prices.select(
        [pl.col("median_price").quantile(q).alias(f"_q{i}") for i, q in enumerate(quantiles)]
    ).row(0)
    breaks = sorted({float(v) for v in values})

    # to_physical maps each price to its (0, 1, ...) decile index via the
    # half-open cut convention (-inf, b0], (b0, b1], ... — identical to the
    # partition polars.cut documents).
    coded = prices.with_columns(
        pl.col("median_price").cut(breaks=breaks).to_physical().alias("_bin")
    )

    records = []
    for step, conv, base in _FUNNEL_STEPS:
        if step not in _PRICE_CURVE_STEPS:
            continue
        stats = (
            coded.group_by("_bin")
            .agg(
                # Same co-occurrence / basis semantics as _funnel_from_flags:
                # a session converts the step iff it reached both stages, and
                # rates divide by the basis alone.
                (pl.col(conv) & pl.col(base)).sum().alias("_num"),
                pl.col(base).sum().alias("_den"),
                pl.col("median_price").median().alias("_x"),
            )
            .sort("_bin")
        )
        for rec in stats.to_dicts():
            num, den = rec["_num"], rec["_den"]
            if den < 1:
                # Empty basis in this bin: the rate is undefined, so the point
                # is NaN and the plotter leaves a gap (sparse-bin honesty).
                rate = ci_low = ci_high = float("nan")
            else:
                ci_low, ci_high = _wilson_ci(int(num), int(den))
                rate = num / den
            records.append(
                {"step": step, "x": rec["_x"], "rate": rate, "ci_low": ci_low, "ci_high": ci_high}
            )

    return pl.DataFrame(records)
