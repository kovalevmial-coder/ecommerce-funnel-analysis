"""Tests for the chart module (src/plots.py).

The tests pin the file-level contracts of ``plot_funnel``,
``plot_segment_rates`` and ``plot_price_conversion_curve`` — a PNG materialises
at the requested path (creating missing parent directories), is non-empty and
starts with the PNG magic bytes — rather than any pixel content: aesthetics are
deliberately out of scope for CI (the human sanity look happens in the stage
notebook / report). The ``assets_path`` path-by-convention helper, the fail-fast
column guards and the segment/price-curve semantics (step filtering, sorting,
NA handling) are pinned here too.
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest
from matplotlib.collections import PolyCollection

from src.plots import (
    assets_path,
    plot_funnel,
    plot_funnel_stages,
    plot_price_conversion_curve,
    plot_segment_rates,
)

# pyplot is imported for the close-contract assertion below; src.plots already
# forces the Agg backend at import, so this is safe under any display.
import matplotlib.pyplot as plt

# The 3-row funnel_rates-shaped contract frame (Task 2 output schema). Wilson
# CIs are precomputed via ``src.funnel._wilson_ci`` for the hand-set counts so
# the frame is internally consistent, not just shaped:
#   view->cart:      1000 / 2000 = 0.500  CI (0.478108, 0.521892)
#   cart->purchase:   400 / 1000 = 0.400  CI (0.370074, 0.430691)
#   view->purchase:   250 / 2000 = 0.125  CI (0.111221, 0.140217)
_FUNNEL_ROWS = [
    {
        "step": "view->cart",
        "numerator": 1000,
        "denominator": 2000,
        "rate": 0.5,
        "ci_low": 0.478108,
        "ci_high": 0.521892,
    },
    {
        "step": "cart->purchase",
        "numerator": 400,
        "denominator": 1000,
        "rate": 0.4,
        "ci_low": 0.370074,
        "ci_high": 0.430691,
    },
    {
        "step": "view->purchase",
        "numerator": 250,
        "denominator": 2000,
        "rate": 0.125,
        "ci_low": 0.111221,
        "ci_high": 0.140217,
    },
]


def _funnel_frame() -> pl.DataFrame:
    """Build the synthetic rates frame (schema mirrors funnel_rates output)."""
    return pl.DataFrame(_FUNNEL_ROWS)


def _png_signature(path: Path) -> bytes:
    """Return the file's first 8 bytes; the PNG spec fixes them as \x89PNG\\r\\n\\x1a\\n."""
    return path.read_bytes()[:8]


def test_assets_path_follows_assets_convention() -> None:
    """assets_path resolves the repo-root assets/<name>.png convention."""
    assert assets_path("foo") == Path("assets") / "foo.png"
    assert assets_path("funnel_check") == Path("assets") / "funnel_check.png"
    assert isinstance(assets_path("foo"), Path)


def test_plot_funnel_writes_png_into_missing_parent_dirs(tmp_path: Path) -> None:
    """A PNG must land at out_path even when the parent dirs do not exist yet."""
    out = tmp_path / "figures" / "funnel.png"
    assert not out.parent.exists()  # the scenario under test: parents missing

    result = plot_funnel(_funnel_frame(), out)

    assert out.exists()
    assert out.stat().st_size > 0
    assert _png_signature(out) == b"\x89PNG\r\n\x1a\n"
    assert result is None  # contract: returns nothing, figure closed internally
    assert plt.get_fignums() == []  # contract: no figure survives the call


def test_plot_funnel_writes_png_into_deeply_nested_path(tmp_path: Path) -> None:
    """Deeply nested out_path parents (>/1 level) must be created recursively."""
    out = tmp_path / "a" / "b" / "c" / "funnel.png"
    assert not out.parent.exists()

    plot_funnel(_funnel_frame(), out)

    assert out.exists()
    assert out.stat().st_size > 0
    assert _png_signature(out) == b"\x89PNG\r\n\x1a\n"


def test_plot_funnel_fails_fast_on_missing_columns(tmp_path: Path) -> None:
    """Missing contract columns must raise, listing the absent ones (fail-fast)."""
    with pytest.raises(ValueError, match="ci_high"):
        plot_funnel(_funnel_frame().drop("ci_high"), tmp_path / "x.png")


# --- plot_funnel_stages ------------------------------------------------------


def test_plot_funnel_stages_writes_png_into_missing_parent_dirs(tmp_path: Path) -> None:
    """A PNG must land at out_path with parents created; figure closed, nothing returned."""
    out = tmp_path / "figures" / "stages" / "funnel_stages.png"
    assert not out.parent.exists()

    result = plot_funnel_stages(_funnel_frame(), out)

    assert out.exists()
    assert out.stat().st_size > 0
    assert _png_signature(out) == b"\x89PNG\r\n\x1a\n"
    assert result is None
    assert plt.get_fignums() == []


def test_plot_funnel_stages_draws_one_trapezoid_per_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One band per funnel step, ordered top=first step, each with count/rate label.

    The frame's rows are (view->cart 1000, cart->purchase 400, view->purchase
    250); monkeypatching ``plt.close`` captures the axes for inspection because
    the function always closes its own figure.
    """
    captured = _spy_on_close(monkeypatch)

    plot_funnel_stages(_funnel_frame(), tmp_path / "stages.png")

    ax = captured["ax"]
    # fill_betweenx leaves PolyCollection artists; the classic funnel is three
    # tapered bands, so exactly three are expected.
    n_bands = sum(1 for c in ax.collections if isinstance(c, PolyCollection))
    assert n_bands == 3
    texts = [t.get_text() for t in ax.texts]
    # Every absolute count and its conditional rate appear on the chart — the
    # volume-shape claim is built from the same numbers as the table. Labels
    # carry a newline between the count line and the rate line, so exact-match.
    assert "1,000 of 2,000\n(50.0%)" in texts
    assert "250 of 2,000\n(12.5%)" in texts
    assert "400 of 1,000\n(40.0%)" in texts
    assert "conversion funnel" in ax.get_title()
    assert plt.get_fignums() == []


def test_plot_funnel_stages_fails_fast_on_missing_columns(tmp_path: Path) -> None:
    """Missing contract columns must raise, listing the absent ones (fail-fast)."""
    with pytest.raises(ValueError, match="numerator"):
        plot_funnel_stages(_funnel_frame().drop("numerator"), tmp_path / "x.png")


# --- plot_segment_rates ---------------------------------------------------


# segment_funnel output schema, hand-set to two steps with internally
# consistent Wilson CIs (precomputed via ``src.funnel._wilson_ci``):
#   view->cart:   NA 5/50=0.10  CI (0.043475, 0.213605)   n=50
#                 weak 30/100=0.30 CI (0.218948, 0.395850) n=100
#                 strong 120/200=0.60 CI (0.530835, 0.665395) n=200
#   cart->purchase: weak 12/60=0.20 CI (0.118284, 0.317820) n=100
#                   strong 48/120=0.40 CI (0.316763, 0.489441) n=200
# The view->cart rows are deliberately NOT ordered by rate so the test can prove
# the function sorts; the cart->purchase rows exist only to prove step filtering.
_SEGMENT_ROWS = [
    {
        "step": "view->cart",
        "numerator": 30,
        "denominator": 100,
        "rate": 0.3,
        "ci_low": 0.218948,
        "ci_high": 0.39585,
        "segment": "weak",
        "n_sessions": 100,
    },
    {
        "step": "view->cart",
        "numerator": 5,
        "denominator": 50,
        "rate": 0.1,
        "ci_low": 0.043475,
        "ci_high": 0.213605,
        "segment": "NA",
        "n_sessions": 50,
    },
    {
        "step": "view->cart",
        "numerator": 120,
        "denominator": 200,
        "rate": 0.6,
        "ci_low": 0.530835,
        "ci_high": 0.665395,
        "segment": "strong",
        "n_sessions": 200,
    },
    {
        "step": "cart->purchase",
        "numerator": 12,
        "denominator": 60,
        "rate": 0.2,
        "ci_low": 0.118284,
        "ci_high": 0.31782,
        "segment": "weak",
        "n_sessions": 100,
    },
    {
        "step": "cart->purchase",
        "numerator": 48,
        "denominator": 120,
        "rate": 0.4,
        "ci_low": 0.316763,
        "ci_high": 0.489441,
        "segment": "strong",
        "n_sessions": 200,
    },
]


def _segment_frame() -> pl.DataFrame:
    """Build the synthetic segment_funnel-shaped frame (schema mirrors its output)."""
    return pl.DataFrame(_SEGMENT_ROWS)


def _spy_on_close(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Capture the axes of the figure a function closes before it is discarded.

    The plot functions always ``plt.close`` their figure, so to assert on the
    drawing (sorting, filtering, colours) the test intercepts ``plt.close``,
    keeps a reference to the axes, then delegates to the real close so the
    close-contract (``plt.get_fignums() == []``) still holds.
    """
    captured: dict = {}
    real_close = plt.close  # capture BEFORE monkeypatching (the patched name is _spy)

    def _spy(fig=None) -> None:
        if fig is not None and fig.axes:
            captured["ax"] = fig.axes[0]
        real_close(fig)

    monkeypatch.setattr(plt, "close", _spy)
    return captured


def test_plot_segment_rates_writes_png_into_missing_parent_dirs(tmp_path: Path) -> None:
    """A PNG must land at out_path with parents created; figure closed, nothing returned."""
    out = tmp_path / "figures" / "segments" / "seg.png"
    assert not out.parent.exists()

    result = plot_segment_rates(_segment_frame(), "view->cart", out)

    assert out.exists()
    assert out.stat().st_size > 0
    assert _png_signature(out) == b"\x89PNG\r\n\x1a\n"
    assert result is None
    assert plt.get_fignums() == []


def test_plot_segment_rates_sorts_by_rate_and_draws_only_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the requested step's segments are drawn, one bar each, sorted ascending by rate."""
    captured = _spy_on_close(monkeypatch)

    plot_segment_rates(_segment_frame(), "view->cart", tmp_path / "seg.png")

    ax = captured["ax"]
    ylabels = [t.get_text() for t in ax.get_yticklabels()]
    assert ylabels == ["NA", "weak", "strong"]  # ascending rate, highest on top
    assert len(ax.patches) == 3  # bars == segments of THIS step only (5 rows in frame)
    assert "view->cart" in ax.get_title()
    assert plt.get_fignums() == []


def test_plot_segment_rates_reports_na_segment_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The NA price tier must be shown, never silently dropped before Stage 04/05."""
    na_frame = _segment_frame().filter(pl.col("segment").is_in(["NA", "weak"]))
    captured = _spy_on_close(monkeypatch)

    plot_segment_rates(na_frame, "view->cart", tmp_path / "seg.png")

    ylabels = [t.get_text() for t in captured["ax"].get_yticklabels()]
    assert "NA" in ylabels
    assert len(captured["ax"].patches) == 2  # both the NA and the regular tier render


def test_plot_segment_rates_fails_fast_on_missing_columns(tmp_path: Path) -> None:
    """Missing contract columns (labels need n_sessions, whiskers need ci_*) must raise."""
    with pytest.raises(ValueError, match="n_sessions"):
        plot_segment_rates(_segment_frame().drop("n_sessions"), "view->cart", tmp_path / "x.png")


def test_plot_segment_rates_fails_fast_on_unknown_step(tmp_path: Path) -> None:
    """A step not present in the frame means empty output — raise, not silently plot."""
    with pytest.raises(ValueError, match="not in seg_rates"):
        plot_segment_rates(_segment_frame(), "view->purchase", tmp_path / "x.png")


# --- plot_price_conversion_curve ------------------------------------------


def _price_sessions_frame(n: int = 50) -> pl.DataFrame:
    """Deterministic sessions_features-shaped fixture for the price curve.

    Prices rise linearly so the decile bins separate cleanly. Flags follow small
    interleaved patterns (every session viewed, every third session did not
    cart, a quarter of carted sessions purchased) so every bin keeps non-zero
    denominators for both plotted steps.
    """
    rows = [
        {
            "median_price": 1.0 + 3.0 * i,
            "has_view": True,
            "has_cart": i % 3 != 0,
            "has_purchase": i % 3 != 0 and i % 4 == 0,
        }
        for i in range(n)
    ]
    return pl.DataFrame(rows)


def test_plot_price_conversion_curve_writes_png_into_missing_parent_dirs(tmp_path: Path) -> None:
    """A PNG must land at out_path with parents created; figure closed, nothing returned."""
    out = tmp_path / "figures" / "price.png"
    assert not out.parent.exists()

    result = plot_price_conversion_curve(_price_sessions_frame(), out)

    assert out.exists()
    assert out.stat().st_size > 0
    assert _png_signature(out) == b"\x89PNG\r\n\x1a\n"
    assert result is None
    assert plt.get_fignums() == []


def test_plot_price_conversion_curve_draws_two_banded_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One line per funnel step (view->cart, cart->purchase) with a CI ribbon each."""
    captured = _spy_on_close(monkeypatch)

    plot_price_conversion_curve(_price_sessions_frame(), tmp_path / "price.png")

    ax = captured["ax"]
    line_labels = [line.get_label() for line in ax.get_lines()]
    assert line_labels == ["view->cart", "cart->purchase"]
    n_ribbons = sum(1 for c in ax.collections if isinstance(c, PolyCollection))
    assert n_ribbons == 2
    assert "median price" in ax.get_title().lower()
    assert plt.get_fignums() == []


def test_plot_price_conversion_curve_drops_null_prices_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Null median_price rows are documented-excluded (unpriceable), not silently kept.

    The five nulled rows were the five most expensive ones (136..148 EUR). If
    they leaked into the quantiles/bins, the top bin's x would reach into that
    price region; because the plot's x values are in-bin medians of the *priced*
    rows (max 133 EUR), any x >= 136 proves the nulls were fed to the bins.
    """
    frame = _price_sessions_frame().with_columns(
        pl.when(pl.int_range(pl.len()) >= 45)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("median_price"))
        .alias("median_price")
    )
    captured = _spy_on_close(monkeypatch)

    plot_price_conversion_curve(frame, tmp_path / "price.png")

    ax = captured["ax"]
    plotted_xs = [x for line in ax.get_lines() for x in line.get_xdata()]
    assert plotted_xs, "expected at least one plotted rate line"
    assert min(plotted_xs) >= 1.0  # lowest priced row is 1 EUR
    assert max(plotted_xs) < 136.0, "nulled rows (136+) must not anchor any bin"
    assert (tmp_path / "price.png").exists()
    assert (tmp_path / "price.png").stat().st_size > 0
    assert plt.get_fignums() == []


def test_plot_price_conversion_curve_raises_when_no_binable_rows(tmp_path: Path) -> None:
    """A frame whose median_price is entirely null has nothing to plot — raise loudly."""
    all_null = _price_sessions_frame(10).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("median_price")
    )
    with pytest.raises(ValueError, match="median_price"):
        plot_price_conversion_curve(all_null, tmp_path / "x.png")


def test_plot_price_conversion_curve_skips_bins_without_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bin with no has_cart rows must not break cart->purchase — that point is NaN.

    Prices are 0..8 plus an isolated 1000 EUR outlier; the 90% quantile break
    lands at ~107 EUR, so the purely-expensive top bin holds exactly the outlier
    row, whose has_cart is False. Its cart->purchase basis is therefore empty and
    the renderer must leave exactly one NaN gap there rather than fabricating a
    number or a fake join across the gap.
    """
    rows = [
        {"median_price": float(i), "has_view": True, "has_cart": True, "has_purchase": True}
        for i in range(9)
    ]
    rows.append({"median_price": 1000.0, "has_view": True, "has_cart": False, "has_purchase": False})
    captured = _spy_on_close(monkeypatch)

    plot_price_conversion_curve(pl.DataFrame(rows), tmp_path / "price.png")

    cart_line = next(line for line in captured["ax"].get_lines() if line.get_label() == "cart->purchase")
    ydata = [float(y) for y in cart_line.get_ydata()]
    assert sum(1 for y in ydata if math.isnan(y)) == 1, "exactly the empty-basis bin must be NaN"
    assert (tmp_path / "price.png").exists()
    assert (tmp_path / "price.png").stat().st_size > 0
    assert plt.get_fignums() == []


def test_plot_price_conversion_curve_fails_fast_on_missing_columns(tmp_path: Path) -> None:
    """Missing contract columns must raise, listing the absent ones (fail-fast)."""
    with pytest.raises(ValueError, match="has_cart"):
        plot_price_conversion_curve(
            _price_sessions_frame().drop("has_cart"), tmp_path / "x.png"
        )
