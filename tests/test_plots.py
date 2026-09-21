"""Tests for the funnel chart module (src/plots.py).

The tests pin the file-level contract of ``plot_funnel`` — a PNG materialises
at the requested path (creating missing parent directories), is non-empty and
starts with the PNG magic bytes — rather than any pixel content: aesthetics are
deliberately out of scope for CI (the human sanity look happens in the stage
notebook / report). The ``assets_path`` path-by-convention helper and the
fail-fast column guard are pinned here too.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.plots import assets_path, plot_funnel

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


def test_plot_funnel_writes_png_into_deeply_nested_path(tmp_path: Path) -> None:
    """Deeply nested out_path parents (>/1 level) must be created recursively."""
    out = tmp_path / "a" / "b" / "c" / "funnel.png"
    assert not out.parent.exists()

    plot_funnel(_funnel_frame(), out)

    assert out.exists()
    assert out.stat().st_size > 0
    assert _png_signature(out) == b"\x89PNG\r\n\x1a\n"


def test_plot_funnel_fails_fast_on_missing_columns() -> None:
    """Missing contract columns must raise, listing the absent ones (fail-fast)."""
    with pytest.raises(ValueError, match="ci_high"):
        plot_funnel(_funnel_frame().drop("ci_high"), Path("whatever.png"))