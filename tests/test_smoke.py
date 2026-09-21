"""Scaffold smoke test: proves the venv, package layout, and pytest config work."""
from __future__ import annotations

import src


def test_src_importable_with_version() -> None:
    """The src package must be importable and expose the project version."""
    assert src is not None
    assert src.__version__ == "0.1.0"