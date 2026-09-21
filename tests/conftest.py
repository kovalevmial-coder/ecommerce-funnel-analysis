"""Shared fixtures for the test suite.

The data fixture exposes a path helper for future synthetic-data tests so they
never hard-code absolute paths; nothing else is needed at scaffold time.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _REPO_ROOT / "data"


@pytest.fixture
def data_dir() -> Path:
    """Return the absolute path to the repository data/ directory."""
    return DATA_DIR