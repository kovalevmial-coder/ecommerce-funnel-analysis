"""Reproducible notebook build-and-execute runner.

Stage 05 gate keeps notebooks generated from scripts, not hand-edited: a
``.ipynb`` is an opaque JSON assembly, and a hand-edited one cannot prove its
outputs came from code. Every notebook in the project is therefore built by a
``notebooks/build_*.py`` script that declares markdown/code cells, executes
them against the pinned ``ecommerce-funnel`` kernel, saves the executed file
with outputs embedded, and fails loudly if any cell errored — executed
notebooks are committed, so ``errors=none`` is verifiable from git alone.
"""
from __future__ import annotations

import nbclient
import nbformat
from nbformat.notebooknode import NotebookNode
from pathlib import Path

KERNEL_NAME = "ecommerce-funnel"


def _title_metadata(title: str | None) -> NotebookNode:
    """Return an optional Jupyter markdown title cell for a notebook.

    nbformat has no first-class title; notebooks conventionally carry it as a
    level-1 markdown cell, which is what the plan (and readers) expect.
    """
    if title is None:
        return None
    return nbformat.v4.new_markdown_cell(f"# {title}")


def build_and_execute(
    name: str,
    cells: list[dict],
    out_dir: Path,
    title: str | None = None,
) -> Path:
    """Build and execute a notebook named ``name`` under ``out_dir``.

    Args:
        name: notebook filename without the ``.ipynb`` suffix.
        cells: ordered list of ``{"kind": "markdown"|"code", "source": str}``
            dicts; each dict is converted with the corresponding nbformat v4
            factory.
        out_dir: destination directory (created if missing).
        title: optional level-1 markdown title cell prepended to ``cells``.

    Returns:
        Path to the written, executed notebook.

    Raises:
        RuntimeError: if execution produced an error output, with the 0-based
            index of the offending cell in the message — errors must never be
            silently embedded in a committed notebook.

    Note:
        ``cells`` is consumed as ordered input, not stored, so callers may
        pass a generator if a script builds hundreds of cells.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.ipynb"

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "name": KERNEL_NAME,
        "display_name": "Python 3.12 (ecommerce-funnel)",
        "language": "python",
    }
    title_cell = _title_metadata(title)
    if title_cell is not None:
        notebook.cells.append(title_cell)
    for cell in cells:
        kind = cell["kind"]
        if kind == "markdown":
            notebook.cells.append(nbformat.v4.new_markdown_cell(cell["source"]))
        elif kind == "code":
            notebook.cells.append(nbformat.v4.new_code_cell(cell["source"]))
        else:
            raise ValueError(f"unsupported cell kind: {kind!r}")

    client = nbclient.NotebookClient(nb=notebook, kernel_name=KERNEL_NAME, timeout=600)
    client.execute()  # raises CellExecutionError on the first failed cell

    error_indexes = [
        i for i, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and any(output.output_type == "error" for output in cell.get("outputs", []))
    ]
    if error_indexes:
        # A CellExecutionError is itself fatal, but the explicit sweep guards
        # against kernels that die mid-way and leave partial error outputs.
        raise RuntimeError(
            f"{name}.ipynb: execution errors in cell(s) {error_indexes} were "
            "not raised by nbclient; refusing to write an error notebook"
        )

    nbformat.write(notebook, out_path)
    return out_path


if __name__ == "__main__":
    # Developer utility: run directly to build one cell-less notebook for
    # kernel wiring checks. Real notebooks always go through build_*.py.
    raise SystemExit("import build_notebook; not a standalone script")