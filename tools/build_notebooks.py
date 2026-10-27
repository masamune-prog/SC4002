"""Assemble the notebooks from the code cells in tools/core.py plus the explanations below.

    python tools/build_notebooks.py [main|stage1_tuning|stage2_experiments|all]

Building writes fresh, un-executed notebooks; run them top to bottom (or with tools/run_notebook.py) to fill in outputs.
"""
import os
import sys

import nbformat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from notebook_text import LAYOUTS  # noqa: E402


def load_cells():
    src = open(os.path.join(ROOT, "tools", "core.py"), encoding="utf-8").read()
    cells = {}
    for chunk in src.split("\n# %% ")[1:]:
        name, body = chunk.split("\n", 1)
        cells[name.strip()] = body.strip("\n")
    return cells


def build(name):
    cells = load_cells()
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, content in LAYOUTS[name]:
        if kind == "md":
            nb.cells.append(nbformat.v4.new_markdown_cell(content.strip("\n")))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(cells[content]))
    path = os.path.join(ROOT, f"{name}.ipynb")
    nbformat.write(nb, path)
    print("wrote", path)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    for n in (LAYOUTS if target == "all" else [target]):
        build(n)
