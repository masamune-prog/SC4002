"""Execute a notebook top to bottom in a fresh kernel and save it with all outputs.

    python tools/run_notebook.py main.ipynb
"""
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
start = time.time()
NotebookClient(nb, timeout=None, kernel_name="python3",
               resources={"metadata": {"path": os.path.dirname(os.path.abspath(path))}}).execute()
nbformat.write(nb, path)
print(f"executed {path} in {(time.time() - start) / 60:.1f} min")
