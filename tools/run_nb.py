"""Execute a notebook top to bottom in a fresh kernel and save it in place with all outputs kept.

Usage (from anywhere): python tools/run_nb.py [nb.ipynb] [working_dir]
Defaults: <repo>/main.ipynb, executed with <repo> as the working directory. Exits non-zero if a cell fails
(the notebook is still saved, including the error output).
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO = Path(__file__).resolve().parent.parent
path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "main.ipynb"
cwd = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO

nb = nbformat.read(path, as_version=4)
t0 = time.time()
client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(cwd)}})
failed = None
try:
    client.execute()
except CellExecutionError as e:
    failed = str(e)[-3000:]
nbformat.write(nb, path)
print(f"{'FAILED' if failed else 'OK'} ({time.time() - t0:.1f}s): {path}")

for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    for out in cell.get("outputs", []):
        if out.output_type == "stream":
            print(out.text, end="")
        elif out.output_type in ("execute_result", "display_data"):
            print(out["data"].get("text/plain", ""))
        elif out.output_type == "error":
            print("ERROR:", out.ename, out.evalue)
if failed:
    print(failed)
    sys.exit(1)
