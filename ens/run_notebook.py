"""Execute a notebook in place (outputs saved into the file).  python ens/run_notebook.py main.ipynb [--smoke OUT]
--smoke writes a 2-epoch copy to OUT (Stage 1 and Stage 2) that scores dev in place of test (code check only;
never touches test)."""
import os, sys, time
import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1]
nb = nbformat.read(src, as_version=4)
out = src
if "--smoke" in sys.argv:
    out = sys.argv[sys.argv.index("--smoke") + 1]
    for c in nb.cells:
        c.source = (c.source.replace("epochs=40", "epochs=2").replace("swa_start=30", "swa_start=2")
                    .replace('run_ensemble("test")', 'run_ensemble("dev")').replace('data["test"]', 'data["dev"]')
                    .replace('s1_run_split("test")', 's1_run_split("dev")')
                    .replace('"stage1"', '"stage1_smoke"').replace('"stage2"', '"stage2_smoke"'))
t0 = time.time()
try:
    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": ROOT}}).execute()
finally:
    nbformat.write(nb, out)
    print(f"executed {src} -> {out} in {(time.time() - t0) / 60:.1f} min", flush=True)
