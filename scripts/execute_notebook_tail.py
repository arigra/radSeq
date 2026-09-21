"""Execute only the appended cells of radSeq.ipynb and store their outputs.

The notebook is read, not re-run, so its results have to be saved in it. Only
the tail is executed: the earlier cells load DiT checkpoints and sample from
them, which needs the GPU and many minutes, and re-running them would fight
whatever training is in flight.

A temporary notebook is built from the setup prelude plus the tail, executed,
and the outputs are copied back cell by cell.
"""
import argparse
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "radSeq.ipynb"
PRELUDE = (3, 4)          # imports, then ROOT / sys.path / device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker", default="## Step 2 — does generated data actually help?")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    nb = nbformat.reads(NB.read_text(), as_version=4)
    start = next(i for i, c in enumerate(nb.cells)
                 if args.marker in "".join(c.source))
    tail = list(range(start, len(nb.cells)))
    code_idx = [i for i in tail if nb.cells[i].cell_type == "code"]
    print(f"executing {len(code_idx)} code cells from cell {start}")

    tmp = nbformat.v4.new_notebook(metadata=nb.metadata)
    tmp.cells = [nb.cells[i] for i in PRELUDE] + [nb.cells[i] for i in code_idx]
    # Matplotlib must render into the notebook rather than pop a window.
    tmp.cells.insert(0, nbformat.v4.new_code_cell("%matplotlib inline"))

    client = NotebookClient(tmp, timeout=args.timeout,
                            kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT / "notebooks")}})
    client.execute()

    offset = 1 + len(PRELUDE)
    for n, i in enumerate(code_idx):
        executed = tmp.cells[offset + n]
        nb.cells[i].outputs = executed.outputs
        nb.cells[i].execution_count = executed.execution_count
        errors = [o for o in executed.outputs if o.output_type == "error"]
        status = f"ERROR {errors[0].ename}" if errors else f"{len(executed.outputs)} output(s)"
        print(f"  cell {i}: {status}")

    NB.write_text(nbformat.writes(nb) + "\n")
    print(f"wrote {NB}")


if __name__ == "__main__":
    main()
