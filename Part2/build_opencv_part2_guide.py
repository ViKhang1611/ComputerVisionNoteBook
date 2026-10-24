"""Build and execute Huong_Dan_OpenCV_Phan2_Lo_Trinh_Tiep_Theo.ipynb from nb2_part1..3."""
import json
import os
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook
from nbclient import NotebookClient

import nb2_part1
import nb2_part2  # noqa: F401
import nb2_part3  # noqa: F401

root = Path(__file__).parent
out_path = root / "Huong_Dan_OpenCV_Phan2_Lo_Trinh_Tiep_Theo.ipynb"

os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(root / ".cvip-mpl")
os.environ["IPYTHONDIR"] = str(root / ".cvip-ipython")
os.environ["JUPYTER_RUNTIME_DIR"] = str(root / ".cvip-jupyter-runtime")
os.environ["JUPYTER_PATH"] = str(root / ".cvip-jupyter")

kernel_dir = root / ".cvip-jupyter" / "kernels" / "cvip-verify"
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / "kernel.json").write_text(
    json.dumps(
        {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "CVIP verification",
            "language": "python",
        }
    ),
    encoding="utf-8",
)

nb = new_notebook(
    cells=nb2_part1.CELLS,
    metadata={
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": sys.version.split()[0],
        },
    },
)

nbformat.validate(nb)
nbformat.write(nb, out_path)
print(f"Wrote unexecuted notebook with {len(nb.cells)} cells to {out_path.name}")

client = NotebookClient(
    nb,
    timeout=300,
    kernel_name="cvip-verify",
    resources={"metadata": {"path": str(root)}},
)
client.execute()

nbformat.validate(nb)
nbformat.write(nb, out_path)

n_code = sum(1 for c in nb.cells if c.cell_type == "code")
n_md = sum(1 for c in nb.cells if c.cell_type == "markdown")
n_figs = sum(
    1
    for c in nb.cells
    for out in c.get("outputs", [])
    if "image/png" in out.get("data", {})
)
print(
    f"SUCCESS: Executed {n_code} code cells ({n_md} markdown cells), "
    f"embedded {n_figs} figures into {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)."
)
