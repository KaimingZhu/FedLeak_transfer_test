"""
@Encoding:      UTF-8
@File:          run_notebook.py

@Introduction:  script to run jupyter notebook in background.
@Author:        Kaiming Zhu
@Date:          2026/1/2 20:38:21
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


nb_path = Path(str(Path(__file__).parent.absolute()) + "/sanity_check_resize_showcase.ipynb")
out_path = Path(str(Path(__file__).parent.absolute()) + "/sanity_check_resize_showcase.result.ipynb")   # keep original untouched (recommended)


nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(
    nb,
    timeout=None,
    kernel_name="python3",
    allow_errors=True,
)


# Correct context manager in nbclient 0.10.0:
with client.setup_kernel():
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        client.execute_cell(cell, idx)
        nbformat.write(nb, out_path)
        print(f"Saved after cell {idx} -> {out_path}")