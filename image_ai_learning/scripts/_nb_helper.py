"""
Module hỗ trợ khởi tạo và xác thực file Jupyter Notebook chuẩn nbformat v4.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def create_validated_notebook(
    out_path: Path,
    cells_spec: List[Tuple[str, str]],
) -> Path:
    """
    Tạo một notebook chuẩn `nbformat` v4 từ danh sách `(cell_type, source)`.
    - `cell_type`: `'md'` (Markdown) hoặc `'code'` (Code Python).
    - Tự động kiểm tra cú pháp `ast.parse` cho mọi code cell.
    - Tự động kiểm tra không có chuỗi `'TODO'` hoặc `'...'` làm placeholder code.
    - Tự động gọi `nbformat.validate(nb)`.
    """
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3.12 (image_ai_learning)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.10",
        },
    }

    for idx, (ctype, content) in enumerate(cells_spec):
        clean_src = content.strip() + "\n"
        if ctype == "md":
            nb.cells.append(new_markdown_cell(clean_src))
        elif ctype == "code":
            try:
                ast.parse(clean_src)
            except SyntaxError as exc:
                raise SyntaxError(f"Lỗi cú pháp tại {out_path.name} cell #{idx}: {exc}\n{clean_src}") from exc
            if "TODO" in clean_src:
                raise ValueError(f"Phát hiện TODO trong code cell #{idx} của {out_path.name}")
            nb.cells.append(new_code_cell(clean_src))
        else:
            raise ValueError(f"Loại cell không hợp lệ: {ctype}")

    nbformat.validate(nb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return out_path
