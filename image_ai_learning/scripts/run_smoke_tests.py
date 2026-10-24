"""
Script thực thi kiểm thử thực tế (Real Execution & Smoke Test) cho toàn bộ 23 Jupyter Notebooks:
- Thực thi từng notebook từ cell đầu tiên đến cell cuối cùng trong không gian biến độc lập (`fresh namespace`),
  đảm bảo không notebook nào phụ thuộc biến của notebook khác.
- Ghi nhận kết quả `stdout` thực tế và hình ảnh đồ thị `matplotlib` thực tế (`image/png` base64)
  vào đúng chuẩn `nbformat` v4 (`stream` và `display_data` outputs).
- Xác nhận lại `nbformat.validate(nb)` sau khi ghi output.
- Lưu báo cáo thời gian chạy và trạng thái từng notebook vào `outputs/reports/execution_summary.json`.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Đảm bảo Matplotlib dùng backend Agg (không bật cửa sổ GUI khi chạy tự động)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_output

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def execute_notebook_in_isolated_namespace(nb_path: Path) -> dict:
    """
    Chạy tuần tự mọi code cell của `nb_path` trong một dictionary `ns` mới hoàn toàn.
    Bắt toàn bộ stdout và các hình vẽ `plt.show()` thành output chuẩn của Jupyter Notebook.
    """
    nb = nbformat.read(nb_path, as_version=4)
    ns: dict = {"__name__": "__main__", "__file__": str(nb_path)}

    start_t = time.perf_counter()
    executed_cells = 0
    captured_figures = 0

    old_cwd = Path.cwd()
    os.chdir(NOTEBOOKS_DIR)

    try:
        for cell_idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue

            executed_cells += 1
            cell.execution_count = executed_cells
            cell.outputs = []

            stdout_buf = io.StringIO()
            plt.close("all")

            # Thay thế tạm plt.show() để chụp ảnh PNG của các figure đang mở
            figures_png_b64: list[str] = []

            def _custom_show(*args, **kwargs):
                for num in plt.get_fignums():
                    fig = plt.figure(num)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
                    buf.seek(0)
                    figures_png_b64.append(base64.b64encode(buf.read()).decode("ascii"))
                plt.close("all")

            orig_show = plt.show
            plt.show = _custom_show
            try:
                with contextlib.redirect_stdout(stdout_buf):
                    exec(compile(cell.source, filename=f"{nb_path.name}[cell_{cell_idx}]", mode="exec"), ns)
            finally:
                plt.show = orig_show

            out_text = stdout_buf.getvalue()
            if out_text:
                cell.outputs.append(new_output(output_type="stream", name="stdout", text=out_text))

            for b64_img in figures_png_b64:
                captured_figures += 1
                cell.outputs.append(
                    new_output(
                        output_type="display_data",
                        data={"image/png": b64_img, "text/plain": "<Figure>"},
                        metadata={},
                    )
                )

        nbformat.validate(nb)
        with nb_path.open("w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        elapsed = time.perf_counter() - start_t
        return {
            "notebook": nb_path.name,
            "status": "PASSED",
            "code_cells_executed": executed_cells,
            "figures_generated": captured_figures,
            "elapsed_sec": round(elapsed, 2),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start_t
        return {
            "notebook": nb_path.name,
            "status": "FAILED",
            "code_cells_executed": executed_cells,
            "elapsed_sec": round(elapsed, 2),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        os.chdir(old_cwd)
        plt.close("all")


def main() -> None:
    nb_files = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    results = []
    total_start = time.perf_counter()

    for nb_file in nb_files:
        res = execute_notebook_in_isolated_namespace(nb_file)
        results.append(res)
        status_str = f"[{res['status']}] {res['notebook']:52s} | {res['code_cells_executed']:2d} cells | {res['elapsed_sec']:5.2f}s"
        if res["status"] == "PASSED":
            status_str += f" | {res['figures_generated']} figs"
        else:
            status_str += f" | ERROR: {res.get('error')}"
        print(status_str, flush=True)

    total_elapsed = round(time.perf_counter() - total_start, 2)
    summary_path = PROJECT_ROOT / "outputs" / "reports" / "execution_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "total_notebooks": len(results),
        "passed": sum(1 for r in results if r["status"] == "PASSED"),
        "failed": sum(1 for r in results if r["status"] == "FAILED"),
        "total_elapsed_sec": total_elapsed,
        "results": results,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTổng kết: {summary_payload['passed']}/{summary_payload['total_notebooks']} notebooks PASSED trong {total_elapsed}s.")

    if summary_payload["failed"] > 0:
        for r in results:
            if r["status"] == "FAILED":
                print(f"\n--- CHI TIẾT LỖI TẠI {r['notebook']} ---")
                print(r.get("traceback", ""))
        sys.exit(1)


if __name__ == "__main__":
    main()
