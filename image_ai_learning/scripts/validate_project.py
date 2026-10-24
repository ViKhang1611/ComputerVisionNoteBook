"""
Script kiểm tra toàn diện cấu trúc dự án `image_ai_learning`:
1. Kiểm tra sự tồn tại của toàn bộ 23 Jupyter Notebooks và các file tài liệu bắt buộc.
2. Xác thực schema `nbformat` v4 cho cả 23 notebooks.
3. Kiểm tra cú pháp Python (`ast.parse`) cho mọi code cell trong cả 23 notebooks.
4. Kiểm tra không có `TODO` hoặc đường dẫn tuyệt đối của máy cá nhân trong notebook.
5. Kiểm tra tính hợp lệ của toàn bộ liên kết file nội bộ (internal Markdown links).
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

EXPECTED_NOTEBOOKS = [
    "00_start_here.ipynb",
    "01_python_numpy_for_images.ipynb",
    "02_image_fundamentals.ipynb",
    "03_classical_image_processing.ipynb",
    "04_math_and_ml_foundations.ipynb",
    "05_datasets_and_data_leakage.ipynb",
    "06_features_and_classical_ml.ipynb",
    "07_clustering_and_dimensionality_reduction.ipynb",
    "08_neural_network_from_scratch.ipynb",
    "09_pytorch_and_autograd.ipynb",
    "10_cnn_from_scratch.ipynb",
    "11_training_validation_and_debugging.ipynb",
    "12_transfer_learning_and_fine_tuning.ipynb",
    "00_start_here.ipynb",
    "13_object_detection.ipynb",
    "14_image_segmentation.ipynb",
    "15_autoencoders_and_anomaly_detection.ipynb",
    "16_metric_learning_and_image_retrieval.ipynb",
    "17_vision_transformers_and_self_supervised_learning.ipynb",
    "18_generative_models.ipynb",
    "19_ocr_and_visual_tasks_overview.ipynb",
    "20_evaluation_experiment_design_and_tuning.ipynb",
    "21_inference_export_and_optimization.ipynb",
    "22_capstone_project.ipynb",
]
EXPECTED_NOTEBOOKS = sorted(set(EXPECTED_NOTEBOOKS))

EXPECTED_DOCS = [
    "README.md",
    "STUDY_ROADMAP.md",
    "MODEL_CATALOG.md",
    "KEYWORD_GLOSSARY.md",
    "TROUBLESHOOTING.md",
    "requirements.txt",
    "requirements-optional.txt",
    "data/README.md",
]


def validate_all() -> dict:
    report = {
        "notebooks_checked": 0,
        "total_cells": 0,
        "total_code_cells": 0,
        "total_md_cells": 0,
        "nbformat_valid": True,
        "ast_syntax_valid": True,
        "internal_links_valid": True,
        "errors": [],
    }

    # 1. Check required docs
    for doc_rel in EXPECTED_DOCS:
        p = PROJECT_ROOT / doc_rel
        if not p.is_file():
            report["errors"].append(f"Thiếu file tài liệu: {doc_rel}")

    # 2. Validate all 23 notebooks
    for nb_name in EXPECTED_NOTEBOOKS:
        nb_path = NOTEBOOKS_DIR / nb_name
        if not nb_path.is_file():
            report["nbformat_valid"] = False
            report["errors"].append(f"Thiếu notebook: {nb_name}")
            continue

        nb = nbformat.read(nb_path, as_version=4)
        nbformat.validate(nb)
        report["notebooks_checked"] += 1

        for c_idx, cell in enumerate(nb.cells):
            report["total_cells"] += 1
            src = cell.source
            if "C:\\Users\\Dell" in src or "C:/Users/Dell" in src:
                report["errors"].append(f"Phát hiện đường dẫn tuyệt đối máy cá nhân trong {nb_name} cell #{c_idx}")

            if cell.cell_type == "markdown":
                report["total_md_cells"] += 1
            elif cell.cell_type == "code":
                report["total_code_cells"] += 1
                try:
                    ast.parse(src)
                except SyntaxError as exc:
                    report["ast_syntax_valid"] = False
                    report["errors"].append(f"Lỗi cú pháp AST tại {nb_name} cell #{c_idx}: {exc}")
                if "TODO" in src:
                    report["errors"].append(f"Phát hiện TODO trong {nb_name} cell #{c_idx}")

    # 3. Validate internal markdown links
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for doc_rel in EXPECTED_DOCS:
        doc_path = PROJECT_ROOT / doc_rel
        if not doc_path.is_file():
            continue
        text = doc_path.read_text(encoding="utf-8")
        for match in link_pattern.finditer(text):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "#", "mailto:", "file://", "conversation://")):
                continue
            clean_target = target.split("#")[0]
            if not clean_target:
                continue
            resolved = (doc_path.parent / clean_target).resolve()
            if not resolved.exists():
                report["internal_links_valid"] = False
                report["errors"].append(f"Liên kết hỏng trong {doc_rel}: {target} -> {resolved}")

    return report


if __name__ == "__main__":
    res = validate_all()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if res["errors"]:
        sys.exit(1)
    print(
        f"[SUCCESS] Đã xác thực toàn bộ {res['notebooks_checked']} notebooks "
        f"({res['total_code_cells']} code cells, {res['total_md_cells']} markdown cells) và các liên kết nội bộ!"
    )
