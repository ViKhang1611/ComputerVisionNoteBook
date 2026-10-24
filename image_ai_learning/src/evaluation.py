"""
Các hàm tính toán độ đo đánh giá (evaluation metrics) cho từng nhóm bài toán:
1. Phân loại (Classification): Accuracy, Precision, Recall, Specificity, F1-score (Macro/Micro/Weighted), Confusion Matrix.
2. Phát hiện vật thể (Object Detection): Bounding Box IoU, Non-Maximum Suppression (NMS), Average Precision (AP).
3. Phân vùng ảnh (Image Segmentation): Pixel Accuracy, Intersection over Union (IoU / Jaccard), hệ số Dice.
4. Hiệu năng suy luận (Inference Performance): Độ trễ trung bình (Latency ms) và thông lượng (Throughput FPS).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def compute_confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Tính ma trận nhầm lẫn (Confusion Matrix) kích thước `(num_classes, num_classes)`:
    Hàng `i` là nhãn thực tế (`y_true = i`), cột `j` là nhãn dự đoán (`y_pred = j`).
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Tính bộ chỉ số đầy đủ cho bài toán phân loại đa lớp:
    - `accuracy`: Tỷ lệ dự đoán đúng tổng thể.
    - `per_class`: `precision`, `recall`, `specificity`, `f1`, `support` cho từng lớp.
    - `macro_f1`, `weighted_f1`, `macro_precision`, `macro_recall`, `macro_specificity`.
    """
    if class_names is None:
        num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
        class_names = [f"class_{i}" for i in range(num_classes)]
    else:
        num_classes = len(class_names)

    cm = compute_confusion_matrix_np(y_true, y_pred, num_classes)
    total_samples = int(cm.sum())
    accuracy = float(np.trace(cm) / max(1, total_samples))

    per_class: Dict[str, Dict[str, float]] = {}
    precisions, recalls, specificities, f1s, supports = [], [], [], [], []

    for c_idx, c_name in enumerate(class_names):
        tp = float(cm[c_idx, c_idx])
        fp = float(cm[:, c_idx].sum() - tp)
        fn = float(cm[c_idx, :].sum() - tp)
        tn = float(total_samples - tp - fp - fn)
        sup = float(cm[c_idx, :].sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[c_name] = {
            "precision": prec,
            "recall": rec,
            "specificity": spec,
            "f1": f1,
            "support": sup,
        }
        precisions.append(prec)
        recalls.append(rec)
        specificities.append(spec)
        f1s.append(f1)
        supports.append(sup)

    supports_arr = np.array(supports, dtype=np.float64)
    weights = supports_arr / max(1.0, supports_arr.sum())

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_specificity": float(np.mean(specificities)),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.sum(np.array(f1s) * weights)),
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def compute_bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Tính chỉ số Intersection over Union (IoU) giữa hai khung giới hạn (bounding boxes)
    theo định dạng `[x_min, y_min, x_max, y_max]`.
    """
    x_left = max(float(box_a[0]), float(box_b[0]))
    y_top = max(float(box_a[1]), float(box_b[1]))
    x_right = min(float(box_a[2]), float(box_b[2]))
    y_bottom = min(float(box_a[3]), float(box_b[3]))

    inter_w = max(0.0, x_right - x_left)
    inter_h = max(0.0, y_bottom - y_top)
    inter_area = inter_w * inter_h

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_segmentation_iou_dice(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Tính độ đo `IoU` (Jaccard Index), hệ số `Dice` (F1-score mức điểm ảnh) và `Pixel Accuracy`
    cho bài toán phân vùng nhị phân (`0` và `1`).
    """
    p = (pred_mask > 0).astype(np.float32).ravel()
    t = (true_mask > 0).astype(np.float32).ravel()

    intersection = float((p * t).sum())
    union = float(p.sum() + t.sum() - intersection)
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (float(p.sum() + t.sum()) + smooth)
    pixel_acc = float((p == t).mean())

    return {
        "iou": float(iou),
        "dice": float(dice),
        "pixel_accuracy": pixel_acc,
    }


@torch.no_grad()
def measure_inference_latency(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
    device: Optional[torch.device] = None,
    warmup_runs: int = 5,
    timed_runs: int = 20,
) -> Dict[str, float]:
    """
    Đo độ trễ suy luận trung bình (`latency_ms`) và thông lượng (`throughput_fps`).
    Có bước chạy làm nóng (`warmup_runs`) để tránh tính chi phí khởi tạo bộ nhớ lần đầu.
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    model.to(device)
    dummy = torch.randn(*input_shape, device=device)

    for _ in range(warmup_runs):
        _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_t = time.perf_counter()
    for _ in range(timed_runs):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_t

    batch_size = input_shape[0]
    avg_latency_ms = (elapsed / max(1, timed_runs)) * 1000.0
    throughput_fps = (timed_runs * batch_size) / max(1e-9, elapsed)
    return {
        "latency_ms_per_batch": float(avg_latency_ms),
        "throughput_samples_per_sec": float(throughput_fps),
    }
