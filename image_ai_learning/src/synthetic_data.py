"""
Bộ sinh dữ liệu ảnh tổng hợp (synthetic image datasets) phục vụ học tập và kiểm thử pipeline offline.
Lưu ý sư phạm:
- Dữ liệu tổng hợp giúp hiểu rõ bản chất toán học, kiểm tra nhanh pipeline mà không phụ thuộc internet.
- Kết quả đạt được trên dữ liệu tổng hợp KHÔNG dùng để kết luận độ chính xác trên ảnh tự nhiên phức tạp.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


CLASS_NAMES_SHAPES: List[str] = ["circle", "square", "triangle", "cross"]


def _draw_single_shape(
    canvas: np.ndarray,
    shape_name: str,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Vẽ một hình học lên `canvas` (RGB) và trả về mặt nạ nhị phân (binary mask) của hình đó."""
    h, w = canvas.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = center

    if shape_name == "circle":
        cv2.circle(mask, (cx, cy), radius, 255, -1, lineType=cv2.LINE_AA)
    elif shape_name == "square":
        rect = ((float(cx), float(cy)), (float(radius * 1.8), float(radius * 1.8)), float(angle_deg))
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillConvexPoly(mask, box, 255, lineType=cv2.LINE_AA)
    elif shape_name == "triangle":
        angles = np.deg2rad(np.array([90.0, 210.0, 330.0]) + angle_deg)
        pts = np.stack(
            [
                cx + radius * 1.15 * np.cos(angles),
                cy - radius * 1.15 * np.sin(angles),
            ],
            axis=1,
        ).astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 255, lineType=cv2.LINE_AA)
    elif shape_name == "cross":
        thickness = max(3, int(radius * 0.55))
        cv2.rectangle(
            mask,
            (cx - radius, cy - thickness // 2),
            (cx + radius, cy + thickness // 2),
            255,
            -1,
        )
        cv2.rectangle(
            mask,
            (cx - thickness // 2, cy - radius),
            (cx + thickness // 2, cy + radius),
            255,
            -1,
        )
        if abs(angle_deg) > 1.0:
            rot_mat = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle_deg), 1.0)
            mask = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST)
    else:
        raise ValueError(f"Unknown shape_name: {shape_name}")

    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    fg = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    blended = canvas.astype(np.float32) * (1.0 - alpha) + fg * alpha
    canvas[:] = np.clip(blended, 0, 255).astype(np.uint8)
    return (mask > 127).astype(np.uint8)


def generate_synthetic_shapes_dataset(
    num_samples: int = 320,
    image_size: int = 32,
    seed: int = 42,
    noise_std: float = 10.0,
) -> Dict[str, np.ndarray | List[str]]:
    """
    Tạo bộ dữ liệu phân loại ảnh 4 lớp (`circle`, `square`, `triangle`, `cross`).
    Trả về dictionary gồm:
    - `images`: mảng `uint8` có shape `(N, H, W, 3)` theo quy ước RGB.
    - `labels`: mảng `int64` có shape `(N,)` với giá trị `0..3`.
    - `group_ids`: mảng `int64` mô phỏng ID phiên chụp/vật thể (`0..num_groups-1`) để thực hành Group Split.
    - `class_names`: danh sách tên lớp.
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((num_samples, image_size, image_size, 3), dtype=np.uint8)
    labels = np.zeros(num_samples, dtype=np.int64)
    group_ids = np.zeros(num_samples, dtype=np.int64)

    palette = [
        (225, 75, 65),   # Đỏ cam
        (60, 145, 230),  # Xanh dương
        (65, 185, 110),  # Xanh lá
        (235, 180, 50),  # Vàng hổ phách
        (165, 95, 215),  # Tím
    ]

    num_groups = max(8, num_samples // 8)
    for idx in range(num_samples):
        cls_idx = idx % len(CLASS_NAMES_SHAPES)
        shape_name = CLASS_NAMES_SHAPES[cls_idx]
        grp_id = (idx // 4) % num_groups

        # Nền gradient nhẹ kèm nhiễu Gaussian
        bg_base = rng.integers(20, 60, size=(3,), dtype=np.int32)
        y_grid, x_grid = np.mgrid[0:image_size, 0:image_size]
        grad = ((x_grid + y_grid) / (2.0 * image_size) * 25.0).astype(np.int32)[..., None]
        canvas = np.clip(bg_base + grad, 0, 255).astype(np.uint8)

        margin = max(6, image_size // 4)
        cx = int(rng.integers(margin, image_size - margin))
        cy = int(rng.integers(margin, image_size - margin))
        radius = int(rng.integers(max(5, image_size // 5), max(7, image_size // 3)))
        color = palette[int(rng.integers(0, len(palette)))]
        angle = float(rng.uniform(-25.0, 25.0))

        _draw_single_shape(canvas, shape_name, (cx, cy), radius, color, angle_deg=angle)

        if noise_std > 0:
            noise = rng.normal(0.0, noise_std, size=canvas.shape)
            canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        images[idx] = canvas
        labels[idx] = cls_idx
        group_ids[idx] = grp_id

    return {
        "images": images,
        "labels": labels,
        "group_ids": group_ids,
        "class_names": list(CLASS_NAMES_SHAPES),
    }


def generate_detection_dataset(
    num_images: int = 80,
    image_size: int = 96,
    max_objects: int = 3,
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """
    Tạo dữ liệu cho bài toán Phát hiện vật thể (Object Detection).
    Nhãn `0` dành riêng cho nền (`background`), các lớp vật thể được đánh số từ `1..3`:
    `1: circle`, `2: square`, `3: triangle`.
    Mỗi phần tử gồm:
    - `image`: mảng `uint8` shape `(H, W, 3)`
    - `boxes`: mảng `float32` shape `(M, 4)` theo định dạng `[x_min, y_min, x_max, y_max]`
    - `labels`: mảng `int64` shape `(M,)`
    """
    rng = np.random.default_rng(seed)
    det_shapes = ["circle", "square", "triangle"]
    palette = [(230, 80, 70), (70, 155, 235), (75, 195, 115), (240, 185, 60)]
    dataset: List[Dict[str, np.ndarray]] = []

    for _ in range(num_images):
        canvas = np.full((image_size, image_size, 3), 32, dtype=np.uint8)
        num_objs = int(rng.integers(1, max_objects + 1))
        boxes: List[List[float]] = []
        labels: List[int] = []
        occupied_centers: List[Tuple[int, int]] = []

        for _obj_i in range(num_objs):
            cls_id = int(rng.integers(1, len(det_shapes) + 1))
            shape_name = det_shapes[cls_id - 1]
            radius = int(rng.integers(image_size // 9, image_size // 5))
            margin = radius + 4

            # Thử chọn vị trí ít chồng lấp quá mức
            cx, cy = image_size // 2, image_size // 2
            for _attempt in range(15):
                cand_x = int(rng.integers(margin, image_size - margin))
                cand_y = int(rng.integers(margin, image_size - margin))
                if all((cand_x - ox) ** 2 + (cand_y - oy) ** 2 > (radius * 1.6) ** 2 for ox, oy in occupied_centers):
                    cx, cy = cand_x, cand_y
                    break

            occupied_centers.append((cx, cy))
            color = palette[int(rng.integers(0, len(palette)))]
            obj_mask = _draw_single_shape(canvas, shape_name, (cx, cy), radius, color, angle_deg=float(rng.uniform(-15, 15)))
            ys, xs = np.where(obj_mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min, x_max = float(xs.min()), float(xs.max() + 1)
            y_min, y_max = float(ys.min()), float(ys.max() + 1)
            if x_max - x_min >= 4 and y_max - y_min >= 4:
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(cls_id)

        noise = rng.normal(0.0, 6.0, size=canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        dataset.append(
            {
                "image": canvas,
                "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
                "labels": np.array(labels, dtype=np.int64),
            }
        )

    return dataset


def generate_segmentation_dataset(
    num_images: int = 120,
    image_size: int = 48,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Tạo dữ liệu Phân vùng ảnh nhị phân (Binary Semantic Segmentation):
    - `images`: `(N, H, W, 3)` kiểu `uint8`
    - `masks`: `(N, H, W)` kiểu `uint8` với giá trị `0` (nền) và `1` (vật thể)
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((num_images, image_size, image_size, 3), dtype=np.uint8)
    masks = np.zeros((num_images, image_size, image_size), dtype=np.uint8)
    shapes = ["circle", "square", "triangle"]
    palette = [(220, 85, 75), (75, 160, 230), (80, 195, 120)]

    for i in range(num_images):
        canvas = np.full((image_size, image_size, 3), 38, dtype=np.uint8)
        full_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        num_objs = int(rng.integers(1, 3))
        for _ in range(num_objs):
            s_name = shapes[int(rng.integers(0, len(shapes)))]
            radius = int(rng.integers(image_size // 7, image_size // 4))
            cx = int(rng.integers(radius + 3, image_size - radius - 3))
            cy = int(rng.integers(radius + 3, image_size - radius - 3))
            color = palette[int(rng.integers(0, len(palette)))]
            m = _draw_single_shape(canvas, s_name, (cx, cy), radius, color, angle_deg=float(rng.uniform(-20, 20)))
            full_mask = np.maximum(full_mask, m)

        noise = rng.normal(0.0, 8.0, size=canvas.shape)
        images[i] = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        masks[i] = full_mask

    return {"images": images, "masks": masks}


def generate_anomaly_dataset(
    num_normal: int = 160,
    num_anomalous: int = 40,
    image_size: int = 48,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Tạo dữ liệu kiểm tra lỗi bề mặt công nghiệp giả lập (Surface Anomaly Detection):
    - Ảnh bình thường (`label = 0`): bề mặt kim loại phay xước định kỳ cộng đĩa tròn đồng tâm.
    - Ảnh bất thường (`label = 1`): xuất hiện vết xước chéo sáng/tối hoặc vết lõm bất thường kèm mặt nạ lỗi `anomaly_masks`.
    """
    rng = np.random.default_rng(seed)
    total = num_normal + num_anomalous
    images = np.zeros((total, image_size, image_size, 3), dtype=np.uint8)
    labels = np.zeros(total, dtype=np.int64)
    anomaly_masks = np.zeros((total, image_size, image_size), dtype=np.uint8)

    y_grid, x_grid = np.mgrid[0:image_size, 0:image_size]

    for i in range(total):
        is_anomaly = i >= num_normal
        phase = float(rng.uniform(0, 2 * np.pi))
        stripes = 12.0 * np.sin(2.0 * np.pi * y_grid / 6.0 + phase)
        base_gray = np.clip(115.0 + stripes, 0, 255)
        canvas = np.stack([base_gray, base_gray, base_gray + 8.0], axis=-1).astype(np.uint8)

        # Vẽ vòng đệm chuẩn ở tâm
        center = (image_size // 2, image_size // 2)
        radius_outer = image_size // 3
        radius_inner = image_size // 6
        cv2.circle(canvas, center, radius_outer, (175, 180, 190), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, radius_inner, (95, 100, 110), -1, lineType=cv2.LINE_AA)

        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        if is_anomaly:
            labels[i] = 1
            defect_type = int(rng.integers(0, 2))
            if defect_type == 0:
                # Vết xước sáng/tối
                pt1 = (int(rng.integers(8, image_size // 2)), int(rng.integers(8, image_size // 2)))
                pt2 = (int(rng.integers(image_size // 2, image_size - 8)), int(rng.integers(image_size // 2, image_size - 8)))
                cv2.line(canvas, pt1, pt2, (245, 50, 50), 2)
                cv2.line(mask, pt1, pt2, 1, 2)
            else:
                # Đốm bẩn/lõm
                dx = int(rng.integers(image_size // 3, 2 * image_size // 3))
                dy = int(rng.integers(image_size // 3, 2 * image_size // 3))
                r_def = int(rng.integers(3, 6))
                cv2.circle(canvas, (dx, dy), r_def, (25, 230, 230), -1)
                cv2.circle(mask, (dx, dy), r_def, 1, -1)

        noise = rng.normal(0.0, 4.0, size=canvas.shape)
        images[i] = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        anomaly_masks[i] = mask

    return {
        "images": images,
        "labels": labels,
        "anomaly_masks": anomaly_masks,
    }
