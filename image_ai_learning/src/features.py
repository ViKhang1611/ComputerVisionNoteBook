"""
Các hàm trích xuất đặc trưng thủ công (hand-crafted feature extraction) cho Machine Learning truyền thống:
1. Pixel Flattening (Trải phẳng giá trị điểm ảnh làm đường cơ sở - baseline).
2. Color Histogram (Biểu đồ tần suất màu sắc).
3. HOG - Histogram of Oriented Gradients (Biểu đồ hướng đạo hàm bậc nhất - bắt hình dáng/cạnh).
4. LBP - Local Binary Patterns (Mẫu nhị phân cục bộ - bắt kết cấu/vân bề mặt).

Lưu ý quan trọng:
- HOG và LBP là phương pháp TRÍCH XUẤT ĐẶC TRƯNG (Feature Extraction), KHÔNG phải mô hình phân loại.
- Đầu ra của chúng là các vector đặc trưng để đưa vào k-NN, Logistic Regression, SVM, Random Forest...
"""

from __future__ import annotations

import cv2
import numpy as np


def extract_pixel_flatten(images: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Trải phẳng mỗi ảnh `(H, W, C)` thành vector 1 chiều `(H * W * C,)`.
    """
    feats = images.reshape(len(images), -1).astype(np.float32)
    if normalize:
        feats = feats / 255.0
    return feats


def extract_color_histogram(images: np.ndarray, bins_per_channel: int = 16) -> np.ndarray:
    """
    Tính biểu đồ tần suất màu (color histogram) chuẩn hóa cho từng kênh R, G, B rồi nối lại.
    Kích thước đầu ra: `(N, 3 * bins_per_channel)`.
    """
    n_samples = len(images)
    out = np.zeros((n_samples, 3 * bins_per_channel), dtype=np.float32)
    for idx, img in enumerate(images):
        channel_hists = []
        for ch in range(3):
            hist, _ = np.histogram(img[:, :, ch], bins=bins_per_channel, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-8
            channel_hists.append(hist)
        out[idx] = np.concatenate(channel_hists, axis=0)
    return out


def extract_hog_manual(
    images: np.ndarray,
    cell_size: int = 8,
    num_bins: int = 9,
) -> np.ndarray:
    """
    Cài đặt tường minh thuật toán HOG (Histogram of Oriented Gradients) không góc dấu (`0..180` độ):
    1. Chuyển ảnh sang thang xám (grayscale) `[0, 1]`.
    2. Tính đạo hàm theo phương ngang `gx` và phương dọc `gy` bằng bộ lọc Sobel.
    3. Tính độ lớn (magnitude) và góc hướng (orientation) của gradient tại mỗi pixel.
    4. Chia ảnh thành các ô (`cell_size x cell_size`) và cộng dồn độ lớn vào `num_bins` ngăn góc.
    5. Chuẩn hóa L2 toàn cục để bền vững với thay đổi độ sáng.
    """
    feats_list = []
    bin_edges = np.linspace(0.0, 180.0, num_bins + 1)

    for img in images:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = img.astype(np.float32) / 255.0

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.hypot(gx, gy)
        angle = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0

        h, w = gray.shape
        n_cells_y = h // cell_size
        n_cells_x = w // cell_size
        hog_cells = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float32)

        bin_idx = np.clip(np.digitize(angle, bin_edges) - 1, 0, num_bins - 1)
        for cy in range(n_cells_y):
            for cx in range(n_cells_x):
                y0, y1 = cy * cell_size, (cy + 1) * cell_size
                x0, x1 = cx * cell_size, (cx + 1) * cell_size
                cell_mag = magnitude[y0:y1, x0:x1].ravel()
                cell_bin = bin_idx[y0:y1, x0:x1].ravel()
                hog_cells[cy, cx] = np.bincount(cell_bin, weights=cell_mag, minlength=num_bins)

        vec = hog_cells.ravel()
        vec = vec / (np.linalg.norm(vec) + 1e-6)
        feats_list.append(vec)

    return np.stack(feats_list, axis=0)


def extract_lbp_manual(images: np.ndarray, num_bins: int = 16) -> np.ndarray:
    """
    Cài đặt tường minh thuật toán LBP (Local Binary Patterns) 8 lân cận (`3x3`):
    So sánh từng pixel trung tâm với 8 pixel xung quanh để tạo mã nhị phân 8-bit (`0..255`),
    sau đó lập biểu đồ tần suất chuẩn hóa gồm `num_bins` ngăn.
    """
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]
    out = np.zeros((len(images), num_bins), dtype=np.float32)

    for idx, img in enumerate(images):
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        center = gray[1:-1, 1:-1]
        lbp_code = np.zeros_like(center, dtype=np.uint8)
        for bit_i, (dy, dx) in enumerate(offsets):
            neighbor = gray[1 + dy : gray.shape[0] - 1 + dy, 1 + dx : gray.shape[1] - 1 + dx]
            lbp_code |= ((neighbor >= center).astype(np.uint8) << bit_i)

        hist, _ = np.histogram(lbp_code, bins=num_bins, range=(0, 256))
        hist = hist.astype(np.float32)
        out[idx] = hist / (hist.sum() + 1e-8)

    return out


def extract_combined_features(images: np.ndarray, cell_size: int = 8) -> np.ndarray:
    """
    Kết hợp đặc trưng hình dáng (HOG), màu sắc (Color Histogram) và kết cấu (LBP).
    """
    f_hog = extract_hog_manual(images, cell_size=cell_size, num_bins=9)
    f_col = extract_color_histogram(images, bins_per_channel=8)
    f_lbp = extract_lbp_manual(images, num_bins=16)
    return np.concatenate([f_hog, f_col, f_lbp], axis=1)
