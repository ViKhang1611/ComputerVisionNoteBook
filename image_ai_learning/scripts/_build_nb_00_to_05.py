"""
Script khởi tạo các Jupyter Notebook từ 00 đến 05 (Khởi động + Giai đoạn A + đầu Giai đoạn B).
"""

from __future__ import annotations

from pathlib import Path
from _nb_helper import create_validated_notebook

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def build_nb00() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 00: Khởi động — Làm chủ Môi trường và Bức tranh Toàn cảnh Xử lý Ảnh & AI

## 1. Mục tiêu học tập (Learning Objectives)
Sau khi hoàn thành bài học này, bạn sẽ:
- Kiểm tra được phiên bản `Python`, các thư viện cốt lõi (`NumPy`, `OpenCV`, `Pillow`, `scikit-learn`, `PyTorch`, `torchvision`) và thiết bị tính toán (`CPU` / `GPU CUDA`).
- Hiểu cơ chế xác định thư mục gốc dự án (`project_root`) bằng đường dẫn tương đối để code chạy ổn định trên mọi máy tính.
- Hiểu ý nghĩa của cờ cấu hình `QUICK_RUN` (chế độ kiểm tra nhanh pipeline so với chế độ huấn luyện đầy đủ).
- Phân biệt rõ 4 khái niệm nền tảng: **Image Processing (Xử lý ảnh)**, **Computer Vision (Thị giác máy tính)**, **Machine Learning (Học máy)** và **Deep Learning (Học sâu)**.
- Phân biệt phương pháp xử lý ảnh bằng **thuật toán cố định (Rule-based)** với **mô hình học từ dữ liệu (Data-driven model)**.

## 2. Kiến thức cần có (Prerequisites)
- Biết thao tác mở file trong JupyterLab hoặc VS Code.
- Không yêu cầu kiến thức toán cao cấp hay học sâu trước đó.

## 3. Thời gian học dự kiến (Estimated Time)
- Khoảng **45 – 60 phút** (chỉ là ước tính tham khảo; hãy ưu tiên hiểu sâu thay vì chạy nhanh).

## 4. Bài toán thực tế đơn giản (Practical Problem)
Giả sử bạn cần xây dựng chương trình kiểm tra xem một bức ảnh chụp sản phẩm trên băng chuyền là **sáng rõ hay bị tối**, và nhận diện hình dạng vật thể (`circle`, `square`, `triangle`, `cross`).
- Với yêu cầu "kiểm tra ảnh sáng hay tối": Một công thức tính trung bình độ sáng cố định là đủ.
- Với yêu cầu "phân loại hình dạng khi vật thể bị xoay, đổi màu, nhiễu nền": Luật cố định rất dễ vỡ, cần mô hình học từ dữ liệu.

## 5. Giải thích trực quan & Phân biệt Khái niệm
```
┌─────────────────────────────────────────────────────────────────────────┐
│ Artificial Intelligence (AI - Trí tuệ nhân tạo)                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ Machine Learning (ML - Học máy: Học quy luật từ dữ liệu mẫu)    │   │
│   │   ┌─────────────────────────────────────────────────────────┐   │   │
│   │   │ Deep Learning (DL - Học sâu: Mạng nơ-ron nhiều tầng     │   │   │
│   │   │ tự động học cả đặc trưng lẫn bộ phân loại: CNN, ViT...) │   │   │
│   │   └─────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```
- **Image Processing (Xử lý ảnh)**: Đầu vào là **Ảnh** → Đầu ra thường là **Ảnh đã biến đổi** (làm nét, khử nhiễu, cân bằng sáng, tách biên).
- **Computer Vision (Thị giác máy tính)**: Đầu vào là **Ảnh/Video** → Đầu ra là **Thông tin hiểu biết / Quyết định** (nhãn lớp, tọa độ khung vật thể, văn bản đọc được).

## 6. Thuật ngữ mới & Giải thích Keyword
| Thuật ngữ tiếng Anh | Nghĩa tiếng Việt | Nhóm | Vai trò & Giải thích chi tiết |
|---|---|---|---|
| `project_root` | Thư mục gốc dự án | Triển khai | Đường dẫn gốc `image_ai_learning/` giúp nạp dữ liệu và lưu kết quả bằng đường dẫn tương đối (`relative path`), tránh lỗi đường dẫn cứng (`hardcoded absolute path`). |
| `seed` | Hạt giống ngẫu nhiên | Huấn luyện | Số nguyên dương (ví dụ `42`) cố định chuỗi số giả ngẫu nhiên trong `random`, `numpy`, `torch` để kết quả có tính tái lập (`reproducibility`). |
| `device` | Thiết bị tính toán | Triển khai | Đối tượng `torch.device('cpu')` hoặc `torch.device('cuda')` chỉ định nơi lưu trữ Tensor và thực hiện phép nhân ma trận. |
| `QUICK_RUN` | Chế độ chạy kiểm tra nhanh | Huấn luyện | Biến `bool` (`True`/`False`). Khi `True`, giảm số mẫu và số `epochs` để kiểm tra luồng code trong vài giây trên CPU. **Lưu ý**: Không dùng điểm số của `QUICK_RUN` để kết luận chất lượng tối đa của mô hình. |

## 7. Công thức cần thiết & Ví dụ số nhỏ
Khi đánh giá độ sáng trung bình $\mu$ của một bức ảnh thang xám (Grayscale) có chiều cao $H$ và chiều rộng $W$:
$$\mu = \frac{1}{H \times W} \sum_{y=0}^{H-1} \sum_{x=0}^{W-1} I(y, x)$$
- Ký hiệu:
  - $I(y, x) \in \{0, 1, \dots, 255\}$: Cường độ sáng tại hàng $y$, cột $x$ (`0` là đen tuyệt đối, `255` là trắng tuyệt đối).
  - $H, W$: Chiều cao (`height`) và chiều rộng (`width`) tính theo số điểm ảnh (`pixels`).
- **Ví dụ số nhỏ**: Với ảnh `2x2` có ma trận điểm ảnh $\begin{bmatrix} 10 & 20 \\ 30 & 200 \end{bmatrix}$, độ sáng trung bình là:
  $$\mu = \frac{10 + 20 + 30 + 200}{2 \times 2} = \frac{260}{4} = 65.0$$""",
        ),
        (
            "code",
            """# Cell 1: Thiết lập đường dẫn tương đối tới thư mục gốc dự án và kiểm tra môi trường
import sys
import platform
from pathlib import Path

# Tự động tìm thư mục gốc image_ai_learning dù chạy từ bất kỳ đâu
current_dir = Path.cwd().resolve()
project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
if (project_root / "image_ai_learning").is_dir():
    project_root = project_root / "image_ai_learning"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import PIL
import sklearn
import torch
import torchvision

from src.utils import find_project_root, set_seed, get_device
from src.synthetic_data import generate_synthetic_shapes_dataset

QUICK_RUN = True
SEED = 42
set_seed(SEED)
device = get_device()

print(f"Project Root (tương đối): {find_project_root().name}")
print(f"Python version          : {platform.python_version()}")
print(f"NumPy version           : {np.__version__}")
print(f"OpenCV version          : {cv2.__version__}")
print(f"scikit-learn version    : {sklearn.__version__}")
print(f"PyTorch version         : {torch.__version__}")
print(f"Torchvision version     : {torchvision.__version__}")
print(f"Active Device           : {device} (CUDA available: {torch.cuda.is_available()})")
print(f"QUICK_RUN mode          : {QUICK_RUN}")""",
        ),
        (
            "code",
            """# Cell 2: Tạo tập dữ liệu ảnh tổng hợp đầu tiên và hiển thị 4 lớp hình học
dataset_demo = generate_synthetic_shapes_dataset(
    num_samples=16 if QUICK_RUN else 64,
    image_size=32,
    seed=SEED,
    noise_std=8.0,
)
images = dataset_demo["images"]
labels = dataset_demo["labels"]
class_names = dataset_demo["class_names"]

print(f"Images shape: {images.shape}, dtype: {images.dtype}, min={images.min()}, max={images.max()}")
print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}, classes: {class_names}")

fig, axes = plt.subplots(1, 4, figsize=(10, 2.6))
for i in range(4):
    axes[i].imshow(images[i])
    axes[i].set_title(f"Mẫu #{i}: {class_names[labels[i]]}\\nμ={images[i].mean():.1f}")
    axes[i].axis("off")
plt.suptitle("4 lớp hình học cơ bản trong bộ dữ liệu thực hành offline", fontsize=12)
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 3: Minh họa sự khác biệt giữa Luật cố định (Rule-based) và Học từ dữ liệu (Data-driven)
# Giả sử ta thử phân loại 4 hình trên chỉ bằng 1 ngưỡng "độ sáng trung bình" cố định:
mean_intensities = images.mean(axis=(1, 2, 3))
for cls_idx, cls_name in enumerate(class_names):
    cls_means = mean_intensities[labels == cls_idx]
    print(f"Lớp {cls_name:8s} -> Độ sáng trung bình: {cls_means.mean():.2f} ± {cls_means.std():.2f}")

# Nhận xét: Các lớp chồng lấp độ sáng rất mạnh, nên luật ngưỡng độ sáng cố định không thể tách 4 hình dạng!""",
        ),
        (
            "code",
            """# Cell 4: Thí nghiệm thay đổi một yếu tố (Single-Factor Experiment)
# Thay đổi mức độ nhiễu cảm biến (noise_std) từ 0.0 -> 10.0 -> 35.0 và quan sát tác động lên ảnh
noise_levels = [0.0, 10.0, 35.0]
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for idx, n_std in enumerate(noise_levels):
    ds_temp = generate_synthetic_shapes_dataset(num_samples=4, image_size=32, seed=SEED, noise_std=n_std)
    img_sample = ds_temp["images"][0]
    axes[idx].imshow(img_sample)
    axes[idx].set_title(f"noise_std = {n_std}\\nStd pixel = {img_sample.std():.1f}")
    axes[idx].axis("off")
plt.suptitle("Thí nghiệm: Ảnh hưởng của tham số noise_std lên chất lượng tín hiệu ảnh", fontsize=11)
plt.tight_layout()
plt.show()""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- Ở **Cell 2**, mảng `images` có `shape = (16, 32, 32, 3)` nghĩa là gồm `16` bức ảnh, mỗi ảnh cao `32` pixel, rộng `32` pixel và có `3` kênh màu (`R, G, B`), kiểu dữ liệu `uint8` (`0..255`).
- Ở **Cell 3**, độ sáng trung bình giữa 4 lớp (`circle`, `square`, `triangle`, `cross`) xấp xỉ nhau với độ lệch chuẩn chồng lấn mạnh. Điều này giải thích vì sao các bài toán nhận dạng hình ảnh cần trích xuất đặc trưng hình học hoặc học sâu thay vì dùng một ngưỡng cường độ sáng cố định.
- Ở **Cell 4**, khi tăng `noise_std` lên `35.0`, biên của hình tròn bị lẫn vào nhiễu nền, báo hiệu các thuật toán đạo hàm bậc nhất nhạy nhiễu sẽ cần bước làm mượt trước khi tách biên.

## 13. Lỗi thường gặp và cách sửa (Common Pitfalls)
1. **Chọn sai Jupyter Kernel**: Nếu gặp `ModuleNotFoundError: No module named 'torch'`, hãy nhìn góc trên bên phải cửa sổ notebook và chọn kernel `.venv (Python 3.12)`.
2. **Dùng đường dẫn tuyệt đối của máy cá nhân**: Không viết `C:/Users/TenBan/...` trong code. Hãy luôn dùng `project_root / "data" / ...` như ở Cell 1.
3. **Quên cố định `seed`**: Khi so sánh hai thuật toán mà không gọi `set_seed(42)`, sự khác biệt kết quả có thể đến từ may rủi lúc khởi tạo ngẫu nhiên chứ không phải do thuật toán tốt hơn.

## 14. Bài tập tự làm (Exercises)
- **Mức Cơ bản**: Tạo bộ dữ liệu gồm `8` ảnh với kích thước `image_size = 48` và in ra `shape`, `dtype`, giá trị nhỏ nhất (`min`), lớn nhất (`max`) của mảng ảnh.
- **Mức Trung cấp**: Viết hàm `check_brightness_status(img, low_thresh=40.0, high_thresh=200.0)` trả về `"too_dark"`, `"normal"`, hoặc `"too_bright"` dựa trên độ sáng trung bình của ảnh.
- **Mức Nâng cao**: Thử nghiệm thay đổi `seed` qua 5 giá trị `[1, 2, 3, 4, 5]`, tính độ sáng trung bình của toàn bộ tập `16` ảnh ở mỗi `seed` để kiểm chứng mức độ ổn định thống kê.

## 15. Gợi ý và Lời giải (Hints & Complete Solutions)
- *Gợi ý*: Dùng `.mean()` trên mảng NumPy và cấu trúc rẽ nhánh `if / elif / else`. Cell bên dưới cung cấp lời giải chạy được hoàn chỉnh.""",
        ),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho 3 bài tập tự làm
# 1. Bài tập Cơ bản
ds_ex1 = generate_synthetic_shapes_dataset(num_samples=8, image_size=48, seed=42)
imgs_ex1 = ds_ex1["images"]
print("[Cơ bản] Shape:", imgs_ex1.shape, "| Dtype:", imgs_ex1.dtype, "| Min/Max:", imgs_ex1.min(), imgs_ex1.max())

# 2. Bài tập Trung cấp
def check_brightness_status(img: np.ndarray, low_thresh: float = 40.0, high_thresh: float = 200.0) -> str:
    mean_val = float(img.mean())
    if mean_val < low_thresh:
        return "too_dark"
    if mean_val > high_thresh:
        return "too_bright"
    return "normal"

print("[Trung cấp] Trạng thái ảnh #0:", check_brightness_status(imgs_ex1[0]))

# 3. Bài tập Nâng cao
seed_means = []
for s in [1, 2, 3, 4, 5]:
    ds_s = generate_synthetic_shapes_dataset(num_samples=16, image_size=32, seed=s)
    seed_means.append(float(ds_s["images"].mean()))
print("[Nâng cao] Độ sáng trung bình qua 5 seeds:", [round(v, 2) for v in seed_means])""",
        ),
        (
            "md",
            r"""## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Đã xác nhận import thành công `numpy`, `cv2`, `sklearn`, `torch`, `torchvision`.
- [x] Đã hiểu ý nghĩa của `project_root`, `seed`, `device`, `QUICK_RUN`.
- [x] Sẵn sàng chuyển sang **Bài 01: `01_python_numpy_for_images.ipynb`** để nắm vững cấu trúc mảng đa chiều cho xử lý ảnh.

## 17. Nguồn tham khảo (References)
- Tài liệu chính thức PyTorch: https://pytorch.org/docs/stable/index.html
- Tài liệu chính thức NumPy: https://numpy.org/doc/stable/
- Szeliski, R. (2022), *Computer Vision: Algorithms and Applications* (2nd ed.), Springer.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "00_start_here.ipynb", cells)


def build_nb01() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 01: Python và NumPy cho Xử lý Ảnh — Từ Danh sách Số đến Tensor Đa chiều

## 1. Mục tiêu học tập
- Ôn tập nhanh các cấu trúc Python thiết yếu trong pipeline AI: `list`, `dict`, `tuple`, hàm (`def`) và lớp (`class`).
- Hiểu sâu cấu trúc mảng `numpy.ndarray`: `ndim` (số chiều), `shape` (kích thước từng chiều), `dtype` (kiểu dữ liệu số), `axis` (trục tính toán).
- Thành thạo 3 kỹ thuật cốt lõi giúp xử lý ảnh nhanh gấp hàng chục lần vòng lặp thuần Python:
  1. `Slicing` & `Boolean Indexing` (Cắt vùng ảnh `ROI` và lọc pixel theo ngưỡng).
  2. `Broadcasting` (Tự động mở rộng chiều khi cộng/nhân hai mảng khác kích thước).
  3. `Vectorization` (Vector hóa phép toán trên toàn bộ mảng ảnh).

## 2. Kiến thức cần có
- Đã hoàn thành `00_start_here.ipynb`.

## 3. Thời gian học dự kiến
- Khoảng **90 – 120 phút** (ước tính).

## 4. Bài toán thực tế đơn giản
Khi cần tăng độ sáng cho kênh màu Đỏ (`Red`) thêm `30` đơn vị, cắt lấy vùng trung tâm bức ảnh (`crop`) và xóa nền tối về màu đen: nếu dùng 3 vòng lặp `for` lồng nhau `(y, x, c)` trong Python thuần trên ảnh Full HD (`1080 x 1920 x 3` = `6.2 triệu` phép tính), chương trình sẽ rất chậm. Với `NumPy`, toàn bộ thao tác chỉ mất vài mili-giây.

## 5. Giải thích trực quan: Scalar → Vector → Matrix → 3D/4D Tensor
- **Scalar (Vô hướng, `0D`)**: Một con số đơn lẻ, ví dụ độ sáng của 1 điểm ảnh `128`, `shape = ()`.
- **Vector (Vector, `1D`)**: Một dãy số, ví dụ bộ màu RGB của 1 pixel `[220, 80, 60]`, `shape = (3,)`.
- **Matrix (Ma trận, `2D`)**: Một bảng số hàng × cột, ví dụ 1 bức ảnh xám cao `H`, rộng `W`, `shape = (H, W)`.
- **3D Array / Tensor (`3D`)**: Một bức ảnh màu gồm 3 mặt phẳng kênh ghép lại, `shape = (H, W, 3)`.
- **4D Batch Tensor (`4D`)**: Một lô gồm $N$ bức ảnh màu, `shape = (N, H, W, 3)` (trong NumPy) hoặc `(N, 3, H, W)` (trong PyTorch).

## 6. Thuật ngữ mới & Giải thích Keyword
| Keyword | Nghĩa tiếng Việt | Nhóm | Kiểu / Đơn vị | Vai trò & Ảnh hưởng khi thay đổi |
|---|---|---|---|---|
| `ndarray` | Mảng N-chiều đồng nhất kiểu | Dữ liệu | `np.ndarray` | Lưu trữ các phần tử liên tiếp nhau trong bộ nhớ C, cho phép tính toán song song SIMD. |
| `shape` | Kích thước các chiều | Dữ liệu | `tuple[int, ...]` | Ví dụ `(32, 48, 3)` tương ứng `(Height=32, Width=48, Channels=3)`. Lưu ý: `shape[0]` là chiều cao `H` (số hàng `y`), `shape[1]` là chiều rộng `W` (số cột `x`). |
| `dtype` | Kiểu dữ liệu phần tử | Dữ liệu | `np.uint8`, `np.float32`, `np.int64` | `uint8` chỉ chứa số nguyên không dấu `0..255` (1 byte/pixel). Nếu cộng vượt `255` trên `uint8` sẽ bị tràn số quay vòng (`250 + 20 = 14`!). |
| `axis` | Trục thực hiện phép gộp | Dữ liệu | `int` hoặc `tuple[int]` | Xác định chiều nào sẽ bị **thu gọn (collapsed)** khi gọi `.sum()`, `.mean()`, `.max()`. Ví dụ trên ảnh `(H, W, C)`, `img.mean(axis=(0, 1))` gộp hết chiều `H` và `W`, giữ lại vector trung bình `(C,)` của 3 kênh màu. |
| `broadcasting` | Cơ chế tự khớp chiều | Dữ liệu | Quy tắc NumPy | Cho phép thực hiện phép toán giữa ảnh `(H, W, 3)` và vector màu `(1, 1, 3)` hoặc `(3,)` mà không cần nhân bản bộ nhớ. |
| `vectorization` | Vector hóa phép toán | Tối ưu hóa | Kỹ thuật lập trình | Thay vòng lặp `for` Python bằng phép toán cấp mảng của NumPy/PyTorch. |

## 7. Công thức & Quy tắc Broadcasting, Ví dụ số nhỏ
- **Quy tắc Broadcasting của NumPy**: Khi thực hiện phép toán giữa hai mảng `A` và `B`, NumPy so sánh kích thước các chiều **từ phải sang trái**. Hai chiều tương ứng là hợp lệ nếu:
  1. Chúng **bằng nhau**, HOẶC
  2. Một trong hai chiều **bằng `1`** (hoặc tự động chèn `1` vào bên trái mảng ít chiều hơn).
- **Ví dụ số nhỏ**:
  - Ảnh `A` có `shape = (32, 32, 3)`.
  - Vector hệ số cân bằng trắng `w` có `shape = (3,)` → được hiểu là `(1, 1, 3)` → tương thích hoàn hảo với `(32, 32, 3)`!
  - Nhưng nếu vector `v` có `shape = (32,)` → so từ phải sang trái: `3 != 32` → Báo lỗi `ValueError: operands could not be broadcast together`!""",
        ),
        (
            "code",
            """# Cell 1: Khởi tạo môi trường và ôn tập cấu trúc dữ liệu Python cho quản lý cấu hình AI
import sys
import time
from pathlib import Path

current_dir = Path.cwd().resolve()
project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
if (project_root / "image_ai_learning").is_dir():
    project_root = project_root / "image_ai_learning"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from src.utils import set_seed

set_seed(42)

# Cấu trúc class và dict chuẩn để quản lý thông tin một mẫu ảnh
class ImageSampleRecord:
    def __init__(self, sample_id: int, image: np.ndarray, label_name: str) -> None:
        self.sample_id = sample_id
        self.image = image
        self.label_name = label_name

    def summary(self) -> dict:
        return {
            "id": self.sample_id,
            "label": self.label_name,
            "shape": self.image.shape,
            "dtype": str(self.image.dtype),
            "mean": round(float(self.image.mean()), 2),
        }

toy_img = np.zeros((16, 24, 3), dtype=np.uint8)
toy_img[4:12, 6:18] = [220, 90, 50]  # Slicing: gán màu cam cho vùng chữ nhật giữa ảnh
rec = ImageSampleRecord(sample_id=1, image=toy_img, label_name="rectangle_roi")
print("Thông tin mẫu ảnh:", rec.summary())""",
        ),
        (
            "code",
            """# Cell 2: Kiểm tra sâu về shape, axis và bẫy tràn số (Overflow) của kiểu uint8
# Bẫy kinh điển: Cộng trực tiếp số vào mảng uint8
pixel_u8 = np.array([240, 250], dtype=np.uint8)
overflow_result = pixel_u8 + np.uint8(30)  # 240+30=270 -> 270 % 256 = 14!
safe_result = np.clip(pixel_u8.astype(np.int32) + 30, 0, 255).astype(np.uint8)

print(f"Giá trị gốc (uint8)         : {pixel_u8}")
print(f"Cộng trực tiếp bị tràn số   : {overflow_result}  <-- SAI (270 quay vòng về 14)")
print(f"Cộng an toàn với np.clip    : {safe_result} <-- ĐÚNG (giới hạn ở 255)")

# Minh họa ý nghĩa của tham số axis trên ảnh màu (H=16, W=24, C=3)
mean_per_channel = toy_img.mean(axis=(0, 1))  # Gộp chiều H(0) và W(1), còn lại chiều C(2)
mean_grayscale = toy_img.mean(axis=2)         # Gộp chiều C(2), còn lại ma trận 2D (H, W)
print(f"Mean theo từng kênh R,G,B (shape {mean_per_channel.shape}): {mean_per_channel.round(2)}")
print(f"Ảnh trung bình kênh (shape {mean_grayscale.shape})")""",
        ),
        (
            "code",
            """# Cell 3: Thực hành Slicing, Cropping, Boolean Masking và Broadcasting
# Tạo ảnh gradient tổng hợp kích thước (64, 64, 3)
y_idx, x_idx = np.mgrid[0:64, 0:64]
img_grad = np.stack([x_idx * 3, y_idx * 3, np.full_like(x_idx, 100)], axis=-1).astype(np.uint8)

# 1. Cắt vùng quan tâm (ROI Cropping): lấy từ hàng 16..48, cột 16..48
roi_crop = img_grad[16:48, 16:48].copy()

# 2. Broadcasting: nhân hệ số cân bằng trắng [1.2, 0.9, 0.7] có shape (3,) vào ảnh (64, 64, 3)
wb_gains = np.array([1.2, 0.9, 0.7], dtype=np.float32)
img_wb = np.clip(img_grad.astype(np.float32) * wb_gains, 0, 255).astype(np.uint8)

# 3. Boolean Masking: tạo mặt nạ hình tròn ở tâm và đặt nền ngoài hình tròn về màu đen
dist_sq = (x_idx - 32) ** 2 + (y_idx - 32) ** 2
circle_mask = dist_sq <= (22 ** 2)  # Mảng bool (64, 64)
img_masked = img_grad.copy()
img_masked[~circle_mask] = 0

fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
for ax, im, title in zip(
    axes,
    [img_grad, roi_crop, img_wb, img_masked],
    ["1. Ảnh gốc (64x64x3)", "2. Cắt ROI (32x32x3)", "3. Broadcasting WB", "4. Boolean Masking"],
):
    ax.imshow(im)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 4: Thí nghiệm thay đổi một yếu tố — So sánh tốc độ Vòng lặp Python vs Vectorization
test_img = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)

# Cách 1: Dùng vòng lặp for thuần Python để đảo ngược màu (255 - pixel)
t0 = time.perf_counter()
out_loop = np.empty_like(test_img)
h, w, c = test_img.shape
for y in range(h):
    for x in range(w):
        for ch in range(c):
            out_loop[y, x, ch] = 255 - test_img[y, x, ch]
loop_ms = (time.perf_counter() - t0) * 1000.0

# Cách 2: Dùng Vectorization của NumPy
t1 = time.perf_counter()
out_vec = 255 - test_img
vec_ms = (time.perf_counter() - t1) * 1000.0

assert np.array_equal(out_loop, out_vec), "Kết quả hai cách phải giống hệt nhau!"
speedup = loop_ms / max(vec_ms, 1e-4)
print(f"Thời gian chạy vòng lặp Python : {loop_ms:.3f} ms")
print(f"Thời gian chạy Vectorization   : {vec_ms:.4f} ms")
print(f"--> Vectorization nhanh gấp    : ~{speedup:.1f} lần!")""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- Phép cắt lát `img[y1:y2, x1:x2]` trong NumPy mặc định trả về một **view** (tham chiếu chung vùng nhớ với ảnh gốc). Vì vậy ở Cell 3 ta gọi `.copy()` khi muốn chỉnh sửa `roi_crop` độc lập mà không làm hỏng `img_grad`.
- Ở Cell 4, `Vectorization` (`255 - test_img`) cho kết quả giống hệt 3 vòng lặp `for` nhưng nhanh hơn hàng trăm lần nhờ thực thi ở tầng C đã biên dịch sẵn.

## 13. Lỗi thường gặp và cách sửa
1. **Nhầm thứ tự tọa độ `(x, y)` với chỉ số mảng `(row, col) = (y, x)`**: Trong NumPy, chỉ số đầu tiên luôn là **hàng (`y` - chiều dọc)**, chỉ số thứ hai mới là **cột (`x` - chiều ngang)**: `img[y, x]`.
2. **Sửa nhầm ảnh gốc do quên `.copy()` trên slice**: Luôn dùng `sub_img = img[y1:y2, x1:x2].copy()` nếu bạn định gán giá trị mới lên `sub_img`.
3. **Tràn số `uint8` khi cộng/trừ độ sáng**: Luôn ép sang `float32` hoặc `int32`, tính toán, rồi dùng `np.clip(..., 0, 255).astype(np.uint8)`.

## 14. Bài tập tự làm
- **Cơ bản**: Tạo một ảnh bàn cờ vua (`checkerboard`) kích thước `(64, 64)` kiểu `uint8` gồm các ô vuông `8x8` xen kẽ màu đen (`0`) và trắng (`255`) bằng slicing hoặc phép chia lấy nguyên `//`.
- **Trung cấp**: Cho một lô gồm `10` ảnh màu `(10, 32, 32, 3)` kiểu `float32`. Dùng `broadcasting` để chuẩn hóa từng kênh màu theo công thức `(X - mean) / std` trong đó `mean` và `std` có shape `(3,)`.
- **Nâng cao**: Không dùng vòng lặp `for`, hãy tính khoảng cách Euclidean từ một vector truy vấn `q` có shape `(128,)` tới toàn bộ `500` vector cơ sở dữ liệu `DB` có shape `(500, 128)`.

## 15. Gợi ý và Lời giải
- *Gợi ý*: Với bàn cờ, dùng `(y // 8 + x // 8) % 2`. Với khoảng cách Euclidean, dùng `np.linalg.norm(DB - q, axis=1)`.""",
        ),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 01
# 1. Cơ bản: Tạo ảnh bàn cờ vua 64x64 với ô 8x8
yy, xx = np.mgrid[0:64, 0:64]
checkerboard = (((yy // 8) + (xx // 8)) % 2 * 255).astype(np.uint8)
print("[Cơ bản] Checkerboard shape:", checkerboard.shape, "unique values:", np.unique(checkerboard))

# 2. Trung cấp: Chuẩn hóa lô 10 ảnh bằng broadcasting
batch_imgs = np.random.rand(10, 32, 32, 3).astype(np.float32)
ch_mean = batch_imgs.mean(axis=(0, 1, 2))          # shape (3,)
ch_std = batch_imgs.std(axis=(0, 1, 2)) + 1e-6     # shape (3,)
batch_norm = (batch_imgs - ch_mean) / ch_std
print("[Trung cấp] Mean sau chuẩn hóa:", batch_norm.mean(axis=(0, 1, 2)).round(4))

# 3. Nâng cao: Tính khoảng cách Euclidean vector hóa
db_vectors = np.random.randn(500, 128).astype(np.float32)
query_vec = np.random.randn(128).astype(np.float32)
distances = np.linalg.norm(db_vectors - query_vec, axis=1)
nearest_idx = int(np.argmin(distances))
print(f"[Nâng cao] Distances shape: {distances.shape}, mẫu gần nhất: #{nearest_idx} (dist={distances[nearest_idx]:.3f})")""",
        ),
        (
            "md",
            r"""## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Đã nắm vững `shape`, `dtype`, `axis`, `slicing`, `broadcasting`, `vectorization`.
- [x] Không còn mắc lỗi tràn số `uint8` hoặc nhầm `(y, x)` với `(x, y)`.
- [x] Sẵn sàng chuyển sang **Bài 02: `02_image_fundamentals.ipynb`**.

## 17. Nguồn tham khảo
- NumPy Broadcasting Documentation: https://numpy.org/doc/stable/user/basics.broadcasting.html
- Harris, C. R. et al. (2020), *Array programming with NumPy*, Nature 585, 357–362.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "01_python_numpy_for_images.ipynb", cells)


def build_nb02() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 02: Bản chất Ảnh số — Kênh màu, Quy ước Chiều (HWC vs NCHW), Nội suy và Histogram

## 1. Mục tiêu học tập
- Hiểu cấu trúc vật lý và toán học của ảnh số: `pixel` (điểm ảnh), độ phân giải không gian (`spatial resolution`), độ sâu bit (`bit depth`) và số kênh màu (`Grayscale`, `RGB`, `BGR`, `RGBA`).
- Nắm vững quy ước chuyển đổi thứ tự chiều giữa **OpenCV/Matplotlib (`HWC`)** và **PyTorch (`CHW` / `NCHW`)**.
- Chuyển đổi chính xác ảnh giữa kiểu số nguyên `uint8 [0, 255]` và số thực `float32 [0.0, 1.0]`.
- Thực hành các phép biến đổi hình học và cường độ sáng: `Resize`, `Crop`, `Interpolation` (`NEAREST`, `LINEAR`, `CUBIC`), `Histogram`, `Brightness` (độ sáng) và `Contrast` (độ tương phản).
- Với mỗi phép biến đổi: Quan sát **Ảnh gốc → Phép biến đổi → Ảnh kết quả → Giải thích tác dụng và giới hạn**.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 01 (`NumPy array`, `shape`, `dtype`, `broadcasting`).

## 3. Thời gian học dự kiến
- Khoảng **2.0 – 2.5 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
Một mô hình mạng nơ-ron yêu cầu ảnh đầu vào phải có kích thước cố định `(Batch, 3, 32, 32)` ở hệ màu `RGB` và giá trị nằm trong `[0.0, 1.0]`. Trong khi đó, ảnh đọc từ camera công nghiệp bằng OpenCV lại có kích thước `(480, 640, 3)` ở hệ màu `BGR` và bị thiếu sáng. Bạn cần xây dựng bước tiền xử lý chuẩn xác mà không làm biến dạng thông tin.

## 5. Giải thích trực quan: Các Không gian Kênh và Thứ tự Chiều
1. **Grayscale (Ảnh đa mức xám)**: `1` kênh duy nhất `(H, W)` hoặc `(H, W, 1)`. Công thức chuẩn ITU-R BT.601 chuyển từ `RGB` sang `Gray`:
   $$Y = 0.299 R + 0.587 G + 0.114 B$$
   *(Mắt người nhạy nhất với màu xanh lá `G`, nên hệ số của `G` lớn nhất là `0.587`)*.
2. **`RGB` vs `BGR`**: Cùng chứa 3 kênh Đỏ, Lá, Dương nhưng `OpenCV` xếp thứ tự `[Blue, Green, Red]` do lịch sử phần cứng camera đời đầu.
3. **Kênh `Alpha` (`RGBA`, 4 kênh)**: Kênh thứ 4 biểu diễn độ đục (`opacity`: `0` là trong suốt hoàn toàn, `255` hoặc `1.0` là đặc hoàn toàn). Công thức trộn ảnh tiền cảnh $I_{fg}$ lên nền $I_{bg}$ (**Alpha Blending**):
   $$I_{\text{out}} = \alpha \cdot I_{fg} + (1 - \alpha) \cdot I_{bg}, \quad \alpha \in [0, 1]$$
4. **Quy ước chiều `HWC` vs `CHW` và `NCHW`**:
   - Thư viện hiển thị và xử lý ảnh (`OpenCV`, `Pillow`, `Matplotlib`): lưu từng điểm ảnh cạnh nhau → **`HWC`** `(Height, Width, Channels)`.
   - `PyTorch`: lưu từng mặt phẳng kênh liên tiếp nhau để tối ưu phép tích chập trên bộ nhớ đệm → **`CHW`** `(Channels, Height, Width)`, và khi gom theo lô là **`NCHW`** `(Batch_size, Channels, Height, Width)`.

## 6. Thuật ngữ mới & Giải thích Keyword
| Keyword | Nghĩa tiếng Việt | Nhóm | Kiểu / Giá trị | Vai trò, Ảnh hưởng & Lưu ý lỗi |
|---|---|---|---|---|
| `interpolation` | Nội suy điểm ảnh khi Resize | Dữ liệu | `cv2.INTER_NEAREST`, `INTER_LINEAR`, `INTER_CUBIC`, `INTER_AREA` | Xác định cách tính giá trị pixel tại tọa độ mới. **Quy tắc**: Khi resize **ảnh thường** dùng `INTER_LINEAR` (phóng to) hoặc `INTER_AREA` (thu nhỏ chống răng cưa); nhưng khi resize **mặt nạ nhãn (`segmentation mask`)** **BẮT BUỘC dùng `INTER_NEAREST`** để không tạo ra số thập phân làm hỏng ID nhãn lớp! |
| `histogram` | Biểu đồ tần suất mức sáng | Dữ liệu | Mảng 1D `(bins,)` | Đếm số lượng pixel ở từng mức sáng `0..255`. Giúp phát hiện ảnh bị cháy sáng (dồn về `255`), thiếu sáng (dồn về `0`) hoặc kém tương phản (co cụm ở giữa). |
| `alpha` (Contrast gain) & `beta` (Brightness bias) | Hệ số tương phản & Độ lệch sáng | Dữ liệu | Số thực ($\alpha > 0$, $\beta \in \mathbb{R}$) | Trong biến đổi tuyến tính $I'(y,x) = \text{clip}(\alpha \cdot I(y,x) + \beta, 0, 255)$: Tăng $\alpha > 1$ làm dãn histogram (tăng tương phản); tăng $\beta > 0$ dịch toàn bộ histogram sang phải (tăng sáng). |""",
        ),
        (
            "code",
            """# Cell 1: Khởi tạo và minh họa chuyển đổi RGB <-> BGR, Grayscale, RGBA Alpha Blending
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
if (project_root / "image_ai_learning").is_dir():
    project_root = project_root / "image_ai_learning"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.synthetic_data import generate_synthetic_shapes_dataset
from src.utils import set_seed

set_seed(42)
ds = generate_synthetic_shapes_dataset(num_samples=4, image_size=48, seed=42)
img_rgb = ds["images"][0]  # Hình tròn màu đỏ cam trên nền tối

# Chuyển sang BGR (như khi đọc bằng cv2.imread) và Grayscale
img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# Tạo kênh Alpha (trong suốt dần từ trái sang phải) và trộn lên nền caro trắng-xám
h, w, _ = img_rgb.shape
alpha_channel = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))[..., None]
yy, xx = np.mgrid[0:h, 0:w]
checker_bg = (((yy // 6 + xx // 6) % 2) * 80 + 140).astype(np.float32)[..., None]
img_blended = np.clip(alpha_channel * img_rgb.astype(np.float32) + (1.0 - alpha_channel) * checker_bg, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
axes[0].imshow(img_rgb)
axes[0].set_title("1. Ảnh chuẩn RGB")
axes[1].imshow(img_bgr)
axes[1].set_title("2. Hiển thị nhầm BGR\\n(Đỏ bị đổi thành Xanh!)")
axes[2].imshow(img_gray, cmap="gray")
axes[2].set_title("3. Ảnh Grayscale (1 kênh)")
axes[3].imshow(img_blended)
axes[3].set_title("4. Alpha Blending")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 2: Chuyển đổi HWC (uint8) <-> CHW / NCHW (float32) cho PyTorch
# Bước 1: Ảnh đơn (H, W, C) uint8 [0..255] -> Tensor (C, H, W) float32 [0.0..1.0]
tensor_chw = torch.from_numpy(img_rgb).permute(2, 0, 1).to(torch.float32) / 255.0

# Bước 2: Lô ảnh (N, H, W, C) -> Batch Tensor (N, C, H, W)
batch_hwc = ds["images"]
batch_nchw = torch.from_numpy(batch_hwc).permute(0, 3, 1, 2).to(torch.float32) / 255.0

# Bước 3: Chuyển ngược từ Tensor (C, H, W) về NumPy (H, W, C) uint8 để hiển thị
reconstructed_rgb = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)

print(f"NumPy đơn ảnh (HWC)  : shape={img_rgb.shape}, dtype={img_rgb.dtype}, range=[{img_rgb.min()}, {img_rgb.max()}]")
print(f"PyTorch đơn ảnh (CHW): shape={tuple(tensor_chw.shape)}, dtype={tensor_chw.dtype}, range=[{tensor_chw.min():.3f}, {tensor_chw.max():.3f}]")
print(f"PyTorch Batch (NCHW) : shape={tuple(batch_nchw.shape)}")
assert np.array_equal(img_rgb, reconstructed_rgb), "Chuyển đổi hai chiều phải bảo toàn tuyệt đối giá trị pixel!" """,
        ),
        (
            "code",
            """# Cell 3: Resize & Interpolation — Tác dụng và Giới hạn khi phóng to / thu nhỏ ảnh và mask
# Thu nhỏ ảnh 48x48 xuống 12x12 (mất thông tin tần số cao) rồi phóng to lại 48x48 bằng 3 kiểu nội suy
small_12 = cv2.resize(img_rgb, (12, 12), interpolation=cv2.INTER_AREA)
up_nearest = cv2.resize(small_12, (48, 48), interpolation=cv2.INTER_NEAREST)
up_linear = cv2.resize(small_12, (48, 48), interpolation=cv2.INTER_LINEAR)
up_cubic = cv2.resize(small_12, (48, 48), interpolation=cv2.INTER_CUBIC)

# Kiểm chứng lỗi nghiêm trọng nếu dùng INTER_LINEAR trên mặt nạ nhãn (mask có nhãn 0 và 2)
mask_gt = np.zeros((24, 24), dtype=np.uint8)
mask_gt[6:18, 6:18] = 2  # Nhãn lớp 2
mask_wrong = cv2.resize(mask_gt, (48, 48), interpolation=cv2.INTER_LINEAR)
mask_correct = cv2.resize(mask_gt, (48, 48), interpolation=cv2.INTER_NEAREST)
print("Các giá trị nhãn trong mask gốc            :", np.unique(mask_gt))
print("Resize mask bằng INTER_LINEAR (BỊ TẠO NHÃN GIẢ 1!):", np.unique(mask_wrong))
print("Resize mask bằng INTER_NEAREST (CHÍNH XÁC) :", np.unique(mask_correct))

fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
for ax, im, title in zip(
    axes,
    [img_rgb, up_nearest, up_linear, up_cubic],
    ["Gốc (48x48)", "NEAREST (Vỡ hạt ô vuông)", "LINEAR (Mượt nhưng mờ biên)", "CUBIC (Sắc cạnh hơn)"],
):
    ax.imshow(im)
    ax.set_title(title, fontsize=9.5)
    ax.axis("off")
plt.suptitle("Ảnh gốc -> Thu nhỏ 12x12 -> Phóng đại lại 48x48 (Không thể khôi phục chi tiết đã mất!)", fontsize=10.5)
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 4: Thí nghiệm thay đổi Brightness, Contrast và Cân bằng Histogram (Histogram Equalization)
# Tạo phiên bản ảnh thiếu sáng & kém tương phản (nén vào dải [60..110])
low_contrast = np.clip(img_gray.astype(np.float32) * 0.35 + 60.0, 0, 255).astype(np.uint8)

# Cách 1: Biến đổi tuyến tính với alpha=2.2 (tăng contrast), beta=-80 (chỉnh lại điểm đen)
linear_stretched = np.clip(low_contrast.astype(np.float32) * 2.2 - 80.0, 0, 255).astype(np.uint8)

# Cách 2: Cân bằng biểu đồ tần suất tự động (Histogram Equalization)
hist_eq = cv2.equalizeHist(low_contrast)

fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
variants = [
    (low_contrast, "1. Ảnh gốc kém tương phản"),
    (linear_stretched, "2. Tăng Contrast tuyến tính (α=2.2, β=-80)"),
    (hist_eq, "3. Cân bằng Histogram (equalizeHist)"),
]
for col, (im, title) in enumerate(variants):
    axes[0, col].imshow(im, cmap="gray", vmin=0, vmax=255)
    axes[0, col].set_title(title, fontsize=10)
    axes[0, col].axis("off")

    axes[1, col].hist(im.ravel(), bins=32, range=(0, 256), color="steelblue", edgecolor="black")
    axes[1, col].set_title(f"Histogram (std={im.std():.1f})", fontsize=9.5)
    axes[1, col].set_xlim(0, 255)
plt.tight_layout()
plt.show()""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Về Resize & Interpolation (Cell 3)**: Một khi ảnh đã bị thu nhỏ xuống `12x12`, thông tin tần số cao đã mất vĩnh viễn. Phóng to lại lên `48x48` chỉ là ước lượng nội suy giữa các điểm lân cận. Đặc biệt, kết quả in ra chứng minh nếu dùng `INTER_LINEAR` trên mask nhãn `[0, 2]`, các điểm biên bị nội suy thành số `1` (tạo ra nhãn lớp giả không tồn tại!).
- **Về Histogram (Cell 4)**: Ảnh kém tương phản có biểu đồ co cụm ở giữa (`60..110`). Sau khi kéo dãn tuyến tính hoặc `equalizeHist`, biểu đồ trải rộng ra toàn dải `0..255`, độ lệch chuẩn (`std`) tăng vọt giúp vật thể nổi bật rõ rệt trên nền. Giới hạn của `equalizeHist` toàn cục là có thể khuếch đại luôn cả nhiễu ở vùng nền phẳng.

## 13. Lỗi thường gặp và cách sửa
1. **Dùng `cv2.resize(..., width, height)` nhầm với `shape = (height, width)`**: Lưu ý `cv2.resize(img, dsize=(new_W, new_H))` nhận **`(width, height)`**, ngược với thứ tự `(height, width)` của `img.shape[:2]`!
2. **Quên chia `255.0` khi chuyển sang `float32`**: Nếu để tensor `float32` mang giá trị `0.0..255.0` đưa vào mạng có sẵn hoặc hàm `plt.imshow(float_img)`, ảnh sẽ bị bão hòa trắng xóa hoặc làm nổ gradient.

## 14. Bài tập tự làm
- **Cơ bản**: Viết hàm `letterbox_resize(img, target_size=64)` thay đổi kích thước ảnh có chiều `(H, W)` bất kỳ (ví dụ `(30, 60, 3)`) về hình vuông `(64, 64, 3)` mà **giữ nguyên tỉ lệ khung hình (aspect ratio)** bằng cách đệm viền xám (`114`).
- **Trung cấp**: Thực hiện phép hiệu chỉnh Gamma (**Gamma Correction**): $I_{\text{out}} = 255 \times (I_{\text{in}} / 255)^\gamma$ với $\gamma = 0.5$ (làm sáng vùng tối) và $\gamma = 2.0$ (làm tối vùng trung gian), vẽ đồ thị so sánh.
- **Nâng cao**: So sánh cân bằng histogram toàn cục (`cv2.equalizeHist`) với cân bằng histogram thích nghi giới hạn độ tương phản **CLAHE** (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))`) trên ảnh có độ sáng không đều.

## 15. Gợi ý và Lời giải
- *Gợi ý*: Trong `letterbox_resize`, hệ số co giãn là `scale = min(target_size / H, target_size / W)`. Xem code lời giải đầy đủ bên dưới.""",
        ),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 02
# 1. Cơ bản: Letterbox resize bảo toàn tỉ lệ khung hình
def letterbox_resize(img: np.ndarray, target_size: int = 64, pad_val: int = 114) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), pad_val, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas

rect_img = cv2.resize(img_rgb, (60, 30))  # Ảnh chữ nhật H=30, W=60
lb_img = letterbox_resize(rect_img, target_size=64)
print("[Cơ bản] Ảnh gốc:", rect_img.shape, "-> Sau letterbox:", lb_img.shape)

# 2. Trung cấp: Gamma Correction
norm_gray = img_gray.astype(np.float32) / 255.0
gamma_05 = np.clip((norm_gray ** 0.5) * 255.0, 0, 255).astype(np.uint8)
gamma_20 = np.clip((norm_gray ** 2.0) * 255.0, 0, 255).astype(np.uint8)
print(f"[Trung cấp] Mean gốc={img_gray.mean():.1f}, Gamma 0.5={gamma_05.mean():.1f}, Gamma 2.0={gamma_20.mean():.1f}")

# 3. Nâng cao: CLAHE chống khuếch đại nhiễu quá mức
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
img_clahe = clahe.apply(low_contrast)
print(f"[Nâng cao] Std sau equalizeHist={hist_eq.std():.1f} vs CLAHE={img_clahe.std():.1f}")""",
        ),
        (
            "md",
            r"""## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Đã phân biệt rõ `RGB`, `BGR`, `Grayscale`, `RGBA` và chuyển đổi thành thạo `HWC <-> NCHW`.
- [x] Hiểu nguyên tắc chọn `interpolation` cho ảnh thường và cho mặt nạ nhãn (`INTER_NEAREST`).
- [x] Sẵn sàng chuyển sang **Bài 03: `03_classical_image_processing.ipynb`**.

## 17. Nguồn tham khảo
- Gonzalez, R. C., & Woods, R. E. (2018), *Digital Image Processing* (4th ed.), Pearson.
- OpenCV Geometric Image Transformations: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "02_image_fundamentals.ipynb", cells)



def build_nb03() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 03: Xử lý Ảnh Cổ điển — Bộ lọc, Phân ngưỡng, Hình thái học, Tách biên và Đường viền

## 1. Mục tiêu học tập
- Hiểu cơ chế tích chập không gian 2D (`2D Spatial Filtering / Convolution`) với nhân lọc (`kernel`).
- Thực hành và phân biệt **Tác dụng & Giới hạn** của:
  1. Làm mờ & Khử nhiễu: `Box Blur`, `Gaussian Filter`, `Median Filter` (đặc trị nhiễu muối tiêu).
  2. Làm sắc nét: `Sharpening Filter` (Unsharp Masking / Laplacian).
  3. Phân ngưỡng tách nền: `Global Thresholding`, `Otsu's Thresholding`, `Adaptive Thresholding`.
  4. Phép toán Hình thái học (`Morphology`): `Erosion` (co), `Dilation` (giãn), `Opening` (mở - xóa đốm nhiễu trắng), `Closing` (đóng - lấp lỗ thủng đen).
  5. Phát hiện biên (`Edge Detection`): đạo hàm bậc nhất `Sobel` và thuật toán đa bước `Canny`.
  6. Tìm đường viền (`Contour Detection`) để đếm, đo diện tích và đóng khung vật thể.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 01 và Bài 02.

## 3. Thời gian học dự kiến
- Khoảng **2.5 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
Camera trên dây chuyền chụp ảnh các linh kiện hình học nhưng bị dính **nhiễu muối tiêu (salt-and-pepper noise)** do bụi cảm biến và **bóng đổ ánh sáng không đều** từ đèn chiếu nghiêng. Bạn cần khử nhiễu, tách mặt nạ nhị phân sạch và đếm chính xác số lượng linh kiện cùng diện tích của chúng.

## 5. Giải thích trực quan & Công thức Toán học
1. **Phép lọc tuyến tính 2D (2D Spatial Filtering)**:
   $$(I * K)(y, x) = \sum_{u=-a}^{a} \sum_{v=-b}^{b} I(y + u, x + v) \cdot K(u, v)$$
   - `Box Filter 3x3`: Trung bình cộng đều 9 pixel lân cận ($K_{uv} = 1/9$). Làm mờ nhanh nhưng làm nhòe biên mạnh.
   - `Gaussian Filter`: Trọng số giảm dần theo hàm chuông Gauss từ tâm ra ngoài ($G(u,v) \propto e^{-(u^2+v^2)/(2\sigma^2)}$). Khử nhiễu Gaussian mượt tự nhiên hơn Box Filter.
   - `Median Filter` (Bộ lọc phi tuyến): Thay pixel tâm bằng **trung vị (`median`)** của các pixel trong cửa sổ. Loại bỏ hoàn hảo các điểm nhiễu cực đại/cực tiểu (`0` hoặc `255`) mà **giữ nguyên độ sắc của cạnh**!
2. **Đạo hàm biên Sobel**:
   $$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * I, \quad |\nabla I| = \sqrt{G_x^2 + G_y^2}$$

## 6. Thuật ngữ mới & Giải thích Keyword
| Keyword | Nghĩa tiếng Việt | Nhóm | Kiểu / Đơn vị | Giá trị, Ảnh hưởng khi Tăng/Giảm & Lưu ý |
|---|---|---|---|---|
| `ksize` (`kernel_size`) | Kích thước cửa sổ lọc | Xử lý ảnh | Số nguyên lẻ dương (`3, 5, 7...`) | Tăng `ksize` giúp làm mượt mạnh hơn nhưng làm mất các chi tiết nhỏ và tăng chi phí tính toán theo $K^2$. |
| `sigmaX` ($\sigma$) | Độ lệch chuẩn hàm Gauss | Xử lý ảnh | `float > 0` | Quyết định độ rộng của chuông Gauss trong `cv2.GaussianBlur`. Tăng $\sigma$ làm ảnh mờ nhiều hơn. |
| `THRESH_OTSU` | Thuật toán tự tìm ngưỡng Otsu | Xử lý ảnh | Cờ OpenCV | Tự động tìm ngưỡng $T^*$ cực đại hóa phương sai giữa 2 lớp sáng/tối (yêu cầu histogram có dạng 2 đỉnh – bimodal). |
| `Morphology Opening / Closing` | Phép Mở / Phép Đóng hình thái | Xử lý ảnh | `MORPH_OPEN`, `MORPH_CLOSE` | `Opening = Erosion -> Dilation` (xóa hạt nhiễu trắng nhỏ ngoài nền); `Closing = Dilation -> Erosion` (hàn gắn lỗ thủng nhỏ bên trong vật thể). |
| `Canny threshold1, threshold2` | Ngưỡng trễ dưới & trên của Canny | Xử lý ảnh | `float` (ví dụ `50, 150`) | Biên mạnh (`> threshold2`) được giữ chắc chắn; biên yếu nằm giữa `[threshold1, threshold2]` chỉ được giữ nếu nối liền với biên mạnh (Hysteresis). |""",
        ),
        (
            "code",
            """# Cell 1: Khử nhiễu & Làm sắc nét (Ảnh gốc -> Nhiễu -> Box / Gaussian / Median / Sharpen)
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
if (project_root / "image_ai_learning").is_dir():
    project_root = project_root / "image_ai_learning"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.synthetic_data import generate_synthetic_shapes_dataset
from src.utils import set_seed

set_seed(42)
ds = generate_synthetic_shapes_dataset(num_samples=4, image_size=64, seed=42, noise_std=2.0)
clean_gray = cv2.cvtColor(ds["images"][1], cv2.COLOR_RGB2GRAY)

# Tạo ảnh dính nhiễu muối tiêu (Salt & Pepper 6% điểm ảnh)
rng = np.random.default_rng(42)
noisy_sp = clean_gray.copy()
rand_map = rng.random(clean_gray.shape)
noisy_sp[rand_map < 0.03] = 0
noisy_sp[rand_map > 0.97] = 255

# Áp dụng 3 bộ lọc làm mượt và 1 bộ lọc làm sắc nét
blur_box = cv2.blur(noisy_sp, ksize=(5, 5))
blur_gauss = cv2.GaussianBlur(noisy_sp, ksize=(5, 5), sigmaX=1.2)
blur_median = cv2.medianBlur(noisy_sp, ksize=5)

sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
sharpened = cv2.filter2D(blur_median, ddepth=-1, kernel=sharpen_kernel)

fig, axes = plt.subplots(1, 6, figsize=(14, 2.6))
items = [
    (clean_gray, "1. Ảnh gốc sạch"),
    (noisy_sp, "2. Nhiễu muối tiêu"),
    (blur_box, "3. Box Blur 5x5\\n(Lem nhiễu, mờ biên)"),
    (blur_gauss, "4. Gaussian 5x5\\n(Vẫn còn đốm mờ)"),
    (blur_median, "5. Median 5x5\\n(Sạch nhiễu, giữ biên!)"),
    (sharpened, "6. Sharpen sau Median\\n(Tăng độ gắt cạnh)"),
]
for ax, (im, title) in zip(axes, items):
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=8.8)
    ax.axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 2: Phân ngưỡng (Thresholding) trên ảnh có bóng đổ không đều: Global vs Otsu vs Adaptive
# Tạo ảnh có vật thể kèm độ dốc ánh sáng mạnh từ trái sang phải (mô phỏng đèn chiếu lệch)
h, w = clean_gray.shape
illumination_ramp = np.tile(np.linspace(-70, 90, w, dtype=np.float32), (h, 1))
uneven_img = np.clip(clean_gray.astype(np.float32) + illumination_ramp, 0, 255).astype(np.uint8)

# 1. Ngưỡng cố định (Global = 127)
_, th_global = cv2.threshold(uneven_img, 127, 255, cv2.THRESH_BINARY)
# 2. Ngưỡng tự động toàn cục Otsu
otsu_val, th_otsu = cv2.threshold(uneven_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 3. Ngưỡng thích nghi cục bộ (Adaptive Gaussian Thresholding)
th_adaptive = cv2.adaptiveThreshold(
    uneven_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=15, C=-5
)

fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
for ax, im, title in zip(
    axes,
    [uneven_img, th_global, th_otsu, th_adaptive],
    [
        "1. Ảnh bị chói lệch một bên",
        "2. Global Threshold (T=127)\\n(Nền bên phải bị nhận nhầm)",
        f"3. Otsu toàn cục (T={otsu_val:.0f})\\n(Vẫn hỏng do ánh sáng dốc)",
        "4. Adaptive Threshold\\n(Tách đúng nhờ tính cục bộ)",
    ],
):
    ax.imshow(im, cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 3: Hình thái học (Morphology) — Erosion, Dilation, Opening (xóa đốm trắng), Closing (lấp lỗ đen)
# Tạo mặt nạ nhị phân có cả đốm nhiễu trắng ngoài nền và lỗ thủng đen bên trong vật thể
_, base_mask = cv2.threshold(clean_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
corrupted_mask = base_mask.copy()
corrupted_mask[6:9, 6:9] = 255     # Đốm nhiễu trắng ngoài nền
corrupted_mask[52:55, 10:13] = 255 # Đốm nhiễu trắng ngoài nền
corrupted_mask[28:32, 28:32] = 0   # Lỗ thủng đen giữa vật thể

kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded = cv2.erode(corrupted_mask, kernel_morph, iterations=1)
dilated = cv2.dilate(corrupted_mask, kernel_morph, iterations=1)
opened = cv2.morphologyEx(corrupted_mask, cv2.MORPH_OPEN, kernel_morph)
closed_after_open = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_morph)

fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.6))
for ax, im, title in zip(
    axes,
    [corrupted_mask, eroded, dilated, opened, closed_after_open],
    [
        "1. Mask lỗi (đốm + lỗ)",
        "2. Erosion (Co nhỏ)",
        "3. Dilation (Phình to)",
        "4. Opening (Xóa đốm trắng)",
        "5. Opening + Closing (Sạch hoàn toàn)",
    ],
):
    ax.imshow(im, cmap="gray")
    ax.set_title(title, fontsize=8.8)
    ax.axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 4: Phát hiện biên (Sobel vs Canny) và Tìm đường viền (Contours) để đếm & đo diện tích
# Tạo ảnh chứa 3 vật thể tách rời để thực hành đếm bằng Contour
canvas_multi = np.zeros((80, 160, 3), dtype=np.uint8)
cv2.circle(canvas_multi, (30, 40), 16, (230, 90, 70), -1)
cv2.rectangle(canvas_multi, (68, 22), (104, 58), (70, 170, 235), -1)
pts_tri = np.array([[135, 20], [118, 58], [152, 58]], dtype=np.int32)
cv2.fillConvexPoly(canvas_multi, pts_tri, (80, 200, 110))
gray_multi = cv2.cvtColor(canvas_multi, cv2.COLOR_RGB2GRAY)

# Tính đạo hàm biên Sobel và biên Canny
gx = cv2.Sobel(gray_multi, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray_multi, cv2.CV_32F, 0, 1, ksize=3)
sobel_mag = np.clip(np.hypot(gx, gy), 0, 255).astype(np.uint8)
canny_edges = cv2.Canny(gray_multi, threshold1=50, threshold2=150)

# Tìm Contours trên ảnh nhị phân
_, bin_multi = cv2.threshold(gray_multi, 30, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(bin_multi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

vis_contours = canvas_multi.copy()
print(f"Tổng số vật thể phát hiện qua Contour: {len(contours)}")
for idx, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, closed=True)
    x, y, bw, bh = cv2.boundingRect(cnt)
    cv2.rectangle(vis_contours, (x, y), (x + bw, y + bh), (255, 255, 0), 1)
    print(f"  - Vật thể #{idx+1}: bbox=(x={x}, y={y}, w={bw}, h={bh}), Diện tích={area:.1f} px², Chu vi={perimeter:.1f} px")

fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
axes[0].imshow(sobel_mag, cmap="gray")
axes[0].set_title("1. Độ lớn đạo hàm Sobel (Biên dày)")
axes[1].imshow(canny_edges, cmap="gray")
axes[1].set_title("2. Biên Canny (Mảnh 1 pixel nhờ NMS)")
axes[2].imshow(vis_contours)
axes[2].set_title(f"3. Tìm thấy {len(contours)} Contours & Bounding Boxes")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả & Phân tích Giới hạn
- **Box/Gaussian vs Median (Cell 1)**: Với nhiễu xung (muối tiêu `0` và `255`), `Box` và `Gaussian` lấy trung bình cộng nên kéo giá trị nhiễu lan ra xung quanh làm mờ biên. Ngược lại, `Median` sắp xếp các điểm trong cửa sổ và lấy điểm chính giữa nên loại bỏ triệt để điểm cực đoan. Tuy nhiên, nếu mật độ nhiễu `> 40%` hoặc cần giữ kết cấu hạt mịn, `Median` với cửa sổ lớn sẽ làm bệt chi tiết.
- **Otsu vs Adaptive Threshold (Cell 2)**: `Otsu` tìm một ngưỡng duy nhất cho toàn bức ảnh nên thất bại khi ánh sáng chênh lệch giữa hai nửa ảnh. `Adaptive Threshold` tính ngưỡng riêng cho từng vùng lân cận `blockSize x blockSize`, nhưng có giới hạn là nhạy cảm với nhiễu ở các mảng nền phẳng rộng nếu hằng số `C` chọn quá nhỏ.
- **Khi nào Xử lý ảnh cổ điển là ĐỦ, và khi nào cần Học máy (ML/DL)?**
  - Nếu điều kiện chiếu sáng cố định trong buồng kín, vật thể tách biệt rõ với nền đồng màu → Chuỗi `Median -> Threshold -> Morphology -> Contour` (Cell 4) chạy trong `< 1 ms` trên CPU và đạt độ chính xác tuyệt vời mà không cần thu thập hàng nghìn ảnh để huấn luyện AI!
  - Nhưng nếu vật thể chồng lấp nhau, nền lộn xộn ngoài trời, hoặc các lớp chỉ khác nhau ở ngữ nghĩa phức tạp (mèo vs chó) → Các ngưỡng cố định hoàn toàn bất lực, lúc đó ta bắt buộc phải chuyển sang **Machine Learning / Deep Learning** từ Bài 04 trở đi.

## 13. Lỗi thường gặp và cách sửa
1. **Truyền `ksize` chẵn vào `GaussianBlur` hoặc `medianBlur`**: OpenCV yêu cầu `ksize` phải là **số nguyên dương lẻ** (`3, 5, 7...`) để cửa sổ có đúng 1 pixel tâm đối xứng.
2. **Dùng `cv2.Sobel(..., cv2.CV_8U)` làm mất toàn bộ đạo hàm âm**: Khi chuyển từ vùng sáng sang vùng tối, đạo hàm mang dấu âm. Nếu để `ddepth=cv2.CV_8U`, mọi số âm bị cắt thành `0` (mất một nửa số cạnh!). Hãy luôn tính Sobel với `cv2.CV_32F` rồi lấy trị tuyệt đối hoặc `np.hypot(gx, gy)`.

## 14. Bài tập tự làm
- **Cơ bản**: Thử thay đổi `ksize` của `cv2.GaussianBlur` qua `[3, 7, 13]` và tính độ lớn biên Sobel trung bình của ảnh sau làm mờ để chứng minh làm mờ càng mạnh thì độ dốc biên càng giảm.
- **Trung cấp**: Tính chỉ số **Độ tròn (`Circularity`)** của mỗi contour theo công thức $C = \frac{4 \pi \times \text{Area}}{\text{Perimeter}^2}$ (hình tròn lý tưởng có $C \approx 1.0$, hình vuông/tam giác có $C$ nhỏ hơn) để tự động phân biệt hình tròn với hình vuông và tam giác ở Cell 4!
- **Nâng cao**: Viết hàm tự động chọn ngưỡng Canny dựa trên trung vị cường độ sáng $v = \text{median}(I)$: `lower = max(0, 0.67 * v)`, `upper = min(255, 1.33 * v)`.

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 03
# 1. Cơ bản: Ảnh hưởng của ksize làm mờ lên độ mạnh biên Sobel
for k in [3, 7, 13]:
    g_blur = cv2.GaussianBlur(clean_gray, (k, k), sigmaX=0)
    sx = cv2.Sobel(g_blur, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g_blur, cv2.CV_32F, 0, 1, ksize=3)
    print(f"[Cơ bản] Gaussian ksize={k:2d} -> Độ lớn biên Sobel trung bình: {np.hypot(sx, sy).mean():.2f}")

# 2. Trung cấp: Phân loại hình dạng bằng độ tròn (Circularity) từ Contour
for idx, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, closed=True)
    circularity = (4.0 * np.pi * area) / (peri ** 2 + 1e-6)
    x, _, _, _ = cv2.boundingRect(cnt)
    print(f"[Trung cấp] Contour tại x={x:3d}: Circularity = {circularity:.3f}")

# 3. Nâng cao: Auto-Canny dựa trên median
def auto_canny(gray_img: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = float(np.median(gray_img))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray_img, lower, upper)

edges_auto = auto_canny(clean_gray)
print("[Nâng cao] Số pixel biên phát hiện bởi Auto-Canny:", int((edges_auto > 0).sum()))""",
        ),
        (
            "md",
            r"""## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Đã nắm vững bộ công cụ Xử lý ảnh cổ điển (`Filter`, `Threshold`, `Morphology`, `Sobel/Canny`, `Contour`).
- [x] Hiểu rõ khi nào xử lý ảnh cổ điển là đủ và khi nào cần chuyển sang mô hình học từ dữ liệu.
- [x] Sẵn sàng bước vào **Giai đoạn B — Bài 04: `04_math_and_ml_foundations.ipynb`**.

## 17. Nguồn tham khảo
- Canny, J. (1986), *A Computational Approach to Edge Detection*, IEEE TPAMI.
- Otsu, N. (1979), *A Threshold Selection Method from Gray-Level Histograms*, IEEE Trans. SMC.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "03_classical_image_processing.ipynb", cells)


def build_nb04() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 04: Toán Nền tảng và Nguyên lý Cốt lõi của Machine Learning — Từ Đạo hàm đến Gradient Descent

## 1. Mục tiêu học tập
- Nắm được trực quan toán học vừa đủ dùng (không cần biết trước giải tích/đại số nặng): Tích vô hướng (`Dot Product`), Nhân ma trận (`Matrix Multiplication`), Đạo hàm (`Derivative`) và Vector độ dốc (`Gradient`).
- Phân biệt rõ 4 nhánh Học máy: **Supervised (Có giám sát)**, **Unsupervised (Không giám sát)**, **Semi-supervised (Bán giám sát)** và **Self-supervised Learning (Tự giám sát)**.
- Phân biệt 3 bài toán: **Classification (Phân loại)**, **Regression (Hồi quy)** và **Clustering (Phân cụm)**.
- Phân biệt **Feature Engineering (Thiết kế đặc trưng thủ công)** và **Feature Learning (Tự học đặc trưng)**.
- Hiểu sâu về **Overfitting (Học vẹt)**, **Underfitting (Chưa học đủ)** và đánh đổi **Bias – Variance**.
- Tự tay tính toán và lập trình thuật toán **Gradient Descent** từng bước bằng con số cụ thể và hình minh họa.

## 2. Kiến thức cần có
- Đã hoàn thành Giai đoạn A (`00` đến `03`).

## 3. Thời gian học dự kiến
- Khoảng **2.0 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
Giả sử từ mỗi bức ảnh bạn đo được 2 con số đặc trưng $x = [x_1, x_2]$ (ví dụ: độ tròn và độ tương phản biên) và muốn tìm bộ trọng số $w = [w_1, w_2]$ cùng độ lệch $b$ để dự đoán nhãn của ảnh. Làm thế nào máy tính **tự động chỉnh dần** $w, b$ để sai số ngày càng nhỏ đi? Đó chính là vai trò của **Hàm mất mát (`Loss Function`)** và **Hạ dốc đạo hàm (`Gradient Descent`)**.

## 5. Giải thích trực quan: Các Nhánh Học máy & Bias–Variance
1. **4 Nhánh Học máy theo cách sử dụng nhãn**:
   - **Supervised Learning (Học có giám sát)**: Mỗi ảnh $x_i$ đều có nhãn đáp án $y_i$ do con người gán (`Classification` nếu $y_i$ là lớp rời rạc; `Regression` nếu $y_i$ là số thực liên tục như tọa độ, góc xoay).
   - **Unsupervised Learning (Học không giám sát)**: Chỉ có ảnh $x_i$, **không có nhãn** $y_i$. Máy tự gom các mẫu giống nhau thành cụm (`Clustering` như K-Means) hoặc nén số chiều (`PCA`).
   - **Semi-supervised Learning (Học bán giám sát)**: Có một lượng nhỏ ảnh đã gán nhãn (ví dụ 500 ảnh) kết hợp với lượng lớn ảnh chưa gán nhãn (10,000 ảnh) để tăng độ chính xác.
   - **Self-supervised Learning (Học tự giám sát)**: Không dùng nhãn con người, nhưng tự tạo bài toán giải đố từ chính bức ảnh (ví dụ che một mảnh ảnh bắt đoán lại, hoặc xoay ảnh bắt đoán góc xoay) để học bộ trích xuất đặc trưng mạnh mẽ (như SimCLR, DINO, MAE).
2. **Feature Engineering vs Feature Learning**:
   - *Feature Engineering (ML cổ điển)*: Con người tự viết công thức toán (như HOG, LBP) biến ảnh thành vector rồi mới đưa vào SVM/Random Forest.
   - *Feature Learning (Deep Learning)*: Mạng CNN/ViT tự học luôn các bộ lọc trích xuất đặc trưng trực tiếp từ pixel thô thông qua quá trình tối ưu hàm loss.
3. **Underfitting vs Overfitting (Bias – Variance Tradeoff)**:
   - **Underfitting (Bias cao)**: Mô hình quá đơn giản (ví dụ dùng đường thẳng để chia dữ liệu hình vòng tròn) → Sai số cao trên **cả tập Train lẫn tập Validation**.
   - **Overfitting (Variance cao)**: Mô hình quá phức tạp hoặc học quá lâu, uốn lượn qua từng điểm nhiễu của tập Train → Sai số tập Train cực thấp (`~0`), nhưng sai số tập Validation lại cao vọt!

## 6. Thuật ngữ mới & Giải thích Keyword
| Keyword | Nghĩa tiếng Việt | Nhóm | Công thức / Kiểu | Vai trò & Ảnh hưởng khi Tăng/Giảm |
|---|---|---|---|---|
| `dot product` | Tích vô hướng 2 vector | Toán nền tảng | $w \cdot x = \sum_{j=1}^D w_j x_j$ | Đo mức độ "cùng hướng" (khớp mẫu) giữa vector đặc trưng $x$ và vector mẫu trọng số $w$. |
| `loss function` | Hàm mất mát | Tối ưu hóa | $\mathcal{L}(\theta) \in \mathbb{R}$ | Thước đo khả vi định lượng độ sai lệch giữa dự đoán $\hat{y}$ và đích $y$. |
| `gradient` ($\nabla_\theta \mathcal{L}$) | Vector đạo hàm riêng | Tối ưu hóa | Vector cùng chiều với $\theta$ | Chỉ hướng mà tại đó hàm mất mát **tăng nhanh nhất**. Muốn giảm mất mát, ta đi theo hướng **ngược lại** ($-\nabla_\theta \mathcal{L}$). |
| `learning_rate` ($\eta$) | Tốc độ học | Tối ưu hóa | `float > 0` | Bước nhảy cập nhật: $\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla_\theta \mathcal{L}$. Quá nhỏ → bò rất chậm; quá lớn → nhảy vọt qua đáy và phân kỳ! |

## 7. Công thức Cần thiết & Ví dụ Số Nhỏ Từng Bước
Xét hàm mất mát bình phương đơn giản nhất với 1 tham số $w$:
$$\mathcal{L}(w) = (w \cdot x - y)^2$$
- Đạo hàm của $\mathcal{L}$ theo $w$ (theo quy tắc hàm hợp):
  $$\frac{d\mathcal{L}}{dw} = 2 (w \cdot x - y) \cdot x$$
- **Ví dụ số nhỏ**: Cho $x = 2.0$, đáp án thật $y = 10.0$ (nghiệm đúng là $w^* = 5.0$). Khởi tạo $w_0 = 1.0$, tốc độ học $\eta = 0.1$:
  1. Dự đoán lần đầu: $\hat{y}_0 = w_0 \cdot x = 1.0 \times 2.0 = 2.0$.
  2. Giá trị mất mát: $\mathcal{L}(w_0) = (2.0 - 10.0)^2 = (-8.0)^2 = 64.0$.
  3. Đạo hàm tại $w_0$: $g_0 = 2 \times (2.0 - 10.0) \times 2.0 = -32.0$ (dấu âm báo hiệu cần **tăng** $w$ thì loss mới giảm!).
  4. Cập nhật trọng số: $w_1 = w_0 - \eta \cdot g_0 = 1.0 - 0.1 \times (-32.0) = 4.2$.
  5. Kiểm tra lại mất mát mới: $\hat{y}_1 = 4.2 \times 2.0 = 8.4 \implies \mathcal{L}(w_1) = (8.4 - 10.0)^2 = 2.56$ (giảm mạnh từ `64.0` xuống `2.56` chỉ sau 1 bước!).""",
        ),
        (
            "code",
            """# Cell 1: Kiểm chứng Ví dụ số nhỏ về Gradient Descent bằng code Python
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
if (project_root / "image_ai_learning").is_dir():
    project_root = project_root / "image_ai_learning"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from src.utils import set_seed

set_seed(42)

x_val, y_true = 2.0, 10.0
w = 1.0
lr = 0.1

print("Bước |   w_hiện_tại | Dự_đoán (w*x) |   Loss   | Gradient (dL/dw)")
print("-" * 66)
for step in range(6):
    y_pred = w * x_val
    loss = (y_pred - y_true) ** 2
    grad = 2.0 * (y_pred - y_true) * x_val
    print(f"  {step}  |    {w:8.4f}  |    {y_pred:8.4f}   | {loss:8.4f} |    {grad:9.4f}")
    w = w - lr * grad""",
        ),
        (
            "code",
            """# Cell 2: Thí nghiệm thay đổi một yếu tố — Tác động của Learning Rate (Quá nhỏ vs Vừa phải vs Quá lớn)
def run_gd_trajectory(lr_val: float, steps: int = 12) -> tuple[list[float], list[float]]:
    w_curr = 1.0
    w_hist, loss_hist = [w_curr], [(w_curr * x_val - y_true) ** 2]
    for _ in range(steps):
        grad = 2.0 * (w_curr * x_val - y_true) * x_val
        w_curr = w_curr - lr_val * grad
        w_hist.append(w_curr)
        loss_hist.append((w_curr * x_val - y_true) ** 2)
    return w_hist, loss_hist

w_grid = np.linspace(-2.0, 12.0, 200)
loss_grid = (w_grid * x_val - y_true) ** 2

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
configs = [
    (0.01, "1. LR quá nhỏ (η=0.01)\\nHội tụ rất chậm"),
    (0.10, "2. LR hợp lý (η=0.10)\\nHội tụ nhanh và êm"),
    (0.26, "3. LR quá lớn (η=0.26 > 0.25)\\nNhảy vọt qua đáy và phân kỳ!"),
]
for ax, (lr_test, title) in zip(axes, configs):
    w_h, l_h = run_gd_trajectory(lr_test, steps=8)
    ax.plot(w_grid, loss_grid, color="lightgray", lw=2, label="Hàm Loss L(w)")
    ax.plot(w_h, l_h, "o-", color="crimson", markersize=5, label="Quỹ đạo cập nhật w")
    ax.scatter([5.0], [0.0], color="green", s=80, zorder=5, label="Đáy tối ưu w*=5")
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("Tham số w")
    ax.set_ylabel("Loss")
    ax.set_ylim(-5, 160)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 3: Minh họa trực quan Underfitting vs Just Right vs Overfitting (Bias-Variance Tradeoff)
rng = np.random.default_rng(42)
x_train_1d = np.sort(rng.uniform(0.0, 1.0, size=12))
y_train_1d = np.sin(2.0 * np.pi * x_train_1d) + rng.normal(0.0, 0.22, size=len(x_train_1d))

x_dense = np.linspace(0.0, 1.0, 200)
y_true_clean = np.sin(2.0 * np.pi * x_dense)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
for ax, deg, title in zip(
    axes,
    [1, 3, 10],
    [
        "Bậc 1: Underfitting (High Bias)\\nQuá cứng nhắc, sai cả Train & Val",
        "Bậc 3: Cân bằng tốt (Good Fit)\\nNắm đúng quy luật tổng quát",
        "Bậc 10: Overfitting (High Variance)\\nHọc thuộc cả nhiễu, sai nặng ở mẫu mới",
    ],
):
    coeffs = np.polyfit(x_train_1d, y_train_1d, deg=deg)
    y_fit_dense = np.polyval(coeffs, x_dense)
    train_mse = np.mean((np.polyval(coeffs, x_train_1d) - y_train_1d) ** 2)
    val_mse = np.mean((y_fit_dense - y_true_clean) ** 2)

    ax.plot(x_dense, y_true_clean, "--", color="seagreen", label="Quy luật thật")
    ax.scatter(x_train_1d, y_train_1d, color="navy", s=35, label="Dữ liệu Train (có nhiễu)")
    ax.plot(x_dense, y_fit_dense, color="darkorange", lw=2, label=f"Mô hình bậc {deg}")
    ax.set_ylim(-1.8, 1.8)
    ax.set_title(f"{title}\\nTrain MSE={train_mse:.3f} | Val MSE={val_mse:.3f}", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower left")
plt.tight_layout()
plt.show()""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Ở Cell 1 & 2**: Khi `lr = 0.10`, chỉ sau 5 bước cập nhật, $w$ đã tiến sát nghiệm đúng $5.0$ và `Loss` giảm về `0.0001`. Nhưng khi tăng `lr = 0.26` (vượt ngưỡng ổn định $1 / x^2 = 0.25$), mỗi bước nhảy lại văng sang bờ dốc đối diện xa hơn bước trước, làm `Loss` tăng vọt lên vô cùng (hiện tượng bùng nổ thành `NaN` khi train mạng sâu).
- **Ở Cell 3**: Đa thức bậc 10 đạt `Train MSE = 0.003` (gần như bằng 0 vì đi xuyên qua mọi điểm nhiễu), nhưng `Val MSE` lại vọt lên rất cao. Đây là lý do **không bao giờ được đánh giá mô hình chỉ dựa trên điểm số của tập Train**.

## 13. Lỗi thường gặp và cách sửa
1. **Nhầm dấu trong công thức cập nhật Gradient Descent**: Phải là **trừ** đạo hàm (`w = w - lr * grad`) khi muốn **cực tiểu hóa (minimize)** hàm mất mát. Nếu viết cộng (`w + lr * grad`), bạn đang leo lên đỉnh núi (**Gradient Ascent** - cực đại hóa mất mát)!
2. **Đánh giá sai nguyên nhân khi mô hình đạt điểm thấp**:
   - Nếu `Train Loss` còn rất cao → Mô hình đang **Underfitting** (cần tăng kích thước mạng, train thêm epoch, hoặc chỉnh lại `learning_rate`). Đừng thêm `Dropout` hay `weight_decay` lúc này vì sẽ làm mô hình càng yếu hơn!
   - Nếu `Train Loss` rất thấp nhưng `Val Loss` cao và đang đi lên → Mô hình đang **Overfitting** (lúc này mới cần thêm dữ liệu, `Augmentation`, `Dropout`, `weight_decay`, hoặc `Early Stopping`).

## 14. Bài tập tự làm
- **Cơ bản**: Kiểm tra đạo hàm giải tích $g = 2(wx - y)x$ bằng công thức **đạo hàm số (Finite Difference)**: $\frac{\mathcal{L}(w + \epsilon) - \mathcal{L}(w - \epsilon)}{2\epsilon}$ với $\epsilon = 10^{-5}$ tại $w = 1.0$.
- **Trung cấp**: Mở rộng Gradient Descent cho mô hình hồi quy tuyến tính 2 biến $\hat{y} = w_1 x_1 + w_2 x_2 + b$ trên `50` mẫu dữ liệu giả lập.
- **Nâng cao**: Thêm thành phần phạt chuẩn $L_2$ (**Ridge Regularization**): $\mathcal{L}_{\text{reg}}(w) = (wx - y)^2 + \lambda w^2$, suy ra đạo hàm mới và kiểm tra xem nghiệm hội tụ $w^*$ thay đổi thế nào khi tăng $\lambda$ từ `0.0` lên `1.0`.

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 4: Lời giải hoàn chỉnh cho bài tập Bài 04
# 1. Cơ bản: Kiểm tra đạo hàm giải tích bằng đạo hàm số (Numerical Gradient Check)
eps = 1e-5
loss_fn = lambda w_param: (w_param * x_val - y_true) ** 2
analytic_grad = 2.0 * (1.0 * x_val - y_true) * x_val
numeric_grad = (loss_fn(1.0 + eps) - loss_fn(1.0 - eps)) / (2.0 * eps)
print(f"[Cơ bản] Đạo hàm giải tích: {analytic_grad:.6f} | Đạo hàm số: {numeric_grad:.6f} | Sai lệch: {abs(analytic_grad - numeric_grad):.2e}")

# 2. Trung cấp: Gradient Descent cho mô hình 2 biến y = 3*x1 - 2*x2 + 4
rng = np.random.default_rng(42)
X_2d = rng.normal(0, 1, size=(50, 2))
y_2d = 3.0 * X_2d[:, 0] - 2.0 * X_2d[:, 1] + 4.0
w_vec = np.zeros(2)
b_val = 0.0
for _ in range(150):
    err = (X_2d @ w_vec + b_val) - y_2d
    grad_w = (2.0 / len(X_2d)) * (X_2d.T @ err)
    grad_b = (2.0 / len(X_2d)) * np.sum(err)
    w_vec -= 0.1 * grad_w
    b_val -= 0.1 * grad_b
print(f"[Trung cấp] Trọng số học được: w={w_vec.round(3)} (đích [3, -2]), b={b_val:.3f} (đích 4.0)")

# 3. Nâng cao: Tác động của hệ số phạt L2 (lambda) làm co nhỏ trọng số w
for lam in [0.0, 0.5, 2.0]:
    w_reg = 1.0
    for _ in range(50):
        grad_total = 2.0 * (w_reg * x_val - y_true) * x_val + 2.0 * lam * w_reg
        w_reg -= 0.05 * grad_total
    print(f"[Nâng cao] lambda={lam:.1f} -> Nghiệm hội tụ w* = {w_reg:.4f}")""",
        ),
        (
            "md",
            r"""## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Hiểu trực quan về Đạo hàm, Gradient Descent, Loss Function và tốc độ học `learning_rate`.
- [x] Phân biệt được 4 nhánh học máy và biết chẩn đoán Underfitting vs Overfitting.
- [x] Sẵn sàng chuyển sang **Bài 05: `05_datasets_and_data_leakage.ipynb`**.

## 17. Nguồn tham khảo
- Hastie, T., Tibshirani, R., & Friedman, J. (2009), *The Elements of Statistical Learning*, Springer.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016), *Deep Learning*, MIT Press (Chương 4 & 5).""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "04_math_and_ml_foundations.ipynb", cells)


def build_nb05() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 05: Thiết kế Tập dữ liệu, Chống Rò rỉ Dữ liệu (Data Leakage) và Xử lý Mất cân bằng Lớp

## 1. Mục tiêu học tập
- Hiểu vai trò tách biệt của 3 tập dữ liệu: **`Train`**, **`Validation`** và **`Test`**.
- Thực hành chia dữ liệu đúng chuẩn bằng `Stratified Split` (giữ nguyên tỷ lệ lớp) và **`Group Split`** (giữ các ảnh chụp cùng một vật thể / cùng chuỗi video / cùng bệnh nhân nằm trọn trong 1 tập).
- Chứng minh bằng thực nghiệm sự nguy hiểm của **Data Leakage (Rò rỉ dữ liệu)** khi gọi `fit` bộ tiền xử lý (`StandardScaler`, `PCA`) trước khi chia tập hoặc nhân bản ảnh trước khi chia tập.
- Thực hành các kỹ thuật xử lý **Mất cân bằng lớp (`Class Imbalance`)**: Trọng số lớp (`class_weight`) và lấy mẫu có trọng số (`WeightedRandomSampler`).

## 2. Kiến thức cần có
- Đã hoàn thành Bài 04.

## 3. Thời gian học dự kiến
- Khoảng **2.0 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
Trong một dự án phân loại ảnh linh kiện lỗi, kỹ sư A lấy 100 ảnh gốc, mỗi ảnh xoay nhẹ 4 góc tạo thành 400 ảnh, chuẩn hóa toàn bộ 400 ảnh bằng `PCA`, rồi mới chia ngẫu nhiên `80% Train / 20% Test`. Kỹ sư A báo cáo độ chính xác trên tập Test đạt `99%`, nhưng khi đem ra nhà máy chạy trên linh kiện mới thì độ chính xác tụt xuống `60%`. Chuyện gì đã xảy ra? Bài học này sẽ mổ xẻ chính xác nguyên nhân!

## 5. Giải thích trực quan: Quy tắc "Bức tường Lửa" Giữa Train – Val – Test
```
[Toàn bộ Dữ liệu Gốc]
        │
        ├──► (Chia tập TRƯỚC TIÊN: Stratified hoặc Group Split)
        │
        ├──► 1. Tập TRAIN (70%) ──► Fit(Scaler/PCA) + Transform + Augmentation ngẫu nhiên
        │
        ├──► 2. Tập VAL   (15%) ──► Chỉ Transform (dùng μ, σ của Train) ──► Chọn Hyperparameter & Checkpoint
        │
        └──► 3. Tập TEST  (15%) ──► Chỉ Transform (dùng μ, σ của Train) ──► Đánh giá đúng 1 lần cuối cùng!
```

## 6. Thuật ngữ mới & Giải thích Keyword
| Keyword | Nghĩa tiếng Việt | Nhóm | Vai trò, Quy tắc sử dụng & Lỗi thường gặp |
|---|---|---|---|
| `stratify` | Phân tầng theo tỷ lệ nhãn | Dữ liệu | Tham số trong `train_test_split(..., stratify=y)` đảm bảo tỷ lệ phần trăm mỗi lớp ở `train`, `val`, `test` giống hệt nhau. |
| `GroupShuffleSplit` / `GroupKFold` | Chia tập theo nhóm đối tượng | Dữ liệu | Đảm bảo mọi mẫu có cùng `group_id` (cùng 1 bệnh nhân, cùng 1 video clip) không bao giờ vừa xuất hiện ở `train` vừa xuất hiện ở `val/test`. |
| `fit_transform` vs `transform` | Học tham số & Biến đổi | Dữ liệu | **Chỉ gọi `scaler.fit_transform(X_train)` trên tập Train**. Với `X_val` và `X_test`, **chỉ được gọi `scaler.transform(...)`**. |
| `class_weight` | Trọng số phạt theo lớp | Huấn luyện | Tăng hệ số phạt trong hàm loss cho lớp thiểu số theo tỷ lệ nghịch với tần suất xuất hiện: $w_c = \frac{N}{C \times N_c}$. |

## 7. Công thức tính Trọng số Lớp Cân bằng (`Balanced Class Weight`) & Ví dụ số nhỏ
Với tập dữ liệu có tổng cộng $N$ mẫu, $C$ lớp, và lớp thứ $c$ có $N_c$ mẫu:
$$w_c = \frac{N}{C \times N_c}$$
- **Ví dụ số nhỏ**: Cho $N = 100$ ảnh gồm $C = 2$ lớp: Lớp `0` (Bình thường) có $N_0 = 90$ ảnh, Lớp `1` (Lỗi) chỉ có $N_1 = 10$ ảnh:
  - $w_0 = \frac{100}{2 \times 90} \approx 0.556$
  - $w_1 = \frac{100}{2 \times 10} = 5.000$
  $\implies$ Mỗi khi đoán sai 1 ảnh Lớp `1` (Lỗi), mô hình bị phạt nặng gấp $5.0 / 0.556 = 9$ lần so với đoán sai 1 ảnh Lớp `0`!""",
        ),
        (
            "code",
            """# Cell 1: Chia tập Train / Val / Test đúng chuẩn (Stratified Split) vs Chia ngẫu nhiên không phân tầng
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
if (project_root / "image_ai_learning").is_dir():
    project_root = project_root / "image_ai_learning"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from src.synthetic_data import generate_synthetic_shapes_dataset
from src.utils import set_seed

set_seed(42)

ds = generate_synthetic_shapes_dataset(num_samples=200, image_size=32, seed=42)
X_all = ds["images"].reshape(len(ds["images"]), -1).astype(np.float32)
y_all = ds["labels"]

# Chia 70% Train, 15% Val, 15% Test có dùng stratify=y
X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.30, random_state=42, stratify=y_all
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Tổng mẫu: {len(X_all)} -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print("Phân phối nhãn tập Train:", np.bincount(y_train))
print("Phân phối nhãn tập Val  :", np.bincount(y_val))
print("Phân phối nhãn tập Test :", np.bincount(y_test))""",
        ),
        (
            "code",
            """# Cell 2: Thí nghiệm chứng minh Rò rỉ dữ liệu (Data Leakage) khi Nhân bản/Biến đổi trước khi chia tập
# Giả lập tình huống phổ biến: Có 60 vật thể gốc, mỗi vật thể chụp 3 tấm liên tiếp gần giống nhau (tổng 180 ảnh)
rng = np.random.default_rng(42)
n_objects = 60
base_features = rng.normal(0, 1, size=(n_objects, 256))
obj_labels = rng.integers(0, 4, size=n_objects)

# Mỗi vật thể sinh ra 3 ảnh chỉ lệch nhau một chút nhiễu nhỏ, có cùng group_id = mã vật thể
X_seq = np.repeat(base_features, repeats=3, axis=0) + rng.normal(0, 0.08, size=(n_objects * 3, 256))
y_seq = np.repeat(obj_labels, repeats=3, axis=0)
groups_seq = np.repeat(np.arange(n_objects), repeats=3, axis=0)

# CÁCH SAI (Rò rỉ dữ liệu): Chia ngẫu nhiên từng ảnh -> Ảnh chụp lần 1 vào Train, ảnh chụp lần 2 của CÙNG vật thể vào Test!
X_tr_leak, X_te_leak, y_tr_leak, y_te_leak = train_test_split(X_seq, y_seq, test_size=0.25, random_state=42)
clf_leak = LogisticRegression(max_iter=500).fit(X_tr_leak, y_tr_leak)
acc_leaked = clf_leak.score(X_te_leak, y_te_leak)

# CÁCH ĐÚNG (Group Split): Mọi ảnh của cùng 1 vật thể (group_id) bắt buộc nằm cùng 1 phía (hoặc Train, hoặc Test)
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X_seq, y_seq, groups=groups_seq))
overlap_groups = set(groups_seq[train_idx]).intersection(set(groups_seq[test_idx]))

clf_clean = LogisticRegression(max_iter=500).fit(X_seq[train_idx], y_seq[train_idx])
acc_clean = clf_clean.score(X_seq[test_idx], y_seq[test_idx])

print(f"[SAI - Random Split gây rò rỉ] Độ chính xác ảo trên Test   : {acc_leaked*100:.1f}% (do học thuộc vật thể đã gặp!)")
print(f"[ĐÚNG - Group Split chuẩn]     Số group trùng nhau         : {len(overlap_groups)}")
print(f"[ĐÚNG - Group Split chuẩn]     Độ chính xác thực sự (dữ liệu ngẫu nhiên): {acc_clean*100:.1f}% (kỳ vọng ~25% cho 4 lớp ngẫu nhiên)")""",
        ),
        (
            "code",
            """# Cell 3: Quy trình Chuẩn hóa (StandardScaler) và PCA KHÔNG rò rỉ dữ liệu
scaler = StandardScaler()
pca = PCA(n_components=32, random_state=42)

# CHỈ FIT TRÊN TẬP TRAIN
X_train_scaled = scaler.fit_transform(X_train)
X_train_pca = pca.fit_transform(X_train_scaled)

# TẬP VAL VÀ TEST CHỈ DÙNG TRANSFORM (Áp dụng mean, std và trục PCA đã học từ Train)
X_val_scaled = scaler.transform(X_val)
X_val_pca = pca.transform(X_val_scaled)

X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

print("X_train_pca shape:", X_train_pca.shape, "| Mean tập Train sau scale:", round(float(X_train_scaled.mean()), 5))
print("X_val_pca shape  :", X_val_pca.shape,   "| Mean tập Val sau scale  :", round(float(X_val_scaled.mean()), 5))""",
        ),
        (
            "code",
            """# Cell 4: Xử lý Mất cân bằng lớp (Class Imbalance) bằng Class Weights và WeightedRandomSampler
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from src.evaluation import compute_classification_metrics

# Tạo tập dữ liệu mất cân bằng nặng: Lớp 0,1,2 mỗi lớp 45 mẫu, riêng Lớp 3 (lớp hiếm) chỉ có 6 mẫu!
keep_indices = np.concatenate([
    np.where(y_train == 0)[0][:45],
    np.where(y_train == 1)[0][:45],
    np.where(y_train == 2)[0][:45],
    np.where(y_train == 3)[0][:6],
])
X_imb = X_train_pca[keep_indices]
y_imb = y_train[keep_indices]

# So sánh LogisticRegression mặc định vs class_weight='balanced'
clf_unweighted = LogisticRegression(max_iter=500, random_state=42).fit(X_imb, y_imb)
clf_balanced = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42).fit(X_imb, y_imb)

m_unw = compute_classification_metrics(y_val, clf_unweighted.predict(X_val_pca), ds["class_names"])
m_bal = compute_classification_metrics(y_val, clf_balanced.predict(X_val_pca), ds["class_names"])

print("Số lượng mẫu mỗi lớp trong tập Imbalanced Train:", np.bincount(y_imb))
print(f"Không dùng class_weight -> Recall lớp hiếm 'cross': {m_unw['per_class']['cross']['recall']*100:5.1f}% | Macro-F1: {m_unw['macro_f1']:.3f}")
print(f"Có dùng class_weight    -> Recall lớp hiếm 'cross': {m_bal['per_class']['cross']['recall']*100:5.1f}% | Macro-F1: {m_bal['macro_f1']:.3f}")

# Minh họa WeightedRandomSampler cho PyTorch DataLoader
class_counts = np.bincount(y_imb)
class_weights = 1.0 / np.maximum(class_counts, 1)
sample_weights = class_weights[y_imb]
sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(y_imb), replacement=True)
imb_loader = DataLoader(TensorDataset(torch.from_numpy(X_imb), torch.from_numpy(y_imb)), batch_size=32, sampler=sampler)
first_batch_x, first_batch_y = next(iter(imb_loader))
print("Phân phối nhãn trong 1 batch 32 mẫu nhờ WeightedRandomSampler:", np.bincount(first_batch_x.shape[0] and first_batch_y.numpy(), minlength=4))

fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.8))
axes[0].bar(ds["class_names"], np.bincount(y_imb, minlength=4), color="indianred")
axes[0].set_title("Tập Imbalanced Train (Lớp cross rất ít)")
axes[1].bar(ds["class_names"], np.bincount(first_batch_y.numpy(), minlength=4), color="seagreen")
axes[1].set_title("1 Batch 32 mẫu qua WeightedRandomSampler")
plt.tight_layout()
plt.show()""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Ở Cell 2 (Data Leakage)**: Trên dữ liệu hoàn toàn ngẫu nhiên 4 lớp (độ chính xác thực tế chỉ có thể quanh `25%`), việc chia ngẫu nhiên các ảnh chụp liên tiếp của cùng một vật thể đã đẩy độ chính xác trên tập Test lên mức **ảo tưởng `>80%`**! Khi dùng `GroupShuffleSplit`, điểm số trở về đúng thực tế `~25%`.
- **Ở Cell 4 (Class Imbalance)**: Khi lớp `cross` chỉ có `6` mẫu so với `45` mẫu của các lớp khác, mô hình mặc định dễ bỏ qua lớp `cross` (`Recall` thấp). Khi bật `class_weight='balanced'` hoặc dùng `WeightedRandomSampler`, lớp thiểu số được chú trọng công bằng hơn và trong mỗi batch của PyTorch cả 4 lớp đều xuất hiện cân đối.

## 13. Lỗi thường gặp và cách sửa
1. **Dùng tập `Test` để chọn mô hình (Test Set Leakage)**: Nếu bạn thử 20 cấu hình siêu tham số và chọn cấu hình có điểm `Test` cao nhất, tập `Test` đã biến thành tập `Validation`! Hãy luôn dùng tập `Val` để chọn cấu hình.
2. **Đặt cả `shuffle=True` và `sampler=...` trong `DataLoader`**: Khi đã dùng `WeightedRandomSampler`, bản thân `sampler` đã lấy mẫu ngẫu nhiên có trọng số rồi, nên phải để `shuffle=False` (mặc định).

## 14. Bài tập tự làm
- **Cơ bản**: Tính tay vector `class_weights` theo công thức $w_c = \frac{N}{C \times N_c}$ cho mảng nhãn `y_imb` ở Cell 4 và so sánh với `sklearn.utils.class_weight.compute_class_weight('balanced', classes=np.arange(4), y=y_imb)`.
- **Trung cấp**: Viết hàm `verify_no_group_leakage(train_groups, val_groups, test_groups)` trả về `True` nếu 3 tập hoàn toàn rời nhau về mặt `group_id`, ngược lại ném ra `AssertionError`.
- **Nâng cao**: Thực hiện kiểm định chéo 5 phần có phân tầng (**5-Fold Stratified Cross-Validation**) bằng `StratifiedKFold(n_splits=5)` mà trong mỗi fold đều tạo mới `StandardScaler()` chỉ `fit` trên phần `train_fold`.

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 05
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

# 1. Cơ bản: Đối chiếu công thức tính tay class_weight với scikit-learn
counts = np.bincount(y_imb)
manual_weights = len(y_imb) / (len(counts) * counts)
sklearn_weights = compute_class_weight(class_weight="balanced", classes=np.arange(4), y=y_imb)
print("[Cơ bản] Trọng số tính tay :", manual_weights.round(4))
print("[Cơ bản] Trọng số sklearn  :", sklearn_weights.round(4))
assert np.allclose(manual_weights, sklearn_weights)

# 2. Trung cấp: Hàm kiểm tra rò rỉ nhóm (Group Leakage Guard)
def verify_no_group_leakage(tr_g: np.ndarray, va_g: np.ndarray, te_g: np.ndarray) -> bool:
    s_tr, s_va, s_te = set(tr_g), set(va_g), set(te_g)
    assert len(s_tr & s_va) == 0, f"Rò rỉ nhóm giữa Train và Val: {s_tr & s_va}"
    assert len(s_tr & s_te) == 0, f"Rò rỉ nhóm giữa Train và Test: {s_tr & s_te}"
    assert len(s_va & s_te) == 0, f"Rò rỉ nhóm giữa Val và Test: {s_va & s_te}"
    return True

print("[Trung cấp] Kiểm tra độc lập nhóm:", verify_no_group_leakage(groups_seq[train_idx], np.array([-1]), groups_seq[test_idx]))

# 3. Nâng cao: 5-Fold Stratified CV không rò rỉ StandardScaler
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for fold_tr, fold_va in skf.split(X_train, y_train):
    fold_scaler = StandardScaler()
    X_f_tr = fold_scaler.fit_transform(X_train[fold_tr])
    X_f_va = fold_scaler.transform(X_train[fold_va])
    clf_f = LogisticRegression(max_iter=1000, C=0.1, random_state=42).fit(X_f_tr, y_train[fold_tr])
    cv_scores.append(clf_f.score(X_f_va, y_train[fold_va]))
print(f"[Nâng cao] 5-Fold CV Accuracy: {np.mean(cv_scores)*100:.2f}% ± {np.std(cv_scores)*100:.2f}%")""",
        ),
        (
            "md",
            r"""## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Hiểu và áp dụng đúng quy tắc chia `Train / Val / Test` trước khi `fit` bất kỳ bước tiền xử lý nào.
- [x] Biết dùng `GroupShuffleSplit` cho dữ liệu ảnh theo chuỗi/vật thể và xử lý `Class Imbalance`.
- [x] Sẵn sàng chuyển sang **Bài 06: `06_features_and_classical_ml.ipynb`** để trích xuất đặc trưng HOG/LBP/Color Histogram và huấn luyện 7 mô hình Machine Learning truyền thống!

## 17. Nguồn tham khảo
- Kaufman, S. et al. (2012), *Leakage in Data Mining: Formulation, Detection, and Avoidance*, ACM TKDD.
- scikit-learn Cross-validation documentation: https://scikit-learn.org/stable/modules/cross_validation.html""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "05_datasets_and_data_leakage.ipynb", cells)


if __name__ == "__main__":
    for builder in [build_nb00, build_nb01, build_nb02, build_nb03, build_nb04, build_nb05]:
        path = builder()
        print("Created and validated:", path.name)
