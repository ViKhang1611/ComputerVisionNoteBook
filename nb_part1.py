from nbformat.v4 import new_markdown_cell, new_code_cell

CELLS = []


def md(s):
    CELLS.append(new_markdown_cell(s.strip("\n")))


def code(s):
    CELLS.append(new_code_cell(s.strip("\n")))


# =====================================================================
md(r'''
# 📘 Xử lý ảnh với OpenCV — Hướng dẫn chi tiết từ cơ bản đến nâng cao

> **Mục tiêu:** Nắm vững *các thao tác xử lý trên MỘT bức ảnh* với OpenCV (Python): từ pixel, màu sắc, biến đổi hình học, lọc, phát hiện cạnh, hình thái học, contour… đến phân vùng ảnh và một dự án hoàn chỉnh (Document Scanner).

Mỗi phần đều có cấu trúc:

1. 🧠 **Lý thuyết** — ý tưởng, công thức, khi nào dùng.
2. 💻 **Code ví dụ** — có chú thích từng dòng.
3. 🖼️ **Kết quả** — ảnh/biểu đồ ngay bên dưới (notebook đã được chạy sẵn).
4. 🔍 **Nhận xét** — đọc kết quả thế nào, lỗi hay gặp.
5. ✍️ **Bài tập nhỏ** — tự làm để nhớ lâu (lời giải ở cuối notebook).

### 🗺️ Mục lục

| # | Chủ đề | Hàm chính |
|---|---|---|
| 0 | Chuẩn bị môi trường & hàm tiện ích | `show`, `load_sample` |
| 1 | Ảnh số là gì? | NumPy `ndarray`, `shape`, `dtype` |
| 2 | Đọc – hiển thị – lưu ảnh | `imread`, `imwrite`, `imencode` |
| 3 | Pixel, ROI, tách/gộp kênh, viền | `split`, `merge`, `copyMakeBorder` |
| 4 | Không gian màu | `cvtColor`, `inRange` |
| 5 | Vẽ hình & chữ | `line`, `rectangle`, `circle`, `putText` |
| 6 | Phép toán số học & bitwise | `add`, `addWeighted`, `bitwise_*` |
| 7 | Biến đổi hình học | `resize`, `warpAffine`, `warpPerspective`, `remap` |
| 8 | Độ sáng, tương phản, gamma | `convertScaleAbs`, `LUT`, `normalize` |
| 9 | Histogram & cân bằng histogram | `calcHist`, `equalizeHist`, `CLAHE` |
| 10 | Phân ngưỡng | `threshold`, Otsu, `adaptiveThreshold` |
| 11 | Tích chập & lọc làm mịn / khử nhiễu | `filter2D`, `GaussianBlur`, `medianBlur`, `bilateralFilter` |
| 12 | Làm sắc nét | kernel sharpen, unsharp mask |
| 13 | Phát hiện cạnh | `Sobel`, `Scharr`, `Laplacian`, `Canny` |
| 14 | Hình thái học | `erode`, `dilate`, `morphologyEx` |
| 15 | Contour & phân tích hình dạng | `findContours`, `moments`, `approxPolyDP` |
| 16 | Connected Components | `connectedComponentsWithStats` |
| 17 | Hough: đường thẳng & đường tròn | `HoughLinesP`, `HoughCircles` |
| 18 | Template Matching | `matchTemplate` |
| 19 | Góc & đặc trưng | `cornerHarris`, `goodFeaturesToTrack`, ORB, SIFT |
| 20 | Image Pyramid & pha trộn ảnh | `pyrDown`, `pyrUp` |
| 21 | Miền tần số (Fourier) | `np.fft`, lọc thông thấp/cao, notch |
| 22 | Phân vùng ảnh | `kmeans`, `watershed`, `grabCut` |
| 23 | Phục hồi ảnh (Inpainting) | `inpaint` |
| 24 | 🚀 Dự án: Document Scanner | tổng hợp |
| 25 | Tra cứu nhanh, lỗi hay gặp, bài tập & lời giải | |

> 💡 **Cách học hiệu quả:** Đừng chỉ đọc! Hãy chạy lại từng ô (`Shift + Enter`), **thay đổi tham số** (kích thước kernel, ngưỡng, góc xoay…) rồi quan sát kết quả thay đổi thế nào.
''')

# =====================================================================
md(r'''
---
## 0. Chuẩn bị môi trường

Cài đặt (chỉ cần một lần, chạy trong terminal hoặc trong ô notebook với `%pip`):

```bash
pip install opencv-python numpy matplotlib
```

- `opencv-python`: thư viện OpenCV (module `cv2`).
- `numpy`: OpenCV biểu diễn ảnh dưới dạng **mảng NumPy**.
- `matplotlib`: hiển thị ảnh ngay trong notebook.

> ⚠️ `cv2.imshow()` mở cửa sổ riêng và **không hoạt động tốt trong Jupyter** (dễ treo kernel). Trong notebook ta dùng Matplotlib để hiển thị. `cv2.imshow` chỉ nên dùng trong script `.py` (xem mục 2.3).

Ô dưới đây import thư viện và định nghĩa các **hàm tiện ích** dùng xuyên suốt notebook:

- `show(...)`: hiển thị 1 hoặc nhiều ảnh (tự đổi BGR → RGB).
- `load_sample(name)`: tải ảnh mẫu chính thức của OpenCV từ GitHub về thư mục `images/` (nếu không có mạng sẽ tự tạo ảnh tổng hợp thay thế).
- `plot_hist(...)`: vẽ histogram.
''')

code(r'''
%matplotlib inline
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 80          # độ phân giải hình trong notebook
plt.rcParams['axes.titlesize'] = 10

print('OpenCV    :', cv2.__version__)
print('NumPy     :', np.__version__)

IMG_DIR = Path('images');          IMG_DIR.mkdir(exist_ok=True)   # nơi chứa ảnh mẫu
OUT_DIR = Path('outputs_opencv');  OUT_DIR.mkdir(exist_ok=True)   # nơi lưu kết quả


def show(images, titles=None, cols=3, size=4.0, cmap='gray'):
    """Hiển thị 1 hoặc nhiều ảnh OpenCV.

    - Ảnh 3 kênh được coi là BGR và tự chuyển sang RGB để hiển thị đúng màu.
    - Ảnh 1 kênh (xám) hiển thị bằng colormap `cmap`.
    - Ảnh uint8 luôn được hiển thị trong thang 0..255 (không tự kéo giãn).
    """
    if not isinstance(images, (list, tuple)):
        images = [images]
        titles = [titles] if titles is not None else None
    n = len(images)
    cols = min(cols, n)
    rows = int(np.ceil(n / cols))
    h, w = images[0].shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows * h / w + 0.5))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis('off')
        if i >= n:
            continue
        img = images[i]
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        if img.ndim == 2:
            if img.dtype == np.uint8:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
            else:
                ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if titles is not None:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()


def plot_hist(img, ax=None, title='Histogram', mask=None):
    """Vẽ histogram cho ảnh xám (1 đường) hoặc ảnh màu BGR (3 đường)."""
    ax = ax or plt.gca()
    if img.ndim == 2:
        hist = cv2.calcHist([img], [0], mask, [256], [0, 256])
        ax.plot(hist, color='k')
    else:
        for i, c in enumerate(('b', 'g', 'r')):
            hist = cv2.calcHist([img], [i], mask, [256], [0, 256])
            ax.plot(hist, color=c, label=c.upper())
        ax.legend()
    ax.set_xlim(0, 255)
    ax.set_title(title)
    ax.set_xlabel('Mức xám')
    ax.set_ylabel('Số pixel')


def make_synthetic(name, h=480, w=640):
    """Ảnh tổng hợp thay thế khi không tải được ảnh mẫu."""
    rng = np.random.default_rng(0)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = np.linspace(40, 200, w, dtype=np.uint8)[None, :, None]
    if 'sudoku' in name:
        img[:] = 230
        for i in range(10):
            t = 3 if i % 3 == 0 else 1
            cv2.line(img, (60 + i * 40, 40), (60 + i * 40, 400), (20, 20, 20), t)
            cv2.line(img, (60, 40 + i * 40), (420, 40 + i * 40), (20, 20, 20), t)
        for _ in range(30):
            r, c = rng.integers(0, 9, 2)
            cv2.putText(img, str(rng.integers(1, 10)), (72 + c * 40, 72 + r * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
        shade = np.linspace(0.5, 1.1, w)[None, :, None]
        img = np.clip(img * shade, 0, 255).astype(np.uint8)
    else:
        colors = [(40, 40, 220), (40, 200, 40), (220, 120, 30), (30, 200, 230), (200, 40, 200)]
        for _ in range(25):
            c = tuple(int(x) for x in colors[rng.integers(len(colors))])
            cv2.circle(img, (int(rng.integers(40, w - 40)), int(rng.integers(40, h - 40))),
                       int(rng.integers(15, 40)), c, -1)
    return img


BASE_URL = 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/'


def load_sample(name, flags=cv2.IMREAD_COLOR):
    """Đọc ảnh mẫu của OpenCV (tự tải về lần đầu)."""
    path = IMG_DIR / name
    if not path.exists():
        try:
            urllib.request.urlretrieve(BASE_URL + name, path)
            print(f'Đã tải {name}')
        except Exception as e:
            print(f'Không tải được {name} ({e}) -> dùng ảnh tổng hợp thay thế')
            cv2.imwrite(str(path), make_synthetic(name))
    img = cv2.imread(str(path), flags)
    assert img is not None, f'Không đọc được {path}'
    return img


# Tải trước các ảnh mẫu sẽ dùng
for name in ['fruits.jpg', 'sudoku.png', 'smarties.png', 'messi5.jpg',
             'building.jpg', 'apple.jpg', 'orange.jpg']:
    load_sample(name)
print('Ảnh mẫu trong thư mục images/:', sorted(p.name for p in IMG_DIR.iterdir()))
''')

# =====================================================================
md(r'''
---
## 1. Ảnh số là gì?

### 1.1. Pixel và ma trận

Một bức ảnh số là **lưới các điểm ảnh (pixel)**. Trong OpenCV-Python, ảnh chính là một **mảng NumPy** (`numpy.ndarray`):

| Loại ảnh | `shape` | Ý nghĩa |
|---|---|---|
| Ảnh xám (grayscale) | `(H, W)` | mỗi pixel là 1 số: độ sáng |
| Ảnh màu | `(H, W, 3)` | mỗi pixel có 3 số: **B, G, R** (chú ý thứ tự!) |
| Ảnh có kênh trong suốt | `(H, W, 4)` | B, G, R, **A**lpha |

- `H` (height) = số **hàng**, `W` (width) = số **cột**.
- Kiểu dữ liệu phổ biến nhất: `uint8` → giá trị nguyên **0 … 255** (0 = đen, 255 = trắng).

### 1.2. Hệ tọa độ — điều gây nhầm lẫn số 1!

```
(0,0) ───────────► x (cột, width)
  │
  │      pixel tại (x, y)  ⇔  img[y, x]
  ▼
  y (hàng, height)
```

- **Gốc tọa độ ở góc trên-trái**, trục y hướng **xuống**.
- Khi **truy cập mảng NumPy**: `img[y, x]` (hàng trước, cột sau).
- Khi **gọi hàm OpenCV** (vẽ, cắt bằng điểm…): truyền điểm dạng `(x, y)`.
- `cv2.resize(img, (W, H))` nhận **(rộng, cao)**, ngược với `img.shape = (H, W)`.
''')

code(r'''
# Một ảnh xám 5x5 tự tạo bằng NumPy
tiny = np.array([[  0,  50, 100, 150, 200],
                 [ 50, 100, 150, 200, 250],
                 [100, 150, 200, 250, 200],
                 [150, 200, 250, 200, 150],
                 [200, 250, 200, 150, 100]], dtype=np.uint8)

print('Ma trận ảnh:\n', tiny)
print('shape =', tiny.shape, '| dtype =', tiny.dtype, '| số pixel =', tiny.size)
print('Pixel tại hàng y=0, cột x=4 :', tiny[0, 4])

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(tiny, cmap='gray', vmin=0, vmax=255)
for y in range(5):
    for x in range(5):
        ax.text(x, y, tiny[y, x], ha='center', va='center',
                color='red' if tiny[y, x] > 127 else 'yellow', fontsize=11)
ax.set_xlabel('x (cột)'); ax.set_ylabel('y (hàng)')
ax.set_title('Ảnh xám 5×5: mỗi ô là 1 pixel')
plt.show()
''')

md(r'''
### 1.3. Ảnh màu: thứ tự kênh **BGR** (không phải RGB!)

Vì lý do lịch sử, OpenCV lưu ảnh màu theo thứ tự **B – G – R**. Matplotlib, PIL, hầu hết thư viện khác dùng **R – G – B**. Nếu đưa thẳng ảnh OpenCV vào `plt.imshow` → màu bị sai (đỏ ↔ xanh dương).
''')

code(r'''
color = np.zeros((2, 3, 3), np.uint8)   # ảnh 2 hàng x 3 cột x 3 kênh
color[0, 0] = (255, 0, 0)     # B=255 -> XANH DƯƠNG
color[0, 1] = (0, 255, 0)     # G=255 -> XANH LÁ
color[0, 2] = (0, 0, 255)     # R=255 -> ĐỎ
color[1, 0] = (0, 255, 255)   # G+R   -> VÀNG
color[1, 1] = (255, 0, 255)   # B+R   -> TÍM HỒNG (magenta)
color[1, 2] = (255, 255, 255) # cả 3  -> TRẮNG

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(color)                                  # SAI: coi BGR là RGB
axes[0].set_title('❌ plt.imshow(img) trực tiếp (sai màu)')
axes[1].imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)) # ĐÚNG
axes[1].set_title('✅ Đổi BGR→RGB trước khi hiển thị')
for a in axes: a.axis('off')
plt.show()
print('Pixel [0,0] (B,G,R) =', color[0, 0])
''')

md(r'''
🔍 **Nhận xét:** Pixel `(255, 0, 0)` trong OpenCV là **xanh dương**, nhưng Matplotlib hiểu là **đỏ**. Hàm `show()` ở mục 0 đã tự xử lý việc này. Cách khác: `img[:, :, ::-1]` (đảo thứ tự kênh bằng NumPy).
''')

# =====================================================================
md(r'''
---
## 2. Đọc – hiển thị – lưu ảnh

### 2.1. `cv2.imread(path, flags)`

| Flag | Kết quả |
|---|---|
| `cv2.IMREAD_COLOR` (mặc định, =1) | ảnh màu BGR 3 kênh (bỏ alpha) |
| `cv2.IMREAD_GRAYSCALE` (=0) | ảnh xám 1 kênh |
| `cv2.IMREAD_UNCHANGED` (=-1) | giữ nguyên (kể cả kênh alpha, ảnh 16-bit) |
| `cv2.IMREAD_REDUCED_COLOR_2/4/8` | đọc và thu nhỏ 2/4/8 lần (nhanh với ảnh lớn) |

> ⚠️ **Quan trọng:** Nếu đường dẫn sai, `imread` **không báo lỗi** mà trả về `None`. Luôn kiểm tra `img is None`!
''')

code(r'''
path = str(IMG_DIR / 'fruits.jpg')

img   = cv2.imread(path)                              # màu (BGR)
gray  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)        # xám
small = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)  # thu nhỏ 1/2

for name, im in [('img', img), ('gray', gray), ('small', small)]:
    print(f'{name:6s}: type={type(im).__name__}, shape={im.shape}, dtype={im.dtype}, '
          f'min={im.min()}, max={im.max()}, mean={im.mean():.1f}')

show([img, gray, small], ['IMREAD_COLOR', 'IMREAD_GRAYSCALE', 'IMREAD_REDUCED_COLOR_2'])

bad = cv2.imread('khong_ton_tai.jpg')
print('Đọc file không tồn tại ->', bad)   # None, không có exception!
if bad is None:
    print('=> Luôn kiểm tra: if img is None: raise FileNotFoundError(...)')
''')

md(r'''
### 2.2. Lưu ảnh: `cv2.imwrite(path, img, params)`

Định dạng được quyết định bởi **đuôi file**. Một số tham số:

- JPEG: `[cv2.IMWRITE_JPEG_QUALITY, 0..100]` (mặc định 95) — nén **mất dữ liệu** (lossy).
- PNG: `[cv2.IMWRITE_PNG_COMPRESSION, 0..9]` — nén **không mất dữ liệu** (lossless), số càng lớn file càng nhỏ nhưng ghi chậm hơn.

Để đo chất lượng ảnh sau nén, ta dùng **PSNR** (Peak Signal-to-Noise Ratio):

$$\text{PSNR} = 10\log_{10}\frac{255^2}{\text{MSE}}, \qquad \text{MSE}=\frac{1}{N}\sum (I-K)^2$$

PSNR càng cao càng giống ảnh gốc (> 40 dB: gần như không phân biệt; < 30 dB: thấy rõ suy giảm; ảnh giống hệt → ∞).
''')

code(r'''
results, crops = [], []
y0, x0, s = 150, 200, 80                       # vùng cắt để phóng to quan sát
for q in [95, 50, 10]:
    p = OUT_DIR / f'fruits_q{q}.jpg'
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, q])
    rec = cv2.imread(str(p))
    results.append((f'JPEG q={q}', p.stat().st_size / 1024, cv2.PSNR(img, rec)))
    crops.append(rec[y0:y0+s, x0:x0+s])

p = OUT_DIR / 'fruits.png'
cv2.imwrite(str(p), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
rec = cv2.imread(str(p))
results.append(('PNG', p.stat().st_size / 1024, cv2.PSNR(img, rec)))

print(f'{"Định dạng":12s} | {"Dung lượng":>10s} | PSNR')
for name, kb, psnr in results:
    print(f'{name:12s} | {kb:8.1f} KB | {psnr:.2f} dB')

zoom = lambda im: cv2.resize(im, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
show([zoom(img[y0:y0+s, x0:x0+s])] + [zoom(c) for c in crops],
     ['Gốc (phóng to x4)', 'JPEG q=95', 'JPEG q=50', 'JPEG q=10'], cols=4, size=3)
''')

md(r'''
🔍 **Nhận xét:**
- PNG cho PSNR = ∞ (OpenCV trả về một số rất lớn, ví dụ 361 dB) vì **giống hệt** ảnh gốc, nhưng file lớn.
- JPEG chất lượng 10 rất nhỏ nhưng xuất hiện **khối vuông 8×8** và viền nhòe (*blocking artifacts*).
- ➜ Lưu **kết quả trung gian / mask / ảnh nhị phân** nên dùng **PNG**; ảnh chụp để chia sẻ dùng JPEG 85–95.

### 2.3. Hiển thị bằng cửa sổ OpenCV (dùng trong file `.py`, không chạy trong notebook)

```python
img = cv2.imread('anh.jpg')
cv2.imshow('Cua so anh', img)   # tên cửa sổ không nên có dấu
key = cv2.waitKey(0)            # chờ nhấn phím (0 = chờ mãi); trả về mã phím
if key == ord('s'):
    cv2.imwrite('da_luu.png', img)
cv2.destroyAllWindows()         # đóng tất cả cửa sổ
```

### 2.4. ⚠️ Đường dẫn có tiếng Việt (Unicode) trên Windows

`cv2.imread('D:/Ảnh/mèo.jpg')` thường trả về `None` trên Windows vì OpenCV không đọc được đường dẫn Unicode. Giải pháp: đọc byte bằng NumPy rồi giải mã bằng `cv2.imdecode`.
''')

code(r'''
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Đọc ảnh từ đường dẫn có ký tự Unicode (tiếng Việt)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, img, params=None):
    """Ghi ảnh ra đường dẫn Unicode."""
    ok, buf = cv2.imencode(Path(path).suffix, img, params or [])
    if ok:
        buf.tofile(str(path))
    return ok


vn_path = OUT_DIR / 'ảnh_trái_cây.png'
print('imwrite_unicode :', imwrite_unicode(vn_path, img))
print('cv2.imread      :', None if cv2.imread(str(vn_path)) is None else 'đọc được')
print('imread_unicode  :', imread_unicode(vn_path).shape)

# imencode cũng rất hữu ích để gửi ảnh qua mạng / lưu vào database dưới dạng bytes
ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
print('Ảnh mã hóa JPEG trong bộ nhớ:', buf.shape, 'bytes')
''')

# =====================================================================
md(r'''
---
## 3. Truy cập pixel, ROI, tách/gộp kênh

### 3.1. Đọc & ghi pixel
''')

code(r'''
img = load_sample('fruits.jpg')
h, w, c = img.shape
print(f'Kích thước: rộng={w}, cao={h}, số kênh={c}')

y, x = 100, 50
b, g, r = img[y, x]                       # đọc 1 pixel -> 3 giá trị B, G, R
print(f'Pixel (x={x}, y={y}) -> B={b}, G={g}, R={r}')
print('Chỉ kênh đỏ  img[y, x, 2] =', img[y, x, 2])

canvas = img.copy()                        # luôn copy nếu không muốn sửa ảnh gốc
canvas[y, x] = (0, 0, 255)                 # ghi 1 pixel (rất nhỏ, khó thấy)
canvas[20:80, 20:200] = (0, 255, 255)      # ghi cả 1 vùng: tô vàng
canvas[:, -30:] = 0                        # 30 cột cuối -> đen
canvas[::10, :] = (255, 255, 255)          # cứ 10 hàng kẻ 1 đường trắng
show([img, canvas], ['Gốc', 'Sau khi gán giá trị pixel'])
''')

md(r'''
### 3.2. ROI (Region of Interest) — vùng quan tâm

ROI = **cắt một vùng chữ nhật** bằng slicing NumPy: `roi = img[y1:y2, x1:x2]`.

> ⚠️ **View vs Copy:** slicing NumPy trả về một **view** (cùng chia sẻ bộ nhớ). Sửa `roi` ⇒ **ảnh gốc cũng bị sửa**! Dùng `.copy()` nếu muốn bản độc lập.
''')

code(r'''
work = img.copy()
roi_view = work[150:300, 150:330]       # VIEW
roi_copy = work[150:300, 150:330].copy()  # COPY

roi_view[:] = 255 - roi_view             # đảo màu trên view -> ảnh 'work' bị đổi theo!
roi_copy[:] = 0                          # sửa copy -> 'work' không đổi

# Sao chép (dán) một vùng sang chỗ khác: hai vùng phải cùng kích thước
patch = img[300:400, 50:200]
work2 = img.copy()
work2[20:120, 300:450] = patch

show([work, roi_copy, work2],
     ['Sửa VIEW → ảnh gốc bị đổi', 'COPY đã tô đen (độc lập)', 'Dán vùng [300:400, 50:200] lên góc trên'])
print('np.shares_memory(work, roi_view) =', np.shares_memory(work, roi_view))
print('np.shares_memory(work, roi_copy) =', np.shares_memory(work, roi_copy))
''')

md(r'''
### 3.3. Tách và gộp kênh màu — `cv2.split` / `cv2.merge`

Mỗi kênh là một ảnh xám: pixel **sáng** trong kênh R nghĩa là vùng đó **có nhiều màu đỏ**.
''')

code(r'''
b, g, r = cv2.split(img)                 # tương đương: b = img[:, :, 0], ...
zeros = np.zeros_like(b)

only_b = cv2.merge([b, zeros, zeros])    # chỉ giữ kênh B
only_g = cv2.merge([zeros, g, zeros])
only_r = cv2.merge([zeros, zeros, r])
swap   = cv2.merge([r, g, b])            # hoán đổi R <-> B

show([b, g, r], ['Kênh B (dạng xám)', 'Kênh G (dạng xám)', 'Kênh R (dạng xám)'])
show([only_b, only_g, only_r, swap], ['Chỉ B', 'Chỉ G', 'Chỉ R', 'Hoán đổi R↔B'], cols=4, size=3)
''')

md(r'''
🔍 **Nhận xét:** Quả chuối / cam (vàng – cam) sáng trong cả kênh **R** và **G** nhưng tối trong kênh **B** (vì vàng = đỏ + lục). `cv2.split` tốn chi phí sao chép; nếu chỉ cần 1 kênh, dùng indexing `img[:, :, 2]` sẽ nhanh hơn.

### 3.4. Thống kê ảnh
''')

code(r'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mean, std = cv2.meanStdDev(img)
print('Trung bình mỗi kênh (B,G,R):', mean.ravel().round(1))
print('Độ lệch chuẩn (B,G,R)      :', std.ravel().round(1))

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)   # chỉ cho ảnh 1 kênh
print(f'Pixel tối nhất = {min_val} tại (x,y)={min_loc}; sáng nhất = {max_val} tại {max_loc}')
print('Số pixel sáng hơn 200:', cv2.countNonZero((gray > 200).astype(np.uint8)))
''')

md(r'''
### 3.5. ⚡ Đừng dùng vòng lặp `for` trên từng pixel!

Python rất chậm khi lặp từng pixel. Hãy dùng **phép toán vector hóa của NumPy** hoặc **hàm OpenCV** (viết bằng C++ và tối ưu SIMD).
''')

code(r'''
small = cv2.resize(img, (200, 200))

t = time.perf_counter()
out_loop = np.empty_like(small)
for yy in range(small.shape[0]):
    for xx in range(small.shape[1]):
        out_loop[yy, xx] = 255 - small[yy, xx]
t_loop = time.perf_counter() - t

t = time.perf_counter(); out_np = 255 - small;            t_np = time.perf_counter() - t
t = time.perf_counter(); out_cv = cv2.bitwise_not(small); t_cv = time.perf_counter() - t

print(f'Vòng lặp Python : {t_loop*1000:8.2f} ms')
print(f'NumPy vector    : {t_np*1000:8.3f} ms  (nhanh hơn ~{t_loop/t_np:,.0f} lần)')
print(f'OpenCV          : {t_cv*1000:8.3f} ms')
print('Ba kết quả giống nhau:', np.array_equal(out_loop, out_np) and np.array_equal(out_np, out_cv))
''')

md(r'''
### 3.6. Thêm viền ảnh — `cv2.copyMakeBorder`

Rất quan trọng vì các bộ lọc (mục 11) cần biết "bên ngoài ảnh" có giá trị gì. Các kiểu viền:

| Kiểu | Minh họa (dãy `abcdefgh`) |
|---|---|
| `BORDER_CONSTANT` | `iiiiii|abcdefgh|iiiiii` (giá trị cố định) |
| `BORDER_REPLICATE` | `aaaaaa|abcdefgh|hhhhhhh` |
| `BORDER_REFLECT` | `fedcba|abcdefgh|hgfedcb` |
| `BORDER_REFLECT_101` (mặc định của bộ lọc) | `gfedcb|abcdefgh|gfedcba` |
| `BORDER_WRAP` | `cdefgh|abcdefgh|abcdefg` |
''')

code(r'''
src = cv2.resize(img, (200, 180))
p = 50
borders = {
    'CONSTANT (đỏ)': cv2.copyMakeBorder(src, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 255)),
    'REPLICATE':     cv2.copyMakeBorder(src, p, p, p, p, cv2.BORDER_REPLICATE),
    'REFLECT':       cv2.copyMakeBorder(src, p, p, p, p, cv2.BORDER_REFLECT),
    'REFLECT_101':   cv2.copyMakeBorder(src, p, p, p, p, cv2.BORDER_REFLECT_101),
    'WRAP':          cv2.copyMakeBorder(src, p, p, p, p, cv2.BORDER_WRAP),
}
show(list(borders.values()), list(borders.keys()), cols=5, size=2.8)
''')

md(r'''
✍️ **Bài tập 3:**
1. Cắt ra quả chuối trong ảnh `fruits.jpg` (thử đoán tọa độ, hiển thị để kiểm tra).
2. Tạo ảnh mà nửa trái là ảnh gốc, nửa phải là ảnh âm bản (`255 - img`).
''')

# =====================================================================
md(r'''
---
## 4. Không gian màu (Color Spaces)

`cv2.cvtColor(img, code)` chuyển đổi giữa hơn 150 không gian màu. Các không gian hay dùng:

| Không gian | Thành phần | Dùng khi |
|---|---|---|
| **BGR** | Blue, Green, Red | mặc định của OpenCV |
| **GRAY** | độ sáng | hầu hết thuật toán (cạnh, ngưỡng, contour…) |
| **HSV** | Hue (sắc độ), Saturation (bão hòa), Value (độ sáng) | **lọc theo màu**, ít nhạy với ánh sáng |
| **LAB** | L (sáng), a (lục↔đỏ), b (lam↔vàng) | so sánh màu theo cảm nhận người, cân bằng sáng |
| **YCrCb** | Y (sáng), Cr, Cb (màu) | nén ảnh, phát hiện da, cân bằng histogram |

### 4.1. Ảnh xám

$$Y = 0.299R + 0.587G + 0.114B$$

Mắt người nhạy với màu lục nhất ⇒ trọng số G lớn nhất. Đây **không phải** trung bình cộng đơn giản.
''')

code(r'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

B, G, R = [img[:, :, i].astype(np.float32) for i in range(3)]
gray_manual = np.round(0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)
gray_mean = img.mean(axis=2).astype(np.uint8)

print('Sai khác lớn nhất cvtColor vs công thức:', np.abs(gray.astype(int) - gray_manual).max())
show([gray, gray_mean, cv2.absdiff(gray, gray_mean) * 4],
     ['cvtColor BGR2GRAY', 'Trung bình (B+G+R)/3', 'Chênh lệch (x4 cho dễ thấy)'])
''')

md(r'''
### 4.2. HSV — không gian lý tưởng để lọc màu

- **H (Hue)**: "màu gì" — góc trên vòng màu 0°–360°. ⚠️ Trong OpenCV với ảnh `uint8`, **H ∈ [0, 179]** (= độ/2) để vừa 1 byte.
- **S (Saturation)**: độ "tươi/đậm" của màu, 0 = xám, 255 = rất tươi.
- **V (Value)**: độ sáng, 0 = đen.

Vì màu (H) tách riêng khỏi độ sáng (V), nên lọc theo H ổn định hơn nhiều so với lọc trên BGR khi ánh sáng thay đổi.
''')

code(r'''
# Thanh màu Hue của OpenCV: H chạy từ 0 -> 179, S = V = 255
hue = np.tile(np.arange(180, dtype=np.uint8), (30, 1))
hsv_bar = cv2.merge([hue, np.full_like(hue, 255), np.full_like(hue, 255)])
bar = cv2.cvtColor(hsv_bar, cv2.COLOR_HSV2BGR)

fig, ax = plt.subplots(figsize=(10, 1.6))
ax.imshow(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB), aspect='auto', extent=[0, 180, 0, 1])
ax.set_yticks([]); ax.set_xticks(range(0, 181, 10))
ax.set_title('Giá trị H trong OpenCV: Đỏ≈0/179, Cam≈10-20, Vàng≈25-35, Lục≈40-80, Lam≈100-130, Tím≈130-160')
plt.show()

# Mẹo: tra giá trị HSV của một màu BGR bất kỳ
for name, bgr in [('Đỏ', (0, 0, 255)), ('Lục', (0, 255, 0)), ('Lam', (255, 0, 0)), ('Vàng', (0, 255, 255))]:
    print(f'{name:5s} BGR={bgr} -> HSV={cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]}')
''')

code(r'''
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv)
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title('Ảnh gốc')
axes[1].imshow(H, cmap='hsv', vmin=0, vmax=179);     axes[1].set_title('H (hiển thị bằng colormap hsv)')
axes[2].imshow(S, cmap='gray', vmin=0, vmax=255);    axes[2].set_title('S: sáng = màu tươi')
axes[3].imshow(V, cmap='gray', vmin=0, vmax=255);    axes[3].set_title('V: độ sáng')
for a in axes: a.axis('off')
plt.tight_layout(); plt.show()
''')

md(r'''
### 4.3. Tách đối tượng theo màu — `cv2.inRange`

`mask = cv2.inRange(hsv, lower, upper)` → ảnh nhị phân: **255** nếu pixel nằm trong khoảng `[lower, upper]` ở **cả 3 kênh**, ngược lại **0**. Sau đó dùng `cv2.bitwise_and(img, img, mask=mask)` để giữ lại phần có màu đó.

> ⚠️ **Màu đỏ "quấn vòng"**: H của đỏ nằm ở **cả hai đầu** (0–10 và 170–179) ⇒ phải dùng **2 khoảng** rồi OR lại.
''')

code(r'''
ranges = {
    'Vàng (chuối)':  [((20, 100, 100), (35, 255, 255))],
    'Lục':           [((36, 60, 40),   (85, 255, 255))],
    'Đỏ (2 khoảng)': [((0, 120, 70),   (10, 255, 255)),
                      ((170, 120, 70), (179, 255, 255))],
}
outs, titles = [], []
for name, rr in ranges.items():
    mask = np.zeros(img.shape[:2], np.uint8)
    for lo, hi in rr:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    outs += [mask, cv2.bitwise_and(img, img, mask=mask)]
    titles += [f'Mask {name}', f'Kết quả {name}']
show(outs, titles, cols=2, size=4)
''')

md(r'''
🔍 **Nhận xét:** Mask chưa hoàn hảo — có lỗ nhỏ và nhiễu lấm tấm. Ở mục 14 (hình thái học) ta sẽ học cách "làm sạch" mask bằng **opening/closing**.

### 4.4. Ứng dụng: đổi màu đối tượng (dịch Hue)
''')

code(r'''
hsv2 = hsv.copy()
mask_y = cv2.inRange(hsv, np.array((20, 100, 100)), np.array((35, 255, 255)))
hsv2[..., 0] = np.where(mask_y > 0, (hsv2[..., 0].astype(int) + 90) % 180, hsv2[..., 0])  # dịch 180°
recolor = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

hsv3 = hsv.copy()
hsv3[..., 1] = cv2.multiply(hsv3[..., 1], 1.6)          # tăng độ bão hòa 60% (tự cắt ở 255)
vivid = cv2.cvtColor(hsv3, cv2.COLOR_HSV2BGR)
show([img, recolor, vivid], ['Gốc', 'Chuối vàng → xanh lam (H + 90)', 'Tăng saturation x1.6'])
''')

md(r'''
### 4.5. LAB và YCrCb
''')

code(r'''
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
L, A, Bc = cv2.split(lab)
Y, Cr, Cb = cv2.split(ycc)
show([L, A, Bc, Y, Cr, Cb],
     ['LAB: L (sáng)', 'LAB: a (lục ← → đỏ)', 'LAB: b (lam ← → vàng)',
      'YCrCb: Y (sáng)', 'YCrCb: Cr (đỏ)', 'YCrCb: Cb (lam)'], cols=3, size=3.6)
''')

md(r'''
🔍 **Nhận xét:** Kênh **L** và **Y** gần như ảnh xám — chứa *độ sáng*; các kênh còn lại chỉ chứa *thông tin màu*. Đây là lý do khi muốn chỉnh sáng/tương phản ảnh màu (mục 9), ta chỉ xử lý L hoặc Y để **không làm lệch màu**.

✍️ **Bài tập 4:** Dùng `inRange` trong HSV để tách các viên kẹo màu xanh lam trong `smarties.png`.
''')

# =====================================================================
md(r'''
---
## 5. Vẽ hình và chữ lên ảnh

Các hàm vẽ **sửa trực tiếp** ảnh truyền vào (in-place). Tham số chung:
- `color`: tuple **BGR**, ví dụ `(0, 0, 255)` = đỏ.
- `thickness`: độ dày nét; **`-1` (hoặc `cv2.FILLED`) = tô kín**.
- `lineType`: `cv2.LINE_AA` cho nét mượt (khử răng cưa).
- Tọa độ điểm là **`(x, y)` kiểu `int`**.
''')

code(r'''
canvas = np.full((420, 700, 3), 255, np.uint8)        # nền trắng

cv2.line(canvas, (20, 20), (220, 120), (255, 0, 0), 3)                     # đường thẳng
cv2.arrowedLine(canvas, (20, 160), (220, 160), (0, 150, 0), 3, tipLength=0.15)
cv2.rectangle(canvas, (260, 20), (420, 120), (0, 0, 255), 3)               # chữ nhật viền
cv2.rectangle(canvas, (450, 20), (600, 120), (0, 200, 255), -1)            # chữ nhật tô kín
cv2.circle(canvas, (90, 280), 60, (200, 0, 200), 4, cv2.LINE_AA)           # đường tròn
cv2.circle(canvas, (240, 280), 50, (255, 200, 0), -1)                      # hình tròn đặc
cv2.ellipse(canvas, (420, 280), (100, 50), 30, 0, 360, (0, 100, 255), 3)   # elip xoay 30°
cv2.ellipse(canvas, (600, 280), (70, 70), 0, 0, 270, (100, 100, 100), -1)  # cung 0..270° (hình Pac-man)

pts = np.array([[300, 170], [380, 150], [420, 200], [350, 230]], np.int32)
cv2.polylines(canvas, [pts], isClosed=True, color=(0, 0, 0), thickness=2)  # đa giác
tri = np.array([[480, 150], [560, 150], [520, 210]], np.int32)
cv2.fillPoly(canvas, [tri], (50, 180, 50))                                  # đa giác tô kín

for i, m in enumerate([cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_DIAMOND, cv2.MARKER_TRIANGLE_UP]):
    cv2.drawMarker(canvas, (620 + (i % 2) * 40, 160 + (i // 2) * 40), (0, 0, 0), m, 25, 2)

fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX]
for i, f in enumerate(fonts):
    cv2.putText(canvas, f'OpenCV font {i}', (20 + i * 230, 395), f, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
show(canvas, 'Các hàm vẽ cơ bản', size=9)
''')

md(r'''
### 5.1. Vẽ nhãn kiểu "object detection" và chữ tiếng Việt

- `cv2.getTextSize` cho biết kích thước chữ ⇒ vẽ được nền chữ vừa khít.
- ⚠️ `cv2.putText` **không hỗ trợ tiếng Việt có dấu** (chỉ ASCII). Muốn viết tiếng Việt, dùng **Pillow (PIL)** với font TrueType.
''')

code(r'''
from PIL import Image, ImageDraw, ImageFont


def draw_label(img, text, x1, y1, x2, y2, color=(0, 200, 0)):
    """Vẽ bounding box + nhãn có nền giống kết quả object detection."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def put_text_vn(img_bgr, text, org, font_size=28, color=(255, 255, 255)):
    """Viết chữ Unicode (tiếng Việt) bằng Pillow. color theo BGR."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = None
    for f in ['C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/Library/Fonts/Arial.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf']:
        if os.path.exists(f):
            font = ImageFont.truetype(f, font_size)
            break
    font = font or ImageFont.load_default()
    draw.text(org, text, font=font, fill=tuple(int(c) for c in color[::-1]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


annot = img.copy()
draw_label(annot, 'banana 0.97', 250, 40, 470, 200, (0, 180, 255))
draw_label(annot, 'apple 0.91', 20, 230, 200, 420, (0, 0, 220))
bad = annot.copy()
cv2.putText(bad, 'Trái cây tươi', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
good = put_text_vn(annot, 'Trái cây tươi ngon', (20, 10), 32, (255, 255, 255))
show([bad, good], ['❌ cv2.putText với tiếng Việt → lỗi ký tự', '✅ Dùng Pillow'], size=5)
''')

# =====================================================================
md(r'''
---
## 6. Phép toán số học & bitwise

### 6.1. ⚠️ Tràn số: NumPy vs OpenCV

Với `uint8`:
- **NumPy** dùng phép toán **modulo** (quấn vòng): `250 + 10 = 260 % 256 = 4` ⇒ pixel sáng bỗng thành đen!
- **OpenCV** dùng **bão hòa (saturate)**: `250 + 10 = 255`, `10 − 20 = 0`. Đây thường là điều ta muốn.
''')

code(r'''
a = np.uint8([[250]]); b = np.uint8([[10]])
print('NumPy  250 + 10 =', (a + b)[0, 0], '  | OpenCV cv2.add      =', cv2.add(a, b)[0, 0])
print('NumPy   10 - 250 =', (b - a)[0, 0], ' | OpenCV cv2.subtract =', cv2.subtract(b, a)[0, 0])

bright_np = img + 80                                    # sai: tràn số
bright_cv = cv2.add(img, np.full_like(img, 80))         # đúng: bão hòa
dark_cv   = cv2.subtract(img, np.full_like(img, 80))
show([img, bright_np, bright_cv, dark_cv],
     ['Gốc', '❌ NumPy img + 80 (tràn)', '✅ cv2.add(img, 80)', 'cv2.subtract(img, 80)'], cols=4, size=3.2)
''')

md(r'''
🔍 **Nhận xét:** Ảnh `img + 80` của NumPy có các mảng màu lạ ở vùng sáng (giá trị > 175 bị "quấn" về gần 0). Nếu muốn dùng NumPy an toàn: chuyển sang kiểu lớn hơn rồi cắt: `np.clip(img.astype(np.int16) + 80, 0, 255).astype(np.uint8)`.

### 6.2. Pha trộn ảnh — `cv2.addWeighted`

$$dst = \alpha \cdot img_1 + \beta \cdot img_2 + \gamma$$

Hai ảnh phải **cùng kích thước và cùng số kênh**.
''')

code(r'''
img2 = cv2.resize(load_sample('building.jpg'), (img.shape[1], img.shape[0]))
blends = [cv2.addWeighted(img, 1 - a, img2, a, 0) for a in [0, 0.3, 0.7, 1.0]]
show(blends, ['α=0 (chỉ fruits)', 'α=0.3', 'α=0.7', 'α=1 (chỉ building)'], cols=4, size=3.2)
''')

md(r'''
### 6.3. Hiệu tuyệt đối — `cv2.absdiff` (phát hiện thay đổi)

$|img_1 - img_2|$ — vùng nào giống nhau sẽ đen, vùng khác nhau sẽ sáng. Đây là nền tảng của phát hiện chuyển động và "tìm điểm khác nhau".
''')

code(r'''
changed = img.copy()
cv2.circle(changed, (120, 120), 35, (0, 255, 0), -1)
cv2.rectangle(changed, (380, 330), (450, 400), (255, 0, 0), -1)
diff = cv2.absdiff(img, changed)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, diff_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
show([changed, diff, diff_mask], ['Ảnh bị thay đổi', 'absdiff', 'Ngưỡng > 30 → vùng thay đổi'])
''')

md(r'''
### 6.4. Phép toán bitwise và mask

| Hàm | Ý nghĩa (với ảnh nhị phân 0/255) |
|---|---|
| `cv2.bitwise_and(a, b)` | giao (trắng khi **cả hai** trắng) |
| `cv2.bitwise_or(a, b)` | hợp (trắng khi **một trong hai** trắng) |
| `cv2.bitwise_xor(a, b)` | trắng khi **chỉ một** trắng |
| `cv2.bitwise_not(a)` | đảo |

Tham số `mask=`: phép toán chỉ thực hiện ở nơi mask ≠ 0, nơi khác = 0.
''')

code(r'''
r1 = np.zeros((200, 200), np.uint8); cv2.rectangle(r1, (30, 30), (140, 170), 255, -1)
c1 = np.zeros((200, 200), np.uint8); cv2.circle(c1, (130, 100), 60, 255, -1)
show([r1, c1, cv2.bitwise_and(r1, c1), cv2.bitwise_or(r1, c1), cv2.bitwise_xor(r1, c1), cv2.bitwise_not(r1)],
     ['A (chữ nhật)', 'B (tròn)', 'A AND B', 'A OR B', 'A XOR B', 'NOT A'], cols=6, size=2.2)

# Dùng mask để giữ lại vùng tròn của ảnh
mask = np.zeros(img.shape[:2], np.uint8)
cv2.circle(mask, (img.shape[1] // 2, img.shape[0] // 2), 170, 255, -1)
keep = cv2.bitwise_and(img, img, mask=mask)
blur_bg = cv2.GaussianBlur(img, (51, 51), 0)
portrait = np.where(mask[..., None] > 0, img, blur_bg)      # hiệu ứng "xóa phông" đơn giản
show([mask, keep, portrait], ['Mask tròn', 'bitwise_and với mask', 'Giữ nét trong mask, làm mờ ngoài'])
''')

md(r'''
### 6.5. Chèn logo / watermark có nền trong suốt (kinh điển)

Ý tưởng: tạo mask từ logo → "khoét lỗ" trên ảnh nền bằng `mask_inv` → lấy phần logo bằng `mask` → cộng hai phần.
''')

code(r'''
# Tạo logo: chữ trắng + vòng tròn màu trên nền đen
logo = np.zeros((90, 230, 3), np.uint8)
cv2.circle(logo, (45, 45), 38, (0, 140, 255), -1)
cv2.circle(logo, (45, 45), 18, (0, 0, 0), -1)
cv2.putText(logo, 'OpenCV', (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(logo_gray, 10, 255, cv2.THRESH_BINARY)   # nơi có logo = 255
mask_inv = cv2.bitwise_not(mask)

dst = img.copy()
lh, lw = logo.shape[:2]
y, x = dst.shape[0] - lh - 10, dst.shape[1] - lw - 10           # góc dưới-phải
roi = dst[y:y+lh, x:x+lw]
bg = cv2.bitwise_and(roi, roi, mask=mask_inv)    # nền: khoét chỗ logo
fg = cv2.bitwise_and(logo, logo, mask=mask)      # logo: bỏ nền đen
dst[y:y+lh, x:x+lw] = cv2.add(bg, fg)

# Watermark bán trong suốt
wm = img.copy()
roi2 = wm[y:y+lh, x:x+lw]
blend = cv2.addWeighted(roi2, 1, logo, 0.5, 0)
roi2[mask > 0] = blend[mask > 0]

show([logo, mask, dst, wm], ['Logo', 'Mask', 'Chèn logo', 'Watermark 50%'], cols=4, size=3.5)
''')

# =====================================================================
md(r'''
---
## 7. Biến đổi hình học (Geometric Transformations)

### 7.1. Thay đổi kích thước — `cv2.resize`

```python
cv2.resize(src, dsize=(W, H))                 # kích thước cụ thể (rộng, cao)
cv2.resize(src, None, fx=0.5, fy=0.5)         # theo tỉ lệ
```

**Phương pháp nội suy** (interpolation) — cách tính giá trị pixel mới:

| Phương pháp | Đặc điểm | Nên dùng |
|---|---|---|
| `INTER_NEAREST` | lấy pixel gần nhất, nhanh nhất, bị răng cưa | mask, ảnh nhãn (không được tạo giá trị mới) |
| `INTER_LINEAR` (mặc định) | song tuyến tính 2×2 | phóng to, đa dụng |
| `INTER_CUBIC` | bicubic 4×4, mịn hơn, chậm hơn | phóng to chất lượng cao |
| `INTER_LANCZOS4` | 8×8, sắc nét nhất | phóng to chất lượng rất cao |
| `INTER_AREA` | lấy trung bình vùng | **thu nhỏ** (tránh nhiễu moiré) |
''')

code(r'''
crop = img[180:230, 260:310]                 # vùng 50x50
methods = {'NEAREST': cv2.INTER_NEAREST, 'LINEAR': cv2.INTER_LINEAR,
           'CUBIC': cv2.INTER_CUBIC, 'LANCZOS4': cv2.INTER_LANCZOS4}
ups = [cv2.resize(crop, None, fx=6, fy=6, interpolation=m) for m in methods.values()]
show(ups, [f'Phóng to x6 - {k}' for k in methods], cols=4, size=3.2)


def resize_keep_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize giữ nguyên tỉ lệ khung hình: chỉ cần cho width HOẶC height."""
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is not None:
        r = width / w; dim = (width, int(h * r))
    else:
        r = height / h; dim = (int(w * r), height)
    return cv2.resize(image, dim, interpolation=inter)


print('Gốc:', img.shape, '-> resize_keep_ratio(width=200):', resize_keep_ratio(img, width=200).shape)
''')

md(r'''
🔍 **Nhận xét:** `NEAREST` cho các ô vuông rõ rệt; `LINEAR` mịn nhưng hơi mờ; `CUBIC`/`LANCZOS4` sắc nét hơn. **Không phương pháp nào tạo ra chi tiết thật** — muốn vậy cần siêu phân giải (super-resolution) bằng deep learning.

### 7.2. Lật và xoay 90°
''')

code(r'''
show([img, cv2.flip(img, 1), cv2.flip(img, 0), cv2.flip(img, -1),
      cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(img, cv2.ROTATE_180)],
     ['Gốc', 'flip(1): lật ngang (gương)', 'flip(0): lật dọc', 'flip(-1): cả hai',
      'ROTATE_90_CLOCKWISE', 'ROTATE_180'], cols=3, size=3.5)
''')

md(r'''
### 7.3. Biến đổi Affine — `cv2.warpAffine(src, M, (W, H))`

Biến đổi affine dùng **ma trận 2×3** $M$, ánh xạ điểm $(x, y)$ thành:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Affine **giữ nguyên tính song song** của các đường thẳng. Bao gồm: tịnh tiến, xoay, co giãn, trượt (shear).

- **Tịnh tiến**: $M = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \end{bmatrix}$
- **Xoay** quanh tâm $(c_x, c_y)$ một góc θ (độ, **dương = ngược chiều kim đồng hồ**), tỉ lệ s: `cv2.getRotationMatrix2D(center, angle, scale)`
- **Từ 3 cặp điểm**: `cv2.getAffineTransform(src_pts, dst_pts)`
''')

code(r'''
h, w = img.shape[:2]

M_t = np.float32([[1, 0, 80], [0, 1, 40]])                  # dịch phải 80, xuống 40
shifted = cv2.warpAffine(img, M_t, (w, h))

M_r = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)      # xoay 30° quanh tâm
rotated = cv2.warpAffine(img, M_r, (w, h))
print('Ma trận xoay 30°:\n', M_r.round(3))

M_rs = cv2.getRotationMatrix2D((w / 2, h / 2), -45, 0.6)    # xoay -45° (cùng chiều kim đồng hồ) và thu nhỏ
rot_scaled = cv2.warpAffine(img, M_rs, (w, h), borderMode=cv2.BORDER_REFLECT)

M_sh = np.float32([[1, 0.3, 0], [0, 1, 0]])                 # trượt ngang (shear)
sheared = cv2.warpAffine(img, M_sh, (int(w + 0.3 * h), h))

show([shifted, rotated, rot_scaled, sheared],
     ['Tịnh tiến (80, 40)', 'Xoay 30° (bị cắt góc)', 'Xoay -45°, scale 0.6, viền REFLECT', 'Shear 0.3'],
     cols=4, size=3.5)
''')

md(r'''
Khi xoay, các góc ảnh bị **cắt mất**. Hàm dưới tính lại kích thước khung để chứa trọn ảnh sau xoay:
''')

code(r'''
def rotate_bound(image, angle):
    """Xoay ảnh mà không bị cắt góc (khung tự mở rộng)."""
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - cx          # dời tâm sang tâm khung mới
    M[1, 2] += new_h / 2 - cy
    return cv2.warpAffine(image, M, (new_w, new_h))


# Affine từ 3 cặp điểm
src_pts = np.float32([[50, 50], [400, 50], [50, 400]])
dst_pts = np.float32([[10, 100], [380, 30], [120, 450]])
M_aff = cv2.getAffineTransform(src_pts, dst_pts)
aff = cv2.warpAffine(img, M_aff, (w, h))
vis = img.copy()
for (px, py), col in zip(src_pts.astype(int), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
    cv2.circle(vis, (px, py), 10, col, -1)
for (px, py), col in zip(dst_pts.astype(int), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
    cv2.circle(aff, (px, py), 10, col, -1)
show([rotate_bound(img, 30), vis, aff], ['rotate_bound 30° (không cắt)', '3 điểm nguồn', 'Sau getAffineTransform'])
''')

md(r'''
### 7.4. Biến đổi phối cảnh (Perspective) — `cv2.warpPerspective`

Dùng **ma trận 3×3** (homography), cần **4 cặp điểm**. Không giữ tính song song ⇒ dùng để "nhìn thẳng" một mặt phẳng bị chụp nghiêng: tài liệu, bảng số xe, bàn cờ, sudoku…

$$\begin{bmatrix} x' w \\ y' w \\ w \end{bmatrix} = H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
''')

code(r'''
sudoku = load_sample('sudoku.png')
# 4 góc của khung sudoku (trên-trái, trên-phải, dưới-trái, dưới-phải) - xác định thủ công.
# Ở mục 24 ta sẽ tìm 4 góc này TỰ ĐỘNG bằng contour.
pts1 = np.float32([[73, 85], [490, 70], [35, 515], [520, 520]])
pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
H_mat = cv2.getPerspectiveTransform(pts1, pts2)
warped = cv2.warpPerspective(sudoku, H_mat, (450, 450))
print('Ma trận homography 3x3:\n', H_mat.round(4))

vis = sudoku.copy()
for p, col in zip(pts1.astype(int), [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 200, 255)]):
    cv2.circle(vis, tuple(int(v) for v in p), 12, col, -1)
show([vis, warped], ['Ảnh nghiêng + 4 điểm góc', 'Sau warpPerspective (nhìn thẳng)'], size=5)
''')

md(r'''
### 7.5. Biến dạng tự do — `cv2.remap`

`remap` cho phép **mỗi pixel đích lấy giá trị từ vị trí bất kỳ** của ảnh nguồn: `dst(y, x) = src(map_y(y,x), map_x(y,x))`. Dùng để sửa méo ống kính, hiệu ứng sóng nước, phóng to "mắt cá"…
''')

code(r'''
h, w = img.shape[:2]
map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

wave = cv2.remap(img, map_x + 12 * np.sin(map_y / 18), map_y,
                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# Hiệu ứng "kính lúp/mắt cá" quanh tâm
cx, cy = w / 2, h / 2
dx, dy = map_x - cx, map_y - cy
r = np.sqrt(dx**2 + dy**2)
R = min(w, h) / 2
factor = np.where(r < R, (r / R) ** 0.6, 1.0).astype(np.float32)   # co bán kính vào tâm -> phóng to tâm
fish = cv2.remap(img, cx + dx * factor, cy + dy * factor, cv2.INTER_LINEAR)

mirror = cv2.remap(img, (w - 1 - map_x), map_y, cv2.INTER_LINEAR)  # = flip ngang, viết bằng remap
show([wave, fish, mirror], ['Sóng nước (sin)', 'Phóng to tâm (fisheye)', 'Lật ngang bằng remap'])
''')

md(r'''
✍️ **Bài tập 7:** Viết hàm `zoom_center(img, factor)` phóng to vùng trung tâm ảnh (crop + resize) và so sánh với dùng `warpAffine` với scale.
''')

# =====================================================================
md(r'''
---
## 8. Độ sáng, tương phản và Gamma

### 8.1. Biến đổi tuyến tính

$$g(x, y) = \alpha \cdot f(x, y) + \beta$$

- $\alpha > 1$: tăng **tương phản**; $0 < \alpha < 1$: giảm tương phản.
- $\beta > 0$: tăng **độ sáng**; $\beta < 0$: giảm.

`cv2.convertScaleAbs(img, alpha=α, beta=β)` thực hiện công thức trên, lấy trị tuyệt đối và bão hòa về `uint8`.
''')

code(r'''
params = [(1.0, 0), (1.0, 60), (1.0, -60), (1.6, 0), (0.5, 0), (1.5, -60)]
outs = [cv2.convertScaleAbs(img, alpha=a, beta=b) for a, b in params]
show(outs, [f'α={a}, β={b}' for a, b in params], cols=3, size=3.6)

x = np.arange(256)
plt.figure(figsize=(5, 4))
for a, b in params:
    plt.plot(x, np.clip(a * x + b, 0, 255), label=f'α={a}, β={b}')
plt.xlabel('Giá trị vào'); plt.ylabel('Giá trị ra'); plt.title('Đường cong biến đổi'); plt.legend(); plt.grid(alpha=.3)
plt.show()
''')

md(r'''
### 8.2. Hiệu chỉnh Gamma — `cv2.LUT`

$$O = 255 \cdot \left(\frac{I}{255}\right)^{\gamma}$$

- $\gamma < 1$: **làm sáng vùng tối** mạnh hơn vùng sáng (rất hợp với ảnh thiếu sáng).
- $\gamma > 1$: làm tối, tăng chi tiết vùng sáng.

Khác với cộng β (nâng đều mọi pixel, dễ "cháy" vùng sáng), gamma là phép biến đổi **phi tuyến**.

**LUT (Look-Up Table)**: vì `uint8` chỉ có 256 giá trị, ta tính trước bảng 256 phần tử rồi tra bảng ⇒ cực nhanh với mọi phép biến đổi điểm.
''')

code(r'''
def adjust_gamma(image, gamma):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


dark = cv2.convertScaleAbs(img, alpha=0.35, beta=0)       # giả lập ảnh thiếu sáng
gammas = [1.0, 0.6, 0.4, 0.25]
show([adjust_gamma(dark, g) for g in gammas] + [cv2.add(dark, np.full_like(dark, 110))],
     [f'γ = {g}' for g in gammas] + ['So sánh: cộng β=110'], cols=5, size=2.8)

plt.figure(figsize=(5, 4))
for g in [0.25, 0.4, 0.6, 1.0, 1.5, 2.5]:
    plt.plot(x, 255 * (x / 255) ** g, label=f'γ={g}')
plt.legend(); plt.grid(alpha=.3); plt.title('Đường cong gamma'); plt.xlabel('Vào'); plt.ylabel('Ra')
plt.show()
''')

md(r'''
🔍 **Nhận xét:** Với γ = 0.4, chi tiết trong vùng tối hiện rõ mà vùng sáng không bị cháy. Cộng β=110 làm ảnh "bạc màu" (mọi pixel bị nâng đều, tương phản giảm).

### 8.3. Kéo giãn tương phản tự động — `cv2.normalize`

Ảnh có tương phản thấp chỉ dùng một dải hẹp (vd 90–160). Kéo giãn tuyến tính dải đó ra 0–255:

$$O = \frac{I - I_{min}}{I_{max} - I_{min}} \times 255$$
''')

code(r'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
low = cv2.convertScaleAbs(gray, alpha=0.3, beta=90)           # giả lập ảnh tương phản thấp
stretched = cv2.normalize(low, None, 0, 255, cv2.NORM_MINMAX)
print('Ảnh tương phản thấp: min =', low.min(), ', max =', low.max())
print('Sau normalize      : min =', stretched.min(), ', max =', stretched.max())

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes[0, 0].imshow(low, cmap='gray', vmin=0, vmax=255); axes[0, 0].set_title('Tương phản thấp'); axes[0, 0].axis('off')
axes[0, 1].imshow(stretched, cmap='gray', vmin=0, vmax=255); axes[0, 1].set_title('Sau NORM_MINMAX'); axes[0, 1].axis('off')
plot_hist(low, axes[1, 0], 'Histogram: dồn vào dải hẹp')
plot_hist(stretched, axes[1, 1], 'Histogram: trải ra 0..255')
plt.tight_layout(); plt.show()
''')
