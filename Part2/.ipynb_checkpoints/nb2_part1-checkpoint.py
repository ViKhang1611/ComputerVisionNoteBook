from nbformat.v4 import new_markdown_cell, new_code_cell

CELLS = []


def md(s):
    CELLS.append(new_markdown_cell(s.strip("\n")))


def code(s):
    CELLS.append(new_code_cell(s.strip("\n")))


# =====================================================================
md(r'''
# 📗 OpenCV Phần 2 — Lộ Trình Nâng Cao: Video, Hình Học Đa Ảnh, Nhiếp Ảnh Tính Toán & AI/DNN

> **Tiếp nối:** Sau khi đã làm chủ các thao tác trên **một bức ảnh** ở [Huong_Dan_Xu_Ly_Anh_OpenCV_Chi_Tiet.ipynb](Huong_Dan_Xu_Ly_Anh_OpenCV_Chi_Tiet.ipynb), notebook này đưa bạn đi trọn vẹn **lộ trình tiếp theo** của thị giác máy tính (Computer Vision) với OpenCV:
>
> 1. 🎬 **Phần I — Video, Phân tích chuyển động & Theo dõi đối tượng (Tracking)**
> 2. 📐 **Phần II — Hình học đa ảnh (Multi-view Geometry), 3D & Nhiếp ảnh tính toán**
> 3. 🤖 **Phần III — Học máy cổ điển (`cv2.ml`), Nhận dạng & Học sâu (`cv2.dnn`)**
> 4. 🚀 **Phần IV — Dự án tổng hợp (Hệ thống giám sát & đếm phương tiện giao thông) + Bài tập có lời giải**

Mỗi mục đều tuân thủ cấu trúc chuẩn:
1. 🧠 **Lý thuyết & công thức toán học** — hiểu bản chất thay vì học vẹt hàm.
2. 💻 **Code ví dụ hoàn chỉnh** — tự sinh dữ liệu video/ảnh mô phỏng chân thực để **chạy 100% không cần webcam hay tải model hàng trăm MB**.
3. 🖼️ **Kết quả trực quan** — chuỗi khung hình (frame strip), bản đồ dòng quang học, quỹ đạo Kalman, ảnh ghép Panorama, bản đồ chiều sâu Stereo… được nhúng sẵn ngay trong notebook.
4. 🔍 **Nhận xét & lưu ý thực tế** — phân tích kết quả và các bẫy hay gặp.
5. ✍️ **Bài tập tự luyện** — kèm lời giải chạy được ở cuối notebook.

---
### 🗺️ Mục lục chi tiết

| # | Chủ đề | Hàm / Lớp OpenCV trọng tâm |
|---|---|---|
| **0** | Chuẩn bị môi trường & bộ tạo video mô phỏng | `show`, `show_frames`, `make_traffic_video` |
| **1** | Đọc, ghi & xử lý luồng Video | `VideoCapture`, `VideoWriter`, `VideoWriter_fourcc` |
| **2** | Phát hiện chuyển động: Frame Differencing & Running Average | `absdiff`, `accumulateWeighted`, `convertScaleAbs` |
| **3** | Trừ nền nâng cao (Background Subtraction) | `createBackgroundSubtractorMOG2`, `createBackgroundSubtractorKNN` |
| **4** | Dòng quang học (Optical Flow): Lucas–Kanade & Farnebäck | `calcOpticalFlowPyrLK`, `calcOpticalFlowFarneback`, `cartToPolar` |
| **5** | Theo dõi đối tượng theo màu: Backprojection, MeanShift & CamShift | `calcBackProject`, `meanShift`, `CamShift` |
| **6** | Bộ lọc Kalman (Kalman Filter) — dự đoán & khử nhiễu quỹ đạo | `KalmanFilter`, `predict`, `correct` |
| **7** | Theo dõi đa đối tượng (Multi-Object Tracking) | Ghép cặp Centroid/IoU + quản lý vòng đời ID |
| **8** | Khớp đặc trưng nâng cao: Lowe's Ratio Test + Homography & RANSAC | `BFMatcher.knnMatch`, `findHomography`, `perspectiveTransform` |
| **9** | Ghép ảnh toàn cảnh (Panorama / Image Stitching) | Homography + Feather Blending, `Stitcher_create` |
| **10** | Hiệu chuẩn Camera & Khử méo ống kính (Camera Calibration) | `findChessboardCorners`, `calibrateCamera`, `undistort` |
| **11** | Ước lượng tư thế 3D (Pose Estimation) & Thực tế tăng cường (AR) | `solvePnP`, `projectPoints` |
| **12** | Thị giác lập thể (Stereo Vision) & Bản đồ chiều sâu (Depth Map) | `StereoBM_create`, `StereoSGBM_create`, $Z = fB/d$ |
| **13** | Nhiếp ảnh tính toán: HDR, Seamless Cloning & Bộ lọc nghệ thuật | `createMergeMertens`, `seamlessClone`, `pencilSketch`, `stylization` |
| **14** | Phát hiện đối tượng cổ điển: Haar Cascade & HOG + SVM | `CascadeClassifier`, `HOGDescriptor_getDefaultPeopleDetector` |
| **15** | Huấn luyện bộ phân loại chữ số với HOG + `cv2.ml` (SVM & k-NN) | `HOGDescriptor`, `cv2.ml.SVM_create`, `cv2.ml.KNearest_create` |
| **16** | Phát hiện & giải mã QR Code | `QRCodeDetector`, `detectAndDecode`, `detectAndDecodeMulti` |
| **17** | Học sâu với OpenCV (`cv2.dnn`): Blob, NMS & Suy luận mạng nơ-ron | `dnn.blobFromImage`, `dnn.NMSBoxes`, `dnn.readNetFromONNX` |
| **18** | 🚀 **Dự án tổng hợp:** Hệ thống Giám sát, Theo dõi & Đếm Xe Giao Thông | Pipeline đầu-cuối: MOG2 + Morphology + Tracker + Speed + Line Counter |
| **19** | Bảng tra cứu nhanh, Lỗi thường gặp, Bài tập & Lời giải chi tiết | |
''')

# =====================================================================
md(r'''
---
## 0. Chuẩn bị môi trường & dữ liệu mô phỏng

Các thư viện sử dụng vẫn là bộ ba quen thuộc: `opencv-python`, `numpy`, `matplotlib`.

Để mọi ví dụ về **video và chuyển động** có thể chạy ngay trong notebook (kể cả trên máy không có webcam) và cho kết quả **lặp lại chính xác 100%**, ô dưới đây tạo sẵn:
- Các hàm hiển thị ảnh `show(...)` và chuỗi khung hình `show_frames(...)`.
- Một **video giao thông mô phỏng** (`outputs_opencv_p2/traffic_sim.avi`, 90 khung hình, 640×360) gồm mặt đường, vạch kẻ, cây cối, 4 phương tiện di chuyển với vận tốc khác nhau, bóng đổ đi kèm, và một biển quảng cáo nhấp nháy.
''')

code(r'''
%matplotlib inline
import os
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 85
plt.rcParams['axes.titlesize'] = 9.5

IMG_DIR = Path('images')
IMG_DIR.mkdir(exist_ok=True)
OUT_DIR = Path('outputs_opencv_p2')
OUT_DIR.mkdir(exist_ok=True)

print('OpenCV :', cv2.__version__)
print('NumPy  :', np.__version__)


def show(images, titles=None, cols=3, size=4.0, cmap='gray'):
    """Hiển thị 1 hoặc nhiều ảnh OpenCV (tự động chuyển BGR -> RGB)."""
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
        im = images[i]
        if im.dtype == bool:
            im = im.astype(np.uint8) * 255
        if im.ndim == 2:
            if im.dtype == np.uint8:
                ax.imshow(im, cmap=cmap, vmin=0, vmax=255)
            else:
                ax.imshow(im, cmap=cmap)
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()


def show_frames(frames, indices=None, cols=4, size=3.4, prefix='Frame'):
    """Hiển thị một số khung hình đại diện từ danh sách frames."""
    if indices is None:
        indices = np.linspace(0, len(frames) - 1, min(cols, len(frames)), dtype=int)
    imgs = [frames[i] for i in indices]
    titles = [f'{prefix} #{i}' for i in indices]
    show(imgs, titles, cols=cols, size=size)


def load_sample(name):
    """Đọc ảnh mẫu từ thư mục images/ (đã có từ Phần 1) hoặc tạo ảnh thay thế."""
    p = IMG_DIR / name
    if p.exists():
        im = cv2.imread(str(p))
        if im is not None:
            return im
    # Fallback tổng hợp nếu chạy độc lập
    rng = np.random.default_rng(42)
    im = np.zeros((480, 640, 3), np.uint8)
    im[:] = np.linspace(40, 200, 640, dtype=np.uint8)[None, :, None]
    for _ in range(20):
        c = tuple(int(v) for v in rng.integers(30, 240, 3))
        cv2.circle(im, (int(rng.integers(40, 600)), int(rng.integers(40, 440))), int(rng.integers(18, 45)), c, -1)
    return im
''')

code(r'''
def generate_traffic_frames(num_frames=90, width=640, height=360, seed=7):
    """Tạo chuỗi khung hình giao thông mô phỏng có bóng đổ và nhiễu cảm biến nhẹ."""
    rng = np.random.default_rng(seed)
    # 1. Tạo cảnh nền tĩnh (mặt đường + lề đường + vạch phân làn + hàng cây)
    bg = np.full((height, width, 3), (75, 130, 85), np.uint8)              # thảm cỏ hai bên
    cv2.rectangle(bg, (0, 80), (width, 290), (72, 72, 75), -1)             # mặt đường nhựa
    cv2.line(bg, (0, 82), (width, 82), (210, 210, 210), 3)                 # vạch lề trên
    cv2.line(bg, (0, 288), (width, 288), (210, 210, 210), 3)               # vạch lề dưới
    for x in range(0, width, 50):
        cv2.rectangle(bg, (x, 182), (x + 28, 188), (200, 220, 240), -1)    # vạch đứt phân làn giữa
    for tx in range(50, width, 120):
        cv2.circle(bg, (tx, 40), 22, (35, 95, 40), -1)                     # hàng cây phía trên
        cv2.circle(bg, (tx + 30, 330), 20, (40, 105, 45), -1)              # hàng cây phía dưới
    # Thêm texture hạt nhựa đường nhẹ
    road_tex = rng.normal(0, 3.5, bg.shape).astype(np.float32)
    bg = np.clip(bg.astype(np.float32) + road_tex, 0, 255).astype(np.uint8)

    # Danh sách phương tiện: (x0, y, w, h, vx, color_bgr, name)
    vehicles = [
        (-70.0,  105, 68, 34,  6.2, (40,  60, 220), 'Car-Red'),     # làn trên: trái -> phải (nhanh)
        (120.0,  140, 82, 36,  4.4, (220, 140, 30), 'Van-Blue'),    # làn trên: trái -> phải (vừa)
        (620.0,  205, 66, 34, -5.5, (40, 200, 230), 'Car-Yellow'),  # làn dưới: phải -> trái
        (360.0,  242, 92, 38, -3.8, (60, 190,  70), 'Bus-Green'),   # làn dưới: phải -> trái
    ]

    frames = []
    for t in range(num_frames):
        frame = bg.copy()
        # Vẽ bóng đổ trước (làm tối mặt đường phía dưới-phải của xe)
        shadow_mask = np.zeros((height, width), np.uint8)
        for x0, y, w, h, vx, col, _ in vehicles:
            cx = int((x0 + vx * t) % (width + 160)) - 80
            sx1, sy1 = max(0, cx + 10), max(0, y + h - 4)
            sx2, sy2 = min(width, cx + w + 14), min(height, y + h + 12)
            if sx2 > sx1 and sy2 > sy1:
                shadow_mask[sy1:sy2, sx1:sx2] = 255
        frame[shadow_mask > 0] = (frame[shadow_mask > 0].astype(np.float32) * 0.62).astype(np.uint8)

        # Vẽ thân xe, kính chắn gió và đèn xe
        for x0, y, w, h, vx, col, _ in vehicles:
            cx = int((x0 + vx * t) % (width + 160)) - 80
            cv2.rectangle(frame, (cx, y), (cx + w, y + h), col, -1)
            cv2.rectangle(frame, (cx, y), (cx + w, y + h), (25, 25, 25), 2)
            # Kính xe
            wx = cx + w - 22 if vx > 0 else cx + 6
            cv2.rectangle(frame, (wx, y + 5), (wx + 16, y + h - 5), (210, 230, 245), -1)
            # Bánh xe / điểm góc đặc trưng (hữu ích cho Optical Flow)
            cv2.circle(frame, (cx + 14, y + h), 5, (20, 20, 20), -1)
            cv2.circle(frame, (cx + w - 14, y + h), 5, (20, 20, 20), -1)

        # Nhiễu cảm biến camera rất nhẹ (σ = 2)
        noise = rng.normal(0, 2.0, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return bg, frames


BG_TRUE, TRAFFIC_FRAMES = generate_traffic_frames(num_frames=90)
print(f'Đã tạo {len(TRAFFIC_FRAMES)} khung hình, kích thước mỗi khung: {TRAFFIC_FRAMES[0].shape}')
show_frames(TRAFFIC_FRAMES, indices=[0, 20, 45, 70], cols=4)
''')

# =====================================================================
md(r'''
---
# PHẦN I — XỬ LÝ VIDEO, CHUYỂN ĐỘNG & THEO DÕI ĐỐI TƯỢNG

## 1. Đọc, ghi và xử lý luồng Video (`cv2.VideoCapture` & `cv2.VideoWriter`)

### 1.1. Bản chất của Video trong OpenCV
Một video đơn giản là **chuỗi các bức ảnh tĩnh (frames)** hiển thị liên tiếp với tốc độ **FPS** (Frames Per Second — thường 24, 30 hoặc 60 khung hình/giây). Mọi thuật toán xử lý 1 ảnh ở Phần 1 đều áp dụng được cho từng frame trong vòng lặp `while`.

| Thao tác | API chính | Ghi chú |
|---|---|---|
| Mở file video / luồng RTSP | `cap = cv2.VideoCapture('video.mp4')` | truyền đường dẫn hoặc URL |
| Mở webcam | `cap = cv2.VideoCapture(0)` | `0` là camera mặc định, `1` là camera thứ 2 |
| Kiểm tra mở thành công | `cap.isOpened()` | luôn kiểm tra trước khi đọc! |
| Đọc 1 frame | `ret, frame = cap.read()` | `ret=False` khi hết video hoặc lỗi đọc |
| Đọc thuộc tính video | `cap.get(cv2.CAP_PROP_...)` | chiều rộng, chiều cao, FPS, tổng số frame |
| Tua đến frame bất kỳ | `cap.set(cv2.CAP_PROP_POS_FRAMES, idx)` | truy cập ngẫu nhiên |
| Ghi video ra file | `writer = cv2.VideoWriter(path, fourcc, fps, (W, H))` | chú ý `(W, H)` phải khớp `frame.shape[1::-1]` |
| Giải phóng tài nguyên | `cap.release()`, `writer.release()` | bắt buộc gọi sau khi xong |

> ⚠️ **Lỗi kinh điển với `VideoWriter`:**
> 1. Truyền `(H, W)` thay vì `(W, H)` ⇒ file video tạo ra bị rỗng (0 KB) mà **không báo lỗi**!
> 2. Ghi ảnh xám 1 kênh `(H, W)` vào `VideoWriter` đang để mặc định `isColor=True` ⇒ hỏng file. Hãy chuyển sang BGR bằng `cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)` hoặc đặt `isColor=False`.
''')

code(r'''
# 1. Ghi danh sách TRAFFIC_FRAMES ra file video .avi (codec MJPG rất tương thích trên mọi hệ điều hành)
video_path = OUT_DIR / 'traffic_sim.avi'
h, w = TRAFFIC_FRAMES[0].shape[:2]
fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h), isColor=True)
assert writer.isOpened(), 'Không mở được VideoWriter!'
for f in TRAFFIC_FRAMES:
    writer.write(f)
writer.release()
print(f'Đã ghi video: {video_path} ({video_path.stat().st_size / 1024:.1f} KB)')

# 2. Đọc lại file video bằng cv2.VideoCapture và kiểm tra các thuộc tính (metadata)
cap = cv2.VideoCapture(str(video_path))
assert cap.isOpened(), 'Không mở được VideoCapture!'

prop_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
prop_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
prop_fps   = cap.get(cv2.CAP_PROP_FPS)
prop_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration   = prop_count / prop_fps
print(f'Thông số đọc từ VideoCapture: {prop_w}x{prop_h} @ {prop_fps:.1f} FPS | '
      f'Tổng {prop_count} frames ({duration:.2f} giây)')

# 3. Tua trực tiếp (seek) tới frame thứ 50 và đọc
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame_50 = cap.read()
cap.release()
print('Đọc thành công frame #50:', ret, '| shape =', frame_50.shape)
show(frame_50, 'Frame #50 đọc từ file traffic_sim.avi bằng VideoCapture', size=5.5)
''')

md(r'''
### 1.2. Mẫu code chuẩn khi làm việc với Webcam / Video thực tế (chạy trong script `.py`)

```python
import cv2, time

cap = cv2.VideoCapture(0)  # 0 = webcam, hoặc đường dẫn 'video.mp4'
if not cap.isOpened():
    raise RuntimeError("Không mở được nguồn video!")

prev_time = time.perf_counter()
while True:
    ret, frame = cap.read()
    if not ret:
        break  # hết video hoặc mất kết nối camera

    # --- Xử lý frame tại đây ---
    now = time.perf_counter()
    fps_live = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now
    cv2.putText(frame, f"FPS: {fps_live:.1f}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Live Video (Nhan q de thoat)", frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
''')

# =====================================================================
md(r'''
---
## 2. Phát hiện chuyển động cơ bản: Frame Differencing & Running Average

Làm thế nào để máy tính biết **vùng nào đang chuyển động** khi camera đứng yên?

### 2.1. Hiệu 2 khung hình liên tiếp (2-Frame Differencing)
Lấy trị tuyệt đối hiệu giữa khung hình hiện tại $I_t$ và khung hình ngay trước đó $I_{t-1}$:

$$D_t(x, y) = |I_t(x, y) - I_{t-1}(x, y)|$$

- **Nhược điểm của hiệu 2 frame:**
  1. Tạo **"bóng ma kép" (double ghost)**: cả vị trí *vừa rời đi* ($t-1$) lẫn vị trí *mới tới* ($t$) đều sáng lên!
  2. Bên trong vật thể đồng màu thì $I_t \approx I_{t-1}$ nên bị **rỗng ruột** (aperture problem).

### 2.2. Hiệu 3 khung hình (3-Frame Differencing) — khử bóng ma
Kết hợp phép **AND** giữa hai hiệu liên tiếp $|I_t - I_{t-1}|$ và $|I_{t+1} - I_t|$:

$$M_t = \big(|I_t - I_{t-1}| > T\big) \;\land\; \big(|I_{t+1} - I_t| > T\big)$$

Vùng nào thay đổi ở **cả hai** bước chuyển tiếp chính là vị trí của vật thể tại thời điểm $t$!

### 2.3. Trung bình trượt mô hình nền (Running Average — `cv2.accumulateWeighted`)
Thay vì chỉ so với frame trước, ta duy trì một **mô hình nền tích lũy** $B_t$ cập nhật chậm theo thời gian:

$$B_t = (1 - \alpha)\, B_{t-1} + \alpha\, I_t \qquad (\alpha \in [0.01, 0.1])$$

- Vật thể chạy nhanh chỉ đi qua trong vài frame nên không kịp làm thay đổi $B_t$.
- Khi ánh sáng môi trường thay đổi từ từ (mặt trời lên/lặn), $B_t$ tự động thích nghi theo!
''')

code(r'''
# So sánh: (1) 2-frame diff, (2) 3-frame diff, (3) Running average background
t_idx = 35
g_prev = cv2.cvtColor(TRAFFIC_FRAMES[t_idx - 3], cv2.COLOR_BGR2GRAY)
g_curr = cv2.cvtColor(TRAFFIC_FRAMES[t_idx],     cv2.COLOR_BGR2GRAY)
g_next = cv2.cvtColor(TRAFFIC_FRAMES[t_idx + 3], cv2.COLOR_BGR2GRAY)

# 1. Hiệu 2 frame
diff2 = cv2.absdiff(g_curr, g_prev)
_, mask_2f = cv2.threshold(diff2, 25, 255, cv2.THRESH_BINARY)

# 2. Hiệu 3 frame
diff_a = cv2.absdiff(g_curr, g_prev)
diff_b = cv2.absdiff(g_next, g_curr)
_, ma = cv2.threshold(diff_a, 25, 255, cv2.THRESH_BINARY)
_, mb = cv2.threshold(diff_b, 25, 255, cv2.THRESH_BINARY)
mask_3f = cv2.bitwise_and(ma, mb)

# 3. Running average với cv2.accumulateWeighted trên toàn bộ 90 frame
avg_bg = TRAFFIC_FRAMES[0].astype(np.float32)
for f in TRAFFIC_FRAMES:
    cv2.accumulateWeighted(f, avg_bg, alpha=0.05)
recovered_bg = cv2.convertScaleAbs(avg_bg)

# Dùng nền ước lượng được để tách vật thể tại frame t_idx
diff_bg = cv2.absdiff(TRAFFIC_FRAMES[t_idx], recovered_bg)
_, mask_bg = cv2.threshold(cv2.cvtColor(diff_bg, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)

print(f'PSNR giữa nền ước lượng (accumulateWeighted) và nền thật không có xe: {cv2.PSNR(BG_TRUE, recovered_bg):.2f} dB')
show([TRAFFIC_FRAMES[t_idx], recovered_bg, mask_2f, mask_3f, mask_bg],
     [f'Frame #{t_idx} (có 4 xe)', 'Nền khôi phục bằng accumulateWeighted',
      '2-Frame Diff (bị bóng đôi)', '3-Frame Diff (khử bóng đôi)', 'Trừ nền ước lượng (đặc & đầy đủ nhất)'],
     cols=3, size=4.2)
''')

md(r'''
🔍 **Nhận xét:**
- Dù **mọi khung hình đều có xe chạy**, phép `cv2.accumulateWeighted` vẫn tái tạo lại mặt đường **sạch bóng xe** (PSNR > 38 dB so với nền gốc)! Đây là kỹ thuật kinh điển để "xóa người đi bộ / xe cộ khỏi ảnh chụp danh lam thắng cảnh" từ một đoạn video ngắn.
- Khi lấy `absdiff` với nền khôi phục, toàn bộ thân xe hiện lên đặc kín chứ không bị rỗng ruột như 2-Frame / 3-Frame Differencing.
''')

# =====================================================================
md(r'''
---
## 3. Trừ nền nâng cao: MOG2 và KNN (`BackgroundSubtractor`)

Trong thực tế, một pixel nền không chỉ có 1 màu cố định: lá cây rung rinh trước gió, mặt nước gợn sóng, bóng râm lay động… Nếu chỉ dùng 1 giá trị trung bình như mục 2.3 sẽ báo động giả liên tục.

OpenCV cung cấp hai thuật toán trừ nền thống kê mạnh mẽ:

| Thuật toán | Hàm khởi tạo | Nguyên lý |
|---|---|---|
| **MOG2** (Mixture of Gaussians) | `cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)` | Mô hình hóa lịch sử mỗi pixel bằng **hỗn hợp 3–5 phân phối Gauss** (Zivkovic, 2004). Tự chọn số cụm Gauss cho từng pixel. |
| **KNN** (K-Nearest Neighbors) | `cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)` | Lưu các mẫu lịch sử và phân loại pixel là nền nếu có đủ $K$ láng giềng gần trong bán kính ngưỡng. |

> 💡 **Cơ chế phát hiện bóng đổ (`detectShadows=True`):**
> Khi một vùng tối đi nhưng **sắc độ màu (chromaticity) ít đổi**, thuật toán nhận diện đó là **bóng đổ** và gán giá trị **`127` (xám)** trên mask đầu ra, trong khi **tiền cảnh thật (vật thể)** có giá trị **`255` (trắng)** và nền là **`0` (đen)**.
>
> ➜ Để **loại bỏ bóng đổ** và chỉ giữ vật thể thật, chỉ cần ngưỡng hóa: `_, fg_only = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)`.
''')

code(r'''
mog2 = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=20, detectShadows=True)
knn  = cv2.createBackgroundSubtractorKNN(history=60, dist2Threshold=400, detectShadows=True)

# Cho mô hình học qua 55 frame đầu
for i in range(56):
    f = TRAFFIC_FRAMES[i]
    mask_mog2 = mog2.apply(f)
    mask_knn  = knn.apply(f)

# Tách riêng tiền cảnh (255) và bóng đổ (127) từ mask_mog2
_, fg_clean = cv2.threshold(mask_mog2, 200, 255, cv2.THRESH_BINARY)
k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_OPEN, k_open)
fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_CLOSE, k_close)

# Vẽ bounding box quanh các phương tiện tìm được
vis_det = TRAFFIC_FRAMES[55].copy()
cnts, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
det_count = 0
for c in cnts:
    if cv2.contourArea(c) < 350:
        continue
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(vis_det, (x, y), (x + w, y + h), (0, 255, 0), 2)
    det_count += 1

print('Các giá trị duy nhất trong mask_mog2:', np.unique(mask_mog2), '(0=nền, 127=bóng đổ, 255=vật thể)')
print('Số phương tiện phát hiện được tại Frame #55:', det_count)

show([TRAFFIC_FRAMES[55], mog2.getBackgroundImage(), mask_mog2, mask_knn, fg_clean, vis_det],
     ['Frame #55', 'Nền học được (getBackgroundImage)',
      'MOG2 Mask (trắng=xe, xám 127=bóng)', 'KNN Mask',
      'Lọc bỏ bóng (>200) + Morphology', f'Phát hiện {det_count} phương tiện'],
     cols=3, size=4.2)
''')

# =====================================================================
md(r'''
---
## 4. Dòng quang học (Optical Flow): Lucas–Kanade & Farnebäck

Trừ nền chỉ cho biết *"pixel nào khác với nền"*, nhưng không cho biết **pixel đó đã dịch chuyển đi đâu và với vận tốc bao nhiêu**. **Optical Flow** giải quyết bài toán này!

### 4.1. Giả thuyết độ sáng không đổi (Brightness Constancy Assumption)
Giả sử một điểm ảnh tại $(x, y)$ ở thời điểm $t$ di chuyển một đoạn $(\Delta x, \Delta y) = (u, v)$ sau khoảng thời gian $\Delta t$ mà độ sáng không đổi:

$$I(x, y, t) = I(x + u, y + v, t + \Delta t)$$

Khai triển Taylor bậc nhất dẫn tới **Phương trình ràng buộc dòng quang học**:

$$I_x u + I_y v + I_t = 0 \quad \Longleftrightarrow \quad \nabla I \cdot \begin{bmatrix} u \\ v \end{bmatrix} = -I_t$$

Một phương trình có **2 ẩn** $(u, v)$ ⇒ chưa giải được nếu chỉ nhìn 1 pixel (gọi là **Aperture Problem**). Hai hướng giải kinh điển:

| Phương pháp | Loại | Ý tưởng giải | Hàm OpenCV |
|---|---|---|---|
| **Lucas–Kanade** | **Thưa (Sparse)** — chỉ tính tại các góc đặc trưng | Giả sử các pixel trong cửa sổ nhỏ $W \times W$ quanh góc có **cùng vector vận tốc $(u, v)$** → giải hệ bình phương tối tiểu trên cửa sổ đó (kết hợp kim tự tháp ảnh để bắt chuyển động lớn). | `cv2.calcOpticalFlowPyrLK(prevGray, nextGray, p0, None, ...)` |
| **Farnebäck** | **Dày đặc (Dense)** — tính $(u, v)$ cho **mọi pixel** | Xấp xỉ vùng lân cận bằng đa thức bậc 2 và ước lượng độ dịch chuyển từ sự thay đổi hệ số đa thức qua kim tự tháp đa tỉ lệ. | `cv2.calcOpticalFlowFarneback(prevGray, nextGray, None, ...)` |
''')

code(r'''
# --- 4.2. Sparse Optical Flow: Pyramidal Lucas-Kanade ---
lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Tìm các góc tốt trên xe ở frame #15 (dùng mask trừ nền để chỉ bắt điểm trên xe)
start_f = 15
prev_gray = cv2.cvtColor(TRAFFIC_FRAMES[start_f], cv2.COLOR_BGR2GRAY)
diff0 = cv2.absdiff(TRAFFIC_FRAMES[start_f], BG_TRUE)
_, vehicle_mask = cv2.threshold(cv2.cvtColor(diff0, cv2.COLOR_BGR2GRAY), 25, 255, cv2.THRESH_BINARY)

p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=80, qualityLevel=0.05, minDistance=7, mask=vehicle_mask)
print(f'Số điểm đặc trưng khởi tạo trên các xe ở Frame #{start_f}: {len(p0)}')

# Theo dõi các điểm này suốt 25 frame tiếp theo
tracks = [[pt.ravel().copy()] for pt in p0]
active = np.ones(len(p0), dtype=bool)
curr_pts = p0.copy()

for idx in range(start_f + 1, start_f + 26):
    next_gray = cv2.cvtColor(TRAFFIC_FRAMES[idx], cv2.COLOR_BGR2GRAY)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, curr_pts, None, **lk_params)
    # Kiểm tra ngược (Forward-Backward check) để loại điểm trôi sai
    back_pts, status_b, _ = cv2.calcOpticalFlowPyrLK(next_gray, prev_gray, next_pts, None, **lk_params)
    fb_err = np.linalg.norm(curr_pts - back_pts, axis=2).ravel()
    good = (status.ravel() == 1) & (status_b.ravel() == 1) & (fb_err < 1.5)

    for k in range(len(tracks)):
        if active[k] and good[k]:
            tracks[k].append(next_pts[k].ravel().copy())
        else:
            active[k] = False
    curr_pts = next_pts
    prev_gray = next_gray

vis_lk = TRAFFIC_FRAMES[start_f + 25].copy()
palette = [(0, 255, 255), (0, 255, 0), (255, 180, 0), (255, 0, 255)]
kept = 0
for k, tr in enumerate(tracks):
    if len(tr) >= 10:
        kept += 1
        pts_arr = np.array(tr, np.int32)
        col = palette[k % len(palette)]
        cv2.polylines(vis_lk, [pts_arr], False, col, 2, cv2.LINE_AA)
        cv2.circle(vis_lk, tuple(pts_arr[-1]), 4, (0, 0, 255), -1)

print(f'Số quỹ đạo điểm giữ được ổn định qua 25 frames (có kiểm tra Forward-Backward): {kept}')
show([TRAFFIC_FRAMES[start_f], vis_lk],
     [f'Frame #{start_f} (khởi tạo điểm)', f'Frame #{start_f+25}: Quỹ đạo Lucas-Kanade (vệt đuôi chuyển động)'],
     cols=2, size=5.5)
''')

code(r'''
# --- 4.3. Dense Optical Flow: Farnebäck ---
f1 = cv2.cvtColor(TRAFFIC_FRAMES[25], cv2.COLOR_BGR2GRAY)
f2 = cv2.cvtColor(TRAFFIC_FRAMES[27], cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(
    f1, f2, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)  # shape = (H, W, 2) chứa (u, v) tại mọi pixel

mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

# Mã hóa sang HSV: Hue = hướng chuyển động (0..179), Value = độ lớn vận tốc
hsv_flow = np.zeros((*f1.shape, 3), np.uint8)
hsv_flow[..., 0] = (ang / 2).astype(np.uint8)
hsv_flow[..., 1] = 255
hsv_flow[..., 2] = np.clip(mag * 18, 0, 255).astype(np.uint8)
bgr_flow = cv2.cvtColor(hsv_flow, cv2.COLOR_HSV2BGR)

# Vẽ trường vector mũi tên (quiver) thưa trên ảnh để trực quan hóa hướng chạy
arrow_vis = TRAFFIC_FRAMES[27].copy()
step = 16
for y in range(step // 2, f1.shape[0], step):
    for x in range(step // 2, f1.shape[1], step):
        dx, dy = flow[y, x]
        if np.hypot(dx, dy) > 1.0:
            p_end = (int(round(x + dx * 3)), int(round(y + dy * 3)))
            cv2.arrowedLine(arrow_vis, (x, y), p_end, (0, 255, 255), 1, tipLength=0.35)

print(f'Vận tốc dịch chuyển cực đại giữa 2 frame: {mag.max():.2f} px')
show([bgr_flow, arrow_vis],
     ['Farnebäck Dense Flow (Màu = hướng: Đỏ sang phải, lục/lam sang trái)',
      'Trường vector vận tốc (mũi tên vàng chỉ đúng chiều chạy của từng làn xe)'],
     cols=2, size=5.5)
''')

md(r'''
🔍 **Nhận xét:**
- Trên bản đồ HSV của Farnebäck, **hai làn xe có hai màu đối lập nhau** (làn trên chạy sang phải $\approx 0^\circ$ màu đỏ; làn dưới chạy sang trái $\approx 180^\circ$ màu xanh lơ), còn nền đường đứng yên có độ sáng $V \approx 0$ (màu đen).
- Ở Lucas–Kanade, kỹ thuật **Forward–Backward check** (tính flow từ $t \to t+1$ rồi tính ngược từ $t+1 \to t$, loại các điểm lệch $> 1.5$ px) là mẹo thực chiến cực kỳ hiệu quả để lọc bỏ điểm trôi sai.
''')

# =====================================================================
md(r'''
---
## 5. Theo dõi đối tượng theo màu: Histogram Backprojection, MeanShift & CamShift

Khi muốn theo dõi một đối tượng có **màu sắc đặc trưng** (ví dụ chiếc xe màu đỏ hoặc màu vàng):

1. **Histogram Backprojection (`cv2.calcBackProject`):**
   - Tính histogram mẫu (thường là kênh Hue trong HSV) của vùng đối tượng ở frame đầu tiên.
   - Với mỗi frame mới, thay giá trị mỗi pixel bằng **xác suất pixel đó thuộc về histogram mẫu** ⇒ tạo ra **bản đồ xác suất (probability map)** sáng rực tại vị trí đối tượng.
2. **MeanShift (`cv2.meanShift`):**
   - Dịch chuyển cửa sổ chữ nhật từng bước về phía **trọng tâm mật độ** (mode) của bản đồ xác suất cho đến khi hội tụ. Kích thước cửa sổ **cố định**.
3. **CamShift (`cv2.CamShift` — Continuously Adaptive MeanShift):**
   - Mở rộng MeanShift: sau khi hội tụ, nó **tự động co giãn kích thước và xoay góc cửa sổ** theo phân bố xác suất ⇒ trả về một hình chữ nhật xoay `((cx, cy), (w, h), angle)` rất hữu ích khi đối tượng tiến lại gần / ra xa camera.
''')

code(r'''
# Theo dõi chiếc xe màu đỏ (Car-Red) từ Frame #10 đến Frame #65 bằng MeanShift & CamShift
init_frame = TRAFFIC_FRAMES[10]
# Tọa độ ban đầu của xe đỏ tại Frame #10: (x, y, w, h)
track_win_ms  = (0, 105, 62, 34)
track_win_cam = (0, 105, 62, 34)

# Tính histogram mẫu trên kênh Hue (trong vùng xe đỏ)
roi = init_frame[105:139, 0:62]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask_roi = cv2.inRange(hsv_roi, np.array((0, 80, 60)), np.array((15, 255, 255)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask_roi, [36], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 1)
snapshots, snap_titles = [], []

for idx in range(10, 66):
    frame = TRAFFIC_FRAMES[idx]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Lọc bớt các pixel bão hòa thấp (mặt đường xám) trước khi backproject
    s_mask = cv2.inRange(hsv, np.array((0, 70, 50)), np.array((180, 255, 255)))
    dst_prob = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    dst_prob = cv2.bitwise_and(dst_prob, dst_prob, mask=s_mask)

    _, track_win_ms = cv2.meanShift(dst_prob, track_win_ms, term_crit)
    ret_rot, track_win_cam = cv2.CamShift(dst_prob, track_win_cam, term_crit)

    if idx in (12, 28, 44, 60):
        vis = frame.copy()
        x, y, w, h = track_win_ms
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)                # MeanShift: xanh lá
        box = cv2.boxPoints(ret_rot).astype(np.int32)
        cv2.polylines(vis, [box], True, (0, 255, 255), 2)                         # CamShift: vàng
        if idx == 28:
            prob_sample = dst_prob.copy()
        snapshots.append(vis)
        snap_titles.append(f'Frame #{idx}: x={x}')

show([prob_sample], ['Bản đồ xác suất Backprojection tại Frame #28 (sáng tại đúng chiếc xe đỏ)'], size=5.5)
show(snapshots, snap_titles, cols=4, size=3.4)
''')

# =====================================================================
md(r'''
---
## 6. Bộ lọc Kalman (`cv2.KalmanFilter`) — Dự đoán & Khử nhiễu quỹ đạo

Trong thực tế, bộ phát hiện (detector) có hai vấn đề lớn:
1. **Nhiễu đo lường:** tọa độ tâm đo được $(z_x, z_y)$ rung lắc qua từng frame.
2. **Mất dấu tạm thời (Occlusion):** xe đi qua dưới tán cây hoặc bị xe khác che khuất trong vài frame ⇒ detector không trả về tọa độ nào!

**Bộ lọc Kalman** giải quyết cả hai bằng chu trình 2 bước tối ưu theo phương sai:
1. **`predict()` (Dự đoán):** dùng mô hình vật lý (ví dụ vận tốc không đổi) để ước lượng trạng thái tiếp theo:
   $$\hat{x}_{k|k-1} = F\, \hat{x}_{k-1}, \qquad \hat{x} = \begin{bmatrix} x & y & v_x & v_y \end{bmatrix}^T, \qquad F = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
2. **`correct(z)` (Hiệu chỉnh / Cập nhật):** khi có phép đo mới $z_k = [z_x, z_y]^T$, kết hợp dự đoán và phép đo theo hệ số **Kalman Gain** $K_k$. Nếu frame đó **mất dấu (bị che khuất)**, ta **chỉ gọi `predict()`** mà không gọi `correct()` — đối tượng vẫn được nội suy mượt mà!
''')

code(r'''
# Khởi tạo KalmanFilter: 4 biến trạng thái (x, y, vx, vy), 2 biến đo lường (x, y)
kf = cv2.KalmanFilter(4, 2)
kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], np.float32)
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)
kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2   # Q: độ tin cậy vào mô hình chuyển động
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 25.0   # R: phương sai nhiễu phép đo (σ = 5 px)
kf.errorCovPost        = np.eye(4, dtype=np.float32)

# Mô phỏng quỹ đạo 1 vật thể bay theo đường cong qua 80 frame,
# bị nhiễu đo lường σ=6 px VÀ bị che khuất hoàn toàn từ frame 35..50
rng = np.random.default_rng(12)
ts = np.arange(80)
true_x = 40 + 7.0 * ts
true_y = 180 + 90 * np.sin(ts / 11.0)

kf.statePost = np.array([[true_x[0]], [true_y[0]], [0.0], [0.0]], np.float32)

meas_pts, kalman_pts = [], []
for t in ts:
    pred = kf.predict()
    px, py = float(pred[0, 0]), float(pred[1, 0])

    # Giả lập mất dấu (occlusion) từ frame 35 đến 50
    if 35 <= t <= 50:
        meas_pts.append((np.nan, np.nan))
        kalman_pts.append((px, py))      # chỉ dùng dự đoán quán tính!
    else:
        zx = true_x[t] + rng.normal(0, 6.0)
        zy = true_y[t] + rng.normal(0, 6.0)
        est = kf.correct(np.array([[np.float32(zx)], [np.float32(zy)]]))
        meas_pts.append((zx, zy))
        kalman_pts.append((float(est[0, 0]), float(est[1, 0])))

meas_arr   = np.array(meas_pts)
kalman_arr = np.array(kalman_pts)
valid = ~np.isnan(meas_arr[:, 0])
rmse_meas   = np.sqrt(np.mean((meas_arr[valid, 0] - true_x[valid])**2 + (meas_arr[valid, 1] - true_y[valid])**2))
rmse_kalman = np.sqrt(np.mean((kalman_arr[valid, 0] - true_x[valid])**2 + (kalman_arr[valid, 1] - true_y[valid])**2))
print(f'Sai số RMSE của phép đo thô: {rmse_meas:.2f} px  -->  Sau lọc Kalman: {rmse_kalman:.2f} px')

plt.figure(figsize=(10, 4.2))
plt.plot(true_x, true_y, 'k--', lw=2, label='Quỹ đạo thật')
plt.scatter(meas_arr[:, 0], meas_arr[:, 1], c='red', s=22, alpha=0.65, label='Phép đo nhiễu (mất dấu frame 35..50)')
plt.plot(kalman_arr[:, 0], kalman_arr[:, 1], 'b-', lw=2.2, label='Ước lượng Kalman (mượt + vượt vùng che khuất)')
plt.axvspan(true_x[35], true_x[50], color='orange', alpha=0.2, label='Vùng bị che khuất (không có phép đo)')
plt.gca().invert_yaxis()
plt.xlabel('x (pixel)'); plt.ylabel('y (pixel)')
plt.title('Kalman Filter: Khử nhiễu đo lường và duy trì quỹ đạo khi mất dấu')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
''')

# =====================================================================
md(r'''
---
## 7. Theo dõi đa đối tượng (Multi-Object Tracking — Detection + Association)

Khi trong khung hình có **nhiều phương tiện cùng di chuyển**, bộ trừ nền ở mỗi frame chỉ trả về một danh sách các hộp chữ nhật vô danh `[box_0, box_1, ...]`. Làm sao biết `box` nào ở frame $t$ chính là chiếc xe mang `ID #2` ở frame $t-1$?

Quy trình **Tracking-by-Detection** (nền tảng của SORT / ByteTrack):
1. **Dự đoán vị trí mới** của các track đang hoạt động từ vận tốc $(v_x, v_y)$.
2. **Tính ma trận khoảng cách** (hoặc IoU) giữa mọi cặp `(track đang có, detection mới)`.
3. **Ghép cặp tham lam / Hungarian:** nối track với detection gần nhất nếu khoảng cách $< d_{\max}$.
4. **Quản lý vòng đời:**
   - Detection chưa ghép với track nào ⇒ **tạo ID mới**.
   - Track không có detection ghép trong `max_disappeared` frame liên tiếp ⇒ **xóa track** (xe đã ra khỏi khung hình).
''')

code(r'''
class SimpleMultiObjectTracker:
    """Bộ theo dõi đa đối tượng dựa trên tâm + vận tốc (Centroid & Velocity Tracker)."""
    def __init__(self, max_disappeared=6, max_distance=55.0):
        self.next_id = 1
        self.tracks = {}  # id -> {'bbox': (x,y,w,h), 'center': (cx,cy), 'vel': (vx,vy), 'missed': 0, 'trace': []}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections):
        # 1. Dự đoán vị trí tiếp theo của các track hiện có
        for tid, tr in self.tracks.items():
            cx, cy = tr['center']
            vx, vy = tr['vel']
            tr['pred_center'] = (cx + vx, cy + vy)

        det_centers = [(x + w / 2.0, y + h / 2.0) for (x, y, w, h) in detections]
        used_tracks, used_dets = set(), set()

        # 2. Tính khoảng cách giữa dự đoán của track và tâm của detection
        pairs = []
        for tid, tr in self.tracks.items():
            px, py = tr['pred_center']
            for j, (dx, dy) in enumerate(det_centers):
                dist = float(np.hypot(px - dx, py - dy))
                if dist <= self.max_distance:
                    pairs.append((dist, tid, j))
        pairs.sort(key=lambda item: item[0])

        # 3. Ghép cặp theo khoảng cách tăng dần
        for dist, tid, j in pairs:
            if tid in used_tracks or j in used_dets:
                continue
            old_cx, old_cy = self.tracks[tid]['center']
            new_cx, new_cy = det_centers[j]
            vx = 0.6 * self.tracks[tid]['vel'][0] + 0.4 * (new_cx - old_cx)
            vy = 0.6 * self.tracks[tid]['vel'][1] + 0.4 * (new_cy - old_cy)
            self.tracks[tid].update(bbox=detections[j], center=(new_cx, new_cy),
                                    vel=(vx, vy), missed=0)
            self.tracks[tid]['trace'].append((int(new_cx), int(new_cy)))
            used_tracks.add(tid)
            used_dets.add(j)

        # 4. Tăng bộ đếm mất dấu cho các track không khớp
        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > self.max_disappeared:
                    del self.tracks[tid]

        # 5. Tạo ID mới cho detection chưa ghép
        for j, bbox in enumerate(detections):
            if j not in used_dets:
                cx, cy = det_centers[j]
                self.tracks[self.next_id] = {
                    'bbox': bbox, 'center': (cx, cy), 'vel': (0.0, 0.0),
                    'missed': 0, 'trace': [(int(cx), int(cy))]
                }
                self.next_id += 1
        return self.tracks


# Chạy thử trên toàn bộ chuỗi TRAFFIC_FRAMES
tracker = SimpleMultiObjectTracker()
vis_tracks = []
for idx, f in enumerate(TRAFFIC_FRAMES):
    diff = cv2.absdiff(f, BG_TRUE)
    _, bw = cv2.threshold(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), 25, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) >= 400]
    active_tracks = tracker.update(dets)

    if idx in (15, 35, 55, 75):
        canvas = f.copy()
        for tid, tr in active_tracks.items():
            x, y, w, h = tr['bbox']
            col = ((tid * 67) % 200 + 55, (tid * 131) % 200 + 55, (tid * 197) % 200 + 55)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), col, 2)
            cv2.putText(canvas, f'ID #{tid}', (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
            if len(tr['trace']) > 1:
                cv2.polylines(canvas, [np.array(tr['trace'], np.int32)], False, col, 2)
        vis_tracks.append((idx, canvas))

show([im for _, im in vis_tracks], [f'Frame #{i} (Giữ nguyên ID & vết quỹ đạo)' for i, _ in vis_tracks], cols=4, size=3.5)
''')

md(r'''
✍️ **Bài tập Phần I (Bài 1 & 2):**
1. Dùng `cv2.calcOpticalFlowFarneback` để tính tốc độ trung bình (theo pixel/frame) của từng chiếc xe trong vùng bounding box của nó.
2. Viết hàm phát hiện "vật thể bị bỏ quên" (abandoned object): so sánh mô hình nền cập nhật nhanh ($\alpha = 0.1$) và mô hình nền cập nhật rất chậm ($\alpha = 0.005$).
''')
