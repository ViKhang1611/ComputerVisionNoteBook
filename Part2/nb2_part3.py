from nb2_part1 import md, code  # noqa: F401  (shares the same CELLS list)

# =====================================================================
md(r'''
---
# PHẦN III — HỌC MÁY, NHẬN DẠNG & DEEP LEARNING VỚI OPENCV (`cv2.ml` & `cv2.dnn`)

## 14. Phát hiện đối tượng cổ điển: Haar Cascade & HOG + SVM

Trước kỷ nguyên Deep Learning, hai trụ cột nổi tiếng nhất của phát hiện đối tượng thời gian thực là **Viola–Jones (Haar Cascade, 2001)** và **Dalal–Triggs (HOG + Linear SVM, 2005)**. Chúng vẫn rất hữu ích khi chạy trên vi điều khiển / CPU yếu mà không có GPU.

### 14.1. Haar Cascade (`cv2.CascadeClassifier`)
- **Nguyên lý:**
  1. **Đặc trưng Haar-like:** hiệu tổng độ sáng giữa các vùng chữ nhật trắng và đen liền kề (ví dụ vùng mắt thường tối hơn vùng sống mũi và gò má).
  2. **Ảnh tích phân (Integral Image):** $II(x, y) = \sum_{x'\le x, y'\le y} I(x', y')$ giúp tính tổng mọi hình chữ nhật bất kỳ chỉ với **4 phép cộng/trừ** ($O(1)$).
  3. **Tầng phân loại (Cascade of Boosted Classifiers):** loại bỏ nhanh 95% cửa sổ nền ngay từ các tầng đầu tiên, chỉ cửa sổ có khả năng cao mới đi qua hết các tầng.
- OpenCV cài sẵn các file XML huấn luyện trước trong thư mục `cv2.data.haarcascades`:
  - `haarcascade_frontalface_default.xml` (khuôn mặt)
  - `haarcascade_eye.xml` (mắt)
  - `haarcascade_fullbody.xml` (toàn thân)

### 14.2. HOG + SVM (`cv2.HOGDescriptor`)
- **Nguyên lý HOG (Histogram of Oriented Gradients):**
  1. Tính độ lớn $m$ và hướng $\theta \in [0^\circ, 180^\circ)$ của gradient tại mỗi pixel.
  2. Chia ảnh thành các ô nhỏ **cell $8\times 8$ px**; mỗi cell lập một **histogram 9 hướng** (mỗi bin $20^\circ$).
  3. Gom $2\times 2$ cell thành một **block $16\times 16$** và chuẩn hóa $L_2$ để **bất biến với thay đổi độ sáng**.
  4. Nối toàn bộ các vector block lại thành một vector đặc trưng duy nhất rồi đưa vào bộ phân loại **SVM tuyến tính**.
''')

code(r'''
# 1. Ảnh tích phân (Integral Image - cv2.integral) & Đặc trưng Haar-like O(1)
messi = load_sample('messi5.jpg')
m_gray = cv2.equalizeHist(cv2.cvtColor(messi, cv2.COLOR_BGR2GRAY))

# cv2.integral trả về ma trận kích thước (H+1, W+1) để không cần xét biên âm
ii = cv2.integral(m_gray)
print('Ảnh xám shape:', m_gray.shape, '--> Integral Image shape:', ii.shape, ii.dtype)


def rect_sum_o1(integral_img, x1, y1, x2, y2):
    """Tính tổng mọi pixel trong [y1:y2, x1:x2] chỉ bằng 4 phép truy cập mảng O(1)."""
    return int(integral_img[y2, x2] - integral_img[y1, x2] - integral_img[y2, x1] + integral_img[y1, x1])


s_fast = rect_sum_o1(ii, 60, 40, 210, 170)
s_slow = int(m_gray[40:170, 60:210].sum())
print(f'Kiểm chứng tổng vùng [40:170, 60:210]: O(1) Integral = {s_fast} | np.sum = {s_slow} (khớp tuyệt đối!)')

# Minh họa Haar Cascade (nếu bản build OpenCV có module legacy CascadeClassifier)
vis_haar = messi.copy()
if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
    cascade_file = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
    if cascade_file.is_file():
        fc = cv2.CascadeClassifier(str(cascade_file))
        for (x, y, w, h) in fc.detectMultiScale(m_gray, 1.1, 4):
            cv2.rectangle(vis_haar, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Minh họa đặc trưng Haar 2-hình-chữ-nhật (vùng mắt tối hơn vùng gò má) trên cửa sổ trượt
cv2.rectangle(vis_haar, (195, 55), (255, 130), (0, 255, 0), 2)
cv2.rectangle(vis_haar, (202, 72), (248, 88), (255, 0, 0), -1)     # dải tối (mắt)
cv2.rectangle(vis_haar, (202, 88), (248, 104), (255, 255, 255), -1) # dải sáng (gò má)
cv2.putText(vis_haar, 'Haar 2-rect feature', (165, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)

# 2. Trực quan hóa đặc trưng HOG (9 hướng gradient trong mỗi ô 8x8)
patch = cv2.resize(m_gray, (128, 128))
gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=1)
mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
ang = ang % 180.0  # HOG dùng hướng không dấu (0..180 độ)

cell = 8
hog_canvas = np.zeros((128, 128), np.uint8)
for cy in range(0, 128, cell):
    for cx in range(0, 128, cell):
        m_cell = mag[cy:cy + cell, cx:cx + cell]
        a_cell = ang[cy:cy + cell, cx:cx + cell]
        hist, _ = np.histogram(a_cell, bins=9, range=(0, 180), weights=m_cell)
        hist = hist / (hist.max() + 1e-6)
        center_x, center_y = cx + cell // 2, cy + cell // 2
        for b in range(9):
            if hist[b] > 0.25:
                rad = np.deg2rad(b * 20 + 10 + 90)
                r = (cell // 2 - 1) * hist[b]
                dx, dy = int(round(r * np.cos(rad))), int(round(r * np.sin(rad)))
                cv2.line(hog_canvas, (center_x - dx, center_y - dy),
                         (center_x + dx, center_y + dy), int(255 * hist[b]), 1)

show([vis_haar, patch, hog_canvas],
     ['Đặc trưng Haar & Integral Image O(1)', 'Ảnh xám chuẩn hóa 128×128', 'Trực quan hóa các ô HOG 8×8 (9 hướng)'],
     cols=3, size=4.2)
''')

# =====================================================================
md(r'''
---
## 15. Huấn luyện bộ phân loại chữ số với HOG + `cv2.ml` (`SVM` & `KNearest`)

Module `cv2.ml` cung cấp các thuật toán học máy cổ điển chạy trực tiếp bằng C++ cực nhanh: `SVM`, `KNearest`, `RTrees` (Random Forest), `ANN_MLP`, `Boost`.

Dưới đây là quy trình **huấn luyện một mô hình nhận dạng chữ số (0–9)** hoàn chỉnh:
1. Tạo bộ dữ liệu 1.000 ảnh chữ số $28\times 28$ có biến dạng xoay, co giãn, dịch chuyển và nhiễu ngẫu nhiên.
2. Tiền xử lý **Deskew (chỉnh thẳng độ nghiêng)** dựa trên moment bậc hai ($\mu_{11}, \mu_{02}$) của ảnh.
3. Trích xuất vector đặc trưng **HOG** bằng `cv2.HOGDescriptor`.
4. Chia tập Train (80%) / Test (20%) và huấn luyện:
   - **k-NN (`cv2.ml.KNearest_create()`)**
   - **SVM (`cv2.ml.SVM_create()`)** với nhân RBF (`SVM_RBF`).
''')

code(r'''
def deskew_digit(img_28):
    """Chỉnh thẳng chữ số nghiêng dựa trên central moments mu11, mu02."""
    m = cv2.moments(img_28)
    if abs(m['mu02']) < 1e-2:
        return img_28.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * 28 * skew], [0, 1, 0]])
    return cv2.warpAffine(img_28, M, (28, 28), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)


def compute_hog_324(img_28):
    """Tính vector đặc trưng HOG 324 chiều cho ảnh 28x28 (tương thích cả OpenCV 4.x và 5.x)."""
    if hasattr(cv2, 'HOGDescriptor'):
        hog_cv = cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (7, 7), 9)
        return hog_cv.compute(img_28).ravel().astype(np.float32)
    gx = cv2.Sobel(img_28, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img_28, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180.0
    # Chia thành lưới 4x4 cell (mỗi cell 7x7 px, 9 bins)
    cells = np.zeros((4, 4, 9), np.float32)
    for r in range(4):
        for c in range(4):
            m_c = mag[r * 7:(r + 1) * 7, c * 7:(c + 1) * 7]
            a_c = ang[r * 7:(r + 1) * 7, c * 7:(c + 1) * 7]
            hist, _ = np.histogram(a_c, bins=9, range=(0, 180), weights=m_c)
            cells[r, c] = hist
    # Chuẩn hóa L2 trên 3x3 block (mỗi block 2x2 cell -> 36 phần tử) => 3*3*36 = 324 đặc trưng
    blocks = []
    for r in range(3):
        for c in range(3):
            blk = cells[r:r + 2, c:c + 2].ravel()
            blocks.append(blk / np.sqrt(np.dot(blk, blk) + 1e-5))
    return np.concatenate(blocks).astype(np.float32)


# Sinh 1.000 mẫu chữ số 0..9 (mỗi lớp 100 mẫu) với font, góc xoay, độ dày và vị trí ngẫu nhiên
rng = np.random.default_rng(2026)
fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX]
X_hog, y_labels, sample_imgs = [], [], []

for digit in range(10):
    for i in range(100):
        canvas = np.zeros((28, 28), np.uint8)
        fnt = fonts[int(rng.integers(len(fonts)))]
        scale = float(rng.uniform(0.68, 0.88))
        thick = int(rng.integers(1, 3))
        ox = int(rng.integers(4, 9))
        oy = int(rng.integers(20, 25))
        cv2.putText(canvas, str(digit), (ox, oy), fnt, scale, 255, thick, cv2.LINE_AA)
        ang = float(rng.uniform(-16, 16))
        M_rot = cv2.getRotationMatrix2D((14, 14), ang, float(rng.uniform(0.9, 1.08)))
        M_rot[0, 1] += float(rng.uniform(-0.15, 0.15))
        canvas = cv2.warpAffine(canvas, M_rot, (28, 28))
        canvas = np.clip(canvas.astype(np.float32) + rng.normal(0, 8, (28, 28)), 0, 255).astype(np.uint8)

        aligned = deskew_digit(canvas)
        X_hog.append(compute_hog_324(aligned))
        y_labels.append(digit)
        if i == 0:
            sample_imgs.append(canvas)

X_hog = np.array(X_hog, dtype=np.float32)
y_labels = np.array(y_labels, dtype=np.int32)
print('Kích thước ma trận đặc trưng HOG:', X_hog.shape, '(1000 mẫu × 324 đặc trưng/mẫu)')

# Chia ngẫu nhiên 800 mẫu Train, 200 mẫu Test
perm = rng.permutation(len(y_labels))
tr_idx, te_idx = perm[:800], perm[800:]
X_train, y_train = X_hog[tr_idx], y_labels[tr_idx]
X_test,  y_test  = X_hog[te_idx], y_labels[te_idx]

if hasattr(cv2, 'ml'):
    # 1. Huấn luyện k-NN (k = 3) bằng cv2.ml
    knn_clf = cv2.ml.KNearest_create()
    knn_clf.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    _, y_pred_knn, _, _ = knn_clf.findNearest(X_test, k=3)
    y_pred_knn = y_pred_knn.ravel().astype(int)

    # 2. Huấn luyện SVM (RBF Kernel) bằng cv2.ml
    svm_clf = cv2.ml.SVM_create()
    svm_clf.setType(cv2.ml.SVM_C_SVC)
    svm_clf.setKernel(cv2.ml.SVM_RBF)
    svm_clf.setC(5.0)
    svm_clf.setGamma(0.5)
    svm_clf.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    _, y_pred_best = svm_clf.predict(X_test)
    y_pred_best = y_pred_best.ravel().astype(int)
else:
    # Ma trận khoảng cách Cosine / L2 cho k-NN (k=3) và Trọng tâm nguyên mẫu RBF (RBF Kernel Classifier)
    dists = np.linalg.norm(X_test[:, None, :] - X_train[None, :, :], axis=2)
    knn3_idx = np.argsort(dists, axis=1)[:, :3]
    y_pred_knn = np.array([np.bincount(y_train[row], minlength=10).argmax() for row in knn3_idx])
    K_rbf = np.exp(-0.5 * dists ** 2)
    scores_c = np.stack([K_rbf[:, y_train == c].sum(axis=1) for c in range(10)], axis=1)
    y_pred_best = scores_c.argmax(axis=1)

acc_knn  = (y_pred_knn == y_test).mean() * 100
acc_best = (y_pred_best == y_test).mean() * 100
print(f'Độ chính xác trên tập Test (200 mẫu) — k-NN (k=3): {acc_knn:.2f}% | RBF Classifier: {acc_best:.2f}%')

# Vẽ mẫu chữ số và Ma trận nhầm lẫn (Confusion Matrix)
conf_mat = np.zeros((10, 10), dtype=int)
for yt, yp in zip(y_test, y_pred_best):
    conf_mat[yt, yp] += 1

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={'width_ratios': [1.3, 1]})
strip = np.hstack([cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=90) for im in sample_imgs])
axes[0].imshow(strip, cmap='gray'); axes[0].axis('off')
axes[0].set_title('Mẫu ảnh chữ số 0..9 (có xoay, nghiêng & nhiễu)')

axes[1].imshow(conf_mat, cmap='Blues')
axes[1].set_title(f'Confusion Matrix ({acc_best:.1f}%)')
axes[1].set_xlabel('Dự đoán'); axes[1].set_ylabel('Nhãn thật')
axes[1].set_xticks(range(10)); axes[1].set_yticks(range(10))
for r in range(10):
    for c in range(10):
        if conf_mat[r, c] > 0:
            axes[1].text(c, r, str(conf_mat[r, c]), ha='center', va='center',
                         color='white' if conf_mat[r, c] > 10 else 'black', fontsize=8)
plt.tight_layout(); plt.show()
''')

# =====================================================================
md(r'''
---
## 16. Phát hiện & Giải mã QR Code (`cv2.QRCodeDetector`)

OpenCV tích hợp sẵn `cv2.QRCodeDetector` (và `cv2.barcode.BarcodeDetector`) hoạt động độc lập không cần thư viện ngoài:
- `data, bbox, straight_qrcode = detector.detectAndDecode(img)`:
  - `data`: chuỗi văn bản giải mã được (`""` nếu không đọc được).
  - `bbox`: tọa độ 4 đỉnh của mã QR trên ảnh (dù bị xoay hay nghiêng phối cảnh).
  - `straight_qrcode`: ma trận nhị phân của các ô module QR sau khi đã được nắn thẳng!
- `ok, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(img)`: giải mã **nhiều mã QR** cùng lúc trong 1 bức ảnh.
''')

code(r'''
# Mã hóa một ma trận QR Code Version 1 (21x21 modules) chuẩn ISO/IEC 18004 rồi đặt nghiêng vào ảnh
qr_bits = np.array([
    [1,1,1,1,1,1,1,0,1,0,0,1,0,0,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,1],
    [1,0,1,1,1,0,1,0,0,0,1,0,0,0,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1,0,0,1,1,0,1,0,1,0,1,1,1,0,1],
    [1,0,1,1,1,0,1,0,1,0,0,0,0,0,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,1,0,1,1,0,1,1,0,1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0],
    [1,0,1,0,1,1,1,1,1,0,0,1,1,1,0,0,1,0,1,1,0],
    [0,1,1,0,1,0,0,0,0,1,1,1,1,0,0,1,0,0,0,1,0],
    [1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,0,1,1,0,0,1],
    [0,1,0,0,1,1,0,1,0,1,0,0,0,1,1,0,0,1,1,1,0],
    [1,1,0,1,1,0,1,1,0,0,1,0,0,1,1,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1,0,1,0],
    [1,1,1,1,1,1,1,0,0,0,1,0,1,0,1,1,0,0,0,0,1],
    [1,0,0,0,0,0,1,0,0,0,1,1,1,0,0,1,0,0,1,1,0],
    [1,0,1,1,1,0,1,0,1,1,0,1,1,1,0,0,1,1,0,1,0],
    [1,0,1,1,1,0,1,0,0,0,0,0,1,1,0,1,0,1,0,0,1],
    [1,0,1,1,1,0,1,0,1,0,0,1,1,1,1,1,1,0,1,0,0],
    [1,0,0,0,0,0,1,0,0,1,1,1,1,0,1,0,1,0,0,1,0],
    [1,1,1,1,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0],
], dtype=np.uint8)

# Phóng to mỗi module thành 10x10 px kèm viền trắng (quiet zone = 4 modules)
qr_padded = np.pad(1 - qr_bits, pad_width=4, mode='constant', constant_values=1) * 255
qr_img = cv2.resize(qr_padded.astype(np.uint8), (290, 290), interpolation=cv2.INTER_NEAREST)
qr_bgr = cv2.cvtColor(qr_img, cv2.COLOR_GRAY2BGR)

# Đặt thẻ QR nghiêng trên một mặt bàn
card_scene = np.full((360, 520, 3), (205, 195, 180), np.uint8)
src_q = np.float32([[0, 0], [290, 0], [290, 290], [0, 290]])
dst_q = np.float32([[130, 35], [405, 65], [375, 325], [95, 295]])
Mq = cv2.getPerspectiveTransform(src_q, dst_q)
warped_q = cv2.warpPerspective(qr_bgr, Mq, (520, 360))
mask_q   = cv2.warpPerspective(np.full((290, 290), 255, np.uint8), Mq, (520, 360))
card_scene = np.where(mask_q[..., None] > 0, warped_q, card_scene)

decoded_text, points, straight_qr = '', None, None
if hasattr(cv2, 'QRCodeDetector'):
    detector = cv2.QRCodeDetector()
    decoded_text, points, straight_qr = detector.detectAndDecode(card_scene)
    if points is None:
        _, points = detector.detect(card_scene)

# Tự động định vị 4 góc thẻ QR & 3 dấu định vị (Finder Patterns 1:1:3:1:1) bằng phân cấp Contour RETR_TREE
gray_qr = cv2.cvtColor(card_scene, cv2.COLOR_BGR2GRAY)
_, bw_qr = cv2.threshold(gray_qr, 210, 255, cv2.THRESH_BINARY)
cnts_q, _ = cv2.findContours(bw_qr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
card_cnt = max(cnts_q, key=cv2.contourArea)
approx_q = cv2.approxPolyDP(card_cnt, 0.02 * cv2.arcLength(card_cnt, True), True)
if points is None and len(approx_q) == 4:
    points = approx_q.reshape(1, 4, 2).astype(np.float32)

# Tìm 3 mắt định vị (Finder Patterns: các ô vuông có >= 2 tầng contour lồng nhau)
_, bw_inv = cv2.threshold(gray_qr, 120, 255, cv2.THRESH_BINARY_INV)
cnts_tree, hier_tree = cv2.findContours(bw_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
finder_centers = []
if hier_tree is not None:
    h_arr = hier_tree.reshape(-1, 4)
    for i, c in enumerate(cnts_tree):
        child = h_arr[i, 2]
        if child != -1 and h_arr[child, 2] != -1 and cv2.contourArea(c) > 200:
            M_c = cv2.moments(c)
            if M_c['m00'] > 0:
                finder_centers.append((int(M_c['m10'] / M_c['m00']), int(M_c['m01'] / M_c['m00'])))

vis_qr = card_scene.copy()
if points is not None:
    pts_int = points.reshape(-1, 2).astype(int)
    cv2.polylines(vis_qr, [pts_int], True, (0, 220, 0), 3, cv2.LINE_AA)
    for i, p in enumerate(pts_int):
        cv2.circle(vis_qr, tuple(p), 6, (0, 0, 255), -1)
        cv2.putText(vis_qr, f'P{i}', (p[0] + 8, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
for fc in finder_centers:
    cv2.circle(vis_qr, fc, 9, (255, 0, 255), 2)

# Nắn thẳng lại thẻ QR bằng warpPerspective
if straight_qr is None or straight_qr.size == 0:
    pts4 = points.reshape(4, 2).astype(np.float32)
    s = pts4.sum(axis=1); d = np.diff(pts4, axis=1).ravel()
    ordered = np.array([pts4[np.argmin(s)], pts4[np.argmin(d)], pts4[np.argmax(s)], pts4[np.argmax(d)]], np.float32)
    M_rect = cv2.getPerspectiveTransform(ordered, np.float32([[0, 0], [290, 0], [290, 290], [0, 290]]))
    straight_qr = cv2.warpPerspective(gray_qr, M_rect, (290, 290))

print(f'Phát hiện {len(finder_centers)} mắt định vị Finder Patterns tại: {finder_centers}')
print('4 góc QR phát hiện được:\n', points.reshape(-1, 2).round(1) if points is not None else None)
show([qr_img, vis_qr, straight_qr],
     ['Mã QR gốc (21×21 modules)', 'Phát hiện 4 đỉnh (đỏ) & 3 mắt Finder (tím)', 'Mã QR sau khi nắn thẳng'],
     cols=3, size=4.0)
''')

# =====================================================================
md(r'''
---
## 17. Học sâu với OpenCV (`cv2.dnn`): Blob, NMS & Suy luận mạng nơ-ron

Module **`cv2.dnn`** cho phép nạp trực tiếp các mô hình đã huấn luyện từ **PyTorch / ONNX (`.onnx`)**, **TensorFlow (`.pb`)**, **Caffe (`.caffemodel`)**, **Darknet YOLO (`.weights`)** để chạy suy luận (inference) cực nhanh bằng C++ mà **không cần cài PyTorch hay TensorFlow** trên máy đích!

### 17.1. Chuẩn bị dữ liệu đầu vào: `cv2.dnn.blobFromImage`
Hầu hết các mạng CNN/ViT không nhận ảnh `(H, W, 3)` kiểu `uint8` BGR, mà nhận một **tensor 4 chiều `float32`** theo thứ tự **`(N, C, H, W)`** (Batch size, Channels, Height, Width):

```python
blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1.0 / 255.0,     # nhân tỉ lệ (vd 1/255 để đưa về [0, 1])
    size=(224, 224),             # resize về kích thước đầu vào của mạng
    mean=(123.68, 116.78, 103.94), # trừ trung bình từng kênh (trừ trước khi nhân scalefactor!)
    swapRB=True,                 # True: đổi BGR -> RGB
    crop=False                   # False: resize thẳng; True: resize giữ tỉ lệ rồi cắt tâm
)
```

### 17.2. Hậu xử lý hộp giới hạn: `cv2.dnn.NMSBoxes`
Các mạng phát hiện đối tượng (YOLO, SSD, Faster R-CNN) thường sinh ra hàng nghìn hộp dự đoán chồng chéo nhau. `cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)` lọc giữ lại hộp tốt nhất cho mỗi đối tượng:
> ⚠️ **Lưu ý quan trọng:** Tham số `boxes` truyền vào `cv2.dnn.NMSBoxes` phải ở định dạng **`[x, y, width, height]`** (góc trên-trái + rộng + cao), **không phải** `[x1, y1, x2, y2]`!
''')

code(r'''
# 1. Minh họa chi tiết cv2.dnn.blobFromImage
sample_img = load_sample('fruits.jpg')
blob = cv2.dnn.blobFromImage(
    sample_img, scalefactor=1.0 / 255.0, size=(224, 224),
    mean=(0, 0, 0), swapRB=True, crop=False
)
print('Ảnh gốc shape       :', sample_img.shape, sample_img.dtype)
print('Blob đầu ra shape   :', blob.shape, blob.dtype, ' -> (N=1, C=3, H=224, W=224)')
print('Khoảng giá trị blob :', f'[{blob.min():.3f}, {blob.max():.3f}]')

# 2. Minh họa cv2.dnn.NMSBoxes trên các hộp dự đoán chồng chéo quanh các xe
frame_nms = TRAFFIC_FRAMES[40].copy()
# Giả sử mạng object detection xuất ra nhiều hộp ứng viên [x, y, w, h] quanh mỗi xe
raw_boxes_xywh = [
    # Quanh xe 1
    [232, 103, 70, 36], [238, 106, 66, 34], [226, 100, 74, 38], [242, 108, 64, 32],
    # Quanh xe 2
    [292, 138, 84, 38], [298, 141, 80, 35], [288, 136, 88, 40],
    # Quanh xe 3
    [396, 203, 68, 35], [402, 206, 64, 33],
    # Một hộp nhiễu điểm thấp ở nền đường
    [80,  300, 50, 30],
]
raw_scores = [0.94, 0.88, 0.79, 0.65,  0.91, 0.84, 0.72,  0.96, 0.81,  0.22]

indices = cv2.dnn.NMSBoxes(raw_boxes_xywh, raw_scores, score_threshold=0.5, nms_threshold=0.4)
indices = np.array(indices).ravel().tolist()
print(f'Số hộp trước NMS: {len(raw_boxes_xywh)}  -->  Chỉ số được giữ sau cv2.dnn.NMSBoxes: {indices}')

vis_before = frame_nms.copy()
for (x, y, w, h), sc in zip(raw_boxes_xywh, raw_scores):
    cv2.rectangle(vis_before, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.putText(vis_before, f'{sc:.2f}', (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

vis_after = frame_nms.copy()
for i in indices:
    x, y, w, h = raw_boxes_xywh[i]
    cv2.rectangle(vis_after, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(vis_after, f'Vehicle {raw_scores[i]:.2f}', (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

show([blob[0, 0], blob[0, 1], blob[0, 2]],
     ['Blob Kênh 0 (R)', 'Blob Kênh 1 (G)', 'Blob Kênh 2 (B)'], cols=3, size=3.2)
show([vis_before, vis_after],
     [f'Trước NMSBoxes ({len(raw_boxes_xywh)} hộp chồng chéo + hộp nhiễu 0.22)',
      f'Sau cv2.dnn.NMSBoxes ({len(indices)} hộp chuẩn xác)'],
     cols=2, size=5.2)
''')

# =====================================================================
md(r'''
---
# PHẦN IV — DỰ ÁN TỔNG HỢP & BÀI TẬP THỰC HÀNH

## 18. 🚀 Dự án thực tế: Hệ thống Giám sát, Đếm xe & Ước lượng Vận tốc Giao thông

Chúng ta sẽ kết hợp toàn bộ kiến thức ở Phần I, II, III để xây dựng một **pipeline giám sát giao thông đầu-cuối (End-to-End Traffic Analytics Pipeline)**:

```mermaid
flowchart LR
  A[Luồng Video] --> B[Trừ nền MOG2 + Khử bóng >200]
  B --> C[Morphology Open/Close + Lọc Contour]
  C --> D[Multi-Object Tracker gán ID & Quỹ đạo]
  D --> E[Kiểm tra cắt Vạch Đếm Ảo theo 2 chiều]
  D --> F[Ước lượng vận tốc km/h từ dịch chuyển tâm]
  E & F --> G[Hiển thị HUD & Thống kê lưu lượng]
```

**Nguyên lý đếm qua vạch ảo (Virtual Line Crossing):**
- Đặt một vạch đếm thẳng đứng tại hoành độ $x_{\text{line}} = 320$ (giữa khung hình).
- Với mỗi xe đang được theo dõi có tâm trước đó $x_{t-1}$ và tâm hiện tại $x_t$ (và ID này chưa từng được đếm):
  - Nếu $x_{t-1} < x_{\text{line}} \le x_t$: xe vừa đi qua vạch theo chiều **Trái $\to$ Phải** (Làn trên) ⇒ tăng `count_east`, đánh dấu ID đã đếm!
  - Nếu $x_{t-1} > x_{\text{line}} \ge x_t$: xe vừa đi qua vạch theo chiều **Phải $\to$ Trái** (Làn dưới) ⇒ tăng `count_west`, đánh dấu ID đã đếm!
''')

code(r'''
def run_traffic_analytics(frames, fps=30.0, meters_per_pixel=0.06):
    """Pipeline giám sát giao thông: Phát hiện + Theo dõi ID + Đếm qua vạch + Đo tốc độ km/h."""
    subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=20, detectShadows=True)
    # Huấn luyện nhanh mô hình nền trên 15 frame đầu để ổn định ngay từ frame 0
    for f in frames[:15]:
        subtractor.apply(f)

    tracker = SimpleMultiObjectTracker(max_disappeared=5, max_distance=50.0)
    line_x = frames[0].shape[1] // 2   # vạch đếm tại x = 320
    counted_ids = set()
    count_right = 0                     # làn trên (Trái -> Phải)
    count_left  = 0                     # làn dưới (Phải -> Trái)
    speed_log = []                      # lưu tốc độ (km/h) khi xe cắt vạch
    out_frames = []

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))

    for idx, frame in enumerate(frames):
        fg_mask = subtractor.apply(frame, learningRate=0.02)
        _, bin_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)  # loại bóng đổ (127)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, k_open)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k_close)

        cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in cnts:
            if cv2.contourArea(c) < 380:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if 80 <= y <= 290 and w >= 35:   # chỉ xét vùng lòng đường
                dets.append((x, y, w, h))

        tracks = tracker.update(dets)
        vis = frame.copy()

        # Vẽ vạch đếm ảo (đổi sang màu sáng rực nếu vừa có xe cắt qua)
        line_color = (0, 255, 255)
        for tid, tr in tracks.items():
            x, y, w, h = tr['bbox']
            vx, vy = tr['vel']
            # Quy đổi pixel/frame -> m/s -> km/h
            speed_kmh = abs(vx) * fps * meters_per_pixel * 3.6

            # Kiểm tra cắt vạch x = line_x
            if len(tr['trace']) >= 2 and tid not in counted_ids:
                prev_x = tr['trace'][-2][0]
                curr_x = tr['trace'][-1][0]
                if prev_x < line_x <= curr_x:
                    count_right += 1
                    counted_ids.add(tid)
                    speed_log.append(('Làn Trên (→)', tid, idx, speed_kmh))
                    line_color = (0, 0, 255)
                elif prev_x > line_x >= curr_x:
                    count_left += 1
                    counted_ids.add(tid)
                    speed_log.append(('Làn Dưới (←)', tid, idx, speed_kmh))
                    line_color = (0, 0, 255)

            col = (0, 230, 100) if tid in counted_ids else (255, 200, 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
            label = f'ID:{tid} {speed_kmh:.0f}km/h'
            cv2.putText(vis, label, (x, max(y - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
            if len(tr['trace']) > 1:
                cv2.polylines(vis, [np.array(tr['trace'][-15:], np.int32)], False, col, 2)

        cv2.line(vis, (line_x, 80), (line_x, 290), line_color, 3)
        # Bảng thông tin HUD ở góc trên
        cv2.rectangle(vis, (10, 8), (430, 66), (25, 25, 25), -1)
        cv2.putText(vis, f'Frame #{idx:02d} | Active: {len(tracks)} vehicles',
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        cv2.putText(vis, f'Count Right (->): {count_right}  |  Count Left (<-): {count_left}',
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
        out_frames.append(vis)

    return out_frames, count_right, count_left, speed_log


analytics_frames, c_right, c_left, speed_log = run_traffic_analytics(TRAFFIC_FRAMES)
print(f'Tổng kết sau 90 frames — Làn Trên (→): {c_right} lượt | Làn Dưới (←): {c_left} lượt')
print('\nNhật ký các lượt xe cắt qua vạch đếm (x = 320):')
for lane, tid, f_idx, spd in speed_log:
    print(f'  - Frame #{f_idx:02d} | {lane:12s} | Xe ID #{tid:<2d} | Tốc độ ước lượng: {spd:5.1f} km/h')

show_frames(analytics_frames, indices=[18, 38, 58, 82], cols=2, size=5.2, prefix='Traffic HUD')
''')

# =====================================================================
md(r'''
---
## 19. Bảng tra cứu nhanh, Lỗi thường gặp & Lời giải bài tập

### 19.1. Bảng tra cứu nhanh các hàm OpenCV Phần 2

| Nhóm chủ đề | Hàm / Lớp OpenCV | Ghi nhớ quan trọng |
|---|---|---|
| **Đọc/Ghi Video** | `cv2.VideoCapture`, `cv2.VideoWriter`, `cv2.VideoWriter_fourcc` | `VideoWriter` nhận kích thước `(W, H)` |
| **Chuyển động cơ bản** | `cv2.absdiff`, `cv2.accumulateWeighted` | `avg` phải là `float32` hoặc `float64` |
| **Trừ nền thống kê** | `cv2.createBackgroundSubtractorMOG2`, `createBackgroundSubtractorKNN` | Bóng đổ có giá trị `127`, vật thể là `255` |
| **Optical Flow** | `cv2.calcOpticalFlowPyrLK`, `cv2.calcOpticalFlowFarneback`, `cv2.cartToPolar` | LK cần điểm `float32` shape `(N, 1, 2)` |
| **MeanShift / CamShift** | `cv2.calcBackProject`, `cv2.meanShift`, `cv2.CamShift` | Lọc bỏ pixel có độ bão hòa $S$ thấp trước |
| **Kalman Filter** | `cv2.KalmanFilter(dynamParams, measureParams)`, `predict`, `correct` | Mất dấu: chỉ gọi `predict()`, bỏ qua `correct()` |
| **Homography & RANSAC** | `BFMatcher.knnMatch`, `cv2.findHomography(..., cv2.RANSAC)`, `cv2.perspectiveTransform` | Lowe's ratio $\tau \approx 0.75$; cần $\ge 4$ cặp điểm |
| **Ghép ảnh Panorama** | `cv2.Stitcher_create()`, `stitcher.stitch([img1, img2])` | Trả về `(status, pano)`, `status == 0` là thành công |
| **Hiệu chuẩn & 3D AR** | `findChessboardCorners`, `calibrateCamera`, `undistort`, `solvePnP`, `projectPoints` | `patternSize = (số_góc_ngang, số_góc_dọc)` |
| **Stereo Vision** | `cv2.StereoBM_create`, `cv2.StereoSGBM_create` | Chia kết quả `compute()` cho `16.0` để ra pixel |
| **Nhiếp ảnh tính toán** | `createMergeMertens`, `seamlessClone`, `detailEnhance`, `pencilSketch`, `stylization` | `seamlessClone` nhận tọa độ tâm `(cx, cy)` |
| **Nhận dạng & ML** | `CascadeClassifier`, `HOGDescriptor`, `cv2.ml.SVM_create`, `QRCodeDetector` | Đặc trưng huấn luyện `cv2.ml` phải là `float32` |
| **Deep Learning DNN** | `cv2.dnn.blobFromImage`, `cv2.dnn.NMSBoxes`, `cv2.dnn.readNetFromONNX` | `NMSBoxes` nhận hộp dạng `[x, y, w, h]` |

### 19.2. Các lỗi người học nâng cao hay gặp nhất

| Triệu chứng lỗi | Nguyên nhân | Cách khắc phục |
|---|---|---|
| File video xuất ra từ `VideoWriter` nặng 0 KB hoặc không mở được | Kích thước truyền vào là `(H, W)` thay vì `(W, H)`, hoặc frame ghi vào có số kênh khác `isColor` | Truyền `(frame.shape[1], frame.shape[0])` và gọi `writer.release()` |
| Bounding box từ MOG2 bao trùm cả bóng râm dài dưới gầm xe | Để mặc định `detectShadows=True` nhưng chỉ ngưỡng `> 0` | Ngưỡng `cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)` để bỏ giá trị bóng `127` |
| `calcOpticalFlowPyrLK` báo lỗi `Assertion failed ... prevPts` | Điểm đầu vào không phải `np.float32` hoặc thiếu chiều `(N, 1, 2)` | Ép kiểu `pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)` |
| `findChessboardCorners` trả về `False` dù nhìn mắt thường thấy bàn cờ | Nhầm **số ô vuông** với **số giao điểm góc bên trong** (`patternSize`), hoặc thiếu viền trắng quanh bàn cờ | Đếm số **giao điểm nội bộ** (bàn cờ $8\times 6$ ô có $7\times 5$ góc trong) |
| Bản đồ Disparity từ `StereoSGBM` có giá trị lên tới 600–800 | Quên chia cho 16 (OpenCV lưu fixed-point 4 bit lẻ) | `disp = stereo.compute(L, R).astype(np.float32) / 16.0` |
| `cv2.dnn.NMSBoxes` lọc sai vị trí hộp | Truyền `[x1, y1, x2, y2]` thay vì `[x, y, w, h]` | Chuyển sang `[x1, y1, x2 - x1, y2 - y1]` trước khi gọi `NMSBoxes` |

---
### 19.3. Lời giải chi tiết cho các bài tập Phần I & Phần II

Ô code dưới đây giải và hiển thị trực tiếp kết quả của các bài tập đã nêu ở cuối mỗi phần:
- **Bài 1:** Đo vận tốc từng phương tiện trực tiếp từ trường **Dense Optical Flow (Farnebäck)** mà không cần tracker.
- **Bài 3:** Dùng ma trận nghịch đảo Homography `np.linalg.inv(H_est)` để cắt phẳng lại tấm poster bị nghiêng & bị che khuất từ bức ảnh `cluttered`.
''')

code(r'''
# --- Lời giải Bài 1: Đo tốc độ từng xe bằng Farnebäck Dense Optical Flow ---
idx_f = 30
g1 = cv2.cvtColor(TRAFFIC_FRAMES[idx_f], cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(TRAFFIC_FRAMES[idx_f + 1], cv2.COLOR_BGR2GRAY)
flow_dense = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

diff_f = cv2.absdiff(TRAFFIC_FRAMES[idx_f], BG_TRUE)
_, mask_f = cv2.threshold(cv2.cvtColor(diff_f, cv2.COLOR_BGR2GRAY), 25, 255, cv2.THRESH_BINARY)
mask_f = cv2.morphologyEx(mask_f, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
cnts_f, _ = cv2.findContours(mask_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

vis_ex1 = TRAFFIC_FRAMES[idx_f].copy()
for c in cnts_f:
    if cv2.contourArea(c) < 380:
        continue
    x, y, w, h = cv2.boundingRect(c)
    roi_flow = flow_dense[y:y + h, x:x + w]
    roi_mask = mask_f[y:y + h, x:x + w] > 0
    vx_med = float(np.median(roi_flow[..., 0][roi_mask]))
    direction = '->' if vx_med > 0 else '<-'
    cv2.rectangle(vis_ex1, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(vis_ex1, f'{direction} vx={vx_med:+.1f}px/f', (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)

# --- Lời giải Bài 3: Khôi phục poster phẳng từ ảnh cluttered bằng nghịch đảo của H_est ---
H_inv = np.linalg.inv(H_est)
recovered_poster = cv2.warpPerspective(cluttered, H_inv, (poster.shape[1], poster.shape[0]))

show([vis_ex1, cluttered, recovered_poster],
     ['Bài 1: Vận tốc vx từng xe từ Farnebäck Flow',
      'Bài 3: Cảnh chứa poster nghiêng & bị che góc',
      'Bài 3: Poster nắn phẳng bằng inv(H_est)'],
     cols=3, size=4.2)
''')

md(r'''
---
## 🎓 Tổng kết toàn bộ khóa học OpenCV (Phần 1 & Phần 2)

Chúc mừng bạn đã hoàn thành trọn vẹn lộ trình từ **Xử lý 1 bức ảnh tĩnh** ([Huong_Dan_Xu_Ly_Anh_OpenCV_Chi_Tiet.ipynb](Huong_Dan_Xu_Ly_Anh_OpenCV_Chi_Tiet.ipynb)) đến **Video, Hình học đa ảnh 3D, Nhiếp ảnh tính toán & Học máy/DNN** trong notebook này!

**Gợi ý thực hành tiếp theo cho riêng bạn:**
1. Thay đường dẫn trong `cv2.VideoCapture` bằng một đoạn video quay từ điện thoại của bạn và chạy lại pipeline ở **Mục 18**.
2. In một tấm bàn cờ chuẩn ra giấy A4, chụp 10 tấm bằng điện thoại và chạy **Mục 10 & 11** để tự hiệu chuẩn camera điện thoại và vẽ vật thể AR 3D lên mặt bàn làm việc của bạn.
3. Kết hợp `cv2.dnn` ở **Mục 17** với mô hình `YOLOv8n.onnx` hoặc `YuNet` (nhận diện khuôn mặt) để nâng cấp bộ phát hiện trong dự án giám sát giao thông! 🚀
''')
