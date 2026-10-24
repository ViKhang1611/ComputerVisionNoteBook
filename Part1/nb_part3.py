from nb_part1 import md, code  # noqa: F401  (shares the same CELLS list)

# =====================================================================
md(r'''
---
## 17. Biến đổi Hough — tìm đường thẳng và đường tròn

### 17.1. Ý tưởng

Một đường thẳng được biểu diễn bằng $(\rho, \theta)$: $\rho = x\cos\theta + y\sin\theta$.
- Mỗi **điểm cạnh** $(x, y)$ "bỏ phiếu" cho **tất cả** các đường $(\rho, \theta)$ có thể đi qua nó (một đường sin trong không gian tham số).
- Các ô trong ma trận tích lũy (accumulator) được **nhiều phiếu** ⇒ là đường thẳng thật.

| Hàm | Kết quả |
|---|---|
| `cv2.HoughLines(edges, ρ_res, θ_res, threshold)` | các đường vô hạn `(ρ, θ)` |
| `cv2.HoughLinesP(edges, ρ_res, θ_res, threshold, minLineLength, maxLineGap)` | các **đoạn thẳng** `(x1,y1,x2,y2)` — thường hữu ích hơn |
| `cv2.HoughCircles(gray, HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius)` | các đường tròn `(x, y, r)` |
''')

code(r'''
sud_edges = cv2.Canny(sud_blur, 50, 150, apertureSize=3)

# HoughLines chuẩn
lines = cv2.HoughLines(sud_edges, 1, np.pi / 180, 200)
vis_std = sudoku.copy()
print('HoughLines tìm được', 0 if lines is None else len(lines), 'đường')
if lines is not None:
    for rho, theta in lines.reshape(-1, 2):
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        p1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        p2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(vis_std, p1, p2, (0, 0, 255), 2)

# HoughLinesP xác suất
segs = cv2.HoughLinesP(sud_edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)
vis_p = sudoku.copy()
print('HoughLinesP tìm được', 0 if segs is None else len(segs), 'đoạn thẳng')
if segs is not None:
    for x1, y1, x2, y2 in segs.reshape(-1, 4):
        cv2.line(vis_p, (x1, y1), (x2, y2), (0, 200, 0), 2)
show([sud_edges, vis_std, vis_p], ['Canny', 'HoughLines (ρ, θ) - đường vô hạn', 'HoughLinesP - đoạn thẳng'], size=4.5)
''')

code(r'''
# Hough Lines trên ảnh tòa nhà: lọc theo góc -> chỉ giữ đường gần thẳng đứng / nằm ngang
bedges = cv2.Canny(bblur, 50, 150)
segs = cv2.HoughLinesP(bedges, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=5)
vis = building.copy()
n_v = n_h = 0
for x1, y1, x2, y2 in segs.reshape(-1, 4):
    ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if ang > 80 and ang < 100:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2); n_v += 1     # dọc: đỏ
    elif ang < 10 or ang > 170:
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2); n_h += 1     # ngang: lam
print(f'Tổng {len(segs)} đoạn; {n_v} đoạn dọc, {n_h} đoạn ngang')
show([bedges, vis], ['Canny', 'Đoạn dọc (đỏ) / ngang (lam)'], size=5)
''')

md(r'''
### 17.2. HoughCircles — đếm đồng xu, viên kẹo, bánh xe…

Tham số quan trọng:
- `dp`: tỉ lệ nghịch độ phân giải accumulator (1 = bằng ảnh).
- `minDist`: khoảng cách tối thiểu giữa 2 tâm (tránh phát hiện trùng).
- `param1`: ngưỡng cao của Canny bên trong.
- `param2`: ngưỡng tích lũy cho tâm — **nhỏ → nhiều vòng tròn (cả giả)**, lớn → ít hơn.
- `minRadius`, `maxRadius`: giới hạn bán kính.

Luôn **làm mờ** (median) trước để giảm vòng tròn giả.
''')

code(r'''
smarties = load_sample('smarties.png')
sm_gray = cv2.medianBlur(cv2.cvtColor(smarties, cv2.COLOR_BGR2GRAY), 5)
rows = sm_gray.shape[0]
outs, titles = [], []
for p2 in [20, 30, 45]:
    circles = cv2.HoughCircles(sm_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=rows / 8,
                               param1=100, param2=p2, minRadius=1, maxRadius=40)
    vis = smarties.copy()
    n = 0
    if circles is not None:
        circles = np.round(circles.reshape(-1, 3)).astype(int)
        n = len(circles)
        for x, y, r in circles:
            cv2.circle(vis, (x, y), r, (255, 0, 255), 3)
            cv2.circle(vis, (x, y), 2, (0, 255, 0), 3)
    outs.append(vis); titles.append(f'param2={p2} → {n} vòng tròn')
show(outs, titles, size=4.5)
''')

md(r'''
🔍 **Nhận xét:** `param2` là tham số cần tinh chỉnh nhiều nhất. Hãy thử thay `minDist`, `minRadius`/`maxRadius` để thấy ảnh hưởng.
''')

# =====================================================================
md(r'''
---
## 18. Template Matching — tìm mẫu trong ảnh

`cv2.matchTemplate(image, template, method)` trượt template trên ảnh và tính độ tương đồng tại mỗi vị trí ⇒ trả về **bản đồ điểm** kích thước `(H−h+1, W−w+1)`. Sau đó dùng `cv2.minMaxLoc` để tìm vị trí tốt nhất.

| Method | Tốt nhất tại |
|---|---|
| `TM_SQDIFF`, `TM_SQDIFF_NORMED` | giá trị **nhỏ nhất** (min) |
| `TM_CCORR`, `TM_CCORR_NORMED`, `TM_CCOEFF`, `TM_CCOEFF_NORMED` | giá trị **lớn nhất** (max) |

`TM_CCOEFF_NORMED` (∈ [−1, 1]) thường đáng tin cậy nhất và dễ đặt ngưỡng.
''')

code(r'''
messi = load_sample('messi5.jpg')
template = messi[280:340, 330:390].copy()     # cắt quả bóng làm mẫu
th, tw = template.shape[:2]
methods = ['TM_SQDIFF_NORMED', 'TM_CCORR_NORMED', 'TM_CCOEFF_NORMED']
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for i, m in enumerate(methods):
    res = cv2.matchTemplate(messi, template, getattr(cv2, m))
    min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc if 'SQDIFF' in m else max_loc
    score = min_v if 'SQDIFF' in m else max_v
    vis = messi.copy()
    cv2.rectangle(vis, top_left, (top_left[0] + tw, top_left[1] + th), (0, 0, 255), 3)
    axes[0, i].imshow(res, cmap='jet'); axes[0, i].set_title(f'Bản đồ điểm {m}'); axes[0, i].axis('off')
    axes[1, i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); axes[1, i].axis('off')
    axes[1, i].set_title(f'Vị trí tốt nhất {top_left}, điểm = {score:.3f}')
plt.tight_layout(); plt.show()
show(template, 'Template (quả bóng)', size=2)
''')

md(r'''
### 18.1. Tìm NHIỀU đối tượng + Non-Maximum Suppression (NMS)

Lấy mọi vị trí có điểm > ngưỡng. Vì các vị trí lân cận một đối tượng đều có điểm cao, ta cần **NMS** để chỉ giữ 1 hộp cho mỗi đối tượng:
1. Sắp xếp hộp theo điểm giảm dần.
2. Lấy hộp điểm cao nhất, loại mọi hộp có **IoU** (Intersection over Union) với nó > ngưỡng.
3. Lặp lại.
''')

code(r'''
def nms(boxes, scores, iou_thresh=0.3):
    """boxes: Nx4 (x1, y1, x2, y2). Trả về chỉ số các hộp được giữ."""
    boxes = np.asarray(boxes, np.float32)
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / (area(boxes[[i]]) + area(boxes[order[1:]]) - inter)
        order = order[1:][iou <= iou_thresh]
    return keep


# Tạo "màn chơi" với nhiều ngôi sao và vài vật gây nhiễu
rng = np.random.default_rng(5)
scene = cv2.GaussianBlur(rng.integers(90, 160, (360, 640, 3), dtype=np.uint8), (7, 7), 0)
icon = np.full((50, 50, 3), 40, np.uint8)
cv2.fillPoly(icon, [star_points(25, 26, 22, 9)], (0, 215, 255))
positions = [(40, 30), (200, 80), (420, 40), (100, 250), (330, 220), (540, 280)]
for x, y in positions:
    scene[y:y+50, x:x+50] = icon
cv2.circle(scene, (520, 150), 24, (0, 215, 255), -1)        # vật gây nhiễu: tròn cùng màu
cv2.rectangle(scene, (250, 300), (295, 345), (0, 215, 255), -1)

res = cv2.matchTemplate(scene, icon, cv2.TM_CCOEFF_NORMED)
ys, xs = np.where(res >= 0.8)
boxes = [(x, y, x + 50, y + 50) for x, y in zip(xs, ys)]
scores = res[ys, xs]
keep = nms(boxes, scores, 0.3)
print(f'Số vị trí vượt ngưỡng 0.8: {len(boxes)}  ->  sau NMS: {len(keep)} đối tượng')

raw, final = scene.copy(), scene.copy()
for (x1, y1, x2, y2) in boxes:
    cv2.rectangle(raw, (x1, y1), (x2, y2), (0, 0, 255), 1)
for i in keep:
    x1, y1, x2, y2 = boxes[i]
    cv2.rectangle(final, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(final, f'{scores[i]:.2f}', (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
show([icon, raw, final], ['Template', 'Trước NMS (rất nhiều hộp trùng)', 'Sau NMS'], size=4.5)
''')

md(r'''
### 18.2. Hạn chế: không bất biến với tỉ lệ và góc xoay
''')

code(r'''
for desc, tpl in [('Gốc', icon),
                  ('Thu nhỏ 70%', cv2.resize(icon, None, fx=0.7, fy=0.7)),
                  ('Phóng to 140%', cv2.resize(icon, None, fx=1.4, fy=1.4)),
                  ('Xoay 30°', cv2.warpAffine(icon, cv2.getRotationMatrix2D((25, 25), 30, 1), (50, 50), borderValue=(40, 40, 40)))]:
    best = cv2.minMaxLoc(cv2.matchTemplate(scene, tpl, cv2.TM_CCOEFF_NORMED))[1]
    print(f'{desc:14s}: điểm cao nhất = {best:.3f}')
print('=> Muốn bất biến tỉ lệ: lặp nhiều scale (multi-scale). Muốn bất biến xoay: dùng đặc trưng (mục 19).')
''')

# =====================================================================
md(r'''
---
## 19. Phát hiện góc & đặc trưng (Corners & Features)

**Góc** là điểm mà cường độ thay đổi mạnh theo **mọi hướng** — dễ định vị và theo dõi hơn nhiều so với cạnh hay vùng phẳng.

### 19.1. Harris Corner

Xét ma trận cấu trúc $M = \sum w(x,y) \begin{bmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{bmatrix}$ với trị riêng $\lambda_1, \lambda_2$:

$$R = \det(M) - k\,(\text{trace}\,M)^2 = \lambda_1\lambda_2 - k(\lambda_1+\lambda_2)^2$$

- $R$ lớn dương ⇒ **góc** (cả hai λ lớn); $R$ âm ⇒ **cạnh**; $|R|$ nhỏ ⇒ **vùng phẳng**.

`cv2.cornerHarris(gray_float32, blockSize, ksize, k)` — thường `blockSize=2`, `ksize=3`, `k=0.04`.
''')

code(r'''
board = np.zeros((320, 320), np.uint8)
for i in range(8):
    for j in range(8):
        if (i + j) % 2:
            board[i*40:(i+1)*40, j*40:(j+1)*40] = 255
Mp = cv2.getPerspectiveTransform(np.float32([[0, 0], [320, 0], [320, 320], [0, 320]]),
                                 np.float32([[40, 20], [300, 50], [280, 300], [10, 280]]))
board = cv2.warpPerspective(board, Mp, (320, 320), borderValue=128)

def harris_vis(gray_img, color_img, thresh=0.01):
    R = cv2.cornerHarris(np.float32(gray_img), blockSize=2, ksize=3, k=0.04)
    vis = color_img.copy()
    vis[cv2.dilate(R, None) > thresh * R.max()] = (0, 0, 255)
    return R, vis

R_b, vis_b = harris_vis(board, cv2.cvtColor(board, cv2.COLOR_GRAY2BGR))
R_bd, vis_bd = harris_vis(bgray, building)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(R_b, cmap='seismic', vmin=-R_b.max(), vmax=R_b.max()); axes[0].set_title('Bản đồ R (đỏ: góc, lam: cạnh)')
axes[1].imshow(cv2.cvtColor(vis_b, cv2.COLOR_BGR2RGB)); axes[1].set_title('Harris trên bàn cờ')
axes[2].imshow(cv2.cvtColor(vis_bd, cv2.COLOR_BGR2RGB)); axes[2].set_title('Harris trên tòa nhà')
for a in axes: a.axis('off')
plt.tight_layout(); plt.show()
''')

md(r'''
### 19.2. Shi-Tomasi — `cv2.goodFeaturesToTrack`

Dùng $R = \min(\lambda_1, \lambda_2)$; cho phép chọn **N góc tốt nhất**, cách nhau tối thiểu `minDistance`. Rất hay dùng làm điểm đầu vào cho theo dõi chuyển động (optical flow).

`cv2.cornerSubPix` tinh chỉnh vị trí góc tới độ chính xác **dưới pixel** (dùng trong hiệu chuẩn camera).
''')

code(r'''
corners = cv2.goodFeaturesToTrack(bgray, maxCorners=150, qualityLevel=0.01, minDistance=10)
vis = building.copy()
for x, y in corners.reshape(-1, 2):
    cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)

bc = cv2.goodFeaturesToTrack(board, 100, 0.1, 15)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
bc_sub = cv2.cornerSubPix(board, bc.copy(), (5, 5), (-1, -1), criteria)
print('Ví dụ tọa độ góc (nguyên)  :', bc[:3].reshape(-1, 2))
print('Sau cornerSubPix (dưới pixel):', bc_sub[:3].reshape(-1, 2).round(2))
vis_b = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
for x, y in bc_sub.reshape(-1, 2):
    cv2.circle(vis_b, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1)
show([vis, vis_b], [f'Shi-Tomasi: {len(corners)} góc tốt nhất', f'Bàn cờ: {len(bc)} góc + subpixel'], size=5)
''')

md(r'''
### 19.3. Keypoint & Descriptor: ORB và SIFT

Góc Harris chưa đủ để **so khớp** giữa các ảnh: cần **descriptor** — vector mô tả vùng quanh điểm, bất biến với xoay/tỉ lệ.

| Thuật toán | Descriptor | Bất biến | Tốc độ | Ghi chú |
|---|---|---|---|---|
| **ORB** | nhị phân 32 byte (256 bit) | xoay, tỉ lệ (pyramid) | rất nhanh | so khớp bằng khoảng cách **Hamming** |
| **SIFT** | 128 số thực | xoay, tỉ lệ, ánh sáng | chậm hơn | chính xác; miễn phí từ OpenCV 4.4 |

`drawKeypoints(..., flags=DRAW_RICH_KEYPOINTS)` vẽ cả **kích thước** (vòng tròn) và **hướng** (đường bán kính) của keypoint.
''')

code(r'''
orb = cv2.ORB_create(nfeatures=500)
kp_orb, des_orb = orb.detectAndCompute(bgray, None)
sift = cv2.SIFT_create(nfeatures=500)
kp_sift, des_sift = sift.detectAndCompute(bgray, None)
print(f'ORB : {len(kp_orb)} keypoint, descriptor shape {des_orb.shape}, dtype {des_orb.dtype}')
print(f'SIFT: {len(kp_sift)} keypoint, descriptor shape {des_sift.shape}, dtype {des_sift.dtype}')
k0 = kp_orb[0]
print(f'Một keypoint ORB: pt={tuple(round(v, 1) for v in k0.pt)}, size={k0.size:.1f}, angle={k0.angle:.1f}°, response={k0.response:.5f}')

v1 = cv2.drawKeypoints(building, kp_orb, None, color=(0, 255, 0))
v2 = cv2.drawKeypoints(building, kp_sift[:150], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show([v1, v2], ['ORB keypoints', 'SIFT (150 điểm) - vẽ kích thước & hướng'], size=5.5)
''')

md(r'''
**Thử nghiệm bất biến:** so khớp ảnh với **chính nó sau khi xoay 30° và thu nhỏ 70%** — template matching thất bại, nhưng ORB vẫn tìm được các cặp điểm tương ứng.
''')

code(r'''
M2 = cv2.getRotationMatrix2D((bgray.shape[1] / 2, bgray.shape[0] / 2), 30, 0.7)
b2 = cv2.warpAffine(building, M2, (bgray.shape[1], bgray.shape[0]))
kp2, des2 = orb.detectAndCompute(cv2.cvtColor(b2, cv2.COLOR_BGR2GRAY), None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)      # Hamming cho descriptor nhị phân
matches = sorted(bf.match(des_orb, des2), key=lambda m: m.distance)
print(f'Số cặp khớp: {len(matches)}; khoảng cách tốt nhất = {matches[0].distance}, tệ nhất = {matches[-1].distance}')
vis = cv2.drawMatches(building, kp_orb, b2, kp2, matches[:40], None,
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
show(vis, '40 cặp ORB khớp tốt nhất giữa ảnh gốc và ảnh xoay 30° + thu nhỏ', size=12)
''')

# =====================================================================
md(r'''
---
## 20. Image Pyramid (kim tự tháp ảnh)

- **Gaussian pyramid**: lặp lại *làm mờ Gauss + bỏ một nửa hàng/cột* — `cv2.pyrDown` (mỗi tầng giảm 1/2 mỗi chiều). `cv2.pyrUp` phóng to ×2 (nhưng **không khôi phục** chi tiết đã mất).
- **Laplacian pyramid**: $L_i = G_i - \text{pyrUp}(G_{i+1})$ ⇒ lưu **chi tiết** ở mỗi tầng. Từ tầng nhỏ nhất + các $L_i$ có thể **tái tạo hoàn hảo** ảnh gốc.

Ứng dụng: xử lý đa tỉ lệ (phát hiện đối tượng ở nhiều kích thước), nén, **trộn ảnh liền mạch**.
''')

code(r'''
G = [img.copy()]
for _ in range(3):
    G.append(cv2.pyrDown(G[-1]))
print('Kích thước các tầng Gaussian:', [g.shape[:2] for g in G])

# Ghép các tầng vào 1 canvas để thấy rõ tỉ lệ
canvas = np.full((G[0].shape[0], G[0].shape[1] + G[1].shape[1], 3), 255, np.uint8)
canvas[:G[0].shape[0], :G[0].shape[1]] = G[0]
yoff = 0
for g in G[1:]:
    canvas[yoff:yoff + g.shape[0], G[0].shape[1]:G[0].shape[1] + g.shape[1]] = g
    yoff += g.shape[0]
show(canvas, 'Gaussian pyramid: mỗi tầng nhỏ đi một nửa', size=9)

up = cv2.pyrUp(cv2.pyrUp(G[2]), dstsize=(G[0].shape[1], G[0].shape[0]))
lap0 = cv2.subtract(G[0], cv2.pyrUp(G[1], dstsize=(G[0].shape[1], G[0].shape[0])))
show([up, cv2.normalize(lap0, None, 0, 255, cv2.NORM_MINMAX)],
     ['pyrUp(pyrUp(G2)): phóng lại nhưng mờ', 'Laplacian tầng 0 (chi tiết bị mất, đã kéo giãn)'], size=5)
''')

md(r'''
### 20.1. Trộn ảnh liền mạch (Pyramid Blending) — "táo-cam" kinh điển

Ghép trực tiếp nửa trái ảnh A với nửa phải ảnh B ⇒ đường nối rất rõ. Pyramid blending ghép **từng tầng Laplacian** rồi tái tạo ⇒ tần số thấp được trộn trên vùng rộng, tần số cao trên vùng hẹp ⇒ **mối nối mượt**.
''')

code(r'''
A = cv2.resize(load_sample('apple.jpg'), (512, 512)).astype(np.float32)
B = cv2.resize(load_sample('orange.jpg'), (512, 512)).astype(np.float32)
levels = 6


def gaussian_pyr(x, n):
    gp = [x]
    for _ in range(n):
        gp.append(cv2.pyrDown(gp[-1]))
    return gp


def laplacian_pyr(gp):
    lp = [gp[-1]]
    for i in range(len(gp) - 1, 0, -1):
        size = (gp[i - 1].shape[1], gp[i - 1].shape[0])
        lp.append(gp[i - 1] - cv2.pyrUp(gp[i], dstsize=size))
    return lp            # lp[0] là tầng nhỏ nhất


lpA, lpB = laplacian_pyr(gaussian_pyr(A, levels)), laplacian_pyr(gaussian_pyr(B, levels))
LS = [np.hstack((la[:, :la.shape[1] // 2], lb[:, la.shape[1] // 2:])) for la, lb in zip(lpA, lpB)]
blend = LS[0]
for i in range(1, len(LS)):
    blend = cv2.pyrUp(blend, dstsize=(LS[i].shape[1], LS[i].shape[0])) + LS[i]
blend = np.clip(blend, 0, 255).astype(np.uint8)
direct = np.hstack((A[:, :256], B[:, 256:])).astype(np.uint8)
show([A.astype(np.uint8), B.astype(np.uint8), direct, blend],
     ['Ảnh A', 'Ảnh B', 'Ghép trực tiếp (thấy rõ đường nối)', 'Pyramid blending (mượt)'], cols=4, size=3.5)
''')

# =====================================================================
md(r'''
---
## 21. Miền tần số — Biến đổi Fourier

Mọi ảnh có thể biểu diễn bằng **tổng các sóng sin/cos** với tần số, hướng khác nhau:
- **Tần số thấp** (gần tâm phổ sau khi `fftshift`) ⇒ vùng thay đổi chậm: màu nền, mảng lớn.
- **Tần số cao** (xa tâm) ⇒ thay đổi nhanh: cạnh, chi tiết, **nhiễu**.

```python
F = np.fft.fft2(gray)             # hoặc cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
F = np.fft.fftshift(F)            # đưa tần số 0 về tâm
mag = np.log1p(np.abs(F))         # phổ biên độ (log để dễ nhìn)
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))
```

**Định lý tích chập:** tích chập trong miền không gian = **phép nhân** trong miền tần số ⇒ lọc ảnh = nhân phổ với một mặt nạ.
''')

code(r'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
h, w = gray.shape
F = np.fft.fftshift(np.fft.fft2(gray))
mag = np.log1p(np.abs(F))

yy, xx = np.mgrid[0:h, 0:w]
dist = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
D0 = 30
masks = {
    f'Low-pass lý tưởng D0={D0}':  (dist <= D0).astype(np.float32),
    f'Low-pass Gauss D0={D0}':     np.exp(-dist ** 2 / (2 * D0 ** 2)).astype(np.float32),
    f'High-pass Gauss D0={D0}':    1 - np.exp(-dist ** 2 / (2 * D0 ** 2)).astype(np.float32),
}
fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
axes[0, 0].imshow(gray, cmap='gray'); axes[0, 0].set_title('Ảnh xám')
axes[1, 0].imshow(mag, cmap='gray'); axes[1, 0].set_title('Phổ biên độ log|F| (tâm = tần số 0)')
for i, (name, m) in enumerate(masks.items(), start=1):
    back = np.real(np.fft.ifft2(np.fft.ifftshift(F * m)))
    axes[0, i].imshow(m, cmap='gray'); axes[0, i].set_title(f'Mặt nạ: {name}')
    axes[1, i].imshow(back if 'Low' in name else np.abs(back), cmap='gray'); axes[1, i].set_title('Kết quả')
for a in axes.ravel(): a.axis('off')
plt.tight_layout(); plt.show()
''')

md(r'''
🔍 **Nhận xét:**
- Low-pass ⇒ **làm mờ** (bỏ chi tiết). Bộ lọc **lý tưởng** (cắt cụt) gây hiện tượng **gợn sóng (ringing)** quanh cạnh; bộ lọc **Gauss** thì không.
- High-pass ⇒ chỉ còn **cạnh/chi tiết**.

### 21.1. Ứng dụng thực tế: xóa nhiễu tuần hoàn (sọc) bằng **notch filter**

Nhiễu sọc (do quét, do lưới điện, moiré…) rất khó xóa trong miền không gian, nhưng trong miền tần số nó chỉ là **vài điểm sáng** ⇒ chỉ cần "khoét" các điểm đó.
''')

code(r'''
stripes = 40 * np.sin(2 * np.pi * (xx / 12 + yy / 25))
noisy = np.clip(gray + stripes, 0, 255).astype(np.float32)
Fn = np.fft.fftshift(np.fft.fft2(noisy))
mag_n = np.log1p(np.abs(Fn))

# Tự tìm các đỉnh nhiễu: bỏ vùng tâm, lấy 2 điểm sáng nhất (đối xứng qua tâm)
search = mag_n.copy()
cv2.circle(search, (w // 2, h // 2), 20, 0, -1)
notch = np.ones((h, w), np.float32)
for _ in range(2):
    _, _, _, peak = cv2.minMaxLoc(search)
    print('Đỉnh nhiễu tại', peak)
    cv2.circle(notch, peak, 8, 0, -1)
    cv2.circle(search, peak, 8, 0, -1)

restored = np.real(np.fft.ifft2(np.fft.ifftshift(Fn * notch)))
restored = np.clip(restored, 0, 255).astype(np.uint8)
g8 = gray.astype(np.uint8)
print(f'PSNR ảnh nhiễu sọc  : {cv2.PSNR(g8, noisy.astype(np.uint8)):.2f} dB')
print(f'PSNR sau notch      : {cv2.PSNR(g8, restored):.2f} dB')
print(f'PSNR nếu dùng Gauss : {cv2.PSNR(g8, cv2.GaussianBlur(noisy, (15, 15), 0).astype(np.uint8)):.2f} dB')

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
axes[0].imshow(noisy, cmap='gray'); axes[0].set_title('Ảnh có nhiễu sọc')
axes[1].imshow(mag_n, cmap='gray'); axes[1].set_title('Phổ: 2 đỉnh sáng = nhiễu')
axes[2].imshow(mag_n * notch, cmap='gray'); axes[2].set_title('Phổ sau khi khoét (notch)')
axes[3].imshow(restored, cmap='gray'); axes[3].set_title('Ảnh phục hồi')
for a in axes: a.axis('off')
plt.tight_layout(); plt.show()
''')

# =====================================================================
md(r'''
---
## 22. Phân vùng ảnh (Image Segmentation)

### 22.1. K-Means — lượng tử hóa màu / phân cụm màu

Coi mỗi pixel là một điểm trong không gian 3D (B, G, R). K-Means chia chúng thành **K cụm**, mỗi pixel được thay bằng **màu tâm cụm** ⇒ ảnh chỉ còn K màu.

```python
compactness, labels, centers = cv2.kmeans(data_float32, K, None, criteria, attempts, flags)
```
- `criteria`: điều kiện dừng `(type, max_iter, epsilon)`.
- `attempts`: số lần chạy với khởi tạo khác nhau (lấy kết quả tốt nhất).
- `flags`: `cv2.KMEANS_PP_CENTERS` (k-means++) hoặc `cv2.KMEANS_RANDOM_CENTERS`.
''')

code(r'''
Z = img.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
outs, titles, palettes = [], [], []
for K in [2, 4, 8, 16]:
    compactness, labels, centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    outs.append(centers[labels.ravel()].reshape(img.shape))
    titles.append(f'K = {K} màu (compactness={compactness:.2e})')
    palettes.append(centers)
show(outs, titles, cols=4, size=3.5)

# Bảng màu (palette) của K=8, sắp xếp theo tỉ lệ pixel
_, labels8, centers8 = cv2.kmeans(Z, 8, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
counts = np.bincount(labels8.ravel(), minlength=8) / len(labels8)
order = np.argsort(counts)[::-1]
bar = np.zeros((50, 500, 3), np.uint8); x0 = 0
for i in order:
    x1 = x0 + int(round(counts[i] * 500))
    bar[:, x0:x1] = centers8[i]; x0 = x1
show(bar, 'Các màu chủ đạo (độ rộng ∝ tỉ lệ pixel)', size=8)
''')

md(r'''
### 22.2. Watershed — tách các đối tượng chạm nhau

Hình dung ảnh xám như **địa hình**: nơi sáng là đỉnh núi, nơi tối là thung lũng. Đổ nước từ các **điểm đánh dấu (markers)**; nơi nước từ các nguồn khác nhau gặp nhau ⇒ dựng **đập** = đường biên phân vùng.

Quy trình chuẩn với OpenCV:
1. Nhị phân hóa (Otsu) + opening khử nhiễu.
2. **Sure background**: dilate vùng đối tượng.
3. **Sure foreground**: **distance transform** (khoảng cách tới nền gần nhất) rồi ngưỡng ⇒ "lõi" của từng đối tượng.
4. **Unknown** = sure_bg − sure_fg.
5. Gán nhãn các lõi bằng `connectedComponents` ⇒ markers; vùng unknown = 0.
6. `cv2.watershed(img_color, markers)` ⇒ biên được đánh dấu **−1**.
''')

code(r'''
rng = np.random.default_rng(2)
coins = np.full((300, 420), 40, np.uint8)
for (x, y, r) in [(80, 100, 45), (150, 110, 45), (115, 175, 40), (260, 120, 50), (320, 170, 40), (300, 240, 35)]:
    cv2.circle(coins, (x, y), r, 190, -1)
coins = np.clip(coins + rng.normal(0, 12, coins.shape), 0, 255).astype(np.uint8)
coins_bgr = cv2.cvtColor(coins, cv2.COLOR_GRAY2BGR)

_, thresh = cv2.threshold(cv2.GaussianBlur(coins, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.6 * dist.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)
unknown = cv2.subtract(sure_bg, sure_fg)

n_markers, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1            # nền chắc chắn = 1 (không phải 0)
markers[unknown == 255] = 0      # vùng chưa biết = 0 -> watershed sẽ quyết định
markers = cv2.watershed(coins_bgr, markers)

result = coins_bgr.copy()
result[markers == -1] = (0, 0, 255)                 # đường biên
n_obj = len(np.unique(markers)) - 2                 # trừ nhãn -1 (biên) và 1 (nền)
for lbl in range(2, markers.max() + 1):
    ys, xs = np.where(markers == lbl)
    cv2.putText(result, str(lbl - 1), (int(xs.mean()) - 8, int(ys.mean()) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

n_cc, _ = cv2.connectedComponents(opening)
print(f'connectedComponents trực tiếp: {n_cc - 1} vùng (sai vì đồng xu chạm nhau)')
print(f'Watershed: {n_obj} đối tượng')
show([coins, opening, cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
      sure_fg, unknown, result],
     ['Ảnh đồng xu chạm nhau', '1. Otsu + opening', '3. Distance transform',
      '3. Sure foreground (lõi)', '4. Unknown', f'6. Watershed: {n_obj} đối tượng'], cols=3, size=4)
''')

md(r'''
### 22.3. GrabCut — tách tiền cảnh tương tác

Chỉ cần vẽ **một hình chữ nhật bao quanh đối tượng**; GrabCut dùng mô hình **Gaussian Mixture** cho màu tiền cảnh/hậu cảnh + **graph cut** để tìm biên tối ưu, lặp nhiều lần.

Mask kết quả có 4 giá trị: `GC_BGD=0` (chắc chắn nền), `GC_FGD=1` (chắc chắn đối tượng), `GC_PR_BGD=2` (có thể nền), `GC_PR_FGD=3` (có thể đối tượng).
''')

code(r'''
mask_gc = np.zeros(messi.shape[:2], np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
rect = (50, 50, 450, 290)                       # (x, y, w, h) bao quanh cầu thủ

t = time.perf_counter()
cv2.grabCut(messi, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
print(f'GrabCut 5 vòng lặp: {time.perf_counter() - t:.2f} s')
fg_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

vis = messi.copy()
cv2.rectangle(vis, rect[:2], (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
cut = cv2.bitwise_and(messi, messi, mask=fg_mask)
show([vis, (mask_gc * 85).astype(np.uint8), cut],
     ['Hình chữ nhật khởi tạo', 'Mask (0 nền, 1 FG, 2 PR_BG, 3 PR_FG)', 'Tiền cảnh tách được'], size=4.5)
''')

md(r'''
🔍 **Nhận xét:** Kết quả có thể còn sót nền (cỏ) hoặc mất một phần đối tượng. Trong ứng dụng thực tế, người dùng sẽ **tô thêm nét** đánh dấu FG/BG lên mask rồi gọi lại `grabCut` với `cv2.GC_INIT_WITH_MASK` để tinh chỉnh.
''')

# =====================================================================
md(r'''
---
## 23. Phục hồi ảnh (Inpainting)

`cv2.inpaint(img, mask, inpaintRadius, method)` "lấp" các pixel trong `mask` (≠ 0) bằng thông tin từ vùng xung quanh — dùng để **xóa vết xước, chữ, watermark, vật thể nhỏ**.

- `cv2.INPAINT_TELEA`: Fast Marching Method — lấp dần từ biên vào trong.
- `cv2.INPAINT_NS`: dựa trên phương trình Navier–Stokes (động lực học chất lỏng) — lan truyền theo đường đẳng sáng.
''')

code(r'''
rng = np.random.default_rng(4)
damaged = img.copy()
dmask = np.zeros(img.shape[:2], np.uint8)
for _ in range(12):                                              # vết xước ngẫu nhiên
    p1 = tuple(int(v) for v in rng.integers(0, [img.shape[1], img.shape[0]]))
    p2 = tuple(int(v) for v in rng.integers(0, [img.shape[1], img.shape[0]]))
    cv2.line(damaged, p1, p2, (255, 255, 255), 3)
    cv2.line(dmask, p1, p2, 255, 3)
cv2.putText(damaged, 'COPYRIGHT 2026', (30, 260), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 4)
cv2.putText(dmask, 'COPYRIGHT 2026', (30, 260), cv2.FONT_HERSHEY_DUPLEX, 1.6, 255, 4)
dmask = cv2.dilate(dmask, np.ones((3, 3), np.uint8))           # mở rộng mask một chút cho chắc

r_telea = cv2.inpaint(damaged, dmask, 3, cv2.INPAINT_TELEA)
r_ns = cv2.inpaint(damaged, dmask, 3, cv2.INPAINT_NS)
print(f'PSNR ảnh hỏng : {cv2.PSNR(img, damaged):.2f} dB')
print(f'PSNR TELEA    : {cv2.PSNR(img, r_telea):.2f} dB')
print(f'PSNR NS       : {cv2.PSNR(img, r_ns):.2f} dB')
show([damaged, dmask, r_telea, r_ns], ['Ảnh bị hỏng', 'Mask vùng hỏng', 'INPAINT_TELEA', 'INPAINT_NS'], cols=2, size=5.5)
''')

md(r'''
🔍 **Nhận xét:** Inpainting cổ điển hoạt động tốt với vùng hỏng **mảnh** (vết xước, chữ). Với vùng lớn (xóa cả một người) nó sẽ để lại vệt nhòe — khi đó cần phương pháp học sâu (LaMa, Stable Diffusion inpainting…).

**Mẹo:** kết hợp với mục 4.3/10 để **tự động tạo mask** — ví dụ ngưỡng các pixel rất trắng để xóa chữ watermark trắng.
''')

# =====================================================================
md(r'''
---
## 24. 🚀 Dự án tổng hợp: Document Scanner (máy quét tài liệu)

Kết hợp gần như mọi kỹ thuật đã học để biến **ảnh chụp nghiêng một tờ giấy** thành **bản scan phẳng, đen trắng rõ nét** (giống CamScanner):

```
Ảnh chụp → Xám → Làm mờ → Canny → Dilate → findContours → Contour 4 đỉnh lớn nhất
        → Sắp xếp 4 góc → getPerspectiveTransform → warpPerspective → Adaptive threshold
```

Vì không có ảnh chụp thật, trước hết ta **tự tạo ảnh chụp giả lập**: một hóa đơn được đặt nghiêng trên mặt bàn gỗ, có bóng đổ và ánh sáng không đều.
''')

code(r'''
# ---------- Tạo ảnh "chụp" giả lập ----------
rng = np.random.default_rng(8)
doc = np.full((560, 400, 3), 245, np.uint8)
cv2.putText(doc, 'HOA DON BAN HANG', (45, 55), cv2.FONT_HERSHEY_DUPLEX, 1.0, (20, 20, 20), 2)
cv2.line(doc, (30, 75), (370, 75), (40, 40, 40), 2)
total = 0
for i in range(12):
    price = int(rng.integers(10, 500)); total += price
    cv2.putText(doc, f'San pham {i + 1:02d}', (30, 110 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    cv2.putText(doc, f'{price:>4d}.000', (280, 110 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
cv2.line(doc, (30, 470), (370, 470), (40, 40, 40), 2)
cv2.putText(doc, f'TONG: {total}.000 VND', (60, 515), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 160), 2)

H0, W0 = 600, 800
wood = rng.normal(0, 1, (H0, W0)).astype(np.float32)
wood = cv2.GaussianBlur(wood, (0, 0), sigmaX=40, sigmaY=2)                    # vân gỗ kéo dài theo ngang
wood = cv2.normalize(wood, None, 0, 1, cv2.NORM_MINMAX)
table = np.dstack([40 + 40 * wood, 70 + 50 * wood, 110 + 60 * wood]).astype(np.uint8)

src_c = np.float32([[0, 0], [399, 0], [399, 559], [0, 559]])
dst_c = np.float32([[255, 70], [590, 120], [545, 560], [160, 505]])
Mdoc = cv2.getPerspectiveTransform(src_c, dst_c)
warped_doc = cv2.warpPerspective(doc, Mdoc, (W0, H0))
paper_mask = cv2.warpPerspective(np.full(doc.shape[:2], 255, np.uint8), Mdoc, (W0, H0))

shadow = cv2.GaussianBlur(cv2.dilate(paper_mask, np.ones((15, 15), np.uint8)), (41, 41), 0)
shadow = np.roll(shadow, (12, 12), axis=(0, 1))
photo = (table * (1 - 0.5 * shadow[..., None] / 255)).astype(np.uint8)       # bóng đổ của tờ giấy
photo = np.where(paper_mask[..., None] > 0, warped_doc, photo)
light = np.tile(np.linspace(0.6, 1.1, W0), (H0, 1))[..., None]                # ánh sáng không đều
photo = np.clip(photo * light + rng.normal(0, 4, photo.shape), 0, 255).astype(np.uint8)
show(photo, 'Ảnh chụp giả lập: hóa đơn nằm nghiêng trên bàn gỗ', size=7)
''')

code(r'''
# ---------- PIPELINE DOCUMENT SCANNER ----------
def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)              # x + y: nhỏ nhất = TL, lớn nhất = BR
    d = np.diff(pts, axis=1).ravel() # y - x: nhỏ nhất = TR, lớn nhất = BL
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def four_point_transform(image, pts):
    tl, tr, br, bl = rect = order_points(pts)
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))


def scan_document(image, debug=False):
    steps = {}
    gray_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);               steps['1. Xám'] = gray_
    blur_ = cv2.GaussianBlur(gray_, (5, 5), 0)
    edges_ = cv2.Canny(blur_, 50, 150);                             steps['2. Canny'] = edges_
    edges_ = cv2.dilate(edges_, np.ones((3, 3), np.uint8), iterations=2)   # nối các đoạn cạnh đứt
    steps['3. Dilate'] = edges_
    cnts, _ = cv2.findContours(edges_, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]      # các contour lớn nhất
    doc_cnt = None
    for c in cnts:
        if cv2.contourArea(c) < 0.08 * image.shape[0] * image.shape[1]:
            break
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                doc_cnt = approx
                break
        if doc_cnt is not None:
            break
    if doc_cnt is None:
        doc_cnt = cv2.boxPoints(cv2.minAreaRect(cnts[0])).astype(np.int32).reshape(-1, 1, 2)
    vis_ = image.copy()
    cv2.drawContours(vis_, [doc_cnt], -1, (0, 255, 0), 3)
    for i, p in enumerate(order_points(doc_cnt)):
        cv2.circle(vis_, tuple(int(v) for v in p), 10, (0, 0, 255), -1)
        cv2.putText(vis_, ['TL', 'TR', 'BR', 'BL'][i], tuple(int(v) + 12 for v in p), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    steps['4. Contour 4 góc'] = vis_
    flat = four_point_transform(image, doc_cnt);                    steps['5. Warp phối cảnh'] = flat
    flat_gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(flat_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
    steps['6. Adaptive threshold'] = scanned
    return scanned, flat, steps


scanned, flat, steps = scan_document(photo)
show(list(steps.values()), list(steps.keys()), cols=3, size=4.5)
cv2.imwrite(str(OUT_DIR / 'scanned_document.png'), scanned)
print('Đã lưu kết quả:', OUT_DIR / 'scanned_document.png', '| kích thước', scanned.shape)
''')

code(r'''
show([photo, flat, scanned], ['Ảnh chụp', 'Đã làm phẳng (màu)', 'Bản scan cuối cùng'], size=5)
''')

md(r'''
🔍 **Phân tích pipeline:**

| Bước | Kỹ thuật (mục) | Vì sao cần |
|---|---|---|
| Xám + Gaussian | 4, 11 | giảm nhiễu, vân gỗ để Canny không bắt cạnh vụn |
| Canny | 13 | tìm biên tờ giấy |
| Dilate | 14 | nối các đoạn biên bị đứt (do bóng, ánh sáng) |
| findContours + sort theo diện tích | 15 | tờ giấy thường là vật lớn nhất |
| approxPolyDP → 4 đỉnh | 15.2 | xác nhận đó là tứ giác |
| order_points + warpPerspective | 7.4 | "nhìn thẳng" tài liệu |
| Adaptive threshold | 10.3 | khử ánh sáng không đều, cho bản đen trắng sạch |

✍️ **Mở rộng:** (1) Áp dụng `scan_document` cho ảnh `sudoku.png` để tự tìm 4 góc thay vì nhập tay ở mục 7.4. (2) Thêm bước khử bóng bằng *chia cho nền*: `cv2.divide(gray, cv2.medianBlur(cv2.dilate(gray, np.ones((7,7))), 21), scale=255)`.
''')

code(r'''
# Mở rộng (1): tự động tìm 4 góc sudoku bằng chính pipeline trên
sud_scan, sud_flat, sud_steps = scan_document(sudoku)
show([sud_steps['4. Contour 4 góc'], sud_flat, sud_scan], ['Tự tìm 4 góc', 'Warp', 'Scan'], size=4.5)
''')

# =====================================================================
md(r'''
---
## 25. Tra cứu nhanh, lỗi hay gặp, bài tập & lời giải

### 25.1. Bảng tra cứu hàm

| Nhóm | Hàm | Ghi nhớ |
|---|---|---|
| I/O | `imread`, `imwrite`, `imdecode`, `imencode` | `imread` trả `None` nếu lỗi |
| Màu | `cvtColor(img, COLOR_BGR2GRAY / HSV / LAB / YCrCb)` | H ∈ [0, 179] |
| Mask màu | `inRange(hsv, lo, hi)` | đỏ cần 2 khoảng |
| Số học | `add`, `subtract`, `multiply`, `addWeighted`, `absdiff` | bão hòa, không tràn |
| Bitwise | `bitwise_and/or/xor/not(..., mask=)` | |
| Hình học | `resize`, `flip`, `rotate`, `warpAffine`, `warpPerspective`, `remap` | `dsize = (W, H)` |
| Ma trận | `getRotationMatrix2D`, `getAffineTransform` (3 điểm), `getPerspectiveTransform` (4 điểm) | |
| Điểm ảnh | `convertScaleAbs`, `LUT`, `normalize` | |
| Histogram | `calcHist`, `equalizeHist`, `createCLAHE`, `compareHist` | ảnh màu: xử lý kênh sáng |
| Ngưỡng | `threshold` (+`THRESH_OTSU`), `adaptiveThreshold` | blockSize lẻ |
| Lọc | `filter2D`, `blur`, `GaussianBlur`, `medianBlur`, `bilateralFilter`, `fastNlMeansDenoisingColored` | ksize lẻ |
| Cạnh | `Sobel`, `Scharr`, `Laplacian`, `Canny` | dùng `CV_64F` cho đạo hàm |
| Hình thái | `getStructuringElement`, `erode`, `dilate`, `morphologyEx` | đối tượng màu trắng |
| Contour | `findContours`, `drawContours`, `contourArea`, `arcLength`, `moments`, `boundingRect`, `minAreaRect`, `approxPolyDP`, `convexHull`, `matchShapes` | |
| Vùng | `connectedComponentsWithStats`, `distanceTransform` | nhãn 0 = nền |
| Hough | `HoughLines`, `HoughLinesP`, `HoughCircles` | đầu vào là cạnh / ảnh xám mờ |
| Mẫu | `matchTemplate` + `minMaxLoc` | SQDIFF: lấy min |
| Đặc trưng | `cornerHarris`, `goodFeaturesToTrack`, `ORB_create`, `SIFT_create`, `BFMatcher` | ORB → Hamming |
| Pyramid | `pyrDown`, `pyrUp` | |
| Phân vùng | `kmeans`, `watershed`, `grabCut` | |
| Phục hồi | `inpaint` | mask ≠ 0 là vùng cần lấp |

### 25.2. Những lỗi người mới hay gặp

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| `AttributeError: 'NoneType' object has no attribute 'shape'` | `imread` đọc thất bại (sai đường dẫn / đường dẫn tiếng Việt) | kiểm tra `img is None`, dùng `imread_unicode` |
| Màu hiển thị bị đỏ ↔ xanh | đưa BGR vào Matplotlib | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| Ảnh sáng bỗng có đốm đen | tràn số `uint8` khi cộng NumPy | dùng `cv2.add` hoặc ép kiểu + `np.clip` |
| `error: (-215:Assertion failed) ... ksize.width % 2 == 1` | kích thước kernel chẵn | dùng số lẻ: 3, 5, 7… |
| `resize` ra ảnh bị xoay chiều | truyền `(H, W)` thay vì `(W, H)` | `cv2.resize(img, (width, height))` |
| Pixel truy cập sai vị trí | nhầm `img[x, y]` | đúng là `img[y, x]` |
| `findContours` tìm ra 1 contour bao cả ảnh | đối tượng đen trên nền trắng | dùng `THRESH_BINARY_INV` để đối tượng thành màu trắng |
| `findContours` báo lỗi kiểu dữ liệu | ảnh không phải `uint8` 1 kênh | chuyển xám + ngưỡng trước |
| Sobel mất một nửa cạnh | `ddepth=CV_8U` cắt giá trị âm | `CV_64F` rồi `convertScaleAbs` |
| Ảnh float hiển thị toàn trắng | float phải nằm trong [0, 1] với `imshow` | `normalize` hoặc ép về `uint8` |
| Sửa ROI làm hỏng ảnh gốc | slicing NumPy là view | `.copy()` |
| `cv2.imshow` treo notebook | GUI cửa sổ không hợp Jupyter | dùng Matplotlib |
| `ValueError: not enough values to unpack` với `findContours` | khác phiên bản (OpenCV 3 trả 3 giá trị) | `cnts = cv2.findContours(...)[-2]` |
''')

md(r'''
### 25.3. Bài tập tổng hợp

1. **Đếm kẹo theo màu:** Với `smarties.png`, đếm có bao nhiêu viên kẹo mỗi màu (gợi ý: HoughCircles để lấy vị trí, lấy giá trị H trung bình trong mỗi vòng tròn).
2. **Phát hiện điểm khác nhau:** Tạo 2 ảnh gần giống nhau (sửa vài chỗ) và khoanh tròn các điểm khác bằng `absdiff` + threshold + contour.
3. **Bộ lọc "hoạt hình" (cartoon):** bilateral filter nhiều lần + cạnh từ adaptive threshold + `bitwise_and`.
4. **Đo kích thước vật thể:** Biết một vật tham chiếu có chiều rộng thật 2 cm, hãy tính kích thước các vật còn lại trong ảnh hình học ở mục 15 (pixel → cm).
5. **Làm mờ biển số / khuôn mặt:** cho một hình chữ nhật, áp dụng hiệu ứng pixelate (resize nhỏ bằng `INTER_LINEAR` rồi phóng lại bằng `INTER_NEAREST`).
6. **Xoay thẳng văn bản (deskew):** tạo ảnh chữ bị nghiêng, dùng `minAreaRect` trên các pixel chữ để tìm góc nghiêng và xoay lại.

### 25.4. Lời giải gợi ý cho các bài tập nhỏ

<details><summary><b>Bài 3, 4, 10, 13 (nhấn để mở)</b></summary>

```python
# Bài 3.2: nửa trái gốc, nửa phải âm bản
out = img.copy(); w = img.shape[1]
out[:, w // 2:] = 255 - out[:, w // 2:]

# Bài 4: kẹo xanh lam trong smarties
hsv = cv2.cvtColor(smarties, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (95, 120, 60), (130, 255, 255))

# Bài 10: Otsu trên kênh S
s = cv2.cvtColor(smarties, cv2.COLOR_BGR2HSV)[..., 1]
_, m = cv2.threshold(cv2.GaussianBlur(s, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Bài 13: ảnh phác thảo bút chì
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(255 - g, (21, 21), 0)
sketch = cv2.divide(g, 255 - blur, scale=256)
```
</details>

Ô dưới đây chạy lời giải cho một số bài để bạn đối chiếu:
''')

code(r'''
# --- Bài 4 & 10 ---
sm_hsv = cv2.cvtColor(smarties, cv2.COLOR_BGR2HSV)
blue_mask = cv2.inRange(sm_hsv, np.array((95, 120, 60)), np.array((130, 255, 255)))
_, s_otsu = cv2.threshold(cv2.GaussianBlur(sm_hsv[..., 1], (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- Bài 13: phác thảo bút chì ---
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sketch = cv2.divide(g, 255 - cv2.GaussianBlur(255 - g, (21, 21), 0), scale=256)

# --- Bài tổng hợp 3: cartoon ---
color_c = img.copy()
for _ in range(5):
    color_c = cv2.bilateralFilter(color_c, 9, 50, 50)
edges_c = cv2.adaptiveThreshold(cv2.medianBlur(g, 7), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 4)
cartoon = cv2.bitwise_and(color_c, color_c, mask=edges_c)

# --- Bài tổng hợp 5: pixelate một vùng ---
pix = img.copy()
x, y, w_, h_ = 250, 40, 220, 160
small_roi = cv2.resize(pix[y:y+h_, x:x+w_], (w_ // 12, h_ // 12), interpolation=cv2.INTER_LINEAR)
pix[y:y+h_, x:x+w_] = cv2.resize(small_roi, (w_, h_), interpolation=cv2.INTER_NEAREST)

show([cv2.bitwise_and(smarties, smarties, mask=blue_mask), s_otsu, sketch, cartoon, pix],
     ['Bài 4: kẹo xanh lam', 'Bài 10: Otsu kênh S', 'Bài 13: phác thảo bút chì', 'Cartoon', 'Pixelate vùng'],
     cols=5, size=3.2)
''')

code(r'''
# --- Bài tổng hợp 1: đếm kẹo theo màu ---
circles = cv2.HoughCircles(sm_gray, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=1, maxRadius=40)
circles = np.round(circles.reshape(-1, 3)).astype(int)


def hue_name(hv, sv):
    if sv < 60:   return 'xam/trang'
    if hv < 8 or hv >= 165: return 'do'
    if hv < 22:   return 'cam'
    if hv < 35:   return 'vang'
    if hv < 85:   return 'luc'
    if hv < 130:  return 'lam'
    return 'tim/hong'


vis = smarties.copy()
counts = {}
for x, y, r in circles:
    m = np.zeros(sm_gray.shape, np.uint8)
    cv2.circle(m, (x, y), max(r - 6, 3), 255, -1)
    hv = cv2.mean(sm_hsv, mask=m)[:2]
    # Hue là góc: lấy trung vị thay vì trung bình để tránh lỗi quấn vòng ở màu đỏ
    hue_med = int(np.median(sm_hsv[..., 0][m > 0]))
    name = hue_name(hue_med, hv[1])
    counts[name] = counts.get(name, 0) + 1
    cv2.circle(vis, (x, y), r, (255, 255, 255), 2)
    cv2.putText(vis, name, (x - 20, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
print('Tổng số kẹo:', len(circles))
print('Theo màu   :', counts)

# --- Bài tổng hợp 6: deskew ---
txt = np.zeros((220, 520), np.uint8)
cv2.putText(txt, 'Deskew me!', (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 6)
skewed = cv2.warpAffine(txt, cv2.getRotationMatrix2D((260, 110), 12, 1.0), (520, 220))
pts = np.column_stack(np.where(skewed > 0))[:, ::-1].astype(np.float32)   # (x, y)
(cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
if rw < rh:          # chuẩn hóa góc: lấy góc của cạnh dài
    ang = ang - 90
print(f'Góc nghiêng ước lượng: {ang:.2f}°')
deskewed = cv2.warpAffine(skewed, cv2.getRotationMatrix2D((cx, cy), ang, 1.0), (520, 220))
show([vis, skewed, deskewed], ['Bài 1: phân loại màu kẹo', 'Bài 6: chữ nghiêng 12°', 'Sau deskew'], size=4.5)
''')

md(r'''
---
## 🎓 Tổng kết & lộ trình học tiếp

Bạn đã đi qua toàn bộ các thao tác xử lý **một bức ảnh** với OpenCV:

```mermaid
flowchart LR
  A[Đọc ảnh] --> B[Tiền xử lý: màu, resize, khử nhiễu, cân bằng sáng]
  B --> C[Trích xuất: ngưỡng, cạnh, hình thái học]
  C --> D[Phân tích: contour, components, Hough, đặc trưng]
  D --> E[Ứng dụng: đo, đếm, nhận dạng, scan, phục hồi]
```

**Lộ trình tiếp theo:**
1. **Video & camera:** `cv2.VideoCapture`, trừ nền (`createBackgroundSubtractorMOG2`), optical flow, tracking.
2. **Hình học đa ảnh:** homography + RANSAC (`findHomography`), ghép panorama, hiệu chuẩn camera (`calibrateCamera`).
3. **Học máy & học sâu với OpenCV:** module `cv2.dnn` để chạy model YOLO / ONNX, Haar cascade, HOG + SVM.
4. **Tài liệu:** [OpenCV-Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html), sách *Digital Image Processing* (Gonzalez & Woods), *Learning OpenCV* (Bradski & Kaehler).

> 💡 Cách tốt nhất để nhớ: chọn một ảnh **của chính bạn** (ảnh điện thoại) và chạy lại toàn bộ notebook với ảnh đó, chỉnh lại tham số cho phù hợp. Chúc bạn học tốt! 🚀
''')
