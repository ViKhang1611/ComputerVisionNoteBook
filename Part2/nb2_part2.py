from nb2_part1 import md, code  # noqa: F401  (shares the same CELLS list)

# =====================================================================
md(r'''
---
# PHẦN II — HÌNH HỌC ĐA ẢNH, 3D & NHIẾP ẢNH TÍNH TOÁN

## 8. Khớp đặc trưng nâng cao: Lowe's Ratio Test + Homography & RANSAC

Ở Phần 1 (mục 19.3), ta đã biết dùng `BFMatcher(crossCheck=True)` để nối các điểm đặc trưng ORB/SIFT giữa 2 ảnh. Tuy nhiên trong thực tế, **nhiều cặp khớp sẽ bị sai (outliers)** do các chi tiết lặp lại (cửa sổ tòa nhà, gạch lát, chữ cái giống nhau).

Làm sao tìm được phép biến đổi hình học chính xác khi có tới 30–50% cặp khớp sai?

### 8.1. Tỉ lệ Lowe (Lowe's Ratio Test — bước lọc 1)
Với mỗi descriptor ở ảnh 1, dùng `knnMatch(des1, des2, k=2)` để tìm **2 ứng viên giống nhất** $m_1, m_2$ ở ảnh 2.
- Nếu điểm khớp là thật và đặc trưng: ứng viên số 1 sẽ tốt vượt trội so với ứng viên số 2 ($d(m_1) \ll d(m_2)$).
- Nếu vùng đó mơ hồ/lặp lại: $d(m_1) \approx d(m_2)$.
- **Quy tắc Lowe:** chỉ giữ lại cặp khớp nếu:
  $$d(m_1) < \tau \cdot d(m_2) \qquad (\tau \in [0.7, 0.75])$$

### 8.2. Homography & Thuật toán RANSAC (bước lọc 2)
Hai ảnh chụp cùng một **bề mặt phẳng** từ hai góc nhìn bất kỳ liên hệ với nhau qua ma trận **Homography** $H_{3\times 3}$ (8 bậc tự do, cần tối thiểu **4 cặp điểm**):

$$s \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

**RANSAC (Random Sample Consensus)** hoạt động trong `cv2.findHomography(pts1, pts2, cv2.RANSAC, reprojThresh)`:
1. Chọn ngẫu nhiên 4 cặp điểm → giải thử ma trận $H$.
2. Đếm xem có bao nhiêu cặp điểm khác thỏa mãn $\|p'_i - H(p_i)\| < \text{reprojThresh}$ (gọi là **inliers**).
3. Lặp lại hàng trăm lần, giữ lại $H$ có nhiều **inliers** nhất, rồi ước lượng lại $H$ trên toàn bộ tập inliers đó!
''')

code(r'''
# Tạo ảnh mẫu "Poster / Bìa sách" phẳng và đặt nó vào một cảnh lộn xộn (xoay, nghiêng phối cảnh + bị che 1 góc)
poster = load_sample('sudoku.png')
poster = cv2.resize(poster, (320, 320))
ph, pw = poster.shape[:2]

scene_bg = cv2.resize(load_sample('building.jpg'), (640, 440))
# Ma trận phối cảnh thật đặt poster vào trong cảnh
gt_corners = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
dst_quad   = np.float32([[190, 60], [510, 35], [560, 360], [150, 390]])
H_true = cv2.getPerspectiveTransform(gt_corners, dst_quad)

warped_poster = cv2.warpPerspective(poster, H_true, (640, 440))
mask_poster   = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), H_true, (640, 440))
cluttered = np.where(mask_poster[..., None] > 0, warped_poster, scene_bg)
# Giả lập vật thể che khuất một góc dưới-phải của poster (occlusion)
cv2.circle(cluttered, (500, 330), 65, (40, 80, 180), -1)
cv2.putText(cluttered, 'OCCLUDED', (440, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# 1. Trích xuất đặc trưng SIFT
sift = cv2.SIFT_create(nfeatures=800)
kp1, des1 = sift.detectAndCompute(cv2.cvtColor(poster, cv2.COLOR_BGR2GRAY), None)
kp2, des2 = sift.detectAndCompute(cv2.cvtColor(cluttered, cv2.COLOR_BGR2GRAY), None)

# 2. Khớp k-NN (k=2) + Lowe's Ratio Test
bf = cv2.BFMatcher(cv2.NORM_L2)
raw_knn = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in raw_knn if m.distance < 0.75 * n.distance]

# 3. Ước lượng Homography bằng RANSAC
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H_est, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
inliers = int(inlier_mask.sum())
print(f'Tổng cặp k-NN: {len(raw_knn)} | Sau Lowe Ratio (0.75): {len(good_matches)} | '
      f'Inliers sau RANSAC: {inliers} ({100*inliers/len(good_matches):.1f}%)')

# 4. Dùng cv2.perspectiveTransform chiếu 4 góc của poster mẫu sang cảnh để khoanh vùng đối tượng
found_quad = cv2.perspectiveTransform(gt_corners.reshape(-1, 1, 2), H_est)
scene_detected = cluttered.copy()
cv2.polylines(scene_detected, [np.int32(found_quad)], True, (0, 255, 0), 3, cv2.LINE_AA)

vis_matches = cv2.drawMatches(
    poster, kp1, cluttered, kp2, good_matches, None,
    matchColor=(0, 255, 0), singlePointColor=None,
    matchesMask=inlier_mask.ravel().tolist(),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
show(vis_matches, 'Định vị đối tượng phẳng bằng SIFT + Lowe Ratio + RANSAC Homography (dù bị nghiêng & che khuất 1 góc)', size=11)
''')

# =====================================================================
md(r'''
---
## 9. Ghép ảnh toàn cảnh (Panorama / Image Stitching)

Khi chụp hai bức ảnh $I_L$ (trái) và $I_R$ (phải) có **vùng chồng lấp (overlap)** khoảng 30–50% khi xoay camera quanh tâm quang học:
1. Tìm điểm đặc trưng và khớp giữa $I_R \to I_L$ để ước lượng ma trận Homography $H$ biến đổi tọa độ của $I_R$ về hệ tọa độ của $I_L$.
2. Uốn ảnh phải sang khung toàn cảnh bằng `cv2.warpPerspective(I_R, H, (W_pano, H_pano))`.
3. **Pha trộn vùng chồng lấp (Feather Blending):** Nếu chỉ chép đè $I_L$ lên ảnh đã warp, đường giáp ranh sẽ lộ rõ khi hai ảnh chênh lệch phơi sáng nhẹ. Dùng **trọng số giảm dần tuyến tính** $\alpha(x) \in [0, 1]$ dọc vùng giao nhau giúp mối ghép hoàn toàn vô hình!
''')

code(r'''
# Tạo 2 ảnh chụp chồng lấp từ một bức ảnh góc rộng (kèm lệch độ sáng nhẹ như thực tế khi khóa AE khác nhau)
base_wide = cv2.resize(load_sample('building.jpg'), (800, 420))
img_left = base_wide[:, :500].copy()

# Ảnh phải lấy phần x=260..800 và bị xoay/nghiêng phối cảnh nhẹ + tối hơn 8%
crop_right = base_wide[:, 260:].copy()
rh, rw = crop_right.shape[:2]
H_cam = np.array([[0.96, -0.03, 15.0],
                  [0.02,  0.98,  8.0],
                  [-0.00008, 0.00002, 1.0]], dtype=np.float64)
img_right = cv2.warpPerspective(crop_right, H_cam, (rw, rh), borderMode=cv2.BORDER_REFLECT)
img_right = cv2.convertScaleAbs(img_right, alpha=0.92, beta=-4)  # chênh sáng nhẹ giữa 2 lần bấm máy

# --- Pipeline ghép Panorama thủ công ---
sift = cv2.SIFT_create()
kpL, desL = sift.detectAndCompute(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY), None)
kpR, desR = sift.detectAndCompute(cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY), None)

matches = [m for m, n in cv2.BFMatcher().knnMatch(desR, desL, k=2) if m.distance < 0.72 * n.distance]
ptsR = np.float32([kpR[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
ptsL = np.float32([kpL[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
H_RL, _ = cv2.findHomography(ptsR, ptsL, cv2.RANSAC, 3.0)

pano_w, pano_h = 820, 420
warped_R = cv2.warpPerspective(img_right, H_RL, (pano_w, pano_h))
mask_R   = cv2.warpPerspective(np.full((rh, rw), 255, np.uint8), H_RL, (pano_w, pano_h))

# 1. Ghép chép đè trực tiếp (thấy đường nối do chênh sáng)
pano_hard = warped_R.copy()
pano_hard[:, :img_left.shape[1]] = img_left

# 2. Ghép mượt bằng Feather Blending trên vùng chồng lấp
canvas_L = np.zeros_like(warped_R)
canvas_L[:, :img_left.shape[1]] = img_left
mask_L = np.zeros((pano_h, pano_w), np.uint8)
mask_L[:, :img_left.shape[1]] = 255

overlap = (mask_L > 0) & (mask_R > 0)
xs_ov = np.where(overlap.any(axis=0))[0]
x_min, x_max = xs_ov[0], xs_ov[-1]

alpha = np.zeros((pano_h, pano_w), np.float32)
alpha[mask_L > 0] = 1.0
ramp = np.linspace(1.0, 0.0, x_max - x_min + 1, dtype=np.float32)
alpha[:, x_min:x_max + 1] = np.where(overlap[:, x_min:x_max + 1], ramp[None, :], alpha[:, x_min:x_max + 1])

pano_feather = (canvas_L.astype(np.float32) * alpha[..., None] +
                warped_R.astype(np.float32) * (1.0 - alpha[..., None])).astype(np.uint8)

# 3. So sánh với lớp cấp cao cv2.Stitcher_create() của OpenCV
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
status_st, pano_cv = stitcher.stitch([img_left, img_right])
print(f'Vùng chồng lấp: cột x = {x_min}..{x_max} | cv2.Stitcher status = {status_st} (0 = OK)')

show([img_left, img_right], ['Ảnh Trái (500×420)', 'Ảnh Phải (bị nghiêng + tối hơn nhẹ)'], cols=2, size=4.8)
show([pano_hard, pano_feather],
     ['Ghép đè trực tiếp (lộ vệt nối dọc ở x=500)', 'Ghép Homography + Feather Blending (mượt liền mạch)'],
     cols=2, size=5.5)
''')

# =====================================================================
md(r'''
---
## 10. Hiệu chuẩn Camera & Khử méo ống kính (Camera Calibration & Undistortion)

Ống kính thực tế (đặc biệt là camera góc rộng, webcam, action cam) luôn gây **méo hình (distortion)** làm các đường thẳng ngoài rìa bị cong thành hình trống (*barrel distortion*, $k_1 < 0$) hoặc hình gối (*pincushion distortion*, $k_1 > 0$).

### 10.1. Mô hình toán học của Camera
1. **Ma trận nội tại (Intrinsic Matrix $K$):**
   $$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
   trong đó $(f_x, f_y)$ là tiêu cự quy đổi ra pixel, $(c_x, c_y)$ là tâm quang học (principal point).
2. **Hệ số méo (Distortion Coefficients):** `dist = [k1, k2, p1, p2, k3]`
   - **Méo xuyên tâm (Radial — $k_1, k_2, k_3$):**
     $$x_{\text{dist}} = x\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
   - **Méo tiếp tuyến (Tangential — $p_1, p_2$):** do thấu kính lắp không hoàn toàn song song với cảm biến.

### 10.2. Quy trình hiệu chuẩn chuẩn bằng bàn cờ (`calibrateCamera`)
1. Chụp 10–20 ảnh một tấm **bàn cờ phẳng** ở nhiều góc nghiêng khác nhau.
2. Tìm tọa độ các góc ô cờ 2D bằng `cv2.findChessboardCorners` + tinh chỉnh dưới pixel `cv2.cornerSubPix`.
3. Ghép với tọa độ 3D chuẩn trên mặt phẳng bàn cờ $(X, Y, 0)$ và gọi `cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)`.
4. Khử méo cho mọi ảnh chụp sau đó bằng `cv2.undistort(img, K, dist)`.
''')

code(r'''
# Tạo dữ liệu mô phỏng thực tế: một bàn cờ có kích thước 7x5 góc trong (8x6 ô vuông)
# được chiếu qua một camera có thông số thật K_true và độ méo ống kính dist_true
pattern_size = (7, 5)  # (số góc trong theo chiều ngang, dọc)
square_size = 30.0     # mỗi ô cờ rộng 30 mm

# Tọa độ 3D của 35 góc trong hệ tọa độ gắn với bàn cờ (Z = 0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

img_w, img_h = 480, 360
K_true = np.array([[420.0,   0.0, 240.0],
                   [  0.0, 420.0, 180.0],
                   [  0.0,   0.0,   1.0]], dtype=np.float64)
dist_true = np.array([-0.28, 0.08, 0.001, -0.001, 0.0], dtype=np.float64)  # méo hình trống mạnh (k1 = -0.28)


def render_chessboard_view(rvec, tvec, K, dist, w=480, h=360):
    """Vẽ ảnh bàn cờ chụp từ góc (rvec, tvec) có áp dụng méo ống kính (K, dist)."""
    # Tạo ảnh bàn cờ phẳng độ phân giải cao
    bw, bh = (pattern_size[0] + 1) * 40, (pattern_size[1] + 1) * 40
    board_img = np.full((bh + 80, bw + 80, 3), 235, np.uint8)
    for r in range(pattern_size[1] + 1):
        for c in range(pattern_size[0] + 1):
            if (r + c) % 2 == 0:
                cv2.rectangle(board_img, (40 + c * 40, 40 + r * 40), (40 + (c + 1) * 40, 40 + (r + 1) * 40), (20, 20, 20), -1)
    # Chiếu 4 góc ngoài của bảng sang ảnh lý tưởng (chưa méo), rồi làm méo ảnh bằng remap
    src4 = np.float32([[40, 40], [40 + bw, 40], [40 + bw, 40 + bh], [40, 40 + bh]])
    obj4 = np.float32([[-square_size, -square_size, 0],
                       [pattern_size[0] * square_size, -square_size, 0],
                       [pattern_size[0] * square_size, pattern_size[1] * square_size, 0],
                       [-square_size, pattern_size[1] * square_size, 0]])
    pts_ideal, _ = cv2.projectPoints(obj4, rvec, tvec, K, np.zeros(5))
    H_b = cv2.getPerspectiveTransform(src4, pts_ideal.reshape(4, 2).astype(np.float32))
    ideal_view = cv2.warpPerspective(board_img, H_b, (w, h), borderValue=(175, 185, 195))

    # Áp dụng méo ống kính: để tạo ảnh bị méo từ ảnh lý tưởng, với mỗi pixel (u_d, v_d) trên ảnh méo,
    # vị trí tương ứng trên ảnh lý tưởng là undistortPoints(u_d, v_d)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts_d = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).reshape(-1, 1, 2)
    pts_u = cv2.undistortPoints(pts_d, K, dist, P=K).reshape(h, w, 2)
    distorted_view = cv2.remap(ideal_view, pts_u[..., 0], pts_u[..., 1], cv2.INTER_LINEAR, borderValue=(175, 185, 195))
    return distorted_view


# Sinh 8 góc chụp bàn cờ khác nhau và chạy findChessboardCorners + calibrateCamera
poses = [
    ([ 0.12,  0.10,  0.05], [-90, -55, 290]),
    ([-0.18,  0.15, -0.08], [-85, -45, 300]),
    ([ 0.20, -0.16,  0.10], [-95, -60, 310]),
    ([-0.14, -0.18, -0.12], [-80, -50, 295]),
    ([ 0.05,  0.22, -0.15], [-95, -55, 305]),
    ([-0.22,  0.04,  0.14], [-88, -48, 285]),
    ([ 0.16,  0.18, -0.06], [-92, -52, 315]),
    ([-0.08, -0.22,  0.09], [-84, -58, 300]),
]

objpoints, imgpoints, views = [], [], []
for rvec_l, tvec_l in poses:
    view = render_chessboard_view(np.float64(rvec_l), np.float64(tvec_l), K_true, dist_true, img_w, img_h)
    views.append(view)
    gray_v = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray_v, pattern_size, None)
    if found:
        corners_sub = cv2.cornerSubPix(
            gray_v, corners, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners_sub)

rms, K_est, dist_est, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img_w, img_h), None, None)
print(f'Số ảnh tìm thấy đủ 35 góc bàn cờ: {len(objpoints)}/{len(poses)}')
print(f'Sai số chiếu lại (Reprojection RMS error): {rms:.4f} pixel')
print('Tiêu cự thật (fx, fy) = (420.0, 420.0) | Ước lượng =', (round(K_est[0, 0], 1), round(K_est[1, 1], 1)))
print('Hệ số méo k1 thật     = -0.280          | Ước lượng =', round(float(dist_est.ravel()[0]), 3))

# Minh họa phát hiện góc và khử méo (undistort) trên ảnh đầu tiên
vis_corners = views[0].copy()
cv2.drawChessboardCorners(vis_corners, pattern_size, imgpoints[0], True)
undistorted = cv2.undistort(views[0], K_est, dist_est)

show([views[0], vis_corners, undistorted],
     ['Ảnh gốc bị méo hình trống (cạnh bàn cờ bị cong)',
      'findChessboardCorners (35 góc nội suy)',
      'Sau cv2.undistort (các đường thẳng đã thẳng tắp!)'],
     cols=3, size=4.2)
''')

# =====================================================================
md(r'''
---
## 11. Ước lượng tư thế 3D (`cv2.solvePnP`) & Thực tế tăng cường (Augmented Reality)

Khi đã biết thông số nội tại camera $(K, \text{dist})$ từ mục 10, chỉ cần **1 bức ảnh** có các điểm 3D đã biết tọa độ (như tấm bàn cờ, mã ArUco, hoặc các góc của vật thể):
- **`cv2.solvePnP(objectPoints_3D, imagePoints_2D, K, dist)`** tìm ngay **vector xoay `rvec`** và **vector tịnh tiến `tvec`** của vật thể so với camera!
- Sau đó dùng **`cv2.projectPoints(pts_3d, rvec, tvec, K, dist)`** để chiếu bất kỳ mô hình 3D ảo nào (hệ trục tọa độ $OXYZ$, khối lập phương, mô hình 3D) lên đúng mặt phẳng vật thể trên ảnh 2D!
''')

code(r'''
# Dùng K_est, dist_est vừa hiệu chuẩn được để ước lượng tư thế 3D trên góc chụp thứ 3
target_view = views[2].copy()
ok, rvec_pnp, tvec_pnp = cv2.solvePnP(objp, imgpoints[2], K_est, dist_est)
R_mat, _ = cv2.Rodrigues(rvec_pnp)  # đổi vector xoay 3x1 sang ma trận xoay 3x3
print('Vector tịnh tiến tvec (mm):', tvec_pnp.ravel().round(1))
print('Ma trận xoay R (3x3):\n', R_mat.round(3))

# 1. Vẽ hệ trục tọa độ 3D OXYZ tại gốc (0, 0, 0) của bàn cờ (chiều dài 3 ô = 90 mm)
# Chú ý: trục Z âm hướng từ mặt bàn cờ nhô lên phía camera
axis_3d = np.float32([[0, 0, 0], [90, 0, 0], [0, 90, 0], [0, 0, -90]])
axis_2d, _ = cv2.projectPoints(axis_3d, rvec_pnp, tvec_pnp, K_est, dist_est)
pts_a = axis_2d.reshape(-1, 2).astype(int)

ar_axes = target_view.copy()
origin = tuple(pts_a[0])
cv2.arrowedLine(ar_axes, origin, tuple(pts_a[1]), (0, 0, 255), 3, tipLength=0.15)    # Trục X: Đỏ
cv2.arrowedLine(ar_axes, origin, tuple(pts_a[2]), (0, 220, 0), 3, tipLength=0.15)    # Trục Y: Xanh lá
cv2.arrowedLine(ar_axes, origin, tuple(pts_a[3]), (255, 80, 0), 3, tipLength=0.15)   # Trục Z: Xanh dương

# 2. Vẽ một khối lập phương AR 3D cạnh 60 mm đặt tại ô cờ (x=60..120, y=30..90, z=0..-60)
cube_3d = np.float32([
    [60, 30,   0], [120, 30,   0], [120, 90,   0], [60, 90,   0],   # 4 đỉnh đáy (trên mặt bàn cờ)
    [60, 30, -60], [120, 30, -60], [120, 90, -60], [60, 90, -60],   # 4 đỉnh nắp (nổi lên 60 mm)
])
cube_2d, _ = cv2.projectPoints(cube_3d, rvec_pnp, tvec_pnp, K_est, dist_est)
cpts = cube_2d.reshape(-1, 2).astype(int)

ar_cube = target_view.copy()
# Tô mặt đáy, vẽ cột đứng và mặt nắp
cv2.drawContours(ar_cube, [cpts[:4]], -1, (0, 200, 0), 2)
for i in range(4):
    cv2.line(ar_cube, tuple(cpts[i]), tuple(cpts[i + 4]), (0, 255, 255), 2)
cv2.drawContours(ar_cube, [cpts[4:]], -1, (0, 0, 255), 3)

show([ar_axes, ar_cube],
     ['solvePnP: Hệ trục 3D (X-Đỏ, Y-Lục, Z-Lam)', 'Thực tế tăng cường (AR): Khối lập phương 3D gắn trên bàn cờ'],
     cols=2, size=5.2)
''')

# =====================================================================
md(r'''
---
## 12. Thị giác lập thể (Stereo Vision) & Bản đồ chiều sâu (Disparity / Depth Map)

Với 1 camera, ta mất thông tin chiều sâu $Z$. Nhưng nếu dùng **2 camera đặt song song cách nhau đoạn $B$ (baseline)** (giống đôi mắt người):
- Một điểm 3D ở độ sâu $Z$ chiếu lên ảnh trái tại hoành độ $x_L$ và ảnh phải tại hoành độ $x_R$.
- Hiệu số $d = x_L - x_R$ gọi là **thị sai (disparity)**. Theo tam giác đồng dạng:

$$Z = \frac{f \cdot B}{d} = \frac{f \cdot B}{x_L - x_R}$$

- **Quy tắc vàng:** Vật càng **gần** camera ($Z$ nhỏ) $\Longrightarrow$ độ lệch vị trí giữa 2 mắt $d = x_L - x_R$ càng **lớn** (sáng trên bản đồ disparity)! Vật ở rất xa ($Z \to \infty$) $\Longrightarrow$ $d \to 0$.

OpenCV cung cấp hai bộ tính Disparity theo khối trượt dọc dòng cực (epipolar line ngang):
- `cv2.StereoBM_create(numDisparities, blockSize)`: Block Matching — cực nhanh, hợp CPU nhúng.
- `cv2.StereoSGBM_create(...)`: Semi-Global Block Matching (Hirschmüller) — thêm ràng buộc mượt toàn cục nên bản đồ chiều sâu **mịn và chính xác hơn nhiều**.

> ⚠️ Giá trị trả về của `compute(left, right)` trong OpenCV là kiểu số nguyên có dấu 16-bit (`int16`) được **nhân sẵn với hệ số 16** (4 bit phần thập phân). Để lấy thị sai thực (theo pixel), luôn chia cho `16.0`: `disp_px = stereo.compute(L, R).astype(np.float32) / 16.0`.
''')

code(r'''
# Tạo một cặp ảnh Stereo (Left - Right) đã hiệu chỉnh song song (rectified)
# gồm 3 vật thể có kết cấu (texture) đặt ở 3 khoảng cách Z khác nhau:
# - Nền tường xa: disparity = 8 px
# - Khối chữ nhật trung gian: disparity = 22 px
# - Hình tròn tiền cảnh rất gần: disparity = 38 px
rng = np.random.default_rng(99)
sh, sw = 280, 420
# Tạo nền ngẫu nhiên giàu texture để thuật toán Stereo khớp chính xác
base_tex = rng.integers(50, 200, (sh, sw + 60), dtype=np.uint8)
base_tex = cv2.GaussianBlur(base_tex, (3, 3), 0)

obj_rect = rng.integers(60, 230, (120, 130), dtype=np.uint8)
obj_rect = cv2.GaussianBlur(obj_rect, (3, 3), 0)
cv2.rectangle(obj_rect, (0, 0), (129, 119), 240, 2)

obj_disk = rng.integers(40, 220, (110, 110), dtype=np.uint8)
obj_disk = cv2.GaussianBlur(obj_disk, (3, 3), 0)
disk_mask = np.zeros((110, 110), np.uint8)
cv2.circle(disk_mask, (55, 55), 52, 255, -1)

left_img  = np.zeros((sh, sw), np.uint8)
right_img = np.zeros((sh, sw), np.uint8)
gt_disp   = np.full((sh, sw), 8.0, np.float32)

# 1. Đặt nền (d = 8 px: ở ảnh trái tại x, ở ảnh phải tại x - 8)
d_bg = 8
left_img[:]  = base_tex[:, d_bg:d_bg + sw]
right_img[:] = base_tex[:, :sw]

# 2. Đặt khối chữ nhật ở khoảng cách trung bình (d = 22 px)
d_mid = 22
rx, ry, rw_, rh_ = 90, 50, 130, 120
left_img[ry:ry + rh_, rx:rx + rw_] = obj_rect
right_img[ry:ry + rh_, rx - d_mid:rx - d_mid + rw_] = obj_rect
gt_disp[ry:ry + rh_, rx:rx + rw_] = d_mid

# 3. Đặt hình tròn ở rất gần camera (d = 38 px)
d_near = 38
cx, cy = 230, 130
roi_L = left_img[cy:cy + 110, cx:cx + 110]
roi_R = right_img[cy:cy + 110, cx - d_near:cx - d_near + 110]
roi_L[disk_mask > 0] = obj_disk[disk_mask > 0]
roi_R[disk_mask > 0] = obj_disk[disk_mask > 0]
gt_disp[cy:cy + 110, cx:cx + 110][disk_mask > 0] = d_near

# Tính Disparity bằng StereoBM và StereoSGBM (numDisparities phải chia hết cho 16)
stereo_bm = cv2.StereoBM_create(numDisparities=48, blockSize=15)
disp_bm = stereo_bm.compute(left_img, right_img).astype(np.float32) / 16.0

stereo_sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=48, blockSize=7,
    P1=8 * 7 * 7, P2=32 * 7 * 7, disp12MaxDiff=1,
    uniquenessRatio=10, speckleWindowSize=50, speckleRange=2
)
disp_sgbm = stereo_sgbm.compute(left_img, right_img).astype(np.float32) / 16.0

# Tính khoảng cách Z (cm) giả sử f = 500 px, baseline B = 10 cm => Z = 5000 / d
valid_sgbm = disp_sgbm > 4
depth_cm = np.where(valid_sgbm, 5000.0 / np.maximum(disp_sgbm, 1.0), 0)
print(f'Thị sai trung vị đo được (SGBM) — Nền: {np.median(disp_sgbm[:, 360:]):.1f} px (thật 8) | '
      f'Khối giữa: {np.median(disp_sgbm[70:140, 110:190]):.1f} px (thật 22) | '
      f'Đĩa gần: {np.median(disp_sgbm[160:210, 260:310]):.1f} px (thật 38)')

show([left_img, right_img, gt_disp, np.clip(disp_bm, 0, 48), np.clip(disp_sgbm, 0, 48)],
     ['Ảnh Trái (Left)', 'Ảnh Phải (Right - vật gần dịch sang trái nhiều hơn)',
      'Ground-truth Disparity', 'StereoBM (nhanh, có nhiễu rìa)', 'StereoSGBM (mịn & chính xác)'],
     cols=3, size=4.0, cmap='plasma')
''')

# =====================================================================
md(r'''
---
## 13. Nhiếp ảnh tính toán (Computational Photography)

### 13.1. Ảnh dải tương phản động cao (HDR & Exposure Fusion — `createMergeMertens`)
Một cảm biến camera thông thường chỉ thu được dải sáng hạn chế trong 1 lần chụp:
- Chụp **thiếu sáng (under-exposed)**: giữ được chi tiết vùng đèn/bầu trời sáng, nhưng vùng tối đen kịt.
- Chụp **dư sáng (over-exposed)**: thấy rõ vùng bóng tối, nhưng vùng sáng bị cháy trắng (bão hòa 255).

**Exposure Fusion (Mertens et al. — `cv2.createMergeMertens`)** trộn trực tiếp chuỗi ảnh nhiều mức phơi sáng dựa trên 3 trọng số tại mỗi pixel: **Độ tương phản (Contrast)**, **Độ bão hòa màu (Saturation)** và **Độ phơi sáng tốt (Well-exposedness ≈ gần 0.5)** trên kim tự tháp Laplacian mà không cần biết thời gian phơi sáng!
''')

code(r'''
base_scene = load_sample('fruits.jpg')
# Tạo cảnh có dải tương phản cực rộng rồi giả lập 3 mức phơi sáng (Thiếu sáng - Trung bình - Dư sáng)
h, w = base_scene.shape[:2]
illum = np.linspace(0.25, 2.2, w, dtype=np.float32)[None, :, None]
hdr_radiance = np.clip(base_scene.astype(np.float32) * illum, 1.0, 600.0)

def simulate_exposure(radiance, exposure_gain, gamma=1/1.8):
    mapped = np.clip((radiance * exposure_gain / 255.0) ** gamma * 255.0, 0, 255)
    return mapped.astype(np.uint8)

exp_low  = simulate_exposure(hdr_radiance, 0.35)   # thiếu sáng: rõ bên phải, tối đen bên trái
exp_mid  = simulate_exposure(hdr_radiance, 0.85)   # trung bình
exp_high = simulate_exposure(hdr_radiance, 2.10)   # dư sáng: rõ bên trái, cháy trắng bên phải

merge_mertens = cv2.createMergeMertens()
fusion_f32 = merge_mertens.process([exp_low, exp_mid, exp_high])
fusion_u8  = np.clip(fusion_f32 * 255, 0, 255).astype(np.uint8)

show([exp_low, exp_mid, exp_high, fusion_u8],
     ['Thiếu sáng (rõ vùng sáng)', 'Phơi sáng vừa', 'Dư sáng (rõ vùng tối, cháy vùng sáng)',
      'Exposure Fusion (Mertens): Rõ cả hai vùng!'],
     cols=4, size=3.4)
''')

md(r'''
### 13.2. Ghép ảnh liền mạch Poisson (`cv2.seamlessClone`)
Khi cắt một vật thể từ ảnh nguồn (`src`) dán sang ảnh đích (`dst`), dù cắt mask chuẩn đến mấy thì màu nền và độ sáng xung quanh vẫn lệch nhau.

`cv2.seamlessClone(src, dst, mask, center, flags)` giải **phương trình đạo hàm riêng Poisson**: giữ nguyên **trường gradient (chi tiết)** của vùng nguồn nhưng ép **điều kiện biên màu sắc** khớp mượt với ảnh đích!
- `cv2.NORMAL_CLONE`: giữ toàn bộ kết cấu bên trong vật thể nguồn.
- `cv2.MIXED_CLONE`: lấy gradient lớn hơn giữa nguồn và đích tại mỗi điểm ⇒ cho phép nền đích xuyên qua các khoảng trống (rất hợp khi ghép chữ, đám mây, vật bán trong suốt).
''')

code(r'''
dst_sky = cv2.resize(load_sample('building.jpg'), (520, 360))
# Tạo vật thể nguồn (mặt trăng / khinh khí cầu có nền khác màu hoàn toàn với bầu trời đích)
src_patch = np.full((110, 110, 3), (60, 160, 240), np.uint8)  # nền cam sáng
cv2.circle(src_patch, (55, 55), 34, (40, 40, 220), -1)
cv2.ellipse(src_patch, (55, 55), (34, 12), 0, 0, 360, (240, 240, 240), 3)
cv2.rectangle(src_patch, (46, 90), (64, 102), (50, 90, 140), -1)

mask_patch = np.full(src_patch.shape[:2], 255, np.uint8)
center_pt = (130, 85)

# 1. Dán trực tiếp (lộ khung chữ nhật màu cam)
naive_paste = dst_sky.copy()
sh_, sw_ = src_patch.shape[:2]
naive_paste[center_pt[1] - sh_//2 : center_pt[1] + sh_//2,
            center_pt[0] - sw_//2 : center_pt[0] + sw_//2] = src_patch

# 2. Seamless Clone (Poisson editing)
clone_normal = cv2.seamlessClone(src_patch, dst_sky, mask_patch, center_pt, cv2.NORMAL_CLONE)
clone_mixed  = cv2.seamlessClone(src_patch, dst_sky, mask_patch, center_pt, cv2.MIXED_CLONE)

show([src_patch, naive_paste, clone_normal, clone_mixed],
     ['Ảnh nguồn (nền cam lệch màu)', 'Dán trực tiếp (lộ viền vuông)',
      'seamlessClone (NORMAL_CLONE)', 'seamlessClone (MIXED_CLONE)'],
     cols=4, size=3.4)
''')

md(r'''
### 13.3. Các bộ lọc Nhiếp ảnh phi chân thực (Non-Photorealistic Rendering — NPR)
OpenCV tích hợp sẵn các hàm xử lý bảo toàn cạnh và tạo hiệu ứng nghệ thuật chỉ bằng 1 dòng code:
- `cv2.edgePreservingFilter(img, flags, sigma_s, sigma_r)`
- `cv2.detailEnhance(img, sigma_s, sigma_r)` — tăng chi tiết cục bộ cực đẹp (HDR-like).
- `cv2.pencilSketch(img, sigma_s, sigma_r, shade_factor)` — trả về `(sketch_gray, sketch_color)`.
- `cv2.stylization(img, sigma_s, sigma_r)` — hiệu ứng tranh màu nước.
''')

code(r'''
sample_npr = load_sample('fruits.jpg')
epf = cv2.edgePreservingFilter(sample_npr, flags=1, sigma_s=60, sigma_r=0.4)
det = cv2.detailEnhance(sample_npr, sigma_s=10, sigma_r=0.15)
sk_gray, sk_col = cv2.pencilSketch(sample_npr, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
styl = cv2.stylization(sample_npr, sigma_s=60, sigma_r=0.45)

show([sample_npr, det, sk_gray, sk_col, styl],
     ['Ảnh gốc', 'detailEnhance (tăng chi tiết)', 'pencilSketch (chì đen trắng)',
      'pencilSketch (chì màu)', 'stylization (màu nước)'],
     cols=5, size=3.1)
''')

md(r'''
✍️ **Bài tập Phần II (Bài 3 & 4):**
3. Từ ma trận Homography `H_est` ở mục 8, hãy dùng `cv2.warpPerspective` với ma trận nghịch đảo `np.linalg.inv(H_est)` để cắt và nắn thẳng lại tấm poster bị nghiêng từ bức ảnh `cluttered`.
4. Trong mục 12 (Stereo Vision), hãy kiểm chứng rằng nếu giảm khoảng cách baseline $B$ đi một nửa thì thị sai $d$ cũng giảm đi một nửa và sai số ước lượng độ sâu ở xa tăng lên.
''')
