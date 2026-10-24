from nb_part1 import md, code  # noqa: F401  (shares the same CELLS list)

# =====================================================================
md(r'''
---
## 9. Histogram và cân bằng histogram

### 9.1. Histogram là gì?

**Histogram** đếm số pixel ở mỗi mức xám 0..255. Nó cho biết "phân bố độ sáng" của ảnh:
- Dồn về **bên trái** → ảnh tối; dồn về **bên phải** → ảnh sáng.
- Dồn vào một **dải hẹp** → tương phản thấp; trải rộng → tương phản cao.

```python
hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
```
''')

code(r'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
print('hist.shape =', hist.shape, '| tổng =', int(hist.sum()), '= số pixel', gray.size)

variants = {'Gốc': gray,
            'Tối (×0.4)': cv2.convertScaleAbs(gray, alpha=0.4),
            'Sáng (+100)': cv2.add(gray, np.full_like(gray, 100)),
            'Tương phản thấp': cv2.convertScaleAbs(gray, alpha=0.3, beta=100)}
fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
for i, (name, im) in enumerate(variants.items()):
    axes[0, i].imshow(im, cmap='gray', vmin=0, vmax=255); axes[0, i].set_title(name); axes[0, i].axis('off')
    plot_hist(im, axes[1, i], f'Histogram - {name}')
plt.tight_layout(); plt.show()
''')

code(r'''
# Histogram màu và histogram có mask (chỉ tính trong 1 vùng)
mask = np.zeros(gray.shape, np.uint8)
mask[30:200, 250:480] = 255                         # vùng chứa chuối
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].imshow(cv2.cvtColor(cv2.bitwise_and(img, img, mask=mask), cv2.COLOR_BGR2RGB)); axes[0].axis('off')
axes[0].set_title('Vùng mask')
plot_hist(img, axes[1], 'Histogram màu toàn ảnh')
plot_hist(img, axes[2], 'Histogram màu trong mask', mask=mask)
plt.tight_layout(); plt.show()
''')

md(r'''
### 9.2. Cân bằng histogram — `cv2.equalizeHist`

Ý tưởng: biến đổi mức xám sao cho histogram **trải đều** hơn, dùng **hàm phân bố tích lũy (CDF)**:

$$O(v) = \text{round}\left(\frac{CDF(v) - CDF_{min}}{N - CDF_{min}} \times 255\right)$$

Kết quả: tăng tương phản toàn cục, tự động, không cần tham số. Chỉ áp dụng cho ảnh **1 kênh**.
''')

code(r'''
low = cv2.convertScaleAbs(gray, alpha=0.4, beta=60)
eq = cv2.equalizeHist(low)


def plot_hist_cdf(im, ax, title):
    hist = cv2.calcHist([im], [0], None, [256], [0, 256]).ravel()
    cdf = hist.cumsum()
    ax.bar(range(256), hist, color='gray', width=1)
    ax2 = ax.twinx(); ax2.plot(cdf / cdf.max(), 'r'); ax2.set_ylim(0, 1.05); ax2.set_ylabel('CDF', color='r')
    ax.set_xlim(0, 255); ax.set_title(title)


fig, axes = plt.subplots(2, 2, figsize=(11, 7))
axes[0, 0].imshow(low, cmap='gray', vmin=0, vmax=255); axes[0, 0].set_title('Trước'); axes[0, 0].axis('off')
axes[0, 1].imshow(eq, cmap='gray', vmin=0, vmax=255); axes[0, 1].set_title('Sau equalizeHist'); axes[0, 1].axis('off')
plot_hist_cdf(low, axes[1, 0], 'Histogram + CDF trước')
plot_hist_cdf(eq, axes[1, 1], 'Histogram + CDF sau (CDF gần đường thẳng)')
plt.tight_layout(); plt.show()
''')

md(r'''
### 9.3. CLAHE — Cân bằng histogram thích nghi có giới hạn

`equalizeHist` dùng **một** phép biến đổi cho cả ảnh ⇒ có thể làm cháy vùng sáng, khuếch đại nhiễu. **CLAHE** (Contrast Limited Adaptive Histogram Equalization):
1. Chia ảnh thành lưới ô nhỏ (`tileGridSize`, vd 8×8).
2. Cân bằng histogram **trong từng ô**.
3. **Cắt** histogram tại `clipLimit` để không khuếch đại nhiễu quá mức.
4. Nội suy song tuyến tính giữa các ô để không thấy đường biên ô.

Rất hay dùng cho ảnh y tế, ảnh chụp ngược sáng, ảnh trước khi nhận dạng.
''')

code(r'''
# Ảnh có chiếu sáng không đều: bên trái tối, bên phải sáng
h, w = gray.shape
shade = np.tile(np.linspace(0.25, 1.2, w), (h, 1))
uneven = np.clip(gray * shade, 0, 255).astype(np.uint8)

outs = [uneven, cv2.equalizeHist(uneven)]
titles = ['Chiếu sáng không đều', 'equalizeHist (toàn cục)']
for clip, tile in [(2.0, 8), (4.0, 8), (2.0, 16)]:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    outs.append(clahe.apply(uneven)); titles.append(f'CLAHE clip={clip}, tile={tile}x{tile}')
show(outs, titles, cols=5, size=3)
''')

md(r'''
### 9.4. Cân bằng histogram cho ảnh màu — làm ĐÚNG cách

❌ **Sai:** cân bằng riêng từng kênh B, G, R ⇒ tỉ lệ giữa các kênh bị phá vỡ ⇒ **lệch màu**.

✅ **Đúng:** chuyển sang không gian tách độ sáng (YCrCb / LAB / HSV), **chỉ cân bằng kênh độ sáng**, rồi chuyển ngược lại.
''')

code(r'''
dark_color = cv2.convertScaleAbs(img, alpha=0.5, beta=10)

# Sai: từng kênh
wrong = cv2.merge([cv2.equalizeHist(ch) for ch in cv2.split(dark_color)])

# Đúng 1: kênh Y trong YCrCb
ycc = cv2.cvtColor(dark_color, cv2.COLOR_BGR2YCrCb)
ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
right_y = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

# Đúng 2: CLAHE trên kênh L của LAB (thường đẹp nhất)
lab = cv2.cvtColor(dark_color, cv2.COLOR_BGR2LAB)
lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
right_lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

show([dark_color, wrong, right_y, right_lab],
     ['Ảnh tối', '❌ equalize từng kênh BGR', '✅ equalize kênh Y (YCrCb)', '✅ CLAHE kênh L (LAB)'], cols=4, size=3.3)
''')

md(r'''
### 9.5. So sánh histogram — `cv2.compareHist`

Có thể dùng histogram như một "chữ ký màu" để so sánh độ giống nhau giữa các ảnh (tìm ảnh tương tự). Phương pháp `HISTCMP_CORREL`: 1 = giống hệt.
''')

code(r'''
def hsv_hist(image):
    hsv_ = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hst = cv2.calcHist([hsv_], [0, 1], None, [50, 60], [0, 180, 0, 256])   # histogram 2D theo H và S
    return cv2.normalize(hst, hst).flatten()


base = hsv_hist(img)
cands = {'fruits sáng hơn': cv2.add(img, np.full_like(img, 40)),
         'fruits lật ngang': cv2.flip(img, 1),
         'building': load_sample('building.jpg'),
         'smarties': load_sample('smarties.png')}
for name, cimg in cands.items():
    print(f'{name:18s}: correlation = {cv2.compareHist(base, hsv_hist(cimg), cv2.HISTCMP_CORREL):.3f}')
''')

md(r'''
🔍 **Nhận xét:** Ảnh lật ngang có histogram **giống hệt** (histogram không quan tâm vị trí pixel) — vừa là ưu điểm (bất biến với xoay/lật) vừa là nhược điểm (hai ảnh khác nhau hoàn toàn vẫn có thể cùng histogram).
''')

# =====================================================================
md(r'''
---
## 10. Phân ngưỡng (Thresholding) — tạo ảnh nhị phân

Phân ngưỡng chuyển ảnh xám thành ảnh **nhị phân** (đen/trắng) — bước tiền xử lý quan trọng cho contour, OCR, đo đạc…

### 10.1. Ngưỡng toàn cục — `cv2.threshold(src, T, maxval, type)`

| Loại | Công thức (pixel `v`, ngưỡng `T`) |
|---|---|
| `THRESH_BINARY` | `maxval` nếu v > T, ngược lại 0 |
| `THRESH_BINARY_INV` | 0 nếu v > T, ngược lại `maxval` |
| `THRESH_TRUNC` | T nếu v > T, ngược lại giữ v |
| `THRESH_TOZERO` | giữ v nếu v > T, ngược lại 0 |
| `THRESH_TOZERO_INV` | 0 nếu v > T, ngược lại giữ v |

Hàm trả về **2 giá trị**: `(ngưỡng_đã_dùng, ảnh_kết_quả)`.
''')

code(r'''
grad = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (60, 1))   # dải gradient đen -> trắng
types = ['THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
fig, axes = plt.subplots(len(types) + 1, 2, figsize=(12, 10), gridspec_kw={'width_ratios': [1, 1.3]})
axes[0, 0].imshow(grad, cmap='gray', vmin=0, vmax=255); axes[0, 0].set_title('Gradient gốc'); axes[0, 0].axis('off')
axes[0, 1].plot(grad[0]); axes[0, 1].set_title('Giá trị pixel theo x'); axes[0, 1].set_ylim(-5, 260)
for i, t in enumerate(types, start=1):
    _, out = cv2.threshold(grad, 127, 255, getattr(cv2, t))
    axes[i, 0].imshow(out, cmap='gray', vmin=0, vmax=255); axes[i, 0].set_title(f'{t} (T=127)'); axes[i, 0].axis('off')
    axes[i, 1].plot(out[0], 'r'); axes[i, 1].set_ylim(-5, 260); axes[i, 1].axvline(127, ls='--', c='gray')
plt.tight_layout(); plt.show()
''')

md(r'''
### 10.2. Ngưỡng Otsu — tự động chọn ngưỡng

Chọn ngưỡng tay rất khó. **Otsu** tự tìm T **tối đa hóa phương sai giữa hai lớp** (nền và đối tượng):

$$T^* = \arg\max_T \; \omega_0(T)\,\omega_1(T)\,[\mu_0(T) - \mu_1(T)]^2$$

Hoạt động tốt khi histogram có **2 đỉnh** (bimodal). Dùng bằng cách cộng cờ: `cv2.THRESH_BINARY + cv2.THRESH_OTSU` (tham số T truyền vào bị bỏ qua, thường để 0).

Ví dụ kinh điển: ảnh nhiễu → Otsu trực tiếp kém; **làm mờ Gauss trước** rồi Otsu → tốt hơn nhiều.
''')

code(r'''
rng = np.random.default_rng(1)
bimodal = np.full((300, 400), 70, np.uint8)
cv2.circle(bimodal, (130, 150), 80, 170, -1)
cv2.rectangle(bimodal, (240, 60), (360, 240), 190, -1)
noisy = np.clip(bimodal + rng.normal(0, 35, bimodal.shape), 0, 255).astype(np.uint8)

_, th_fixed = cv2.threshold(noisy, 127, 255, cv2.THRESH_BINARY)
t_otsu, th_otsu = cv2.threshold(noisy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(noisy, (7, 7), 0)
t_otsu2, th_otsu2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f'Otsu trên ảnh nhiễu chọn T = {t_otsu:.0f}; trên ảnh đã làm mờ chọn T = {t_otsu2:.0f}')

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
rows = [(noisy, th_fixed, 'Ngưỡng cố định T=127', 127),
        (noisy, th_otsu, f'Otsu T={t_otsu:.0f}', t_otsu),
        (blur, th_otsu2, f'Gaussian 7x7 + Otsu T={t_otsu2:.0f}', t_otsu2)]
for i, (src, out, name, t) in enumerate(rows):
    axes[i, 0].imshow(src, cmap='gray', vmin=0, vmax=255); axes[i, 0].set_title('Ảnh vào'); axes[i, 0].axis('off')
    axes[i, 1].hist(src.ravel(), 256, (0, 256), color='gray'); axes[i, 1].axvline(t, c='r', lw=2)
    axes[i, 1].set_title('Histogram + ngưỡng (đỏ)')
    axes[i, 2].imshow(out, cmap='gray'); axes[i, 2].set_title(name); axes[i, 2].axis('off')
plt.tight_layout(); plt.show()
''')

md(r'''
🔍 **Nhận xét:** Sau khi làm mờ, histogram tách thành **hai đỉnh rõ ràng** ⇒ Otsu chọn ngưỡng nằm giữa thung lũng và kết quả sạch hơn nhiều. Bài học: **khử nhiễu trước khi phân ngưỡng**.

### 10.3. Ngưỡng thích nghi — `cv2.adaptiveThreshold`

Khi ánh sáng **không đều** (một bên sáng, một bên tối), một ngưỡng toàn cục không thể đúng cho mọi nơi. Ngưỡng thích nghi tính **ngưỡng riêng cho từng pixel** dựa trên vùng lân cận `blockSize × blockSize`:

$$T(x,y) = \text{mean hoặc gaussian\_mean}(\text{lân cận}) - C$$

```python
cv2.adaptiveThreshold(src, 255, method, type, blockSize, C)
# method: cv2.ADAPTIVE_THRESH_MEAN_C hoặc cv2.ADAPTIVE_THRESH_GAUSSIAN_C
# blockSize: số lẻ (3, 5, 7, ..., 11, 21...)
# C: hằng số trừ đi (tăng C -> ít pixel trắng/nhiễu hơn với THRESH_BINARY)
```
''')

code(r'''
sud_gray = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)
sud_blur = cv2.medianBlur(sud_gray, 5)

_, g1 = cv2.threshold(sud_blur, 127, 255, cv2.THRESH_BINARY)
_, g2 = cv2.threshold(sud_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
a1 = cv2.adaptiveThreshold(sud_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
a2 = cv2.adaptiveThreshold(sud_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
show([sud_gray, g1, g2, a1, a2],
     ['Gốc (ánh sáng không đều)', 'Toàn cục T=127', 'Otsu', 'Adaptive MEAN (11, 2)', 'Adaptive GAUSSIAN (11, 2)'],
     cols=5, size=3.2)

# Ảnh hưởng của blockSize và C
outs, titles = [], []
for bs in [5, 15, 41]:
    for C in [2, 8]:
        outs.append(cv2.adaptiveThreshold(sud_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, C))
        titles.append(f'blockSize={bs}, C={C}')
show(outs, titles, cols=6, size=2.6)
''')

md(r'''
🔍 **Nhận xét:**
- Ngưỡng toàn cục / Otsu làm **mất trắng hoặc đen hẳn** một vùng do bóng đổ.
- Adaptive giữ được nét chữ số và đường kẻ ở **mọi vùng sáng/tối**.
- `blockSize` nhỏ → chi tiết mảnh nhưng nhiễu; lớn → giống ngưỡng toàn cục hơn. `C` lớn → bớt nhiễu nền.

✍️ **Bài tập 10:** Áp dụng ngưỡng Otsu cho kênh S (saturation) của `smarties.png` để tách kẹo khỏi nền.
''')

# =====================================================================
md(r'''
---
## 11. Tích chập và lọc ảnh: làm mịn & khử nhiễu

### 11.1. Tích chập (convolution) là gì?

Ta trượt một **ma trận nhỏ (kernel)** trên ảnh; tại mỗi vị trí, pixel mới = **tổng có trọng số** của các pixel lân cận:

$$g(x,y) = \sum_{i=-k}^{k}\sum_{j=-k}^{k} K(i,j)\, f(x+j,\, y+i)$$

> 📝 Về lý thuyết, tích chập phải *lật* kernel 180°. `cv2.filter2D` thực ra tính **tương quan (correlation)**, không lật kernel. Với kernel đối xứng (blur, Gauss, Laplace) hai cách cho kết quả như nhau.

Ví dụ thủ công với kernel trung bình 3×3:
''')

code(r'''
f = np.array([[10, 10, 10, 10, 10],
              [10, 50, 50, 50, 10],
              [10, 50, 90, 50, 10],
              [10, 50, 50, 50, 10],
              [10, 10, 10, 10, 10]], np.float32)
K = np.ones((3, 3), np.float32) / 9

# Tính tay tại tâm (y=2, x=2)
manual = (f[1:4, 1:4] * K).sum()
print('Vùng 3x3 quanh tâm:\n', f[1:4, 1:4])
print(f'Tính tay tại tâm = tổng(vùng * K) = {manual:.2f}')

out = cv2.filter2D(f, -1, K, borderType=cv2.BORDER_REPLICATE)   # ddepth=-1: cùng kiểu với ảnh vào
print('Kết quả cv2.filter2D:\n', out.round(1))
''')

md(r'''
### 11.2. Các kernel cơ bản và hiệu ứng

Cùng một hàm `cv2.filter2D`, chỉ thay kernel là được nhiều hiệu ứng khác nhau:
''')

code(r'''
kernels = {
    'Identity':        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32),
    'Box blur 5x5':    np.ones((5, 5), np.float32) / 25,
    'Sharpen':         np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),
    'Edge (Laplacian)':np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32),
    'Emboss':          np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32),
    'Motion blur 15':  np.eye(15, dtype=np.float32) / 15,
}
for name, k in list(kernels.items())[:5]:
    print(f'{name}:\n{np.round(k, 3)}\n')
show([cv2.filter2D(img, -1, k) for k in kernels.values()], list(kernels.keys()), cols=3, size=3.8)
''')

md(r'''
### 11.3. Bốn bộ lọc làm mịn chính của OpenCV

| Hàm | Nguyên lý | Ưu điểm | Nhược điểm |
|---|---|---|---|
| `cv2.blur(img, (k,k))` | trung bình đều | nhanh nhất | mờ cả cạnh, tạo "khối" |
| `cv2.GaussianBlur(img, (k,k), σ)` | trung bình theo trọng số Gauss (gần tâm nặng hơn) | tự nhiên, khử **nhiễu Gauss** tốt | vẫn mờ cạnh |
| `cv2.medianBlur(img, k)` | lấy **trung vị** | khử **nhiễu muối tiêu** cực tốt, giữ cạnh | chậm hơn với k lớn |
| `cv2.bilateralFilter(img, d, σColor, σSpace)` | Gauss theo **khoảng cách** × Gauss theo **chênh lệch màu** | **giữ cạnh** rất tốt | chậm |

> ⚠️ Kích thước kernel `k` phải là **số lẻ** (3, 5, 7…) để có pixel tâm. Trong `GaussianBlur`, σ = 0 nghĩa là tự tính từ k.

Hệ số kernel Gauss 1D (5 phần tử):
''')

code(r'''
g1d = cv2.getGaussianKernel(5, 1.0)
print('Gaussian 1D (k=5, σ=1):', g1d.ravel().round(4))
print('Gaussian 2D = g1d · g1dᵀ:\n', (g1d @ g1d.T).round(4))

# Tạo 2 loại nhiễu
def add_gaussian_noise(image, sigma=20, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(image.astype(np.float32) + rng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)


def add_salt_pepper(image, amount=0.05, seed=0):
    rng = np.random.default_rng(seed)
    out = image.copy()
    m = rng.random(image.shape[:2])
    out[m < amount / 2] = 0          # tiêu (đen)
    out[m > 1 - amount / 2] = 255    # muối (trắng)
    return out


clean = img.copy()
n_gauss = add_gaussian_noise(clean, 25)
n_sp = add_salt_pepper(clean, 0.08)

filters = {
    'blur 5x5':        lambda x: cv2.blur(x, (5, 5)),
    'Gaussian 5x5':    lambda x: cv2.GaussianBlur(x, (5, 5), 0),
    'median 5':        lambda x: cv2.medianBlur(x, 5),
    'bilateral 9,75,75': lambda x: cv2.bilateralFilter(x, 9, 75, 75),
}
print(f'\n{"Bộ lọc":18s} | PSNR nhiễu Gauss | PSNR nhiễu muối tiêu')
print(f'{"(chưa lọc)":18s} | {cv2.PSNR(clean, n_gauss):10.2f} dB   | {cv2.PSNR(clean, n_sp):10.2f} dB')
res_g, res_sp = [], []
for name, fn in filters.items():
    a, b = fn(n_gauss), fn(n_sp)
    res_g.append(a); res_sp.append(b)
    print(f'{name:18s} | {cv2.PSNR(clean, a):10.2f} dB   | {cv2.PSNR(clean, b):10.2f} dB')
''')

code(r'''
c = (slice(120, 320), slice(180, 380))     # phóng to một vùng để dễ so sánh
show([n_gauss[c]] + [r[c] for r in res_g], ['Nhiễu Gauss σ=25'] + list(filters.keys()), cols=5, size=3)
show([n_sp[c]] + [r[c] for r in res_sp], ['Nhiễu muối tiêu 8%'] + list(filters.keys()), cols=5, size=3)
''')

md(r'''
🔍 **Nhận xét:**
- **Nhiễu muối tiêu** → `medianBlur` thắng tuyệt đối: các điểm trắng/đen cực trị không bao giờ là trung vị nên bị loại hoàn toàn. Lọc trung bình chỉ "trải" chúng ra thành vệt mờ.
- **Nhiễu Gauss** → Gaussian/bilateral tốt hơn.

### 11.4. Bilateral filter giữ cạnh như thế nào?

Ta vẽ **mặt cắt cường độ** qua một cạnh để thấy rõ:
''')

code(r'''
step = np.zeros((100, 200), np.uint8); step[:, 100:] = 200
step_n = add_gaussian_noise(step, 20, seed=3)
row = 50
outs = {'Nhiễu': step_n,
        'Gaussian 9x9': cv2.GaussianBlur(step_n, (9, 9), 0),
        'median 9': cv2.medianBlur(step_n, 9),
        'bilateral 9,75,75': cv2.bilateralFilter(step_n, 9, 75, 75)}
plt.figure(figsize=(11, 4))
plt.plot(step[row], 'k--', lw=2, label='Cạnh lý tưởng')
for name, o in outs.items():
    plt.plot(o[row], label=name, alpha=.85)
plt.xlim(60, 140); plt.legend(); plt.title(f'Mặt cắt hàng y={row} qua cạnh'); plt.grid(alpha=.3)
plt.show()
''')

md(r'''
🔍 **Nhận xét:** Gaussian biến bước nhảy thành **dốc thoai thoải** (cạnh bị mờ); bilateral và median giữ **bước nhảy dốc đứng** trong khi vẫn làm phẳng nhiễu hai bên.

### 11.5. Ảnh hưởng của kích thước kernel & bộ khử nhiễu mạnh Non-Local Means

`cv2.fastNlMeansDenoisingColored` tìm các **mảng tương tự ở khắp ảnh** và lấy trung bình chúng ⇒ khử nhiễu rất tốt nhưng chậm hơn nhiều.
''')

code(r'''
ks = [3, 9, 21, 41]
show([cv2.GaussianBlur(img, (k, k), 0) for k in ks], [f'GaussianBlur {k}x{k}' for k in ks], cols=4, size=3.2)

t = time.perf_counter()
nlm = cv2.fastNlMeansDenoisingColored(n_gauss, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
t_nlm = time.perf_counter() - t
print(f'NLM: PSNR = {cv2.PSNR(clean, nlm):.2f} dB, thời gian = {t_nlm:.2f} s')
show([n_gauss[c], res_g[1][c], nlm[c]],
     ['Nhiễu Gauss', f'Gaussian 5x5 ({cv2.PSNR(clean, res_g[1]):.1f} dB)', f'Non-Local Means ({cv2.PSNR(clean, nlm):.1f} dB)'],
     size=4)
''')

# =====================================================================
md(r'''
---
## 12. Làm sắc nét (Sharpening)

### 12.1. Kernel sharpen

Kernel `[[0,-1,0],[-1,5,-1],[0,-1,0]]` = ảnh gốc + (ảnh gốc − trung bình lân cận) ⇒ **khuếch đại chênh lệch** tại cạnh.

### 12.2. Unsharp Masking (cách chuyên nghiệp — dùng trong Photoshop)

$$\text{sharp} = \text{img} + \text{amount}\cdot(\text{img} - \text{blur(img)}) = (1+\text{amount})\cdot\text{img} - \text{amount}\cdot\text{blur}$$

- `img − blur` = **chi tiết tần số cao** (cạnh, kết cấu).
- `amount` điều khiển độ mạnh; σ của blur điều khiển độ "to" của chi tiết được làm nét.
''')

code(r'''
soft = cv2.GaussianBlur(img, (5, 5), 1.5)        # giả lập ảnh hơi mờ

k_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
sharp_k = cv2.filter2D(soft, -1, k_sharp)


def unsharp_mask(image, sigma=2.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharp = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    if threshold > 0:                             # chỉ làm nét nơi chênh lệch đủ lớn (tránh khuếch đại nhiễu)
        low_contrast = cv2.absdiff(image, blurred) < threshold
        np.copyto(sharp, image, where=low_contrast)
    return sharp


c = (slice(150, 300), slice(200, 400))
outs = [soft, sharp_k, unsharp_mask(soft, 2, 0.8), unsharp_mask(soft, 2, 2.0)]
show([o[c] for o in outs], ['Ảnh mờ', 'Kernel sharpen', 'Unsharp amount=0.8', 'Unsharp amount=2.0 (quá đà)'],
     cols=4, size=3.5)
detail = cv2.subtract(soft, cv2.GaussianBlur(soft, (0, 0), 2))
show(cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX), 'Thành phần chi tiết img − blur (đã kéo giãn)', size=5)
''')

md(r'''
🔍 **Nhận xét:** `amount` quá lớn tạo **quầng sáng (halo)** quanh cạnh và khuếch đại nhiễu. Làm nét **không khôi phục** được chi tiết đã mất, chỉ tăng tương phản cục bộ tại cạnh.
''')

# =====================================================================
md(r'''
---
## 13. Phát hiện cạnh (Edge Detection)

### 13.1. Cạnh = nơi cường độ thay đổi đột ngột

- **Đạo hàm bậc 1** (gradient): cạnh ứng với **cực đại** của |đạo hàm|. → Sobel, Scharr, Prewitt.
- **Đạo hàm bậc 2**: cạnh ứng với **điểm cắt 0** (zero-crossing). → Laplacian.

Xem trên tín hiệu 1D:
''')

code(r'''
x1 = np.arange(200)
signal = 50 + 150 / (1 + np.exp(-(x1 - 100) / 4))            # cạnh mượt
signal_n = signal + np.random.default_rng(0).normal(0, 2, 200)
d1 = np.gradient(signal_n); d2 = np.gradient(d1)
fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
axes[0].plot(signal_n); axes[0].set_title('Cường độ f(x)')
axes[1].plot(d1, 'r'); axes[1].set_title("Đạo hàm bậc 1 f'(x): CỰC ĐẠI tại cạnh")
axes[2].plot(d2, 'g'); axes[2].axhline(0, c='k', lw=.5); axes[2].set_title("Đạo hàm bậc 2 f''(x): CẮT 0 tại cạnh (rất nhạy nhiễu)")
plt.tight_layout(); plt.show()
''')

md(r'''
### 13.2. Sobel và Scharr

Kernel Sobel 3×3:

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

- $G_x$ phát hiện **cạnh dọc** (thay đổi theo x); $G_y$ phát hiện **cạnh ngang**.
- Độ lớn: $|G| = \sqrt{G_x^2 + G_y^2}$, hướng: $\theta = \arctan(G_y / G_x)$.

> ⚠️ Đạo hàm có thể **âm** (sáng → tối). Nếu dùng `ddepth=cv2.CV_8U` thì giá trị âm bị cắt về 0 ⇒ **mất một nửa số cạnh**! Hãy dùng `cv2.CV_64F` (hoặc `CV_16S`) rồi lấy trị tuyệt đối với `cv2.convertScaleAbs`.
''')

code(r'''
building = load_sample('building.jpg')
bgray = cv2.cvtColor(building, cv2.COLOR_BGR2GRAY)
bblur = cv2.GaussianBlur(bgray, (3, 3), 0)

gx = cv2.Sobel(bblur, cv2.CV_64F, 1, 0, ksize=3)   # dx=1, dy=0
gy = cv2.Sobel(bblur, cv2.CV_64F, 0, 1, ksize=3)   # dx=0, dy=1
mag = cv2.magnitude(gx, gy)
angle = cv2.phase(gx, gy, angleInDegrees=True)

gx_8u_wrong = cv2.Sobel(bblur, cv2.CV_8U, 1, 0, ksize=3)  # mất giá trị âm
show([bgray, cv2.convertScaleAbs(gx), cv2.convertScaleAbs(gy),
      gx_8u_wrong, cv2.convertScaleAbs(mag), cv2.convertScaleAbs(cv2.Scharr(bblur, cv2.CV_64F, 1, 0))],
     ['Ảnh xám', '|Sobel x| → cạnh dọc', '|Sobel y| → cạnh ngang',
      '❌ Sobel x với CV_8U (mất cạnh sáng→tối)', 'Độ lớn gradient |G|', 'Scharr x (chính xác hơn với 3x3)'],
     cols=3, size=4)

# Hiển thị hướng gradient bằng màu (H = hướng, V = độ lớn)
hsv_vis = np.zeros((*bgray.shape, 3), np.uint8)
hsv_vis[..., 0] = (angle / 2).astype(np.uint8)
hsv_vis[..., 1] = 255
hsv_vis[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
show(cv2.cvtColor(hsv_vis, cv2.COLOR_HSV2BGR), 'Hướng gradient (màu) × độ lớn (độ sáng)', size=6)
''')

md(r'''
### 13.3. Laplacian

$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

Phát hiện cạnh theo **mọi hướng** cùng lúc nhưng **rất nhạy nhiễu** ⇒ luôn làm mờ trước (kết hợp thành LoG — Laplacian of Gaussian). Ứng dụng hay: **đo độ nét ảnh** — phương sai của Laplacian càng lớn ⇒ ảnh càng nét.
''')

code(r'''
lap = cv2.Laplacian(bblur, cv2.CV_64F, ksize=3)
lap_noblur = cv2.Laplacian(add_gaussian_noise(bgray, 10), cv2.CV_64F, ksize=3)
show([cv2.convertScaleAbs(lap), cv2.convertScaleAbs(lap_noblur)],
     ['Laplacian (đã làm mờ trước)', 'Laplacian trên ảnh nhiễu, không làm mờ'], size=5)

print('Đo độ nét bằng phương sai Laplacian:')
for k in [1, 5, 11, 21]:
    im = bgray if k == 1 else cv2.GaussianBlur(bgray, (k, k), 0)
    print(f'  Blur {k:2d}x{k:<2d}: var(Laplacian) = {cv2.Laplacian(im, cv2.CV_64F).var():8.1f}')
''')

md(r'''
### 13.4. Canny — bộ phát hiện cạnh tốt nhất (cổ điển)

`cv2.Canny(img, threshold1, threshold2)` thực hiện 4 bước:

1. **Làm mờ Gauss 5×5** để giảm nhiễu.
2. **Tính gradient** (Sobel) — độ lớn và hướng.
3. **Non-Maximum Suppression**: chỉ giữ pixel là cực đại theo hướng gradient ⇒ cạnh **mỏng 1 pixel**.
4. **Hysteresis thresholding** (ngưỡng kép):
   - Gradient > `threshold2` (cao) → **cạnh chắc chắn**.
   - Gradient < `threshold1` (thấp) → loại.
   - Ở giữa → chỉ giữ nếu **nối liền** với cạnh chắc chắn.

Tỉ lệ khuyên dùng: `threshold2 ≈ 2–3 × threshold1`.
''')

code(r'''
pairs = [(10, 30), (50, 150), (100, 200), (200, 400)]
show([cv2.Canny(bblur, a, b) for a, b in pairs], [f'Canny({a}, {b})' for a, b in pairs], cols=4, size=3.5)


def auto_canny(gray_img, sigma=0.33):
    """Chọn ngưỡng tự động dựa trên trung vị độ sáng."""
    v = np.median(gray_img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray_img, lower, upper), lower, upper


edges_auto, lo, hi = auto_canny(bblur)
edges_l2 = cv2.Canny(bblur, 50, 150, L2gradient=True)   # dùng sqrt(gx²+gy²) thay vì |gx|+|gy|
show([edges_auto, edges_l2], [f'auto_canny → ({lo}, {hi})', 'Canny(50,150, L2gradient=True)'], size=5)
''')

md(r'''
🔍 **Nhận xét:** Ngưỡng thấp → nhiều cạnh vụn (kết cấu, nhiễu); ngưỡng cao → chỉ còn đường viền chính nhưng có thể đứt đoạn. So với Sobel, Canny cho cạnh **mảnh, liên tục, nhị phân** — rất phù hợp làm đầu vào cho contour và Hough.

✍️ **Bài tập 13:** Viết hàm nhận ảnh và trả về "ảnh phác thảo bút chì" bằng cách: xám → đảo màu → GaussianBlur → `cv2.divide(gray, 255 - blur, scale=256)`.
''')

# =====================================================================
md(r'''
---
## 14. Hình thái học (Morphological Operations)

Các phép toán trên **ảnh nhị phân** (cũng dùng được cho ảnh xám) dựa trên một **phần tử cấu trúc (structuring element / kernel)**:

| Phép | Định nghĩa | Tác dụng |
|---|---|---|
| **Erosion** (co) | pixel = MIN trong kernel | đối tượng trắng **co lại**, xóa chấm trắng nhỏ, tách đối tượng dính nhau |
| **Dilation** (giãn) | pixel = MAX trong kernel | đối tượng trắng **phình ra**, lấp lỗ nhỏ, nối nét đứt |
| **Opening** | erode → dilate | **xóa nhiễu trắng** nhỏ bên ngoài, giữ kích thước đối tượng |
| **Closing** | dilate → erode | **lấp lỗ đen** nhỏ bên trong đối tượng |
| **Gradient** | dilate − erode | **đường viền** đối tượng |
| **Top-hat** | img − opening | chi tiết **sáng nhỏ** hơn kernel |
| **Black-hat** | closing − img | chi tiết **tối nhỏ** hơn kernel |
''')

code(r'''
for name, shape in [('MORPH_RECT', cv2.MORPH_RECT), ('MORPH_ELLIPSE', cv2.MORPH_ELLIPSE), ('MORPH_CROSS', cv2.MORPH_CROSS)]:
    print(f'{name}:\n{cv2.getStructuringElement(shape, (5, 5))}\n')
''')

code(r'''
rng = np.random.default_rng(7)
bw = np.zeros((160, 560), np.uint8)
cv2.putText(bw, 'OpenCV', (15, 118), cv2.FONT_HERSHEY_SIMPLEX, 3.3, 255, 16)
noise = rng.random(bw.shape)
salt = bw.copy();   salt[noise < 0.03] = 255          # chấm trắng ngoài nền
holes = bw.copy();  holes[noise > 0.95] = 0           # lỗ đen trong chữ

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
show([bw, cv2.erode(bw, k, iterations=2), cv2.dilate(bw, k, iterations=2), cv2.morphologyEx(bw, cv2.MORPH_GRADIENT, k)],
     ['Ảnh nhị phân gốc', 'Erode (×2)', 'Dilate (×2)', 'Morphological Gradient'], cols=2, size=6)
show([salt, cv2.morphologyEx(salt, cv2.MORPH_OPEN, k), holes, cv2.morphologyEx(holes, cv2.MORPH_CLOSE, k)],
     ['Nhiễu trắng ở nền', 'Opening → xóa nhiễu', 'Lỗ đen trong chữ', 'Closing → lấp lỗ'], cols=2, size=6)
''')

md(r'''
🔍 **Nhận xét:** Opening = "erode để xóa các chấm nhỏ hơn kernel, rồi dilate để trả lại kích thước cũ cho chữ". Closing ngược lại. Chọn **kích thước kernel lớn hơn nhiễu nhưng nhỏ hơn chi tiết cần giữ**.

### 14.1. Làm sạch mask màu (tiếp nối mục 4.3)
''')

code(r'''
mask_y = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), np.array((20, 100, 100)), np.array((35, 255, 255)))
k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
clean_mask = cv2.morphologyEx(mask_y, cv2.MORPH_OPEN, k7)
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, k7, iterations=2)
show([mask_y, clean_mask, cv2.bitwise_and(img, img, mask=clean_mask)],
     ['Mask vàng thô', 'Sau Opening + Closing', 'Kết quả sạch'])
''')

md(r'''
### 14.2. Top-hat / Black-hat: tách chữ khỏi nền chiếu sáng không đều
''')

code(r'''
h, w = 200, 600
bg_light = np.tile(np.linspace(60, 230, w), (h, 1)).astype(np.uint8)          # nền sáng dần
text_img = bg_light.copy()
cv2.putText(text_img, 'Black-hat 2026', (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 30, 6)
kbig = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
blackhat = cv2.morphologyEx(text_img, cv2.MORPH_BLACKHAT, kbig)
_, bin_direct = cv2.threshold(text_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, bin_bh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
show([text_img, bin_direct, blackhat, bin_bh],
     ['Chữ tối trên nền không đều', 'Otsu trực tiếp (hỏng)', 'Black-hat (25x25)', 'Black-hat + Otsu'], cols=2, size=6)
''')

md(r'''
### 14.3. Tách đường kẻ ngang / dọc (bảng biểu, sudoku)

Dùng kernel **dài và mảnh**: opening với kernel `(W/15, 1)` chỉ giữ lại các đoạn **nằm ngang dài**; kernel `(1, H/15)` giữ đoạn **dọc**.
''')

code(r'''
sud_bin = cv2.adaptiveThreshold(sud_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
H_, W_ = sud_bin.shape
horiz = cv2.morphologyEx(sud_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (W_ // 15, 1)))
vert = cv2.morphologyEx(sud_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, H_ // 15)))
grid = cv2.bitwise_or(horiz, vert)
digits = cv2.morphologyEx(cv2.subtract(sud_bin, cv2.dilate(grid, np.ones((3, 3), np.uint8))), cv2.MORPH_OPEN,
                          np.ones((2, 2), np.uint8))
show([sud_bin, horiz, vert, digits], ['Nhị phân (INV)', 'Đường ngang', 'Đường dọc', 'Bỏ lưới → còn chữ số'],
     cols=4, size=3.4)
''')

md(r'''
### 14.4. Skeleton (xương) bằng các phép hình thái

Lặp: `skeleton |= img − open(img)`, `img = erode(img)` cho tới khi img rỗng.
''')

code(r'''
def skeletonize(bin_img):
    skel = np.zeros_like(bin_img)
    work = bin_img.copy()
    el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work) > 0:
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, el)
        skel = cv2.bitwise_or(skel, cv2.subtract(work, opened))
        work = cv2.erode(work, el)
    return skel


show([bw, skeletonize(bw)], ['Chữ gốc', 'Skeleton'], cols=1, size=7)
''')

# =====================================================================
md(r'''
---
## 15. Contour và phân tích hình dạng

**Contour** = đường cong nối các điểm liên tục **dọc theo biên** của một vùng cùng màu/cường độ. Là công cụ quan trọng nhất để **đếm, đo, nhận dạng hình dạng** đối tượng.

```python
contours, hierarchy = cv2.findContours(binary, mode, method)
```
- `binary`: ảnh **nhị phân**, **đối tượng màu TRẮNG trên nền ĐEN**.
- `mode`: `RETR_EXTERNAL` (chỉ viền ngoài cùng), `RETR_LIST` (tất cả, không phân cấp), `RETR_TREE` (tất cả + cây phân cấp cha/con), `RETR_CCOMP` (2 cấp).
- `method`: `CHAIN_APPROX_NONE` (giữ mọi điểm), `CHAIN_APPROX_SIMPLE` (nén đoạn thẳng còn 2 đầu mút — tiết kiệm bộ nhớ).
- Mỗi contour là mảng `shape = (N, 1, 2)` chứa các điểm `(x, y)`.

> 📝 OpenCV 3.x trả về 3 giá trị `(image, contours, hierarchy)`; từ OpenCV 4 trở đi chỉ 2 giá trị.
''')

code(r'''
def regular_polygon(cx, cy, r, n, rot=-np.pi / 2):
    a = rot + np.arange(n) * 2 * np.pi / n
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1).astype(np.int32)


def star_points(cx, cy, r_out, r_in, n=5):
    pts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        a = -np.pi / 2 + i * np.pi / n
        pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts, np.int32)


shapes = np.full((420, 720, 3), 255, np.uint8)
cv2.fillPoly(shapes, [np.array([[50, 170], [170, 170], [110, 60]], np.int32)], (60, 60, 220))    # tam giác
cv2.rectangle(shapes, (215, 65), (320, 170), (60, 180, 60), -1)                                   # vuông
cv2.rectangle(shapes, (365, 85), (530, 150), (220, 120, 40), -1)                                  # chữ nhật
cv2.circle(shapes, (630, 115), 58, (40, 200, 230), -1)                                            # tròn
cv2.fillPoly(shapes, [regular_polygon(110, 305, 68, 5)], (200, 60, 200))                          # ngũ giác
cv2.fillPoly(shapes, [regular_polygon(285, 305, 68, 6, 0)], (40, 140, 255))                       # lục giác
cv2.fillPoly(shapes, [star_points(455, 305, 75, 32)], (0, 200, 120))                              # ngôi sao
cv2.ellipse(shapes, (620, 305), (78, 42), 25, 0, 360, (120, 80, 40), -1)                          # elip

sgray = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)
_, sbin = cv2.threshold(sgray, 240, 255, cv2.THRESH_BINARY_INV)      # đối tượng trắng, nền đen
contours, hierarchy = cv2.findContours(sbin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_none, _ = cv2.findContours(sbin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print('Số contour tìm được:', len(contours))
print('Tổng số điểm  CHAIN_APPROX_NONE  :', sum(len(c) for c in contours_none))
print('Tổng số điểm  CHAIN_APPROX_SIMPLE:', sum(len(c) for c in contours))
print('Contour đầu tiên shape =', contours[0].shape)

drawn = shapes.copy()
cv2.drawContours(drawn, contours, -1, (0, 0, 0), 3)                  # -1 = vẽ tất cả
pts_vis = shapes.copy()
for cnt in contours:
    for p in cnt[:, 0]:
        cv2.circle(pts_vis, tuple(int(v) for v in p), 4, (0, 0, 255), -1)
show([shapes, sbin, drawn, pts_vis], ['Ảnh hình học', 'Nhị phân (INV)', 'drawContours', 'Các điểm của CHAIN_APPROX_SIMPLE'],
     cols=2, size=6)
''')

md(r'''
### 15.1. Các đặc trưng của contour

| Đặc trưng | Hàm / công thức |
|---|---|
| Diện tích | `cv2.contourArea(c)` |
| Chu vi | `cv2.arcLength(c, closed=True)` |
| Tâm (centroid) | `M = cv2.moments(c)`; `cx = M['m10']/M['m00']`, `cy = M['m01']/M['m00']` |
| Hộp bao thẳng | `x, y, w, h = cv2.boundingRect(c)` |
| Hộp bao xoay nhỏ nhất | `rect = cv2.minAreaRect(c)`; `box = cv2.boxPoints(rect)` |
| Đường tròn bao nhỏ nhất | `(x, y), r = cv2.minEnclosingCircle(c)` |
| Elip khớp nhất | `cv2.fitEllipse(c)` (cần ≥ 5 điểm) |
| Bao lồi | `hull = cv2.convexHull(c)` |
| Xấp xỉ đa giác | `cv2.approxPolyDP(c, epsilon, True)` (thuật toán Douglas–Peucker) |
| Aspect ratio | `w / h` |
| Extent | `area / (w*h)` |
| Solidity (độ đặc) | `area / area(hull)` |
| Độ tròn (circularity) | $4\pi \cdot area / perimeter^2$ (=1 với hình tròn) |
''')

code(r'''
geo = shapes.copy()
print(f'{"#":>2} | {"area":>7} | {"peri":>6} | {"centroid":>11} | {"w/h":>5} | {"extent":>6} | {"solidity":>8} | {"circ":>5}')
for i, cnt in enumerate(sorted(contours, key=lambda c: cv2.boundingRect(c)[0] + 1000 * (cv2.boundingRect(c)[1] > 200))):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    M = cv2.moments(cnt)
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    x, y, w, h = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    solidity = area / cv2.contourArea(hull)
    circ = 4 * np.pi * area / peri ** 2
    print(f'{i:2d} | {area:7.0f} | {peri:6.1f} | ({cx:3d},{cy:3d})   | {w/h:5.2f} | {area/(w*h):6.2f} | {solidity:8.2f} | {circ:5.2f}')

    cv2.rectangle(geo, (x, y), (x + w, y + h), (0, 255, 0), 1)                           # bbox thẳng: xanh lá
    box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.int32)
    cv2.drawContours(geo, [box], 0, (0, 0, 255), 2)                                      # bbox xoay: đỏ
    (ex, ey), r = cv2.minEnclosingCircle(cnt)
    cv2.circle(geo, (int(ex), int(ey)), int(r), (255, 0, 0), 1)                          # đường tròn bao: lam
    cv2.drawContours(geo, [hull], 0, (0, 0, 0), 1)                                       # bao lồi: đen
    cv2.circle(geo, (cx, cy), 4, (0, 0, 0), -1)
    cv2.putText(geo, str(i), (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
show(geo, 'Xanh lá: boundingRect | Đỏ: minAreaRect | Lam: minEnclosingCircle | Đen: convexHull + tâm', size=10)
''')

md(r'''
### 15.2. Nhận dạng hình dạng bằng `approxPolyDP`

`approxPolyDP(c, ε, True)` đơn giản hóa contour thành đa giác ít đỉnh hơn, sao cho khoảng cách lệch ≤ ε. Thường chọn **ε = 1–4% chu vi**. Sau đó đếm số đỉnh:
- 3 → tam giác; 4 → vuông/chữ nhật (xét tỉ lệ w/h); 5 → ngũ giác; 6 → lục giác.
- Lõm (solidity thấp) → ngôi sao.
- Nhiều đỉnh → tròn hoặc elip (xét tỉ lệ 2 trục của `fitEllipse`).
''')

code(r'''
def classify_shape(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    v = len(approx)
    area = cv2.contourArea(cnt)
    solidity = area / cv2.contourArea(cv2.convexHull(cnt))
    if solidity < 0.8:
        return 'Ngoi sao', approx
    if v == 3:
        return 'Tam giac', approx
    if v == 4:
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        return ('Hinh vuong' if 0.92 <= rw / rh <= 1.08 else 'Chu nhat'), approx
    if v == 5:
        return 'Ngu giac', approx
    if v == 6:
        return 'Luc giac', approx
    (_, _), (ma, MA), _ = cv2.fitEllipse(cnt)
    return ('Hinh tron' if min(ma, MA) / max(ma, MA) > 0.9 else 'Elip'), approx


labeled = shapes.copy()
for cnt in contours:
    name, approx = classify_shape(cnt)
    cv2.drawContours(labeled, [approx], -1, (0, 0, 0), 2)
    for p in approx[:, 0]:
        cv2.circle(labeled, tuple(int(v) for v in p), 5, (0, 0, 255), -1)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.putText(labeled, f'{name} ({len(approx)})', (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
show(labeled, 'Phân loại hình dạng (số trong ngoặc = số đỉnh sau approxPolyDP)', size=10)

# Ảnh hưởng của epsilon
star = min(contours, key=lambda c: cv2.contourArea(c) / cv2.contourArea(cv2.convexHull(c)))  # solidity nhỏ nhất
outs, titles = [], []
for eps in [0.005, 0.02, 0.05, 0.1]:
    canvas = np.full((200, 200, 3), 255, np.uint8)
    off = star - star.reshape(-1, 2).min(0) + 20
    ap = cv2.approxPolyDP(off, eps * cv2.arcLength(off, True), True)
    cv2.drawContours(canvas, [off], -1, (200, 200, 200), -1)
    cv2.drawContours(canvas, [ap], -1, (0, 0, 255), 2)
    outs.append(canvas); titles.append(f'ε = {eps*100:.1f}% chu vi → {len(ap)} đỉnh')
show(outs, titles, cols=4, size=3)
''')

md(r'''
### 15.3. Phân cấp contour (hierarchy) — `RETR_TREE`

`hierarchy[0][i] = [next, previous, first_child, parent]` (−1 nếu không có). Dùng để phân biệt **lỗ bên trong** đối tượng, **đối tượng lồng nhau**.
''')

code(r'''
nest = np.zeros((300, 420), np.uint8)
cv2.rectangle(nest, (20, 20), (280, 280), 255, -1)    # cấp 0
cv2.rectangle(nest, (60, 60), (240, 240), 0, -1)      # lỗ (cấp 1)
cv2.circle(nest, (150, 150), 60, 255, -1)             # đảo trong lỗ (cấp 2)
cv2.circle(nest, (150, 150), 25, 0, -1)               # lỗ trong đảo (cấp 3)
cv2.circle(nest, (350, 150), 50, 255, -1)             # một đối tượng riêng

for mode in ['RETR_EXTERNAL', 'RETR_LIST', 'RETR_CCOMP', 'RETR_TREE']:
    cs, hi = cv2.findContours(nest, getattr(cv2, mode), cv2.CHAIN_APPROX_SIMPLE)
    print(f'{mode:14s}: {len(cs)} contour')

cs, hi = cv2.findContours(nest, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print('\nHierarchy RETR_TREE [next, prev, first_child, parent]:\n', hi[0])


def depth(i):
    d = 0
    while hi[0][i][3] != -1:
        i = hi[0][i][3]; d += 1
    return d


vis = cv2.cvtColor(nest, cv2.COLOR_GRAY2BGR)
palette = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255)]
for i, c in enumerate(cs):
    cv2.drawContours(vis, cs, i, palette[depth(i) % 4], 4)
    x, y, w, h = cv2.boundingRect(c)
    cv2.putText(vis, f'{i}:d{depth(i)}', (x + 3, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 0, 128), 2)
show([nest, vis], ['Ảnh lồng nhau', 'Màu theo độ sâu (d0 đỏ, d1 lục, d2 lam, d3 cam)'], size=5)
''')

md(r'''
### 15.4. So khớp hình dạng — `cv2.matchShapes` (Hu Moments)

7 **Hu moments** bất biến với tịnh tiến, co giãn, xoay (và lật). `cv2.matchShapes` so sánh 2 contour dựa trên chúng: **càng gần 0 càng giống**.
''')

code(r'''
ref = star
big_star = star_points(150, 200, 110, 47).reshape(-1, 1, 2).astype(np.float32)   # sao to hơn
M_rot = cv2.getRotationMatrix2D((150, 200), 37, 1.0)                             # và xoay 37°
big_star = cv2.transform(big_star, M_rot).astype(np.int32)
canvas = np.zeros((400, 900), np.uint8)
cv2.fillPoly(canvas, [big_star], 255)
cv2.fillPoly(canvas, [regular_polygon(500, 200, 80, 5)], 255)
cv2.circle(canvas, (750, 200), 80, 255, -1)
cs2, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vis = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
for c in cs2:
    d = cv2.matchShapes(ref, c, cv2.CONTOURS_MATCH_I1, 0)
    x, y, w, h = cv2.boundingRect(c)
    cv2.putText(vis, f'd={d:.3f}', (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
print('Hu moments của ngôi sao mẫu:', cv2.HuMoments(cv2.moments(ref)).ravel()[:4].round(5), '...')
show(vis, 'Khoảng cách matchShapes tới ngôi sao mẫu (sao xoay + phóng to vẫn ~0)', size=10)
''')

md(r'''
### 15.5. Convexity defects — các "chỗ lõm"

`cv2.convexityDefects(cnt, hull_indices)` trả về các điểm lõm sâu nhất giữa contour và bao lồi — ứng dụng nổi tiếng: **đếm ngón tay**.
''')

code(r'''
hull_idx = cv2.convexHull(star, returnPoints=False)
defects = cv2.convexityDefects(star, hull_idx)
vis = shapes.copy()
cv2.drawContours(vis, [cv2.convexHull(star)], 0, (255, 0, 0), 2)
for s, e, f, d in defects.reshape(-1, 4):
    far = tuple(int(v) for v in star.reshape(-1, 2)[f])
    cv2.circle(vis, far, 6, (0, 0, 255), -1)
    print(f'Điểm lõm {far}, độ sâu = {d / 256:.1f} px')
show(vis[200:400, 350:560], f'{len(defects.reshape(-1, 4))} điểm lõm của ngôi sao (đỏ) và bao lồi (lam)', size=5)
''')

# =====================================================================
md(r'''
---
## 16. Connected Components — gán nhãn vùng liên thông

`cv2.connectedComponentsWithStats(binary, connectivity=8)` gán **một nhãn (số nguyên) cho mỗi vùng trắng liên thông** và trả về luôn thống kê:

- `num_labels`: số nhãn (**bao gồm nền = nhãn 0**).
- `labels`: ảnh nhãn cùng kích thước.
- `stats[i] = [x, y, w, h, area]`.
- `centroids[i] = [cx, cy]`.

So với contour: nhanh hơn, cho diện tích chính xác theo pixel; nhưng không có thông tin hình dạng biên.
''')

code(r'''
rng = np.random.default_rng(11)
blobs = np.zeros((300, 500), np.uint8)
for _ in range(18):
    cv2.circle(blobs, (int(rng.integers(30, 470)), int(rng.integers(30, 270))), int(rng.integers(8, 28)), 255, -1)
blobs[rng.random(blobs.shape) < 0.002] = 255        # nhiễu chấm nhỏ

num, labels, stats, cents = cv2.connectedComponentsWithStats(blobs, connectivity=8)
print('Số vùng (không tính nền):', num - 1)

# Tô màu ngẫu nhiên cho mỗi nhãn
colors = rng.integers(50, 255, (num, 3), dtype=np.uint8); colors[0] = 0
colored = colors[labels]

# Lọc bỏ vùng nhỏ (nhiễu)
min_area = 30
keep = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area]
filtered = np.isin(labels, keep).astype(np.uint8) * 255
vis = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
for i in keep:
    x, y, w, h, a = stats[i]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(vis, str(a), (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
print(f'Sau khi lọc area >= {min_area}: còn {len(keep)} vùng')
show([blobs, colored, vis], ['Ảnh nhị phân (có nhiễu chấm)', f'{num-1} nhãn (mỗi màu 1 nhãn)',
                             f'Lọc area ≥ {min_area}: {len(keep)} vùng (số = diện tích)'], size=4.5)
''')

md(r'''
🔍 **Nhận xét:** Các hình tròn **chạm nhau** bị tính là **một** vùng. Để tách chúng ta cần **Watershed** (mục 22.2) hoặc erosion trước khi đếm.
''')
