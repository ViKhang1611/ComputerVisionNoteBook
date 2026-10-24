# Cẩm nang Chẩn đoán và Khắc phục Lỗi (`TROUBLESHOOTING.md`)

Tài liệu này tổng hợp các lỗi thực tế thường gặp nhất của người mới học Xử lý ảnh, Machine Learning và Deep Learning, kèm **Triệu chứng → Nguyên nhân gốc rễ → Cách sửa bằng code**.

---

## 1. Nhóm Lỗi Môi trường, Cài đặt & Jupyter Kernel

### Lỗi 1.1: `ModuleNotFoundError: No module named 'torch'` (hoặc `'sklearn'`, `'cv2'`) khi chạy cell trong Jupyter
- **Nguyên nhân**: Notebook đang chạy trên kernel Python mặc định của máy (ví dụ Python 3.15) thay vì kernel của môi trường dự án `image_ai_learning/.venv` (Python 3.12).
- **Cách kiểm tra ngay trong cell**:
  ```python
  import sys
  print(sys.executable)
  ```
- **Cách khắc phục**:
  1. Đổi Kernel ở góc trên bên phải của JupyterLab / VS Code sang `.venv (Python 3.12.10)` (tại `image_ai_learning/.venv/Scripts/python.exe`).
  2. Hoặc đăng ký kernel tường minh từ PowerShell:
     ```powershell
     E:\programming\JupyterLab\image_ai_learning\.venv\Scripts\python.exe -m ipykernel install --user --name image-ai-learning --display-name "Python 3.12 (image_ai_learning)"
     ```

### Lỗi 1.2: `ModuleNotFoundError: No module named 'src'`
- **Nguyên nhân**: Thư mục làm việc hiện tại (`os.getcwd()`) là `image_ai_learning/notebooks/` nên Python chưa thấy gói `src/` nằm ở thư mục cha `image_ai_learning/`.
- **Cách khắc phục**: Mọi notebook trong khóa học đều có đoạn code tự động tìm thư mục gốc ở cell đầu tiên. Hãy luôn chạy cell đầu tiên trước:
  ```python
  import sys
  from pathlib import Path
  project_root = Path.cwd().resolve()
  if project_root.name == "notebooks":
      project_root = project_root.parent
  if str(project_root) not in sys.path:
      sys.path.insert(0, str(project_root))
  ```

### Lỗi 1.3: `DataLoader` bị treo hoặc báo lỗi `BrokenPipeError` / `PicklingError` trên Windows
- **Nguyên nhân**: Trên Windows, Python dùng cơ chế `spawn` (tạo tiến trình mới từ đầu) thay vì `fork`. Khi đặt `num_workers > 0` trong `DataLoader` với một class `Dataset` viết trực tiếp trong ô Jupyter Notebook, các tiến trình con không serialize (pickle) được class đó.
- **Cách khắc phục**: Luôn đặt `num_workers=0` khi chạy `DataLoader` bên trong Jupyter Notebook trên Windows:
  ```python
  train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
  ```

---

## 2. Nhóm Lỗi Định dạng Ảnh, Kênh màu và Chiều Tensor (`Shape & Dtype`)

### Lỗi 2.1: Ảnh hiển thị bằng `plt.imshow()` bị sai màu (da người/vật màu đỏ biến thành màu xanh dương)
- **Nguyên nhân**: `cv2.imread()` của OpenCV đọc ảnh theo thứ tự kênh **`BGR`**, trong khi `matplotlib.pyplot.imshow()`, `Pillow` và `torchvision` dùng thứ tự **`RGB`**.
- **Cách khắc phục**:
  ```python
  img_bgr = cv2.imread(str(image_path))
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  plt.imshow(img_rgb)
  ```

### Lỗi 2.2: `RuntimeError: Given groups=1, weight of size [16, 3, 3, 3], expected input[32, 32, 32, 3] to have 3 channels, but got 32 channels instead`
- **Nguyên nhân**: Ảnh trong NumPy/OpenCV có thứ tự chiều **`HWC`** `(Batch, Height, Width, Channels)`, nhưng tầng `nn.Conv2d` của PyTorch bắt buộc thứ tự **`NCHW`** `(Batch, Channels, Height, Width)`.
- **Cách khắc phục**: Hoán vị chiều bằng `.permute(0, 3, 1, 2)` (cho batch) hoặc `.permute(2, 0, 1)` (cho 1 ảnh):
  ```python
  # Chuyển từ (N, H, W, C) uint8 sang (N, C, H, W) float32 [0, 1]
  x_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
  ```

### Lỗi 2.3: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x1024 and 4096x128)`
- **Nguyên nhân**: Số đặc trưng đầu ra sau khi `Flatten()` không khớp với tham số `in_features` của tầng `nn.Linear(in_features, out_features)` (thường do thay đổi kích thước ảnh đầu vào mà quên cập nhật `in_features`).
- **Cách khắc phục**:
  1. Dùng tầng `nn.AdaptiveAvgPool2d((1, 1))` trước `nn.Flatten()` để kích thước vector đặc trưng luôn bằng đúng số kênh `out_channels` bất kể kích thước ảnh $(H, W)$.
  2. Hoặc dùng hàm `summarize_model_layers(model, input_shape)` trong `src/utils.py` để in bảng shape từng tầng.

### Lỗi 2.4: `RuntimeError: expected scalar type Float but found Byte` (hoặc `Double`)
- **Nguyên nhân**: Trọng số của mạng PyTorch mặc định là `torch.float32` (`Float`), nhưng tensor ảnh đưa vào vẫn đang là `torch.uint8` (`Byte`, `0..255`) hoặc `torch.float64` (`Double` từ mảng NumPy mặc định).
- **Cách khắc phục**: Luôn ép kiểu `.float()` (`torch.float32`) cho dữ liệu đầu vào $X$:
  ```python
  inputs = inputs.to(device=device, dtype=torch.float32)
  ```

---

## 3. Nhóm Lỗi Nhãn (`Target / Label`), Hàm Mất mát (`Loss`) và Thiết bị (`Device`)

### Lỗi 3.1: `RuntimeError: expected scalar type Long but found Float` (hoặc `Int`) khi tính `CrossEntropyLoss`
- **Nguyên nhân**: `nn.CrossEntropyLoss()` của PyTorch yêu cầu nhãn `targets` cho bài toán phân loại đa lớp phải có kiểu `torch.long` (`int64`) và có shape 1D `(Batch,)` (không phải `(Batch, 1)`).
- **Cách khắc phục**:
  ```python
  targets = targets.squeeze(-1).to(device=device, dtype=torch.long)
  loss = criterion(logits, targets)
  ```

### Lỗi 3.2: `IndexError: Target 4 is out of bounds.` (hoặc `CUDA error: device-side assert triggered`)
- **Nguyên nhân**: Khi mô hình có `num_classes = 4`, PyTorch yêu cầu nhãn phải đánh số từ **`0` đến `3`** (`[0, C-1]`). Nếu dữ liệu của bạn đánh số từ `1` đến `4` hoặc có giá trị `-1`, hàm loss sẽ báo lỗi vượt chỉ số.
- **Cách kiểm tra trước khi train**:
  ```python
  assert targets.min() >= 0 and targets.max() < num_classes, f"Label range lỗi: {targets.min()}..{targets.max()}"
  ```

### Lỗi 3.3: `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same`
- **Nguyên nhân**: Dữ liệu `inputs` và mô hình `model` đang nằm trên hai thiết bị khác nhau (một bên ở `cpu`, một bên ở `cuda`).
- **Cách khắc phục**: Đưa cả `model`, `inputs` và `targets` về cùng một `device`:
  ```python
  device = get_device()
  model = model.to(device)
  inputs = inputs.to(device)
  targets = targets.to(device)
  ```

---

## 4. Nhóm Sự cố trong Quá trình Huấn luyện (`Training Dynamics`)

### Sự cố 4.1: `Loss` biến thành `NaN` hoặc `Inf` ngay trong vài bước đầu tiên
- **Các nguyên nhân thường gặp & Cách khắc phục**:
  1. **`learning_rate` quá lớn**: Giảm `lr` xuống 10 lần (ví dụ từ `1e-2` xuống `1e-3` hoặc `3e-4`).
  2. **Dữ liệu đầu vào chứa `NaN`/`Inf` hoặc chưa chia `255.0`**: Kiểm tra bằng `torch.isfinite(inputs).all()`.
  3. **Bùng nổ đạo hàm (`Exploding Gradients`)**: Thêm bước cắt chuẩn đạo hàm trước `optimizer.step()`:
     ```python
     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
     ```
  4. **Tự viết hàm `log(p)` mà không cộng $\epsilon$**: Khi $p = 0$, $\log(0) = -\infty$. Luôn dùng sẵn `nn.CrossEntropyLoss()` hoặc `nn.BCEWithLogitsLoss()` thay vì tự lấy `torch.log(torch.softmax(...))`.

### Sự cố 4.2: `Train Loss` không hề giảm sau nhiều epoch (Mạng không học được gì)
- **Danh sách kiểm tra (Debugging Checklist)**:
  1. Có quên gọi `optimizer.zero_grad()`, `loss.backward()`, hoặc `optimizer.step()` không?
  2. Có vô tình đặt `Softmax` trước `nn.CrossEntropyLoss` không?
  3. Có vô tình bọc vòng lặp train trong `with torch.no_grad():` hoặc đóng băng nhầm toàn bộ tham số (`requires_grad=False`) không?
  4. Nhãn `y_train` có bị xáo trộn lệch khỏi ảnh `X_train` không?
  5. **Thử bài kiểm tra vàng ("Overfit 1 Batch Test")**: Lấy đúng 1 batch nhỏ (8–16 ảnh), tắt augmentation và tắt dropout, lặp 100 bước trên đúng 1 batch đó. Một mô hình và vòng lặp code đúng **phải đẩy được `train_loss` về gần `0` và `accuracy` lên `100%` trên 1 batch nhỏ**. Nếu không đạt được, chắc chắn code kiến trúc hoặc pipeline đang có lỗi logic!

### Sự cố 4.3: `Train Accuracy = 99%` nhưng `Validation Accuracy = 55%` (Overfitting nặng)
- **Nguyên nhân**: Mô hình quá lớn so với lượng dữ liệu hoặc huấn luyện quá lâu khiến mạng học thuộc lòng cả nhiễu của tập train.
- **Cách khắc phục theo thứ tự ưu tiên**:
  1. Bổ sung thêm dữ liệu thực hoặc bật `Data Augmentation` hợp lý trên tập `train`.
  2. Sử dụng **Transfer Learning** (khởi tạo từ trọng số pretrained và đóng băng bớt các tầng đầu).
  3. Tăng điều chuẩn: thêm `Dropout(p=0.2..0.4)`, tăng `weight_decay` trong `AdamW` (`1e-4` -> `1e-2`).
  4. Dùng `Early Stopping` để dừng ngay tại epoch mà `val_loss` đạt thấp nhất.

### Sự cố 4.4: Kết quả dự đoán lúc Suy luận (`Inference`) mỗi lần chạy lại ra một nhãn khác nhau trên cùng 1 ảnh!
- **Nguyên nhân**: Quên gọi `model.eval()` trước khi dự đoán, khiến tầng `Dropout` vẫn tiếp tục tắt ngẫu nhiên các nơ-ron.
- **Cách khắc phục chuẩn**:
  ```python
  model.eval()
  with torch.inference_mode():
      logits = model(input_tensor)
      probs = torch.softmax(logits, dim=1)
  ```
