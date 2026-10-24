# Từ điển Giải thích Chi tiết Keyword và Tham số trong AI / Xử lý Ảnh (`KEYWORD_GLOSSARY.md`)

Tài liệu này là **từ điển tra cứu trọng tâm** cho toàn bộ khóa học. Mỗi thuật ngữ tiếng Anh xuất hiện trong mô hình, dữ liệu, huấn luyện, đánh giá và triển khai đều được giải thích theo đúng 10 tiêu chí sư phạm:
1. **Tên tiếng Anh & Nghĩa tiếng Việt**
2. **Nhóm phân loại** (`Dữ liệu` / `Kiến trúc` / `Tối ưu hóa & Huấn luyện` / `Đánh giá` / `Theo bài toán & Triển khai`)
3. **Định nghĩa dễ hiểu**
4. **Vai trò trong pipeline**
5. **Kiểu dữ liệu / Đơn vị**
6. **Ví dụ bằng số hoặc code**
7. **Giá trị trong ví dụ & Lý do chọn**
8. **Ảnh hưởng khi thay đổi (Tăng/Giảm hoặc Bật/Tắt)** *(Lưu ý: Với keyword không mang giá trị số, phân tích theo lựa chọn cấu hình; mọi xu hướng tăng/giảm còn phụ thuộc vào tập dữ liệu và kiến trúc cụ thể)*
9. **Lỗi thường gặp**
10. **Phân biệt với thuật ngữ dễ nhầm**

---

## Phần I. 9 Cặp / Nhóm Khái niệm Cốt lõi Bắt buộc Phân biệt Rạch ròi

Trước khi tra cứu từng từ khóa, bạn cần nắm vững 9 bảng phân biệt nền tảng dưới đây:

### 1. `Parameter` (Tham số mô hình) vs `Hyperparameter` (Siêu tham số)
| Tiêu chí | `Parameter` (Tham số mô hình) | `Hyperparameter` (Siêu tham số) |
|---|---|---|
| **Ai quyết định giá trị?** | **Mô hình tự học** và cập nhật từ dữ liệu thông qua thuật toán tối ưu (`optimizer.step()`). | **Con người (kỹ sư AI) thiết lập** trước khi bắt đầu quá trình huấn luyện. |
| **Ví dụ điển hình** | Ma trận trọng số `weight` ($W$), độ lệch `bias` ($b$) trong tầng `nn.Linear`, `nn.Conv2d`. | Tốc độ học (`learning_rate`), kích thước lô (`batch_size`), số tầng (`depth`), `kernel_size`, `dropout_p`. |
| **Lưu ở đâu?** | Lưu trong `model.state_dict()`. | Lưu trong file cấu hình thí nghiệm (`config.yaml` / dictionary cấu hình). |

### 2. `Epoch` vs `Iteration` (Batch pass) vs `Optimizer Step`
Giả sử tập `train` có $N = 1000$ ảnh, `batch_size` $= 100$, và hệ số cộng dồn đạo hàm `gradient_accumulation_steps` $= 2$:
- **`Epoch` (Kỷ nguyên huấn luyện)**: Khi mô hình duyệt qua **toàn bộ $1000$ mẫu** trong tập huấn luyện đúng 1 lượt.
- **`Iteration` (Lượt duyệt batch)**: Mỗi lần lấy 1 lô (`batch`) gồm $100$ ảnh đưa qua mạng (`forward` + `backward`). Trong 1 `epoch` ở ví dụ trên có $1000 / 100 = 10$ `iterations`.
- **`Optimizer Step` (Bước cập nhật trọng số)**: Mỗi lần gọi `optimizer.step()` để thực sự thay đổi trọng số $W$.
  - Nếu cập nhật mỗi batch (`accumulation = 1`): 1 `epoch` có `10 iterations` = `10 optimizer steps`.
  - Nếu cộng dồn 2 batch mới cập nhật 1 lần (`accumulation = 2`): 1 `epoch` có `10 iterations` nhưng chỉ có `5 optimizer steps`.

### 3. `Loss Function` (Hàm mất mát) vs `Metric` (Độ đo đánh giá)
| Tiêu chí | `Loss Function` (ví dụ `CrossEntropyLoss`, `MSELoss`) | `Metric` (ví dụ `Accuracy`, `F1-score`, `mAP`, `IoU`) |
|---|---|---|
| **Mục đích chính** | Dẫn đường cho thuật toán tối ưu cập nhật trọng số trong lúc `backward()`. | Đo lường chất lượng mô hình theo ngôn ngữ con người và yêu cầu bài toán. |
| **Yêu cầu toán học** | **Bắt buộc phải khả vi** (tính được đạo hàm `gradient` hầu khắp nơi). | **Không cần khả vi** (có thể dùng ngưỡng cứng `argmax`, đếm đúng/sai). |

### 4. Tập `Validation` vs Tập `Test`
- **`Validation set` (Tập kiểm định)**: Dùng **trong quá trình phát triển** để theo dõi overfitting, chọn `checkpoint` tốt nhất, dừng sớm (`early stopping`) và chọn siêu tham số (`learning_rate`, kiến trúc mạng). Vì ta dùng nó để ra quyết định chọn mô hình, thông tin từ tập Validation đã gián tiếp đi vào lựa chọn của ta.
- **`Test set` (Tập kiểm tra cuối cùng)**: Phải được "khóa kín" và **chỉ chạy đánh giá đúng ở bước cuối cùng** sau khi đã chốt xong mô hình. Tuyệt đối không dùng tập `Test` để chỉnh `learning_rate` hay chọn epoch tốt nhất.

### 5. `Softmax` vs `nn.CrossEntropyLoss` trong PyTorch
- **`Softmax`**: Hàm toán học biến vector điểm số thô `logits` $z \in \mathbb{R}^C$ thành phân phối xác suất $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}} \in (0, 1)$ có tổng bằng $1$.
- **`nn.CrossEntropyLoss`**: Trong PyTorch, lớp `nn.CrossEntropyLoss()` đã **tích hợp sẵn `LogSoftmax` + `NLLLoss`** bằng công thức log-sum-exp ổn định số học.
- **Quy tắc sống còn**: Mô hình phân loại đa lớp trong PyTorch **phải trả về `logits` thô** (không đặt `Softmax` ở cuối `forward`). Chỉ gọi `torch.softmax(logits, dim=1)` khi bạn muốn hiển thị phần trăm xác suất lúc suy luận (`inference`).

### 6. `Sigmoid` vs `nn.BCEWithLogitsLoss` trong PyTorch
- **`Sigmoid`**: Hàm $\sigma(z) = \frac{1}{1 + e^{-z}}$ ép một số thực bất kỳ về khoảng $(0, 1)$, dùng cho phân loại nhị phân, đa nhãn hoặc phân vùng nhị phân.
- **`nn.BCEWithLogitsLoss`**: Kết hợp `Sigmoid` và `Binary Cross-Entropy` vào chung một biểu thức toán học để tránh tràn số (`overflow`/`underflow`) khi $|z|$ lớn.
- **Quy tắc sống còn**: Khi dùng `nn.BCEWithLogitsLoss()`, mô hình trả về `logits` thô $(-\infty, +\infty)$. Khi suy luận, ta mới áp dụng `probs = torch.sigmoid(logits)` (hoặc so sánh `logits > 0.0` tương đương với `probs > 0.5`).

### 7. `model.eval()` vs `torch.no_grad()`
Hai lệnh này làm **hai việc hoàn toàn khác nhau** và cần được dùng **cùng nhau** khi đánh giá/suy luận:
- **`model.eval()`**: Thay đổi hành vi của một số tầng đặc biệt trong mạng:
  - Tầng `Dropout`: Ngừng tắt ngẫu nhiên các nơ-ron (giữ nguyên toàn bộ tín hiệu).
  - Tầng `BatchNorm`: Ngừng lấy trung bình/phương sai của batch hiện tại mà chuyển sang dùng `running_mean` và `running_var` đã tích lũy trong lúc train.
  - *Lưu ý*: `model.eval()` **KHÔNG** tắt việc tính đạo hàm `autograd`!
- **`with torch.no_grad():`** (hoặc `with torch.inference_mode():`): Tắt bộ máy lưu đồ thị tính đạo hàm `autograd`, giúp giảm một nửa bộ nhớ RAM/VRAM và tăng tốc độ tính toán.
  - *Lưu ý*: `torch.no_grad()` **KHÔNG** chuyển `BatchNorm` hay `Dropout` sang chế độ suy luận!

### 8. Chuẩn hóa Ảnh (`Image Normalization`) vs Tầng Chuẩn hóa trong Mạng (`Normalization Layer`)
- **Chuẩn hóa Ảnh đầu vào (`Image Normalization / Standardization`)**: Bước tiền xử lý dữ liệu trước khi vào mạng, ví dụ chia `255.0` để đưa pixel về `[0, 1]` rồi trừ `mean`, chia `std` của tập `train`: $x_{\text{norm}} = (x - \mu) / \sigma$. Không có tham số học được bằng gradient.
- **Tầng Chuẩn hóa trong Mạng (`BatchNorm2d`, `LayerNorm`, `GroupNorm`)**: Là các tầng `nn.Module` nằm **xen giữa các tầng tích chập/tuyến tính bên trong mạng**, dùng để ổn định phân phối kích hoạt trung gian và **có chứa các tham số học được** ($\gamma$ `weight` và $\beta$ `bias`).

### 9. `Weight Decay` vs `L2 Penalty` trong các Optimizer (`SGD` vs `Adam` vs `AdamW`)
- **Phạt chuẩn $L_2$ (`L2 Penalty`)**: Cộng thêm $\frac{\lambda}{2} \|W\|_2^2$ vào hàm mất mát $\mathcal{L}$, tức là cộng thêm $\lambda W$ trực tiếp vào đạo hàm $g_t = \nabla \mathcal{L}(W_t) + \lambda W_t$.
- **Suy giảm trọng số (`Decoupled Weight Decay`)**: Thu nhỏ trực tiếp trọng số ở bước cập nhật: $W_{t+1} = (1 - \eta \lambda) W_t - \eta \cdot \text{step}(g_t)$.
- **Sự khác biệt quan trọng**:
  - Với **SGD chuẩn** (không có biến đổi thích nghi phương sai): `Weight Decay` và `L2 Penalty` hoàn toàn tương đương nhau.
  - Với **Adam**: Vì Adam chia đạo hàm cho căn bậc hai của trung bình bình phương đạo hàm ($\sqrt{v_t}$), nếu cộng $\lambda W$ vào đạo hàm (như `optim.Adam(..., weight_decay=...)`), các trọng số có đạo hàm lớn sẽ bị phạt yếu đi không đồng đều. Do đó, **`optim.AdamW`** tách riêng bước `weight_decay` ra khỏi bước chia $\sqrt{v_t}$, giúp điều chuẩn hiệu quả hơn hẳn cho CNN hiện đại và Vision Transformer.

---

## Phần II. Nhóm 1 — Từ khóa về Dữ liệu & Tiền xử lý (`Data & Preprocessing`)

### 1. `sample` (Mẫu dữ liệu)
- **Nhóm**: Dữ liệu | **Kiểu**: Một phần tử trong tập dữ liệu (ví dụ 1 bức ảnh kèm nhãn).
- **Định nghĩa & Vai trò**: Đơn vị quan sát độc lập cơ bản nhất đưa vào bài toán học máy.
- **Ví dụ code**: `sample_img, sample_label = dataset[0]` (`sample_img.shape == (3, 32, 32)`).
- **Giá trị & Lý do**: Mỗi bức ảnh kiểm tra là 1 `sample`. Số lượng `samples` ($N$) càng đa dạng và đại diện tốt cho thực tế thì mô hình càng ít bị học vẹt.
- **Ảnh hưởng khi tăng/giảm số lượng `samples`**: Tăng số mẫu chất lượng cao giúp giảm phương sai (`variance`) và cải thiện khả năng tổng quát hóa; giảm quá ít dễ dẫn tới `overfitting`.
- **Lỗi thường gặp & Phân biệt**: Nhầm 1 `sample` `(C, H, W)` với 1 `batch` `(N, C, H, W)` khi đưa vào PyTorch `nn.Module` (PyTorch luôn kỳ vọng có chiều `batch` ở đầu: `sample_img.unsqueeze(0)`).

### 2. `feature` (Đặc trưng)
- **Nhóm**: Dữ liệu | **Kiểu**: Số thực (`float32`), vector 1D `(D,)` hoặc bản đồ đặc trưng 3D `(C, H, W)`.
- **Định nghĩa & Vai trò**: Các thuộc tính đo đạc được từ dữ liệu đầu vào giúp mô hình phân biệt các đối tượng (ví dụ: cường độ pixel, histogram màu, vector HOG, hoặc feature map của CNN).
- **Ví dụ code**: `X_hog = extract_hog_manual(images, cell_size=8)  # shape: (320, 144)` -> mỗi ảnh có `144 features`.
- **Ảnh hưởng khi thay đổi số lượng/chất lượng `features`**: Đặc trưng giàu thông tin giúp mô hình đơn giản cũng đạt độ chính xác cao; quá nhiều đặc trưng dư thừa/nhiễu so với số mẫu sẽ gây hiện tượng "lời nguyền số chiều" (`curse of dimensionality`).
- **Lỗi thường gặp & Phân biệt**: Đưa các `features` có đơn vị chênh lệch lớn (ví dụ diện tích `0..10000` và độ tròn `0..1`) vào k-NN/SVM mà quên chuẩn hóa. Phân biệt `feature` (đầu vào $X$) với `label`/`target` (đầu ra cần dự đoán $y$).

### 3. `label` / `target` (Nhãn / Giá trị đích)
- **Nhóm**: Dữ liệu | **Kiểu**: `int64` (`torch.long`) cho phân loại đơn lớp; `float32` cho hồi quy hoặc phân loại đa nhãn.
- **Định nghĩa & Vai trò**: Câu trả lời đúng (`ground truth`) đi kèm mỗi mẫu trong học có giám sát để mô hình đối chiếu và tính hàm mất mát (`loss`).
- **Ví dụ code**: `targets = torch.tensor([0, 2, 1, 3], dtype=torch.long)`.
- **Lỗi thường gặp & Phân biệt**:
  - Truyền `targets` kiểu `float` hoặc shape `(N, 1)` thay vì `(N,)` kiểu `torch.long` vào `nn.CrossEntropyLoss`.
  - Nhãn ngoài phạm vi `[0, num_classes - 1]` (ví dụ đánh số lớp từ `1..4` thay vì `0..3`) gây lỗi `IndexError / device-side assert`.
  - Phân biệt: `label` thường chỉ nhãn danh mục (lớp), còn `target` là thuật ngữ tổng quát hơn (bao gồm cả nhãn phân loại, giá trị số thực trong hồi quy, tọa độ hộp, mặt nạ).

### 4. `class` (Lớp / Hạng mục)
- **Nhóm**: Dữ liệu | **Kiểu**: Tên chuỗi (`str`) ánh xạ sang chỉ số nguyên `0..C-1`.
- **Định nghĩa & Vai trò**: Một nhóm đối tượng có chung bản chất mà bài toán phân loại cần nhận diện (ví dụ: `["circle", "square", "triangle", "cross"]`).
- **Ví dụ code**: `class_to_idx = {"circle": 0, "square": 1, "triangle": 2, "cross": 3}`.
- **Lỗi thường gặp & Phân biệt**: Không lưu lại từ điển `class_to_idx` trong checkpoint dẫn đến lúc triển khai gán nhầm tên lớp cho chỉ số dự đoán. Phân biệt `class` trong dữ liệu (nhãn hạng mục) với từ khóa `class` trong lập trình hướng đối tượng Python.

### 5. `annotation` (Chú thích / Dữ liệu gán nhãn)
- **Nhóm**: Dữ liệu | **Kiểu**: Cấu trúc dữ liệu mô tả nhãn (JSON, XML, TXT hoặc mảng mask).
- **Định nghĩa & Vai trò**: Toàn bộ thông tin nhãn gắn cho một bức ảnh, đặc biệt trong các bài toán phức tạp như Object Detection (danh sách tọa độ khung + tên lớp), Segmentation (đa giác hoặc ảnh mask), OCR (tọa độ vùng chữ + nội dung chữ).
- **Ví dụ**: `{"image_id": 12, "boxes": [[14.0, 20.0, 52.0, 58.0]], "labels": [1]}`.
- **Lỗi thường gặp**: Khi áp dụng biến đổi hình học (`resize`, `flip`, `crop`) lên ảnh nhưng quên biến đổi tương ứng các tọa độ hộp/mask trong `annotation`.

### 6. `batch` & `batch_size` (Lô dữ liệu & Kích thước lô)
- **Nhóm**: Dữ liệu & Huấn luyện | **Kiểu**: `batch` là Tensor `(B, C, H, W)`; `batch_size` ($B$) là số nguyên dương (`int`).
- **Định nghĩa & Vai trò**: Tập hợp $B$ mẫu dữ liệu được gom lại để xử lý song song trong một lượt lan truyền tiến (`forward`) và tính đạo hàm trung bình (`backward`).
- **Ví dụ code**: `train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)`.
- **Giá trị trong ví dụ & Lý do chọn**: Chọn `batch_size=16` hoặc `32` trên CPU để cân bằng giữa tốc độ tính toán ma trận và dung lượng RAM, đồng thời đảm bảo tầng `BatchNorm2d` có đủ mẫu để ước lượng trung bình/phương sai ổn định.
- **Ảnh hưởng khi tăng / giảm `batch_size`**:
  - **Tăng `batch_size`** (ví dụ `128`, `256`): Ước lượng gradient mượt và chính xác hơn, tận dụng GPU tốt hơn, nhưng tốn nhiều bộ nhớ (VRAM/RAM) và có ít bước cập nhật trọng số (`optimizer steps`) hơn trong mỗi epoch.
  - **Giảm `batch_size`** (ví dụ `8`, `16`): Tốn ít bộ nhớ, nhiễu ngẫu nhiên của gradient đôi khi giúp thoát điểm cực tiểu địa phương xấu, nhưng nếu quá nhỏ (`batch_size=1` hoặc `2`) sẽ làm `BatchNorm` ước lượng sai lệch nặng (lúc đó nên đổi sang `GroupNorm` hoặc `LayerNorm`).
- **Lỗi thường gặp**: Đặt `batch_size=1` khi huấn luyện mô hình có `nn.BatchNorm2d` gây lỗi `ValueError: Expected more than 1 value per channel when training`.

### 7. `Dataset` & `DataLoader` (Tập dữ liệu & Bộ nạp dữ liệu theo lô)
- **Nhóm**: Dữ liệu | **Kiểu**: Class trong `torch.utils.data`.
- **Định nghĩa & Vai trò**:
  - `Dataset`: Định nghĩa **cách truy xuất 1 mẫu thứ `i`** qua phương thức `__getitem__(self, idx)` và tổng số mẫu qua `__len__(self)`.
  - `DataLoader`: Bao bọc quanh `Dataset` để tự động **gom các mẫu thành từng `batch`**, xáo trộn thứ tự (`shuffle`), và nạp dữ liệu song song.
- **Ví dụ code**:
  ```python
  train_ds = TensorDataset(X_train_t, y_train_t)
  train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
  ```
- **Lỗi thường gặp**: Trên Windows và trong Jupyter Notebook, đặt `num_workers > 0` với một lớp `Dataset` định nghĩa trực tiếp trong cell có thể gây treo hoặc lỗi `PicklingError` do cơ chế `spawn` tiến trình con của Windows; vì vậy trong notebook trên Windows nên để `num_workers=0`.

### 8. `shuffle` & `sampler` (Xáo trộn thứ tự & Bộ lấy mẫu)
- **Nhóm**: Dữ liệu | **Kiểu**: `shuffle` là `bool`; `sampler` là đối tượng `torch.utils.data.Sampler` (ví dụ `WeightedRandomSampler`).
- **Định nghĩa & Vai trò**:
  - `shuffle=True`: Xáo trộn ngẫu nhiên thứ tự các mẫu đầu mỗi epoch để tránh việc mô hình nhận toàn các mẫu cùng một lớp liên tiếp nhau trong 1 batch.
  - `sampler`: Quy định chiến lược chọn chỉ số mẫu tùy chỉnh (ví dụ lấy mẫu lớp thiểu số nhiều hơn để cân bằng dữ liệu).
- **Giá trị & Lý do chọn**:
  - Tập `train`: Luôn đặt `shuffle=True` (trừ khi đã dùng `sampler` tùy chỉnh, vì `shuffle` và `sampler` xung đột nhau).
  - Tập `val` và `test`: Đặt `shuffle=False` để thứ tự dự đoán khớp cố định với danh sách ảnh gốc khi phân tích lỗi.
- **Lỗi thường gặp**: Truyền đồng thời cả `shuffle=True` và `sampler=WeightedRandomSampler(...)` vào `DataLoader` sẽ báo lỗi `ValueError: sampler option is mutually exclusive with shuffle`.

### 9. `augmentation` (Tăng cường dữ liệu)
- **Nhóm**: Dữ liệu | **Kiểu**: Chuỗi các phép biến đổi ngẫu nhiên (`RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `RandomResizedCrop`).
- **Định nghĩa & Vai trò**: Tạo ra các biến thể hợp lý của ảnh huấn luyện (lật, xoay nhẹ, đổi sáng nhẹ) ngay trong lúc nạp batch nhằm tăng sự đa dạng dữ liệu và chống học vẹt (`overfitting`).
- **Quy tắc sống còn**:
  1. **Chỉ áp dụng `augmentation` ngẫu nhiên cho tập `train`**. Tập `validation` và `test` chỉ dùng phép biến đổi tất định (`Resize`, `ToTensor`, `Normalize`).
  2. Phép biến đổi phải **bảo toàn ngữ nghĩa của nhãn** (ví dụ: lật ngang ảnh con mèo vẫn là con mèo, nhưng lật ngang chữ số `6` hoặc biển báo rẽ trái sẽ làm sai nhãn!).

### 10. `normalization` vs `standardization` (Chuẩn hóa miền giá trị vs Chuẩn hóa thống kê Z-score)
- **Nhóm**: Dữ liệu | **Kiểu**: Phép biến đổi số thực `float32`.
- **Định nghĩa & Công thức**:
  - **Min-Max Scaling / Image Scaling (`[0, 1]`)**: Chia giá trị pixel `uint8 [0..255]` cho `255.0`: $x' = x / 255.0$.
  - **Standardization (Z-score Normalization)**: Đưa mỗi kênh màu hoặc đặc trưng về trung bình $0$ và độ lệch chuẩn $1$:
    $$x_{\text{std}} = \frac{x' - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
- **Giá trị thường gặp**:
  - Khi huấn luyện từ đầu: Tính $\mu_{\text{train}}$ và $\sigma_{\text{train}}$ trực tiếp trên tập `train` của bạn.
  - Khi dùng mô hình pretrained trên ImageNet: Dùng thống kê chuẩn của ImageNet `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.
- **Lỗi thường gặp**: Tính `mean` và `std` trên toàn bộ dữ liệu (bao gồm cả `val` và `test`) trước khi chia tập — gây rò rỉ dữ liệu (`data leakage`).

### 11. `class imbalance` (Mất cân bằng lớp)
- **Nhóm**: Dữ liệu | **Kiểu**: Tính chất phân phối nhãn (ví dụ lớp A có `950` ảnh, lớp B chỉ có `50` ảnh).
- **Định nghĩa & Vai trò**: Tình trạng số lượng mẫu giữa các lớp chênh lệch lớn. Nếu không xử lý, mô hình sẽ thiên lệch về lớp đa số (đoán toàn bộ là lớp A cũng đạt `95% Accuracy` nhưng `Recall` của lớp B bằng `0%`).
- **Cách khắc phục thực tế**:
  1. Trọng số lớp trong hàm loss: `nn.CrossEntropyLoss(weight=class_weights)` hoặc `BCEWithLogitsLoss(pos_weight=...)`.
  2. Lấy mẫu cân bằng theo batch: `WeightedRandomSampler`.
  3. Theo dõi `Macro F1`, `PR-AUC` và `Recall` của lớp thiểu số thay vì chỉ nhìn `Accuracy`.

### 12. `data leakage` (Rò rỉ dữ liệu)
- **Nhóm**: Dữ liệu | **Kiểu**: Lỗi phương pháp luận nghiêm trọng.
- **Định nghĩa**: Hiện tượng thông tin từ tập `validation` hoặc `test` vô tình lọt vào quá trình huấn luyện hoặc tiền xử lý, làm điểm đánh giá trong lúc thử nghiệm cao giả tạo nhưng khi chạy thực tế thì thất bại.
- **3 nguồn gây `data leakage` phổ biến nhất trong xử lý ảnh**:
  1. Gọi `scaler.fit(X_all)` hoặc `pca.fit(X_all)` trên toàn bộ dữ liệu trước khi `train_test_split`.
  2. Thực hiện nhân bản dữ liệu bằng `augmentation` trước rồi mới chia `train/test` (khiến hai biến thể của cùng 1 ảnh gốc nằm ở cả 2 tập).
  3. Các khung hình cắt liên tiếp từ cùng 1 video hoặc ảnh chụp cùng 1 bệnh nhân/linh kiện nằm lẫn ở cả `train` và `test` (cần dùng **Group Split** để mọi ảnh của cùng 1 `group_id` chỉ thuộc 1 tập duy nhất).

### 13. `domain shift` (Dịch chuyển miền dữ liệu)
- **Nhóm**: Dữ liệu & Triển khai | **Kiểu**: Sự khác biệt phân phối $P_{\text{train}}(X) \neq P_{\text{deploy}}(X)$.
- **Định nghĩa**: Hiện tượng ảnh lúc triển khai thực tế có điều kiện ánh sáng, góc camera, độ mờ, cảm biến hoặc phông nền khác với bộ ảnh lúc huấn luyện, làm suy giảm độ chính xác của mô hình.
- **Cách giảm thiểu**: Đa dạng hóa nguồn thu thập dữ liệu, dùng `ColorJitter` / `GaussianBlur` augmentation hợp lý, chuẩn hóa ánh sáng đầu vào và kiểm tra độ hiệu chỉnh xác suất (`calibration`).

---

## Phần III. Nhóm 2 — Từ khóa về Kiến trúc Mô hình (`Model Architecture`)

### 14. `weight` & `bias` (Trọng số & Độ lệch)
- **Nhóm**: Kiến trúc | **Kiểu**: `nn.Parameter` (`torch.Tensor` có `requires_grad=True`).
- **Định nghĩa & Vai trò**:
  - `weight` ($W$): Hệ số nhân quyết định mức độ ảnh hưởng của từng đặc trưng đầu vào lên đầu ra ($y = Wx + b$).
  - `bias` ($b$): Hệ số cộng thêm giúp dịch chuyển ngưỡng kích hoạt của nơ-ron.
- **Lưu ý thực hành**: Khi ngay sau tầng `nn.Conv2d` đã có `nn.BatchNorm2d`, phép trừ trung bình và cộng $\beta$ của `BatchNorm2d` đã làm triệt tiêu tác dụng của `bias` trong `Conv2d`; vì vậy ta đặt `bias=False` ở `Conv2d` đó để tiết kiệm tham số.

### 15. `layer` (Tầng / Lớp mạng)
- **Nhóm**: Kiến trúc | **Kiểu**: Một khối biến đổi kế thừa `torch.nn.Module` (ví dụ `nn.Linear`, `nn.Conv2d`, `nn.BatchNorm2d`).
- **Định nghĩa & Vai trò**: Đơn vị cấu trúc thực hiện một phép biến đổi toán học xác định trên tensor đầu vào để tạo ra biểu diễn mức cao hơn.

### 16. `input_dim`, `hidden_dim`, `output_dim` (Số chiều đầu vào, ẩn và đầu ra)
- **Nhóm**: Kiến trúc | **Kiểu**: Số nguyên dương (`int`).
- **Định nghĩa & Ví dụ**: Trong `nn.Linear(in_features=input_dim, out_features=hidden_dim)`:
  - `input_dim`: Số lượng đặc trưng đi vào tầng (ví dụ `3 * 32 * 32 = 3072` cho ảnh RGB `32x32` trải phẳng).
  - `hidden_dim`: Số nơ-ron ở tầng trung gian (ví dụ `128`).
  - `output_dim`: Kích thước đầu ra cuối cùng (ví dụ `num_classes = 4`).
- **Ảnh hưởng khi tăng/giảm `hidden_dim`**: Tăng `hidden_dim` giúp tăng dung lượng biểu diễn (`capacity`) của mạng nhưng làm số tham số tăng bậc hai và dễ overfit trên tập nhỏ; giảm quá thấp gây `underfitting`.

### 17. `channels` (`in_channels`, `out_channels` — Số kênh ảnh / bản đồ đặc trưng)
- **Nhóm**: Kiến trúc | **Kiểu**: Số nguyên dương (`int`).
- **Định nghĩa**:
  - Ở ảnh đầu vào: `in_channels = 1` (ảnh xám Grayscale), `3` (ảnh màu RGB/BGR), `4` (ảnh RGBA có kênh độ trong suốt Alpha).
  - Ở tầng tích chập `nn.Conv2d(in_channels, out_channels, kernel_size)`: `out_channels` chính là **số lượng bộ lọc (`filters` / `kernels`)** khác nhau mà tầng đó học để phát hiện các mẫu hình ảnh khác nhau.
- **Ví dụ**: `nn.Conv2d(3, 16, kernel_size=3)` dùng `16` bộ lọc có kích thước `(3, 3, 3)` để biến ảnh 3 kênh thành `16` bản đồ đặc trưng (`feature maps`).

### 18. `kernel_size`, `stride`, `padding`, `dilation`, `groups` (Bộ tham số cốt lõi của `nn.Conv2d`)
Công thức tổng quát tính kích thước chiều cao/rộng đầu ra $H_{\text{out}}$ của tầng tích chập hoặc gộp (`Conv2d` / `Pool2d`):
$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2 \times \text{padding} - \text{dilation} \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1 \right\rfloor$$

| Tham số | Nghĩa tiếng Việt & Vai trò | Giá trị thường chọn & Lý do | Ảnh hưởng khi Tăng / Giảm |
|---|---|---|---|
| **`kernel_size`** ($K$) | Kích thước cửa sổ lọc trượt (ví dụ `3` tương đương `3x3`). | `3` (chuẩn mực trong ResNet/VGG/U-Net); `1` (để đổi số kênh mà không trộn lân cận); `7` (ở tầng đầu mạng lớn hoặc ConvNeXt). | Tăng $K$ mở rộng vùng nhìn (`receptive field`) trong 1 tầng nhưng số tham số tăng theo $K^2$ (`5x5` tốn gấp `2.78` lần `3x3`). |
| **`stride`** ($S$) | Bước nhảy của cửa sổ lọc theo chiều ngang/dọc mỗi lần trượt. | `1` (giữ nguyên độ phân giải không gian); `2` (giảm một nửa chiều cao và chiều rộng). | Tăng `stride` từ `1` lên `2` làm kích thước feature map giảm 4 lần về diện tích, giảm mạnh chi phí tính toán nhưng mất bớt độ chi tiết vị trí. |
| **`padding`** ($P$) | Số hàng/cột pixel (thường là số `0`) đệm thêm quanh viền ảnh trước khi trượt lọc. | `padding = kernel_size // 2` (ví dụ `K=3 -> P=1`) để khi `stride=1` thì $H_{\text{out}} = H_{\text{in}}$ (Same Padding). | Nếu `padding=0` (`Valid Padding`), ảnh bị co nhỏ dần sau mỗi tầng tích chập và các pixel ở rìa ảnh ít được tham gia tính toán. |
| **`dilation`** ($D$) | Hệ số giãn cách giữa các phần tử trong cùng một bộ lọc (Atrous Convolution). | Mặc định `1` (liên tục); chọn `2, 4, 6` trong DeepLab để mở rộng vùng nhìn mà không tăng số tham số và không làm giảm kích thước ảnh. | Tăng `dilation` quá lớn trên feature map nhỏ có thể gây hiệu ứng kẻ ô bàn cờ (`gridding artifact`). |
| **`groups`** ($G$) | Chia các kênh đầu vào và đầu ra thành $G$ nhóm độc lập. | `groups=1` (tích chập chuẩn); `groups=in_channels` tạo ra **Depthwise Convolution** (mỗi kênh lọc riêng 1 bộ lọc không gian, dùng trong MobileNet). | Tăng `groups = in_channels` giảm số tham số và phép nhân cộng đi gần `in_channels` lần; cần đi kèm một tầng `Conv2d(..., kernel_size=1)` phía sau để trộn thông tin giữa các kênh (Pointwise Conv). |

### 19. `activation` (Hàm kích hoạt phi tuyến)
- **Nhóm**: Kiến trúc | **Kiểu**: Hàm hoặc lớp `nn.Module` (`nn.ReLU`, `nn.GELU`, `nn.LeakyReLU`, `nn.SiLU`).
- **Định nghĩa & Vai trò**: Đặt sau tầng tuyến tính/tích chập để tạo tính **phi tuyến (non-linearity)**.
- **Vì sao bắt buộc phải có?**: Nếu bỏ toàn bộ hàm kích hoạt, việc xếp chồng 100 tầng `Linear` hay `Conv2d` liên tiếp chỉ tương đương với **1 tầng tuyến tính duy nhất** ($W_2(W_1 x) = (W_2 W_1)x = W_{\text{eq}} x$), mạng hoàn toàn mất khả năng học biên quyết định cong phức tạp.
- **Lựa chọn thực tế**: `ReLU(x) = max(0, x)` (nhanh, chuẩn cho CNN cơ bản); `GELU` / `SiLU` (mượt, chuẩn cho Vision Transformer, ConvNeXt, YOLO mới).

### 20. `pooling` (`MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d` — Gộp giảm chiều không gian)
- **Nhóm**: Kiến trúc | **Kiểu**: Lớp không có tham số học được (`total_params = 0`).
- **Định nghĩa & Vai trò**: Thu nhỏ kích thước không gian $(H, W)$ của bản đồ đặc trưng bằng cách lấy giá trị lớn nhất (`MaxPool2d`) hoặc trung bình (`AvgPool2d`) trong từng ô cửa sổ, giúp tăng tính bất biến với dịch chuyển nhỏ và mở rộng `receptive field`.
- **Phân biệt `Flatten` vs `Global Average Pooling` (`nn.AdaptiveAvgPool2d((1, 1))`)**:
  - Nếu dùng `Flatten` trực tiếp trên feature map `(64, 8, 8)` rồi nối vào `Linear(4096, 4)`: Tầng `Linear` tốn tới `16,388` tham số và bắt buộc ảnh đầu vào phải luôn cố định đúng kích thước.
  - Nếu dùng `AdaptiveAvgPool2d((1, 1))` lấy trung bình mỗi kênh thành `(64, 1, 1)` rồi mới qua `Linear(64, 4)`: Chỉ tốn `260` tham số, giảm mạnh nguy cơ overfitting!

### 21. `embedding` (Vector nhúng / Biểu diễn đặc trưng nén)
- **Nhóm**: Kiến trúc & Metric Learning | **Kiểu**: Tensor số thực `(B, D)`.
- **Định nghĩa & Vai trò**: Vector số thực chiều thấp $D$ (ví dụ `D=32`, `128`, `512`) đại diện cho ngữ nghĩa hình ảnh hoặc một token rời rạc trong không gian liên tục.

### 22. `dropout` (`nn.Dropout`, `nn.Dropout2d`)
- **Nhóm**: Kiến trúc & Điều chuẩn (`Regularization`) | **Kiểu**: Xác suất rơi `p ∈ [0.0, 1.0)`.
- **Định nghĩa & Vai trò**: Trong lúc **train**, tắt ngẫu nhiên một tỷ lệ `p` các phần tử về `0` và nhân các phần tử còn lại với $\frac{1}{1-p}$ (Inverted Dropout) để buộc mạng không được phụ thuộc độc quyền vào một vài nơ-ron cá biệt. Trong lúc **eval**, `Dropout` tự động tắt.
- **Ảnh hưởng khi tăng/giảm `p`**: `p = 0.1..0.3` giúp giảm overfitting hiệu quả; đặt `p > 0.5` ở mạng nhỏ sẽ làm mất quá nhiều thông tin khiến mạng `underfit` (khó hội tụ).

### 23. `residual connection` / `skip connection` (Kết nối phần dư / Kết nối tắt)
- **Nhóm**: Kiến trúc | **Công thức**: $y = \mathcal{F}(x) + x$ (trong ResNet) hoặc `torch.cat([decoder_feat, encoder_feat], dim=1)` (trong U-Net).
- **Vai trò cốt lõi**: Tạo "đường cao tốc" cho đạo hàm truyền ngược về các tầng đầu tiên mà không bị nhân suy giảm qua quá nhiều ma trận ($\frac{\partial y}{\partial x} = \frac{\partial \mathcal{F}(x)}{\partial x} + 1$), đồng thời giữ lại thông tin vị trí không gian sắc nét (trong U-Net).

### 24. `receptive field` (Vùng cảm nhận / Vùng nhìn hiệu dụng)
- **Nhóm**: Kiến trúc | **Kiểu**: Kích thước vùng pixel trên ảnh gốc `(R x R)` có ảnh hưởng tới 1 điểm trên bản đồ đặc trưng ở tầng thứ $L$.
- **Công thức đệ quy**:
  $$R_L = R_{L-1} + (K_L - 1) \times D_L \times \prod_{i=1}^{L-1} S_i$$
  trong đó $R_0 = 1$, $K_L$ là `kernel_size`, $D_L$ là `dilation`, $S_i$ là `stride` của tầng thứ $i$.
- **Ý nghĩa thực tế**: Nếu vật thể trong ảnh rộng `80x80` pixel mà `receptive field` ở tầng cuối của mạng chỉ đạt `15x15` pixel, mạng chỉ "nhìn thấy" từng mảnh cục bộ chứ không thể hiểu được tổng thể vật thể!

### 25. `patch_size`, `positional encoding`, `attention`, `head` (Từ khóa Vision Transformer)
- **`patch_size`**: Kích thước mỗi mảnh ảnh vuông khi chia lưới ảnh cho ViT (ví dụ `patch_size = 8` trên ảnh `32x32` tạo ra $(32/8)^2 = 16$ patches). Giảm `patch_size` giúp bắt chi tiết nhỏ tốt hơn nhưng làm số lượng token tăng bậc hai và chi phí Self-Attention tăng bậc bốn!
- **`positional encoding` (`pos_embed`)**: Vector cộng thêm vào mỗi patch embedding để báo cho Transformer biết mảnh ảnh đó nằm ở tọa độ nào (vì phép Self-Attention thuần túy bất biến với thứ tự hoán vị).
- **`attention` (Self-Attention)**: Cơ chế tính ma trận tương quan giữa mọi cặp mảnh ảnh:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **`head` (`num_heads` trong Multi-Head Attention vs `Classification Head`)**:
  - *Nghĩa 1 (`num_heads`)*: Chia không gian vector thành nhiều "đầu chú ý" song song để mỗi đầu tập trung vào một mối quan hệ không gian/màu sắc khác nhau (lưu ý `embed_dim` phải chia hết cho `num_heads`).
  - *Nghĩa 2 (`Task Head`)*: Khối các tầng cuối cùng của mạng (sau `backbone`) chuyên trách xuất kết quả cho bài toán cụ thể (Classification head, Box regression head, Mask head).


---

## Phần IV. Nhóm 3 — Từ khóa về Huấn luyện & Tối ưu hóa (`Training & Optimization`)

### 26. `learning_rate` ($\eta$ / `lr` — Tốc độ học)
- **Nhóm**: Tối ưu hóa | **Kiểu**: Số thực dương (`float`, thường từ `1e-5` đến `1e-1`).
- **Định nghĩa & Vai trò**: Hệ số nhân quyết định độ lớn của mỗi bước cập nhật trọng số ngược hướng đạo hàm: $W \leftarrow W - \eta \cdot \nabla_W \mathcal{L}$. Đây là **siêu tham số quan trọng nhất** khi huấn luyện mạng nơ-ron.
- **Ví dụ code**: `optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)`.
- **Giá trị thường chọn & Lý do**:
  - Với `Adam` / `AdamW` khi train từ đầu: `1e-3` (`0.001`) hoặc `3e-4`.
  - Với `SGD + Momentum`: `1e-2` (`0.01`) đến `1e-1` (`0.1`).
  - Khi `Fine-tuning` một backbone pretrained: `1e-4` đến `1e-5` (để không phá hỏng các đặc trưng tốt đã học sẵn).
- **Ảnh hưởng khi Tăng / Giảm**:
  - **Tăng quá cao** (ví dụ `lr = 1.0` cho Adam): Bước nhảy vọt qua đáy thung lũng hàm mất mát, làm `loss` dao động dữ dội hoặc bùng nổ thành `NaN` (`divergence`).
  - **Giảm quá thấp** (ví dụ `lr = 1e-7`): Mô hình hội tụ cực chậm, gần như đứng yên sau nhiều epoch hoặc mắc kẹt ở vùng phẳng.

### 27. `optimizer`, `momentum`, `weight_decay`, `regularization`
- **`optimizer` (Bộ tối ưu hóa)**: Thuật toán sử dụng `gradient` để cập nhật các `parameters` (tiêu biểu: `SGD`, `Adam`, `AdamW`, `RMSprop`).
- **`momentum` (Động lượng, thường `0.9` trong `SGD`)**: Tích lũy trung bình trượt của các đạo hàm trước đó ($v_t = \mu v_{t-1} + g_t$) giống như viên bi lăn xuống dốc có quán tính, giúp vượt qua các gờ lồi lõm nhỏ và giảm dao động zíc-zắc.
- **`weight_decay` (Suy giảm trọng số, thường `1e-4` đến `1e-2`)**: Kéo nhẹ các trọng số về gần `0` sau mỗi bước cập nhật để ngăn trọng số phình quá lớn, giảm độ phức tạp của mô hình.
- **`regularization` (Điều chuẩn)**: Tập hợp mọi kỹ thuật chống `overfitting` (bao gồm `weight_decay`, `dropout`, `data augmentation`, `early stopping`, `label smoothing`).

### 28. `loss`, `gradient`, `backpropagation`, `autograd`
- **`loss` (Giá trị mất mát)**: Con số vô hướng (`scalar tensor`) đo mức độ sai lệch giữa dự đoán của mô hình và nhãn thật trên batch hiện tại.
- **`gradient` ($\nabla_W \mathcal{L}$ — Đạo hàm riêng / Vector độ dốc)**: Lưu trong thuộc tính `param.grad` của mỗi tham số, cho biết nếu tăng nhẹ tham số đó thì `loss` sẽ tăng hay giảm bao nhiêu.
- **`backpropagation` (Lan truyền ngược)**: Thuật toán áp dụng quy tắc đạo hàm hàm hợp (**Chain Rule**) từ tầng cuối ngược về tầng đầu để tính `gradient` cho mọi tham số.
- **`autograd` (Tính đạo hàm tự động)**: Bộ máy của PyTorch tự động ghi lại các phép toán trên tensor có `requires_grad=True` thành đồ thị tính toán động (`Dynamic Computation Graph`) để thực thi `backpropagation` khi gọi `loss.backward()`.

### 29. Bộ 3 lệnh bất khả phân ly trong Training Loop: `zero_grad()`, `backward()`, `step()`
```python
optimizer.zero_grad()   # 1. Xóa gradient cũ từ batch trước về 0 (hoặc None)
loss.backward()         # 2. Lan truyền ngược tính gradient mới lưu vào param.grad
optimizer.step()        # 3. Cập nhật giá trị param.data dựa trên param.grad
```
- **Lỗi kinh điển**: Quên gọi `optimizer.zero_grad()` ở đầu vòng lặp batch! Mặc định PyTorch **cộng dồn (`+=`)** đạo hàm mới vào `param.grad` cũ, khiến bước cập nhật bị sai lệch hoàn toàn nếu bạn không chủ đích dùng kỹ thuật `gradient accumulation`.

### 30. `scheduler` & `warmup` (Bộ lập lịch tốc độ học & Giai đoạn khởi động ấm)
- **`scheduler` (`StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`)**: Tự động giảm dần `learning_rate` theo số epoch hoặc khi `val_loss` chững lại, giúp mô hình bước nhanh lúc đầu và bước nhỏ mịn lúc tiến gần đáy cực tiểu.
- **`warmup` (Khởi động ấm)**: Tăng dần `learning_rate` từ một giá trị rất nhỏ (`1e-6`) lên giá trị đích (`1e-3`) trong vài epoch đầu tiên. Đặc biệt quan trọng với **Vision Transformer (`ViT`)** để tránh hiện tượng đạo hàm lớn lúc mới khởi tạo làm hỏng cơ chế Self-Attention.

### 31. `early stopping` & `patience` (Dừng sớm & Độ kiên nhẫn)
- **`early stopping`**: Ngừng quá trình huấn luyện trước khi chạy hết tổng số `epochs` nếu chất lượng trên tập `validation` không còn cải thiện nữa, đồng thời khôi phục lại trọng số tại epoch tốt nhất.
- **`patience` (Số nguyên, ví dụ `patience = 5`)**: Số epoch liên tiếp chấp nhận chờ đợi khi `val_loss` không giảm thêm trước khi quyết định dừng hẳn. Tăng `patience` giúp tránh dừng quá sớm do nhiễu nhất thời; giảm `patience` giúp tiết kiệm thời gian tính toán.

### 32. `checkpoint`, `state_dict`, `seed`
- **`state_dict`**: Từ điển Python (`OrderedDict`) ánh xạ tên từng tầng sang tensor trọng số/buffer của nó (`model.state_dict()`, `optimizer.state_dict()`).
- **`checkpoint`**: File lưu trên đĩa (`.pt` / `.pth`) chứa `model_state_dict` cùng metadata cần thiết (`class_names`, `class_to_idx`, `preprocessing_config`, `epoch`, `val_metrics`).
- **`seed` (Hạt giống ngẫu nhiên, ví dụ `seed = 42`)**: Số nguyên cố định bộ sinh số giả ngẫu nhiên để hai lần chạy code tạo ra cùng cách chia dữ liệu và cùng trọng số khởi tạo.

### 33. `gradient clipping`, `gradient accumulation`, `mixed precision`
- **`gradient clipping` (`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`)**: Thu nhỏ toàn bộ vector đạo hàm nếu chuẩn L2 của nó vượt quá `max_norm`, ngăn chặn hiện tượng **bùng nổ đạo hàm (`exploding gradients`)**.
- **`gradient accumulation` (Cộng dồn đạo hàm)**: Khi bộ nhớ chỉ chứa được `batch_size=8` nhưng bạn muốn hiệu ứng của `batch_size=32`, bạn chạy `forward` và `(loss / 4).backward()` qua `4` lô liên tiếp rồi mới gọi `optimizer.step()` và `optimizer.zero_grad()` một lần.
- **`mixed precision` (`torch.amp.autocast`)**: Kết hợp tính toán bằng số thực 16-bit (`float16` / `bfloat16`) cho các phép nhân ma trận nặng và 32-bit (`float32`) cho bước cộng dồn/hàm loss để tăng tốc và giảm ~40% VRAM trên GPU hiện đại.

### 34. `requires_grad`, `freeze`, `unfreeze`, `train()`, `eval()`, `no_grad()`, `inference_mode()`
- **`param.requires_grad`**: Cờ kiểu `bool` báo cho PyTorch có cần tính đạo hàm cho tham số này hay không.
- **`freeze` (Đóng băng tầng)**: Đặt `param.requires_grad = False` cho các tầng `backbone` khi làm Feature Extraction để giữ nguyên trọng số pretrained và tăng tốc độ huấn luyện.
- **`unfreeze` (Mở khóa tầng)**: Đặt lại `param.requires_grad = True` khi chuyển sang giai đoạn Fine-tuning.
- **`torch.inference_mode()`**: Phiên bản tối ưu hơn của `torch.no_grad()` dùng chuyên cho bước suy luận (`inference`), tắt hoàn toàn việc theo dõi phiên bản tensor để đạt tốc độ cao nhất.

---

## Phần V. Nhóm 4 — Từ khóa về Đánh giá Mô hình (`Evaluation Metrics`)

### 35. `confusion matrix`, `accuracy`, `precision`, `recall`, `specificity`, `F1`, `support`
Xét bài toán phân loại đối với một lớp dương tính (`Positive`):
- `TP` (*True Positive*): Thật sự là Dương, đoán đúng là Dương.
- `FP` (*False Positive* - Báo động giả): Thật sự là Âm, đoán nhầm là Dương.
- `FN` (*False Negative* - Bỏ sót): Thật sự là Dương, đoán nhầm là Âm.
- `TN` (*True Negative*): Thật sự là Âm, đoán đúng là Âm.

| Độ đo (Metric) | Công thức toán học | Ý nghĩa trực quan & Khi nào ưu tiên? |
|---|---|---|
| **`Confusion Matrix`** | Ma trận $C \times C$ với $M_{ij}$ là số mẫu thuộc lớp thật $i$ bị dự đoán thành lớp $j$ | Giúp nhìn ngay mô hình đang nhầm lẫn cặp lớp nào với nhau nhiều nhất (ví dụ nhầm `square` với `cross`). |
| **`Accuracy`** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Tỷ lệ đoán đúng trên tổng số mẫu. **Chỉ đáng tin cậy khi các lớp cân bằng nhau**. |
| **`Precision`** (Độ chuẩn xác) | $\frac{TP}{TP + FP}$ | *"Trong những mẫu mô hình báo là Dương, có bao nhiêu % là Dương thật?"* Ưu tiên khi chi phí **báo động giả (`FP`)** rất đắt (ví dụ: lọc thư rác, hủy bỏ linh kiện đắt tiền). |
| **`Recall` / `Sensitivity`** (Độ nhạy / Độ phủ) | $\frac{TP}{TP + FN}$ | *"Trong toàn bộ các mẫu Dương thực tế, mô hình bắt được bao nhiêu %?"* Ưu tiên tuyệt đối khi chi phí **bỏ sót (`FN`)** rất nguy hiểm (ví dụ: tầm soát khối u y tế, phát hiện phanh xe bị nứt). |
| **`Specificity`** (Độ đặc hiệu) | $\frac{TN}{TN + FP}$ | *"Trong toàn bộ các mẫu Âm (bình thường), mô hình nhận diện đúng bao nhiêu % là bình thường?"* Đi cặp với `Recall` trong chẩn đoán y khoa và kiểm định chất lượng. |
| **`F1-score`** | $\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Trung bình điều hòa của `Precision` và `Recall`. Chỉ cao khi **cả hai** cùng cao. |
| **`Support`** | $TP + FN$ | Số lượng mẫu thực tế của từng lớp trong tập đánh giá (giúp biết độ đo của lớp đó được tính trên bao nhiêu mẫu). |

### 36. `macro`, `micro`, `weighted average` (3 cách gộp F1 / Precision / Recall đa lớp)
- **`Macro average`**: Tính `F1` riêng cho từng lớp rồi lấy **trung bình cộng đều** ($\frac{1}{C}\sum_{c=1}^C F1_c$). Coi mọi lớp quan trọng ngang nhau dù lớp đó có ít mẫu — **nên dùng khi dữ liệu mất cân bằng lớp**.
- **`Weighted average`**: Lấy trung bình có trọng số theo tỷ lệ `support` của từng lớp. Phản ánh chất lượng tổng thể theo phân phối mẫu hiện tại.
- **`Micro average`**: Cộng gộp toàn bộ `TP`, `FP`, `FN` của mọi lớp lại rồi mới tính. Trong phân loại đơn lớp (mỗi mẫu đúng 1 nhãn), `Micro-F1 == Micro-Precision == Micro-Recall == Accuracy`.

### 37. `threshold`, `ROC-AUC`, `PR-AUC`, `calibration`
- **`threshold` (Ngưỡng quyết định, mặc định `0.5`)**: Ngưỡng cắt xác suất $p \ge \tau$ để kết luận lớp Dương.
  - **Giảm `threshold`** (ví dụ `0.3`): Mô hình dễ báo Dương hơn → **`Recall` tăng, `Precision` giảm**.
  - **Tăng `threshold`** (ví dụ `0.8`): Mô hình thận trọng hơn → **`Precision` tăng, `Recall` giảm**.
- **`ROC-AUC`**: Diện tích dưới đường cong biểu diễn `TPR (Recall)` theo `FPR (1 - Specificity)` khi quét mọi ngưỡng `threshold`.
- **`PR-AUC` (Average Precision)**: Diện tích dưới đường cong `Precision – Recall`. Khi lớp Dương rất hiếm (ví dụ `< 5%`), **`PR-AUC` phản ánh chất lượng trung thực hơn nhiều so với `ROC-AUC`** (vì `TN` quá lớn làm `FPR` trong ROC luôn nhỏ giả tạo).
- **`calibration` (Độ hiệu chỉnh xác suất — đo bằng `ECE / Expected Calibration Error`)**: Kiểm tra xem khi mô hình dự đoán độ tự tin `80%`, thì trong thực tế có đúng khoảng `80%` các dự đoán đó là chính xác hay không (mạng sâu hiện đại thường bị tự tin thái quá – overconfident).

### 38. `IoU`, `Dice`, `AP`, `mAP` (Độ đo Không gian cho Detection & Segmentation)
- **`IoU` (Intersection over Union / Jaccard Index)**:
  $$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|} \in [0, 1]$$
  Đo mức độ chồng khít giữa hai khung hộp hoặc hai vùng mặt nạ (`IoU >= 0.5` là ngưỡng chấp nhận cơ bản, `>= 0.75` là định vị chặt).
- **`Dice Coefficient` (Hệ số Sørensen–Dice)**:
  $$\text{Dice}(A, B) = \frac{2 |A \cap B|}{|A| + |B|} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$$
  Tương đương với `F1-score` tính trên mức điểm ảnh.
- **`AP` (Average Precision) & `mAP` (mean Average Precision)**:
  - `AP`: Diện tích dưới đường cong Precision–Recall của 1 lớp vật thể tại một ngưỡng `IoU` cố định (ví dụ `AP@0.5`).
  - `mAP`: Trung bình cộng `AP` trên tất cả các lớp vật thể.

### 39. `MAE`, `MSE`, `RMSE` (Độ đo Hồi quy & Khôi phục ảnh)
- **`MAE` (Mean Absolute Error — Sai số tuyệt đối trung bình)**: $\frac{1}{N}\sum |y_i - \hat{y}_i|$. Bền vững với điểm ngoại lai (`outliers`).
- **`MSE` (Mean Squared Error — Sai số bình phương trung bình)**: $\frac{1}{N}\sum (y_i - \hat{y}_i)^2$. Phạt rất nặng các sai số lớn.
- **`RMSE` (Root Mean Squared Error)**: $\sqrt{\text{MSE}}$, đưa đơn vị đo về đúng đơn vị gốc của biến đích (ví dụ số pixel hoặc mét).

### 40. `latency`, `throughput`, `model size` (Chỉ số Triển khai Hệ thống)
- **`latency` (Độ trễ suy luận, đơn vị `ms / ảnh`)**: Thời gian từ lúc đưa 1 ảnh vào đến khi nhận kết quả (quyết định tính thời gian thực cho xe tự hành, dây chuyền băng tải).
- **`throughput` (Thông lượng, đơn vị `ảnh / giây` hoặc `FPS`)**: Tổng số ảnh xử lý được trong 1 giây (thường tăng khi xử lý gom theo lô `batch` trên GPU).
- **`model size` (Dung lượng mô hình, đơn vị `MB` và triệu tham số `M params`)**: Dung lượng file trọng số trên đĩa và trong RAM/VRAM.

---

## Phần VI. Nhóm 5 — Từ khóa Chuyên biệt Theo Bài toán (`Task-Specific Keywords`)

### 41. Từ khóa Object Detection: `bounding box`, `anchor`, `objectness`, `NMS`
- **`bounding box` (Khung giới hạn)**: Hình chữ nhật bao quanh vật thể, biểu diễn bằng `[x_min, y_min, x_max, y_max]` (`xyxy`), `[x_min, y_min, w, h]` (`xywh`) hoặc `[cx, cy, w, h]` (`cxcywh`).
- **`anchor` (Khung neo / Khung mẫu)**: Các hình chữ nhật có tỉ lệ khung hình (`aspect ratio`) và kích thước (`scale`) đặt sẵn tại mỗi vị trí trên bản đồ đặc trưng để mạng dự đoán độ dịch chuyển tương đối so với khung mẫu đó.
- **`objectness` (Điểm có vật thể)**: Xác suất một khung ứng viên có chứa bất kỳ vật thể nào (bất kể thuộc lớp gì) so với phông nền (`background`).
- **`NMS` (Non-Maximum Suppression — Triệt tiêu phi cực đại)**: Thuật toán hậu xử lý loại bỏ các khung dự đoán trùng lặp quanh cùng một vật thể:
  1. Chọn khung có điểm tự tin (`score`) cao nhất đưa vào danh sách kết quả.
  2. Xóa tất cả các khung cùng lớp còn lại có `IoU` với khung vừa chọn lớn hơn ngưỡng `nms_iou_threshold` (ví dụ `0.45`).
  3. Lặp lại cho đến khi hết khung ứng viên.

### 42. Từ khóa Segmentation: `mask`, `semantic`, `instance`, `panoptic`
- **`mask` (Mặt nạ phân vùng)**: Mảng 2D `(H, W)` quy định từng pixel thuộc nền hay thuộc đối tượng nào.
- Xem bảng phân biệt chi tiết `Semantic`, `Instance`, `Panoptic` và `Promptable Segmentation` tại `MODEL_CATALOG.md` (Nhóm 3).

### 43. Từ khóa Metric Learning: `positive pair`, `negative pair`, `triplet margin`
- **`positive pair` (Cặp dương)**: Hai ảnh thuộc cùng một lớp/cùng một thực thể, cần được kéo lại gần nhau trong không gian nhúng.
- **`negative pair` (Cặp âm)**: Hai ảnh thuộc hai lớp khác nhau, cần được đẩy ra xa nhau.
- **`triplet margin` ($m$, thường `0.2` đến `1.0`)**: Khoảng cách an toàn tối thiểu mà khoảng cách cặp âm $d(A, N)$ phải lớn hơn khoảng cách cặp dương $d(A, P)$:
  $$\mathcal{L}_{\text{triplet}} = \max\big(0,\; d(z_A, z_P) - d(z_A, z_N) + m\big)$$
  Nếu tăng `margin` quá lớn, mạng khó thỏa mãn điều kiện và loss giữ ở mức cao; nếu `margin` quá nhỏ, các cụm lớp đứng sát nhau dễ bị nhầm ở vùng biên.

### 44. Từ khóa Autoencoder & Generative Models: `latent vector`, `reconstruction loss`, `KL divergence`, `generator`, `discriminator`, `noise schedule`, `timestep`, `denoising`
- **`latent vector` ($z$ — Vector ẩn / Biểu diễn nút cổ chai)**: Vector chiều thấp chứa thông tin cốt lõi đã nén của bức ảnh.
- **`reconstruction loss` (Hàm mất mát tái tạo)**: Đo độ sai khác giữa ảnh đầu vào $x$ và ảnh tái tạo $\hat{x}$ (`MSE` hoặc `L1`).
- **`KL divergence` ($D_{\text{KL}}$ — Phân kỳ Kullback–Leibler)**: Trong VAE, đo khoảng cách giữa phân phối ẩn $\mathcal{N}(\mu(x), \sigma^2(x))$ và phân phối chuẩn tắc $\mathcal{N}(0, I)$, giúp không gian ẩn liên tục và lấy mẫu sinh ảnh mới được.
- **`generator` ($G$) & `discriminator` ($D$)**: Hai mạng đối kháng trong GAN — $G$ tạo ảnh giả từ nhiễu, $D$ chấm điểm xác suất ảnh là thật hay giả.
- **`timestep` ($t \in \{1, \dots, T\}$) & `noise schedule` ($\beta_t$)**: Trong Diffusion Models, `timestep` là chỉ số bước thời gian thêm nhiễu, còn `noise schedule` quy định lượng phương sai nhiễu $\beta_t$ được cộng thêm ở mỗi bước $t$.
- **`denoising` (Khử nhiễu)**: Quá trình loại bỏ nhiễu khỏi ảnh (cả trong bài toán khôi phục ảnh lẫn bước suy luận ngược sinh ảnh của mô hình Diffusion).
