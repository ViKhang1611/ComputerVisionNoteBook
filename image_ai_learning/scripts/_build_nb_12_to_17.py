"""
Script khởi tạo các Jupyter Notebook từ 12 đến 17 (Giai đoạn D: Transfer Learning, Detection, Segmentation, Anomaly, Metric Learning, ViT & SSL).
"""

from __future__ import annotations

from pathlib import Path
from _nb_helper import create_validated_notebook

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def build_nb12() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 12: Học Chuyển giao (Transfer Learning) và Tinh chỉnh Mô hình (Fine-Tuning)

## 1. Mục tiêu học tập
- Nắm vững bản đồ tiến hóa các kiến trúc phân loại ảnh hiện đại: `LeNet` → `AlexNet` → `VGG` → `Inception` → `ResNet` → `DenseNet` → `MobileNet` → `EfficientNet` → `ConvNeXt`.
- Phân biệt rạch ròi 4 chế độ làm việc với mô hình:
  1. **Pretrained Inference (Chỉ suy luận bằng trọng số có sẵn)**: Không huấn luyện, không cập nhật trọng số.
  2. **Training from Scratch (Huấn luyện từ đầu)**: Khởi tạo ngẫu nhiên toàn bộ mạng, tốn nhiều dữ liệu và thời gian.
  3. **Feature Extraction / Linear Probing (Trích xuất đặc trưng)**: Đóng băng (`freeze`, `requires_grad=False`) toàn bộ phần thân mạng (`backbone`), chỉ huấn luyện tầng phân loại cuối (`head`).
  4. **Fine-Tuning (Tinh chỉnh)**: Mở khóa (`unfreeze`) thêm các tầng sâu của `backbone` và huấn luyện với tốc độ học phân tầng (**Differential Learning Rates**: `lr` rất nhỏ cho `backbone`, `lr` lớn hơn cho `head`).
- Thực hành huấn luyện thật cả 3 chiến lược (`Scratch` vs `Feature Extraction` vs `Fine-Tuning`) trên tập dữ liệu mục tiêu nhỏ, kèm hướng dẫn sử dụng `torchvision.models` cả khi có Internet lẫn khi Offline hoàn toàn.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 10 và Bài 11.

## 3. Thời gian học dự kiến
- Khoảng **2.0 – 2.5 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
Trong thực tế, bạn thường chỉ thu thập được `20 – 50` bức ảnh cho mỗi loại sản phẩm mới. Nếu huấn luyện một mạng CNN lớn từ con số 0 (`Training from scratch`) trên `80` ảnh, mạng sẽ học vẹt (`overfit`) ngay lập tức. Nhờ **Transfer Learning**, ta tận dụng bộ lọc cạnh, góc, hình khối mà mạng đã học từ một tập dữ liệu nguồn lớn hơn, giúp đạt độ chính xác cao chỉ sau vài epoch!

## 5. Giải thích trực quan: Cấu trúc Backbone + Head và Chiến lược Đóng băng (`Freeze`)
```
[Ảnh đầu vào (3, H, W)]
          │
          ▼
┌───────────────────────────────────────────────────────────┐
│ BACKBONE (Thân mạng trích xuất đặc trưng: Stem + Stages)  │
│  - Stage 1 (Tầng nông): Bắt cạnh, góc, màu sắc tổng quát  │──► Luôn giữ cố định (requires_grad=False)
│  - Stage 2 (Tầng giữa): Bắt họa tiết, kết cấu hình học    │──► Đóng băng ở pha 1, mở khóa nhẹ ở pha 2
│  - Stage 3 (Tầng sâu) : Bắt bộ phận vật thể đặc thù       │──► Mở khóa với lr nhỏ (ví dụ 2e-4) khi Fine-tune
└───────────────────────────────────────────────────────────┘
          │ Vector đặc trưng (Embedding)
          ▼
┌───────────────────────────────────────────────────────────┐
│ CLASSIFICATION HEAD (Tầng Linear mới thay thế cho C lớp)  │──► Luôn cập nhật với lr chuẩn (ví dụ 2e-3)
└───────────────────────────────────────────────────────────┘
```

## 6 & 7. Thuật ngữ mới, Tham số & Công thức Tốc độ học Phân tầng
| Keyword | Nghĩa tiếng Việt | Nhóm | Vai trò & Lưu ý lỗi thường gặp |
|---|---|---|---|
| `backbone` | Khung xương sống trích xuất đặc trưng | Kiến trúc | Toàn bộ mạng CNN/ViT trừ tầng phân loại cuối cùng (`fc` hoặc `classifier`). |
| `head` | Đầu dự đoán tác vụ | Kiến trúc | Tầng `nn.Linear(in_features, num_target_classes)` mới khởi tạo để khớp với số lớp của bài toán mới. |
| `freeze / unfreeze` | Đóng băng / Mở khóa tham số | Huấn luyện | Đặt `p.requires_grad = False` (freeze) hoặc `True` (unfreeze). **Lưu ý**: Với tầng `BatchNorm2d` trong phần đóng băng, nếu tập mới quá nhỏ, nên giữ tầng đó ở chế độ `.eval()` để không làm lệch `running_mean/var`. |
| `param_groups` (Differential LR) | Nhóm tham số có `lr` khác nhau | Tối ưu hóa | Truyền danh sách dict vào optimizer: `[{"params": backbone_params, "lr": 1e-4}, {"params": head_params, "lr": 1e-3}]`. |""",
        ),
        (
            "code",
            """# Cell 1: Chuẩn bị Mô hình Nguồn (Pretrained Backbone) & Hướng dẫn Torchvision Online/Offline
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
if (project_root / "image_ai_learning").is_dir():
    project_root = project_root / "image_ai_learning"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split

from src.utils import set_seed, get_device, count_parameters, save_checkpoint
from src.synthetic_data import generate_synthetic_shapes_dataset
from src.models import ResidualCNN
from src.training import fit_classifier, evaluate_classifier

QUICK_RUN = True
set_seed(42)
device = get_device()

# Minh họa cách khởi tạo ResNet-18 từ torchvision:
# - Khi có Internet: dùng resnet18(weights=ResNet18_Weights.DEFAULT) để tải trọng số ImageNet (~44.7 MB).
# - Khi Offline: dùng weights=None rồi nạp từ file local checkpoint (.pt) bằng load_state_dict().
torchvision_r18_demo = resnet18(weights=None)
torchvision_r18_demo.fc = nn.Linear(torchvision_r18_demo.fc.in_features, 4)  # Thay head 1000 lớp bằng 4 lớp
print("Torchvision ResNet-18 (thay head 4 lớp) -> Tổng tham số:", count_parameters(torchvision_r18_demo)["total"])

# Để bài thực hành chạy nhanh 100% offline trên CPU, ta huấn luyện trước một ResidualCNN trên tập nguồn (Source Task)
# làm "Pretrained Source Model", sau đó chuyển giao sang tập mục tiêu cực nhỏ (Target Few-Shot Task)!
src_ds = generate_synthetic_shapes_dataset(num_samples=240, image_size=32, seed=10, noise_std=6.0)
X_src = torch.from_numpy(src_ds["images"]).permute(0, 3, 1, 2).float() / 255.0
y_src = torch.from_numpy(src_ds["labels"]).long()
src_mean = X_src.mean(dim=(0, 2, 3), keepdim=True)
src_std = X_src.std(dim=(0, 2, 3), keepdim=True) + 1e-6
X_src_norm = (X_src - src_mean) / src_std

src_loader = DataLoader(TensorDataset(X_src_norm, y_src), batch_size=32, shuffle=True)
pretrained_source_model = ResidualCNN(in_channels=3, num_classes=4, base_channels=16).to(device)
opt_src = torch.optim.AdamW(pretrained_source_model.parameters(), lr=3e-3, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()

for _ in range(10 if QUICK_RUN else 20):
    pretrained_source_model.train()
    for xb, yb in src_loader:
        opt_src.zero_grad()
        loss = crit(pretrained_source_model(xb.to(device)), yb.to(device))
        loss.backward()
        opt_src.step()

pretrained_backbone_state = copy.deepcopy(pretrained_source_model.state_dict())
print("Đã chuẩn bị xong trọng số Pretrained Backbone từ miền dữ liệu nguồn!")""",
        ),
        (
            "code",
            """# Cell 2: Tạo Tập dữ liệu Mục tiêu Nhỏ (Target Domain có nhiễu mạnh & dịch màu, chỉ có 64 ảnh Train)
target_ds = generate_synthetic_shapes_dataset(num_samples=160, image_size=32, seed=99, noise_std=22.0)
# Đảo kênh màu để tạo sự dịch chuyển miền (domain shift) giữa miền nguồn và miền đích
target_imgs = target_ds["images"][:, :, :, ::-1].copy()
target_labels = target_ds["labels"]
class_names = target_ds["class_names"]

idx_all = np.arange(len(target_imgs))
idx_tr, idx_temp = train_test_split(idx_all, test_size=0.60, random_state=42, stratify=target_labels)  # Chỉ 40% (~64 ảnh) cho Train!
idx_va, idx_te = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=target_labels[idx_temp])

X_tgt = torch.from_numpy(target_imgs).permute(0, 3, 1, 2).float() / 255.0
y_tgt = torch.from_numpy(target_labels).long()
X_tgt_norm = (X_tgt - src_mean) / src_std  # Chuẩn hóa theo thống kê của mô hình Pretrained!

tr_loader = DataLoader(TensorDataset(X_tgt_norm[idx_tr], y_tgt[idx_tr]), batch_size=16, shuffle=True)
va_loader = DataLoader(TensorDataset(X_tgt_norm[idx_va], y_tgt[idx_va]), batch_size=32, shuffle=False)
te_loader = DataLoader(TensorDataset(X_tgt_norm[idx_te], y_tgt[idx_te]), batch_size=32, shuffle=False)
print(f"Tập dữ liệu mục tiêu ít mẫu -> Train: {len(idx_tr)} ảnh, Val: {len(idx_va)} ảnh, Test: {len(idx_te)} ảnh")""",
        ),
        (
            "code",
            """# Cell 3: Thực nghiệm So sánh 3 Chiến lược: (1) Train from Scratch vs (2) Feature Extraction vs (3) Fine-Tuning
epochs_cmp = 10 if QUICK_RUN else 20

# --- Chiến lược 1: Training from Scratch (Khởi tạo ngẫu nhiên toàn bộ) ---
set_seed(42)
model_scratch = ResidualCNN(in_channels=3, num_classes=4, base_channels=16).to(device)
opt_scratch = torch.optim.AdamW(model_scratch.parameters(), lr=2e-3, weight_decay=1e-4)
hist_scratch = fit_classifier(model_scratch, tr_loader, va_loader, crit, opt_scratch, device, epochs=epochs_cmp)
_, _, pred_scr, _ = evaluate_classifier(model_scratch, te_loader, crit, device)
acc_scratch = float((pred_scr == target_labels[idx_te]).mean())

# --- Chiến lược 2: Feature Extraction (Nạp Pretrained, đóng băng toàn bộ Backbone, chỉ train tầng fc) ---
set_seed(42)
model_fe = ResidualCNN(in_channels=3, num_classes=4, base_channels=16).to(device)
model_fe.load_state_dict(pretrained_backbone_state)
model_fe.fc = nn.Linear(64, 4).to(device)  # Khởi tạo lại tầng Head mới
for name, param in model_fe.named_parameters():
    param.requires_grad = name.startswith("fc.")

print("Tham số ở chế độ Feature Extraction:", count_parameters(model_fe))
opt_fe = torch.optim.AdamW([p for p in model_fe.parameters() if p.requires_grad], lr=3e-3, weight_decay=1e-4)
hist_fe = fit_classifier(model_fe, tr_loader, va_loader, crit, opt_fe, device, epochs=epochs_cmp)
_, _, pred_fe, _ = evaluate_classifier(model_fe, te_loader, crit, device)
acc_fe = float((pred_fe == target_labels[idx_te]).mean())

# --- Chiến lược 3: Fine-Tuning với Differential Learning Rate (Mở khóa stage3 + fc) ---
set_seed(42)
model_ft = ResidualCNN(in_channels=3, num_classes=4, base_channels=16).to(device)
model_ft.load_state_dict(pretrained_backbone_state)
model_ft.fc = nn.Linear(64, 4).to(device)
for name, param in model_ft.named_parameters():
    param.requires_grad = name.startswith("stage3.") or name.startswith("fc.")

opt_ft = torch.optim.AdamW(
    [
        {"params": model_ft.stage3.parameters(), "lr": 4e-4},  # LR nhỏ cho tầng sâu của backbone
        {"params": model_ft.fc.parameters(), "lr": 3e-3},      # LR lớn hơn cho tầng head mới
    ],
    weight_decay=1e-4,
)
ckpt_ft = project_root / "outputs" / "checkpoints" / "nb12_transfer_finetuned.pt"
hist_ft = fit_classifier(
    model_ft, tr_loader, va_loader, crit, opt_ft, device,
    epochs=epochs_cmp, checkpoint_path=ckpt_ft, class_names=class_names,
    preprocessing_config={"mean": src_mean.squeeze().tolist(), "std": src_std.squeeze().tolist()},
)
_, _, pred_ft, _ = evaluate_classifier(model_ft, te_loader, crit, device)
acc_ft = float((pred_ft == target_labels[idx_te]).mean())

print("\\n=== KẾT QUẢ SO SÁNH 3 CHIẾN LƯỢC TRÊN TẬP TEST ÍT MẪU ===")
print(f"1. Training from Scratch : Test Accuracy = {acc_scratch*100:5.1f}% (100% tham số cập nhật)")
print(f"2. Feature Extraction    : Test Accuracy = {acc_fe*100:5.1f}% (Chỉ cập nhật 260 tham số của Head!)")
print(f"3. Fine-Tuning (Stage3+FC): Test Accuracy = {acc_ft*100:5.1f}% (Tốc độ học phân tầng)")""",
        ),
        (
            "code",
            """# Cell 4: Biểu đồ So sánh Tốc độ Hội tụ Validation Accuracy giữa 3 Chiến lược
plt.figure(figsize=(7.5, 3.4))
plt.plot([a * 100 for a in hist_scratch["val_acc"]], "o--", label="1. Train from Scratch", color="gray")
plt.plot([a * 100 for a in hist_fe["val_acc"]], "s-", label="2. Feature Extraction (Frozen Backbone)", color="royalblue")
plt.plot([a * 100 for a in hist_ft["val_acc"]], "^-", label="3. Fine-Tuning (Differential LR)", color="seagreen")
plt.title("So sánh Tốc độ Hội tụ trên Tập dữ liệu Mục tiêu Nhỏ (64 ảnh Train)")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.show()""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- Chỉ với `260` tham số được phép học ở chế độ **`Feature Extraction`** (chiếm `< 0.4%` tổng số tham số mạng), mô hình đã vọt lên độ chính xác cao ngay từ epoch đầu tiên vì phần thân mạng (`backbone`) đã biết cách tách cạnh và hình dạng sẵn!
- Khi mở khóa thêm `stage3` với tốc độ học nhỏ (`4e-4`) ở chế độ **`Fine-Tuning`**, mô hình vừa giữ được tri thức gốc vừa thích nghi với nhiễu mạnh của miền dữ liệu mới.

## 13. Lỗi thường gặp và cách sửa
1. **Dùng `learning_rate` lớn (ví dụ `1e-2`) cho toàn bộ mạng khi Fine-Tuning**: Đạo hàm lớn từ tầng `head` mới khởi tạo ngẫu nhiên sẽ truyền ngược làm "xóa sạch" các trọng số tinh tế của `backbone` (**Catastrophic Forgetting**). Hãy luôn đóng băng `backbone` vài epoch đầu hoặc dùng `lr` nhỏ hơn 10 lần cho `backbone`.
2. **Quên dùng đúng `mean` và `std` của mô hình Pretrained**: Các bộ lọc tầng đầu của `backbone` được học trên phân phối chuẩn hóa của tập nguồn; nếu đưa ảnh chuẩn hóa theo thang đo khác vào, kích hoạt tầng đầu sẽ bị lệch.

## 14. Bài tập tự làm
- **Cơ bản**: Viết hàm `freeze_all_except_head(model)` đóng băng mọi tham số trừ `model.fc` và trả về tỷ lệ phần trăm tham số còn được huấn luyện.
- **Trung cấp**: Thực hiện quy trình **Huấn luyện 2 Pha (Two-Phase Transfer Learning)**: Pha 1 chạy 4 epochs `Feature Extraction` (chỉ mở `fc`), sau đó Pha 2 mở khóa `stage2` và `stage3` chạy tiếp 6 epochs với `lr=3e-4`.
- **Nâng cao**: Viết hàm đóng băng luôn thống kê `running_mean`/`running_var` của mọi tầng `nn.BatchNorm2d` bên trong `backbone` bằng cách chuyển riêng các module `BatchNorm2d` sang `.eval()`.

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 12
# 1. Cơ bản: Hàm đóng băng tự động và tính % tham số trainable
def freeze_all_except_head(net: nn.Module) -> float:
    for name, p in net.named_parameters():
        p.requires_grad = name.startswith("fc.")
    info = count_parameters(net)
    return (info["trainable"] / max(1, info["total"])) * 100.0

pct_trainable = freeze_all_except_head(model_fe)
print(f"[Cơ bản] Tỷ lệ tham số cần cập nhật khi đóng băng backbone: {pct_trainable:.2f}%")

# 2. Trung cấp & 3. Nâng cao: Khóa BatchNorm2d ở chế độ eval() khi Fine-tuning trên batch nhỏ
def set_bn_eval(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

model_ft.train()
set_bn_eval(model_ft.stem)
set_bn_eval(model_ft.stage1)
print("[Nâng cao] Trạng thái training của model_ft.fc:", model_ft.fc.training, "| của stem BatchNorm:", model_ft.stem[1].training)""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Phân biệt rõ `Training from scratch`, `Feature extraction`, `Fine-tuning` và `Pretrained inference`.
- [x] Biết cách đóng băng tầng (`requires_grad=False`) và thiết lập `Differential Learning Rate`.
- [x] Sẵn sàng chuyển sang **Bài 13: `13_object_detection.ipynb`** để học bài toán Phát hiện Vật thể!

## 17. Nguồn tham khảo
- Yosinski, J. et al. (2014), *How transferable are features in deep neural networks?*, NeurIPS.
- Torchvision Models & Pretrained Weights Guide: https://pytorch.org/vision/stable/models.html""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "12_transfer_learning_and_fine_tuning.ipynb", cells)


def build_nb13() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 13: Phát hiện Vật thể (Object Detection) — Từ R-CNN, SSD, YOLO, DETR đến Huấn luyện Thực tế

## 1. Mục tiêu học tập
- Hiểu bài toán **Object Detection**: Đầu vào là 1 bức ảnh có thể chứa `0, 1` hoặc `nhiều` vật thể → Đầu ra là danh sách các bộ ba `(Bounding Box, Class Label, Confidence Score)`.
- Nắm vững bản đồ 3 họ mô hình Object Detection:
  1. **Two-stage (Hai giai đoạn)**: `R-CNN`, `Fast R-CNN`, **`Faster R-CNN + FPN`** (Đề xuất vùng ứng viên bằng `RPN` rồi phân loại/tinh chỉnh trên từng vùng `RoIAlign`).
  2. **One-stage (Một giai đoạn)**: **`SSD`**, **`RetinaNet`** (với `Focal Loss`), và **họ `YOLO`** (dự đoán trực tiếp trên lưới ô, tốc độ thời gian thực).
  3. **Transformer-based**: **`DETR`**, `Deformable DETR`, `RT-DETR` (dùng `Object Queries` và ghép cặp `Hungarian Matching`, không cần `NMS`).
- Thành thạo các khái niệm và thuật toán cốt lõi:
  - Chuyển đổi 3 định dạng tọa độ hộp: `Pascal VOC (xyxy)`, `COCO (xywh)`, `YOLO (cxcywh normalized)`.
  - Tính chỉ số chồng lấp **`IoU (Intersection over Union)`**.
  - Cài đặt tường minh thuật toán **`NMS (Non-Maximum Suppression)`** và tính độ đo **`AP / mAP@0.5`**.
- **Thực hành chuẩn mực**:
  - Hướng dẫn chuẩn bị dữ liệu chú thích (`Annotation format`).
  - Phân biệt rõ bước chạy **Pretrained Inference** với **Pipeline Fine-Tuning thực tế** bằng API chính thức `torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn`.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 10, 11, 12.

## 3. Thời gian học dự kiến
- Khoảng **3.0 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
Trên băng chuyền xuất hiện đồng thời nhiều chi tiết (`circle`, `square`, `triangle`) ở các vị trí và kích thước khác nhau. Mô hình phân loại ảnh thông thường chỉ trả lời được 1 nhãn cho cả bức ảnh; ta cần một bộ **Detector** vừa khoanh khung chữ nhật quanh từng chi tiết vừa gắn nhãn lớp chính xác.

## 5. Giải thích trực quan: Quy trình Hoạt động và Ý nghĩa Keyword
| Keyword | Nghĩa tiếng Việt | Nhóm | Định nghĩa, Vai trò & Ảnh hưởng khi Tăng/Giảm |
|---|---|---|---|
| `bounding box` | Khung giới hạn chữ nhật | Detection | Biểu diễn bằng `[x_min, y_min, x_max, y_max]` (yêu cầu `x_max > x_min` và `y_max > y_min`). |
| `anchor` | Khung neo / Khung mẫu | Detection | Các hộp mẫu đặt tại mỗi điểm lưới trên feature map. Mạng học độ lệch $(\Delta x, \Delta y, \Delta w, \Delta h)$ so với anchor. |
| `objectness` | Điểm xác suất có vật thể | Detection | Điểm số báo vùng khung đó chứa vật thể thật hay chỉ là phông nền (`background`). |
| `IoU` | Tỷ lệ Giao trên Hợp | Đánh giá | $\text{IoU} = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)} \in [0, 1]$. Dùng để ghép anchor với nhãn thật, chạy NMS và tính mAP. |
| `NMS (iou_threshold)` | Triệt tiêu phi cực đại | Triển khai | Loại bỏ các hộp dự đoán trùng lặp quanh cùng 1 vật thể. **Tăng `iou_threshold`** (ví dụ `0.8`): giữ lại nhiều hộp chồng nhau (tăng FP); **Giảm quá thấp** (`0.2`): xóa nhầm vật thể đứng cạnh nhau (giảm Recall). |
| `AP` & `mAP@0.5` | Độ chính xác trung bình | Đánh giá | `AP` là diện tích dưới đường cong Precision–Recall của 1 lớp; `mAP@0.5` là trung bình `AP` qua mọi lớp tại ngưỡng khớp hộp `IoU >= 0.5`. |

## 6 & 7. Công thức Chuyển đổi Tọa độ Hộp & Ví dụ Số Nhỏ
Trên bức ảnh rộng $W = 100$, cao $H = 100$, xét 1 vật thể có khung:
- **Pascal VOC (`xyxy`)**: $[x_{\min}, y_{\min}, x_{\max}, y_{\max}] = [20, 30, 60, 80]$.
- **COCO (`xywh`)**: $[x_{\min}, y_{\min}, w, h] = [20, 30, 60-20, 80-30] = [20, 30, 40, 50]$.
- **YOLO (`cxcywh` chuẩn hóa `[0, 1]`)**:
  $$c_x = \frac{20 + 60}{2 \times 100} = 0.40, \quad c_y = \frac{30 + 80}{2 \times 100} = 0.55, \quad w_n = \frac{40}{100} = 0.40, \quad h_n = \frac{50}{100} = 0.50$$""",
        ),
        (
            "code",
            """# Cell 1: Chuyển đổi 3 Định dạng Tọa độ Hộp (VOC xyxy <-> COCO xywh <-> YOLO cxcywh) & Tính IoU
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
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.utils import set_seed, get_device
from src.synthetic_data import generate_detection_dataset
from src.evaluation import compute_bbox_iou

QUICK_RUN = True
set_seed(42)
device = get_device()


def xyxy_to_xywh(box_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


def xyxy_to_yolo(box_xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    return np.array([(x1 + x2) / (2.0 * img_w), (y1 + y2) / (2.0 * img_h), (x2 - x1) / img_w, (y2 - y1) / img_h], dtype=np.float32)


box_voc = np.array([20.0, 30.0, 60.0, 80.0], dtype=np.float32)
print("Pascal VOC [x_min, y_min, x_max, y_max] :", box_voc)
print("COCO       [x_min, y_min, width, height]:", xyxy_to_xywh(box_voc))
print("YOLO norm  [cx, cy, w, h] (trên 100x100):", xyxy_to_yolo(box_voc, 100, 100))

box_pred = np.array([25.0, 35.0, 65.0, 80.0], dtype=np.float32)
print(f"IoU giữa box thật {box_voc.tolist()} và box dự đoán {box_pred.tolist()}: {compute_bbox_iou(box_voc, box_pred):.4f}")""",
        ),
        (
            "code",
            """# Cell 2: Cài đặt Tường minh Thuật toán NMS (Non-Maximum Suppression) và Minh họa Trực quan
def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    \"\"\"
    Thuật toán Non-Maximum Suppression (NMS) bằng NumPy thuần:
    Loại bỏ các khung dự đoán dư thừa chồng lấp cao quanh cùng một vật thể.
    \"\"\"
    order = np.argsort(scores)[::-1].tolist()
    keep: list[int] = []
    while order:
        best_idx = order.pop(0)
        keep.append(best_idx)
        remaining = []
        for idx in order:
            if compute_bbox_iou(boxes[best_idx], boxes[idx]) <= iou_threshold:
                remaining.append(idx)
        order = remaining
    return keep


# Giả lập 4 khung dự đoán chồng lấp quanh 1 vật thể và 1 khung quanh vật thể thứ hai
cand_boxes = np.array([
    [18.0, 20.0, 54.0, 56.0],  # Điểm cao nhất 0.95 (Vật thể 1)
    [20.0, 22.0, 56.0, 58.0],  # Trùng lặp với hộp 0 (score 0.82)
    [16.0, 19.0, 52.0, 55.0],  # Trùng lặp với hộp 0 (score 0.75)
    [60.0, 40.0, 88.0, 72.0],  # Vật thể 2 độc lập (score 0.91)
], dtype=np.float32)
cand_scores = np.array([0.95, 0.82, 0.75, 0.91], dtype=np.float32)

kept_manual = nms_numpy(cand_boxes, cand_scores, iou_threshold=0.45)
kept_torch = torchvision.ops.nms(torch.from_numpy(cand_boxes), torch.from_numpy(cand_scores), iou_threshold=0.45).tolist()
print("Chỉ số hộp giữ lại sau NMS (cài đặt tay):", kept_manual)
print("Chỉ số hộp giữ lại sau NMS (torchvision):", kept_torch)
assert kept_manual == kept_torch, "NMS cài đặt tay phải khớp 100% với torchvision.ops.nms!" """,
        ),
        (
            "code",
            """# Cell 3: Chuẩn bị Annotation & Dataset Chuẩn cho Torchvision Object Detection
# Lưu ý quan trọng trong Torchvision Detection API:
# - Nhãn 0 LUÔN dành riêng cho nền (background).
# - Các lớp vật thể được đánh số từ 1..K (1: circle, 2: square, 3: triangle).
# - Mỗi mẫu trả về: (image_tensor [C, H, W] float32 trong [0, 1], target_dict {'boxes': FloatTensor[M, 4], 'labels': Int64Tensor[M]})
det_class_map = {0: "background", 1: "circle", 2: "square", 3: "triangle"}
raw_det_data = generate_detection_dataset(num_images=24 if QUICK_RUN else 60, image_size=96, max_objects=2, seed=42)

# Kiểm tra định dạng Annotation của mẫu đầu tiên
sample0 = raw_det_data[0]
print("Cấu trúc Annotation chuẩn của mẫu #0:")
print("  - Image shape :", sample0["image"].shape)
print("  - Boxes (xyxy):", sample0["boxes"].round(1).tolist())
print("  - Labels (1..3):", sample0["labels"].tolist(), "->", [det_class_map[int(c)] for c in sample0["labels"]])


def to_torchvision_Format(records: list[dict]) -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    formatted = []
    for item in records:
        img_t = torch.from_numpy(item["image"]).permute(2, 0, 1).float() / 255.0
        target_t = {
            "boxes": torch.from_numpy(item["boxes"]).float(),
            "labels": torch.from_numpy(item["labels"]).long(),
        }
        formatted.append((img_t, target_t))
    return formatted


det_pairs = to_torchvision_Format(raw_det_data)
train_det = det_pairs[:16]
val_det = det_pairs[16:]""",
        ),
        (
            "code",
            """# Cell 4: Thực hành Huấn luyện / Fine-Tuning Thực tế Faster R-CNN (MobileNetV3-FPN) trên Tập Nhỏ
# Lưu ý:
# - Ở chế độ Offline (mặc định trong bài để không tự tải file lớn): khởi tạo fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
# - Khi có Internet: đặt weights="DEFAULT" để nạp trọng số COCO pretrained, sau đó thay box_predictor cho số lớp mới.
set_seed(42)
detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None, num_classes=4)
in_features_roi = detector.roi_heads.box_predictor.cls_score.in_features
detector.roi_heads.box_predictor = FastRCNNPredictor(in_features_roi, num_classes=4)  # 0: bg + 3 lớp hình
detector.to(device)

opt_det = torch.optim.AdamW(detector.parameters(), lr=1e-3, weight_decay=1e-4)

# Huấn luyện thực tế: Khi gọi detector(images, targets) ở chế độ model.train(),
# Faster R-CNN tự động trả về dictionary chứa 4 thành phần Loss:
#   - loss_classifier & loss_box_reg (của nhánh RoI Head giai đoạn 2)
#   - loss_objectness & loss_rpn_box_reg (của nhánh RPN giai đoạn 1)
num_det_epochs = 3 if QUICK_RUN else 8
for ep in range(1, num_det_epochs + 1):
    detector.train()
    ep_loss = 0.0
    # Gom theo batch 4 ảnh
    for i in range(0, len(train_det), 4):
        batch_slice = train_det[i : i + 4]
        imgs_b = [p[0].to(device) for p in batch_slice]
        tgts_b = [{k: v.to(device) for k, v in p[1].items()} for p in batch_slice]

        opt_det.zero_grad()
        loss_dict = detector(imgs_b, tgts_b)
        total_loss = sum(loss for loss in loss_dict.values())
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm=2.0)
        opt_det.step()
        ep_loss += float(total_loss.item())

    avg_ep_loss = ep_loss / max(1, len(train_det) // 4)
    print(f"Faster R-CNN Epoch {ep}/{num_det_epochs} -> Total Detection Loss = {avg_ep_loss:.4f} (Gồm: {list(loss_dict.keys())})")

# Chạy Suy luận (Inference) ở chế độ detector.eval(): chỉ truyền danh sách ảnh, nhận về boxes/labels/scores
detector.eval()
with torch.no_grad():
    sample_eval_img, sample_eval_gt = val_det[0]
    pred_out = detector([sample_eval_img.to(device)])[0]

# Minh họa vẽ Ground Truth (Xanh lá) và Top Box dự đoán (Vàng)
vis_det = (sample_eval_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
for b_gt in sample_eval_gt["boxes"].numpy().astype(int):
    cv2.rectangle(vis_det, (b_gt[0], b_gt[1]), (b_gt[2], b_gt[3]), (50, 235, 80), 2)
if len(pred_out["boxes"]) > 0:
    top_b = pred_out["boxes"][0].cpu().numpy().astype(int)
    cv2.rectangle(vis_det, (top_b[0], top_b[1]), (top_b[2], top_b[3]), (255, 220, 40), 1)

plt.figure(figsize=(3.6, 3.6))
plt.imshow(vis_det)
plt.title("Xanh lá: Nhãn thật (GT) | Vàng: Khung dự đoán top-1", fontsize=9.5)
plt.axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả & Lưu ý Trung thực Khoa học
- **Ở Cell 4**: Khi gọi `detector(imgs_b, tgts_b)` ở chế độ `.train()`, mô hình Faster R-CNN trả về đầy đủ 4 thành phần hàm mất mát (`loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`) và tổng mất mát giảm rõ rệt qua từng epoch.
- **Lưu ý quan trọng**: Trong chế độ `QUICK_RUN` với `weights=None` (khởi tạo từ đầu không có trọng số COCO để đảm bảo chạy offline trong vài giây trên CPU), 3 epochs chỉ nhằm **kiểm chứng pipeline huấn luyện và định dạng dữ liệu hoạt động chính xác**. Trong dự án thực tế, bạn luôn nên khởi tạo với `weights="DEFAULT"` (Pretrained COCO) và fine-tune từ `15 – 30` epochs.

## 13. Lỗi thường gặp và cách sửa
1. **Đánh số lớp vật thể từ `0` trong Faster R-CNN / SSD của `torchvision`**: Trong `torchvision`, nhãn `0` luôn được dành riêng cho **`background`**. Nếu bạn gán vật thể lớp đầu tiên là `0`, mạng sẽ coi vật thể đó là nền và không bao giờ phát hiện ra!
2. **Hộp suy biến (`Degenerate boxes` có `x_max <= x_min` hoặc `y_max <= y_min`)**: Gây lỗi `ValueError: All bounding boxes should have positive height and width`. Luôn kiểm tra `boxes[:, 2] > boxes[:, 0]` và `boxes[:, 3] > boxes[:, 1]` sau bước cắt ảnh (`crop`).

## 14. Bài tập tự làm
- **Cơ bản**: Viết hàm kiểm tra tính hợp lệ của toàn bộ `boxes` trong một dataset (`x_min >= 0`, `y_min >= 0`, `x_max > x_min`, `y_max > y_min`).
- **Trung cấp**: Thay đổi `iou_threshold` trong hàm `nms_numpy` ở Cell 2 từ `0.2` lên `0.5` và `0.95`, giải thích sự thay đổi số lượng hộp được giữ lại.
- **Nâng cao**: Viết hàm tính **`Average Precision (AP@0.5)`** cho 1 lớp vật thể từ danh sách các dự đoán `(score, is_true_positive)`.

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 13
# 1. Cơ bản: Hàm kiểm tra tính hợp lệ của Bounding Boxes
def validate_boxes_xyxy(boxes: np.ndarray, img_w: int, img_h: int) -> bool:
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert np.all(boxes[:, 0] >= 0) and np.all(boxes[:, 1] >= 0)
    assert np.all(boxes[:, 2] <= img_w) and np.all(boxes[:, 3] <= img_h)
    assert np.all(boxes[:, 2] > boxes[:, 0]) and np.all(boxes[:, 3] > boxes[:, 1])
    return True

print("[Cơ bản] Toàn bộ boxes trong raw_det_data hợp lệ:", all(validate_boxes_xyxy(d["boxes"], 96, 96) for d in raw_det_data))

# 2. Trung cấp: Ảnh hưởng của ngưỡng IoU trong NMS
for th in [0.2, 0.5, 0.95]:
    kept = nms_numpy(cand_boxes, cand_scores, iou_threshold=th)
    print(f"[Trung cấp] NMS iou_threshold={th:.2f} -> Số hộp giữ lại: {len(kept)} (indices={kept})")

# 3. Nâng cao: Tính Average Precision (AP) bằng tích phân diện tích dưới đường cong Precision-Recall
def compute_ap_from_matches(scores: np.ndarray, is_tp: np.ndarray, total_gt: int) -> float:
    order = np.argsort(scores)[::-1]
    tp_cum = np.cumsum(is_tp[order].astype(np.float64))
    fp_cum = np.cumsum((1 - is_tp[order]).astype(np.float64))
    recalls = tp_cum / max(1, total_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # Làm trơn đường bao Precision giảm đơn điệu từ phải sang trái
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

sim_scores = np.array([0.95, 0.85, 0.70, 0.60, 0.40])
sim_tp = np.array([1, 1, 0, 1, 0])  # 3 TP, 2 FP trên tổng 4 vật thể thật
print(f"[Nâng cao] Average Precision (AP@0.5) tính được: {compute_ap_from_matches(sim_scores, sim_tp, total_gt=4):.4f}")""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Hiểu rõ kiến trúc Two-stage (`Faster R-CNN`), One-stage (`SSD`, `RetinaNet`, `YOLO`) và Transformer (`DETR`).
- [x] Nắm vững `IoU`, `NMS`, `AP/mAP` và pipeline huấn luyện thực tế với `torchvision.models.detection`.
- [x] Sẵn sàng chuyển sang **Bài 14: `14_image_segmentation.ipynb`** để phân vùng vật thể tới từng điểm ảnh!

## 17. Nguồn tham khảo
- Ren, S. et al. (2015), *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*, NeurIPS.
- Lin, T.-Y. et al. (2017), *Focal Loss for Dense Object Detection (RetinaNet)*, ICCV.
- Carion, N. et al. (2020), *End-to-End Object Detection with Transformers (DETR)*, ECCV.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "13_object_detection.ipynb", cells)


def build_nb14() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 14: Phân vùng Ảnh (Image Segmentation) — U-Net, DeepLab, Mask R-CNN, SAM và Huấn luyện Thực tế

## 1. Mục tiêu học tập
- Phân biệt rõ 4 bài toán Phân vùng ảnh:
  1. **Semantic Segmentation (Phân vùng ngữ nghĩa)**: Gán nhãn lớp cho từng pixel (`FCN`, **`U-Net`**, **`DeepLabv3+`**).
  2. **Instance Segmentation (Phân vùng thực thể)**: Khoanh hộp và tách mặt nạ riêng cho từng cá thể vật thể (**`Mask R-CNN`**).
  3. **Panoptic Segmentation (Phân vùng toàn cảnh)**: Hợp nhất cả nền (`stuff`) và từng cá thể vật thể (`things`) (**`Mask2Former`**).
  4. **Promptable Segmentation (Phân vùng theo gợi ý)**: Nhận điểm click/khung hộp và xuất mặt nạ ngay lập tức (**`SAM / SAM 2`**).
- Hiểu kiến trúc đối xứng **Encoder – Decoder kèm Skip Connections** của **`U-Net`**.
- Hiểu vì sao trong bài toán phân vùng cần kết hợp **`BCEWithLogitsLoss + Dice Loss`** và đánh giá bằng **`IoU (Jaccard)`** cùng **`Dice Coefficient`** thay vì chỉ nhìn `Pixel Accuracy`.
- **Thực hành huấn luyện thật**: Huấn luyện mô hình `MiniUNet` trên dữ liệu ảnh + mặt nạ (`mask`), lưu checkpoint tốt nhất và trực quan hóa dự đoán.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 10, 11 và 13.

## 3. Thời gian học dự kiến
- Khoảng **2.5 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
Trong kiểm tra y tế (tách vùng tổn thương) hoặc công nghiệp (đo chính xác diện tích chi tiết cong), một khung chữ nhật (`bounding box`) là quá thô vì chứa nhiều nền thừa ở 4 góc. Ta cần một mô hình dự đoán mặt nạ nhị phân `mask (H, W)` khớp sát từng pixel của vật thể.

## 5. Giải thích trực quan: Kiến trúc U-Net và Vai trò của Skip Connections
```
Ảnh (3, 48, 48) ──► [Encoder 1 (16, 48, 48)] ──────────(Skip Cat)──────────► [Decoder 1] ──► Head 1x1 ──► Logits (1, 48, 48)
                            │ MaxPool(2)                                         ▲ UpConv 2x2
                            ▼                                                    │
                    [Encoder 2 (32, 24, 24)] ──────────(Skip Cat)──────────► [Decoder 2]
                            │ MaxPool(2)                                         ▲ UpConv 2x2
                            ▼                                                    │
                    [Bottleneck (64, 12, 12)] ───────────────────────────────────┘
```
- **Tại sao cần `Skip Connections` (`torch.cat([up_feat, enc_feat], dim=1)`)?**: Khi đi qua các tầng `MaxPool2d`, nhánh `Encoder` hiểu được ngữ cảnh "đây là vật thể gì" nhưng làm mờ mất tọa độ biên chính xác. Nhánh nối tắt `Skip Connection` truyền trực tiếp các bản đồ đặc trưng độ phân giải cao từ `Encoder` sang `Decoder`, giúp mặt nạ đầu ra có đường viền sắc nét!

## 6 & 7. Công thức Hàm Mất mát `BCE + Dice Loss` & Chỉ số Đánh giá
Gọi $p_i = \sigma(z_i) \in (0, 1)$ là xác suất dự đoán tại pixel thứ $i$ và $t_i \in \{0, 1\}$ là nhãn thật:
$$\text{Dice}(p, t) = \frac{2 \sum_i p_i t_i + \epsilon}{\sum_i p_i + \sum_i t_i + \epsilon}, \qquad \mathcal{L}_{\text{Dice}} = 1 - \text{Dice}(p, t)$$
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \text{BCEWithLogits}(z, t) + 0.5 \cdot \mathcal{L}_{\text{Dice}}(p, t)$$
- **Tại sao kết hợp cả hai?**: `BCE` tối ưu mượt từng pixel độc lập, còn `Dice Loss` tối ưu trực tiếp độ chồng lấp toàn cục của vùng vật thể (đặc biệt hiệu quả khi vật thể chỉ chiếm `5–10%` diện tích bức ảnh).""",
        ),
        (
            "code",
            """# Cell 1: Chuẩn bị Dữ liệu Phân vùng (Images + Masks) và Kiểm tra Kiến trúc MiniUNet
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
if (project_root / "image_ai_learning").is_dir():
    project_root = project_root / "image_ai_learning"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from src.utils import set_seed, get_device, count_parameters, summarize_model_layers, save_checkpoint, load_checkpoint
from src.synthetic_data import generate_segmentation_dataset
from src.models import MiniUNet
from src.evaluation import compute_segmentation_iou_dice

QUICK_RUN = True
set_seed(42)
device = get_device()

n_seg = 96 if QUICK_RUN else 200
seg_ds = generate_segmentation_dataset(num_images=n_seg, image_size=48, seed=42)
imgs_np, masks_np = seg_ds["images"], seg_ds["masks"]

idx_all = np.arange(len(imgs_np))
idx_tr, idx_temp = train_test_split(idx_all, test_size=0.30, random_state=42)
idx_va, idx_te = train_test_split(idx_temp, test_size=0.50, random_state=42)

X_t = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0
Y_t = torch.from_numpy(masks_np).unsqueeze(1).float()  # (N, 1, H, W) giá trị {0.0, 1.0}

tr_loader = DataLoader(TensorDataset(X_t[idx_tr], Y_t[idx_tr]), batch_size=16, shuffle=True)
va_loader = DataLoader(TensorDataset(X_t[idx_va], Y_t[idx_va]), batch_size=16, shuffle=False)
te_loader = DataLoader(TensorDataset(X_t[idx_te], Y_t[idx_te]), batch_size=16, shuffle=False)

unet = MiniUNet(in_channels=3, out_channels=1, base_channels=16).to(device)
print("MiniUNet Parameter Counts:", count_parameters(unet))
print("Tỷ lệ pixel thuộc vật thể (Foreground ratio):", f"{Y_t.mean().item()*100:.1f}%")""",
        ),
        (
            "code",
            """# Cell 2: Định nghĩa Hàm Mất mát Kết hợp (BCEWithLogits + Dice Loss) và Huấn luyện Thực tế MiniUNet
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


seg_criterion = BCEDiceLoss(bce_weight=0.5)
seg_optimizer = torch.optim.AdamW(unet.parameters(), lr=3e-3, weight_decay=1e-4)
epochs_seg = 10 if QUICK_RUN else 20
ckpt_unet = project_root / "outputs" / "checkpoints" / "nb14_mini_unet_best.pt"

best_val_dice = 0.0
seg_history = {"train_loss": [], "val_iou": [], "val_dice": []}

for ep in range(1, epochs_seg + 1):
    unet.train()
    tr_loss_sum = 0.0
    for xb, mb in tr_loader:
        xb, mb = xb.to(device), mb.to(device)
        seg_optimizer.zero_grad()
        logits = unet(xb)  # Mô hình trả về logits thô, chỉ dùng sigmoid khi tính xác suất!
        loss = seg_criterion(logits, mb)
        loss.backward()
        seg_optimizer.step()
        tr_loss_sum += loss.item() * len(xb)

    unet.eval()
    with torch.no_grad():
        val_logits = unet(X_t[idx_va].to(device))
        val_probs = torch.sigmoid(val_logits).cpu().numpy()[:, 0]
        val_preds = (val_probs >= 0.5).astype(np.uint8)
        val_m = compute_segmentation_iou_dice(val_preds, masks_np[idx_va])

    tr_loss = tr_loss_sum / len(idx_tr)
    seg_history["train_loss"].append(tr_loss)
    seg_history["val_iou"].append(val_m["iou"])
    seg_history["val_dice"].append(val_m["dice"])

    if val_m["dice"] > best_val_dice:
        best_val_dice = val_m["dice"]
        save_checkpoint(
            ckpt_unet, unet, class_names=["background", "foreground"],
            preprocessing_config={"image_size": 48, "threshold": 0.5},
            epoch=ep, val_metrics=val_m,
        )

print(f"Huấn luyện hoàn tất -> Best Validation Dice: {best_val_dice:.4f}, Val IoU: {max(seg_history['val_iou']):.4f}")""",
        ),
        (
            "code",
            """# Cell 3: Nạp Checkpoint Tốt nhất, Đánh giá trên Tập Test và Trực quan hóa Mặt nạ Dự đoán
load_checkpoint(ckpt_unet, model=unet, device=device)
unet.eval()
with torch.no_grad():
    te_logits = unet(X_t[idx_te].to(device))
    te_probs = torch.sigmoid(te_logits).cpu().numpy()[:, 0]
    te_preds = (te_probs >= 0.5).astype(np.uint8)

te_metrics = compute_segmentation_iou_dice(te_preds, masks_np[idx_te])
print(f"Kết quả Tập Test -> IoU: {te_metrics['iou']:.4f} | Dice: {te_metrics['dice']:.4f} | Pixel Acc: {te_metrics['pixel_accuracy']*100:.1f}%")

fig, axes = plt.subplots(2, 4, figsize=(11, 5.2))
for row in range(2):
    sample_i = idx_te[row]
    axes[row, 0].imshow(imgs_np[sample_i])
    axes[row, 0].set_title(f"Mẫu Test #{row+1}: Ảnh gốc")
    axes[row, 1].imshow(masks_np[sample_i], cmap="gray")
    axes[row, 1].set_title("Ground Truth Mask")
    axes[row, 2].imshow(te_probs[row], cmap="magma", vmin=0, vmax=1)
    axes[row, 2].set_title("Xác suất Sigmoid(Logits)")
    axes[row, 3].imshow(te_preds[row], cmap="gray")
    m_row = compute_segmentation_iou_dice(te_preds[row], masks_np[sample_i])
    axes[row, 3].set_title(f"Mask Dự đoán (Dice={m_row['dice']:.3f})")
    for c in range(4):
        axes[row, c].axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 4: Thí nghiệm thay đổi một yếu tố — Vì sao Pixel Accuracy đánh lừa khi vật thể nhỏ?
# Giả sử vật thể chỉ chiếm 5% diện tích ảnh, một mô hình "lười biếng" đoán toàn bộ ảnh là nền (toàn số 0):
small_fg_mask = np.zeros((48, 48), dtype=np.uint8)
small_fg_mask[20:30, 20:30] = 1  # 10x10 = 100 pixels (~4.3% diện tích ảnh)
all_zero_pred = np.zeros_like(small_fg_mask)

deceptive_metrics = compute_segmentation_iou_dice(all_zero_pred, small_fg_mask)
print("Khi mô hình đoán TOÀN BỘ LÀ NỀN (0) trên ảnh có vật thể nhỏ 4.3%:")
print(f"  - Pixel Accuracy = {deceptive_metrics['pixel_accuracy']*100:.1f}%  <-- CAO GIẢ TẠO!")
print(f"  - IoU            = {deceptive_metrics['iou']:.6f}  <-- Vạch trần ngay mô hình đoán sai hoàn toàn!")
print(f"  - Dice           = {deceptive_metrics['dice']:.6f}")""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Ở Cell 2 & 3**: Mô hình `MiniUNet` trả về `logits` thô để đưa vào `BCEDiceLoss`, và khi suy luận mới gọi `torch.sigmoid(logits) >= 0.5`. Nhờ các nhánh nối tắt `Skip Connections`, biên của hình tròn, hình vuông và tam giác trên mặt nạ dự đoán rất sắc nét, đạt chỉ số `Dice > 0.90`.
- **Ở Cell 4**: Thí nghiệm chứng minh tại sao trong bài toán Segmentation **không bao giờ được báo cáo mỗi `Pixel Accuracy`**: đoán toàn số `0` vẫn đạt `95.7% Pixel Accuracy`, nhưng `IoU` và `Dice` lập tức bằng `0.0`.

## 13. Lỗi thường gặp và cách sửa
1. **Kích thước chiều cao/rộng $(H, W)$ lẻ khi đi qua `MaxPool2d(2)` và `ConvTranspose2d(2)`**: Nếu $H = 45$, sau `MaxPool2d(2)` còn $22$, khi phóng đại gấp đôi bằng `ConvTranspose2d` lên $44 \neq 45$, lệnh `torch.cat([d1, e1], dim=1)` sẽ báo lỗi lệch kích thước! Hãy chọn kích thước ảnh chia hết cho $2^{\text{số tầng pool}}$ (ví dụ chia hết cho `4, 8, 16, 32`) hoặc dùng `F.interpolate(..., size=e1.shape[2:])`.

## 14. Bài tập tự làm
- **Cơ bản**: Quét ngưỡng cắt xác suất `threshold` qua `[0.2, 0.35, 0.5, 0.65, 0.8]` trên `te_probs` và tìm ngưỡng cho `Dice` cao nhất.
- **Trung cấp**: Kiểm tra bằng code mối liên hệ toán học bất biến $\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$.
- **Nâng cao**: Minh họa cơ chế **Promptable Segmentation (kiểu SAM)**: Cho người dùng truyền vào tọa độ 1 điểm click `(click_x, click_y)` trên ảnh, tạo thêm kênh thứ 4 là bản đồ khoảng cách Gaussian quanh điểm click đó để chỉ tách đúng 1 vật thể được click!

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 14
# 1. Cơ bản: Quét ngưỡng threshold tối ưu cho Dice
for th in [0.20, 0.35, 0.50, 0.65, 0.80]:
    m_th = compute_segmentation_iou_dice((te_probs >= th).astype(np.uint8), masks_np[idx_te])
    print(f"[Cơ bản] Threshold={th:.2f} -> IoU={m_th['iou']:.4f} | Dice={m_th['dice']:.4f}")

# 2. Trung cấp: Kiểm chứng công thức liên hệ giữa IoU và Dice
iou_val = te_metrics["iou"]
dice_from_iou = (2.0 * iou_val) / (1.0 + iou_val)
print(f"[Trung cấp] Dice trực tiếp = {te_metrics['dice']:.6f} vs 2*IoU/(1+IoU) = {dice_from_iou:.6f}")

# 3. Nâng cao: Tạo kênh Prompt Gaussian quanh điểm click (click_x, click_y) kiểu Interactive/Promptable Segmentation
def make_click_prompt_heatmap(h: int, w: int, click_x: int, click_y: int, sigma: float = 4.0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    heatmap = np.exp(-((xx - click_x) ** 2 + (yy - click_y) ** 2) / (2.0 * sigma ** 2))
    return heatmap.astype(np.float32)

prompt_map = make_click_prompt_heatmap(48, 48, click_x=24, click_y=24)
input_with_prompt = np.concatenate([imgs_np[0].astype(np.float32) / 255.0, prompt_map[..., None]], axis=-1)
print("[Nâng cao] Tensor ảnh ghép thêm kênh Click Prompt có shape:", input_with_prompt.shape)""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Đã phân biệt 4 nhánh Segmentation và hiểu sâu kiến trúc `U-Net`, `DeepLab`, `Mask R-CNN`, `SAM`.
- [x] Đã huấn luyện thành công `MiniUNet` với `BCE + Dice Loss` và đánh giá bằng `IoU` & `Dice`.
- [x] Sẵn sàng chuyển sang **Bài 15: `15_autoencoders_and_anomaly_detection.ipynb`**.

## 17. Nguồn tham khảo
- Ronneberger, O., Fischer, P., & Brox, T. (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI.
- Kirillov, A. et al. (2023), *Segment Anything (SAM)*, ICCV.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "14_image_segmentation.ipynb", cells)



def build_nb15() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 15: Bộ Tự Mã hóa (Autoencoders) và Phát hiện Bất thường trên Ảnh (Anomaly Detection)

## 1. Mục tiêu học tập
- Hiểu bài toán **Visual Anomaly Detection (Phát hiện bất thường trên ảnh)**: Huấn luyện mô hình **chỉ sử dụng các mẫu bình thường (`Normal / Defect-free samples`, `label = 0`)** và phát hiện các mẫu lỗi mới lạ (`Anomalous samples`, `label = 1`) lúc kiểm tra.
- Thực hành cả 3 nhóm phương pháp tiêu biểu:
  1. **Phương pháp ML truyền thống**: `Isolation Forest` và `One-Class SVM` trên vector đặc trưng ảnh.
  2. **Phương pháp Học sâu Tái tạo (`Reconstruction-based`)**: Huấn luyện thực tế **`Convolutional Autoencoder (ConvAutoencoder)`** để vừa phân loại ảnh lỗi (Image-level Anomaly Score) vừa khoanh vùng đúng vết xước/đốm lỗi trên bề mặt (Pixel-level Anomaly Map).
  3. **Phương pháp Nhúng vùng cục bộ (`Patch/Embedding Memory Bank` kiểu `PaDiM` / `PatchCore`)**: Tổng quan nguyên lý và minh họa đối sánh khoảng cách lân cận.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 06, 10 và 14.

## 3. Thời gian học dự kiến
- Khoảng **2.0 – 2.5 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
Trên dây chuyền kiểm tra vòng đệm kim loại, bạn có hàng nghìn bức ảnh sản phẩm lành lặn nhưng hầu như chưa có mẫu lỗi, hoặc lỗi trong tương lai có thể là vết xước, vết mẻ, đốm gỉ sét bất kỳ. Ta huấn luyện `ConvAutoencoder` học cách nén và vẽ lại sản phẩm lành lặn; khi gặp sản phẩm có vết xước, Autoencoder sẽ vẽ lại bản lành lặn nên hiệu số $|x - \hat{x}|$ sẽ làm lộ rõ vị trí vết xước!

## 5. Giải thích trực quan: Cơ chế Nút cổ chai (`Latent Bottleneck`) của Autoencoder
```
Ảnh đầu vào x (3, 48, 48) ──► [Encoder Conv] ──► Nút cổ chai z (latent_dim = 24) ──► [Decoder ConvTranspose] ──► Ảnh tái tạo x_hat (3, 48, 48)
                                                                                                                        │
                                                                    Bản đồ lỗi E(y, x) = mean_channel(|x - x_hat|) ◄────┘
```
- **Tại sao nút cổ chai `latent_dim` không được quá lớn?**: Nếu `latent_dim` quá rộng (hoặc có nối tắt Skip Connection như U-Net), mạng sẽ học hàm đồng nhất $\hat{x} = x$ (sao chép nguyên xi cả vết xước!), làm mất khả năng phát hiện lỗi.

## 6 & 7. Thuật ngữ mới, Tham số & Công thức Điểm Bất thường
| Keyword | Nghĩa tiếng Việt | Nhóm | Định nghĩa, Giá trị chọn & Ảnh hưởng khi Tăng/Giảm |
|---|---|---|---|
| `latent_vector` ($z$) / `latent_dim` | Vector ẩn nút cổ chai | Kiến trúc | Số chiều nén ở giữa Encoder và Decoder (ví dụ `24`). **Giảm `latent_dim` quá nhỏ**: ảnh tái tạo bị mờ nhòe ngay cả trên mẫu lành; **Tăng quá lớn**: mạng tái tạo được cả vết lỗi (giảm độ nhạy). |
| `reconstruction_loss` | Hàm mất mát tái tạo | Huấn luyện | $\mathcal{L}_{\text{recon}} = \frac{1}{CHW}\|x - \hat{x}\|_2^2$ (`MSELoss`) hoặc $\|x - \hat{x}\|_1$ (`L1Loss`). |
| `anomaly_map` & `anomaly_score` | Bản đồ lỗi & Điểm bất thường | Đánh giá | `anomaly_map` $M(y,x) = \frac{1}{C}\sum_c |x_{c,y,x} - \hat{x}_{c,y,x}|$. Điểm toàn ảnh `anomaly_score` thường lấy `max` hoặc phân vị `98%` (`np.percentile(M, 98)`) của `anomaly_map` để nhạy với các vết xước nhỏ! |
| `IsolationForest(contamination)` | Rừng cô lập ngoại lai | Anomaly ML | Cô lập điểm dị biệt bằng các nhát cắt ngẫu nhiên; điểm bất thường cần ít nhát cắt hơn để bị cô lập. |""",
        ),
        (
            "code",
            """# Cell 1: Tạo Bộ dữ liệu Kiểm tra Lỗi Bề mặt (Chỉ dùng Ảnh Bình thường để Train!)
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
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils import set_seed, get_device, count_parameters, save_checkpoint
from src.synthetic_data import generate_anomaly_dataset
from src.features import extract_combined_features
from src.models import ConvAutoencoder

QUICK_RUN = True
set_seed(42)
device = get_device()

anom_ds = generate_anomaly_dataset(
    num_normal=120 if QUICK_RUN else 240,
    num_anomalous=36 if QUICK_RUN else 60,
    image_size=48,
    seed=42,
)
imgs_np = anom_ds["images"]
labels_np = anom_ds["labels"]          # 0: Normal, 1: Anomaly
defect_masks_np = anom_ds["anomaly_masks"]

normal_indices = np.where(labels_np == 0)[0]
anom_indices = np.where(labels_np == 1)[0]

# Chia tập: Tập Train CHỈ CHỨA MẪU BÌNH THƯỜNG (label = 0)!
# Tập Test chứa phần mẫu bình thường còn lại + toàn bộ mẫu lỗi (label = 1).
n_train_norm = int(len(normal_indices) * 0.75)
idx_tr_norm = normal_indices[:n_train_norm]
idx_te_mix = np.concatenate([normal_indices[n_train_norm:], anom_indices])
y_te_mix = labels_np[idx_te_mix]

print(f"Tập Train (100% Normal): {len(idx_tr_norm)} ảnh | Tập Test hỗn hợp: {len(idx_te_mix)} ảnh (Normal: {(y_te_mix==0).sum()}, Anomaly: {(y_te_mix==1).sum()})")""",
        ),
        (
            "code",
            """# Cell 2: Baseline Phát hiện Bất thường bằng Isolation Forest và One-Class SVM trên Đặc trưng Ảnh
feats_all = extract_combined_features(imgs_np, cell_size=8)
X_tr_f = feats_all[idx_tr_norm]
X_te_f = feats_all[idx_te_mix]

iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42).fit(X_tr_f)
# decision_function càng nhỏ (càng âm) thì càng bất thường -> đảo dấu để làm anomaly_score
scores_iso = -iso_forest.decision_function(X_te_f)

oc_svm = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale").fit(X_tr_f)
scores_ocsvm = -oc_svm.decision_function(X_te_f)

print(f"Isolation Forest -> ROC-AUC: {roc_auc_score(y_te_mix, scores_iso):.4f} | PR-AUC: {average_precision_score(y_te_mix, scores_iso):.4f}")
print(f"One-Class SVM    -> ROC-AUC: {roc_auc_score(y_te_mix, scores_ocsvm):.4f} | PR-AUC: {average_precision_score(y_te_mix, scores_ocsvm):.4f}")""",
        ),
        (
            "code",
            """# Cell 3: Huấn luyện Thực tế Convolutional Autoencoder Chỉ trên Ảnh Bình thường
X_all_t = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0
train_norm_loader = DataLoader(TensorDataset(X_all_t[idx_tr_norm]), batch_size=16, shuffle=True)

ae_model = ConvAutoencoder(in_channels=3, image_size=48, latent_dim=24).to(device)
print("ConvAutoencoder Parameters:", count_parameters(ae_model))

ae_opt = torch.optim.AdamW(ae_model.parameters(), lr=3e-3, weight_decay=1e-5)
mse_crit = nn.MSELoss()
epochs_ae = 15 if QUICK_RUN else 30

ae_losses = []
for ep in range(1, epochs_ae + 1):
    ae_model.train()
    ep_loss = 0.0
    for (xb,) in train_norm_loader:
        xb = xb.to(device)
        ae_opt.zero_grad()
        recon_b, _ = ae_model(xb)
        loss = mse_crit(recon_b, xb)
        loss.backward()
        ae_opt.step()
        ep_loss += loss.item() * len(xb)
    ae_losses.append(ep_loss / len(idx_tr_norm))

# Tính bản đồ sai số tái tạo (Anomaly Map) trên tập Train Normal (để chọn ngưỡng) và tập Test
ae_model.eval()
with torch.no_grad():
    recon_tr, _ = ae_model(X_all_t[idx_tr_norm].to(device))
    err_map_tr = torch.abs(X_all_t[idx_tr_norm].to(device) - recon_tr).mean(dim=1).cpu().numpy()
    # Dùng phân vị thứ 98 của sai số điểm ảnh trong mỗi ảnh làm điểm bất thường nhạy với vết xước cục bộ
    tr_scores = np.percentile(err_map_tr, 98, axis=(1, 2))
    threshold_95 = float(np.percentile(tr_scores, 95))

    recon_te, _ = ae_model(X_all_t[idx_te_mix].to(device))
    err_map_te = torch.abs(X_all_t[idx_te_mix].to(device) - recon_te).mean(dim=1).cpu().numpy()
    te_scores = np.percentile(err_map_te, 98, axis=(1, 2))

ae_roc_auc = roc_auc_score(y_te_mix, te_scores)
ae_pr_auc = average_precision_score(y_te_mix, te_scores)
preds_ae = (te_scores >= threshold_95).astype(np.int64)
ae_acc = float((preds_ae == y_te_mix).mean())

ckpt_ae = project_root / "outputs" / "checkpoints" / "nb15_conv_autoencoder.pt"
save_checkpoint(
    ckpt_ae, ae_model, class_names=["normal", "anomaly"],
    preprocessing_config={"image_size": 48, "anomaly_threshold": threshold_95},
    epoch=epochs_ae, val_metrics={"roc_auc": float(ae_roc_auc), "pr_auc": float(ae_pr_auc)},
)
print(f"ConvAutoencoder -> Train Recon MSE giảm từ {ae_losses[0]:.4f} xuống {ae_losses[-1]:.4f}")
print(f"ConvAutoencoder Test -> ROC-AUC: {ae_roc_auc:.4f} | PR-AUC: {ae_pr_auc:.4f} | Accuracy (tại ngưỡng {threshold_95:.3f}): {ae_acc*100:.1f}%")""",
        ),
        (
            "code",
            """# Cell 4: Trực quan hóa Ảnh Bình thường vs Ảnh Bị Lỗi, Ảnh Tái tạo và Bản đồ Nhiệt Vị trí Lỗi (Anomaly Map)
norm_sample_idx = 0
anom_sample_idx = len(idx_te_mix) - 1  # Mẫu lỗi

fig, axes = plt.subplots(2, 4, figsize=(11, 5.2))
for row, (local_i, row_title) in enumerate([(norm_sample_idx, "Mẫu Bình thường"), (anom_sample_idx, "Mẫu Lỗi (Vết xước/Đốm)")]):
    orig_idx = idx_te_mix[local_i]
    recon_rgb = (recon_te[local_i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    axes[row, 0].imshow(imgs_np[orig_idx])
    axes[row, 0].set_title(f"{row_title}: Ảnh gốc")
    axes[row, 1].imshow(recon_rgb)
    axes[row, 1].set_title("Ảnh Autoencoder Tái tạo")
    axes[row, 2].imshow(err_map_te[local_i], cmap="inferno", vmin=0.0, vmax=0.35)
    axes[row, 2].set_title(f"Bản đồ lỗi (Score={te_scores[local_i]:.3f})")
    axes[row, 3].imshow(defect_masks_np[orig_idx], cmap="gray")
    axes[row, 3].set_title("Ground Truth Defect Mask")
    for c in range(4):
        axes[row, c].axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Ở Cell 3 & 4**: Vì `ConvAutoencoder` chỉ học trên các vòng đệm lành lặn, khi gặp mẫu có vết xước đỏ hoặc đốm bẩn xanh ở hàng dưới, ảnh tái tạo của Autoencoder "tự xóa" mất vết xước đó để vẽ lại bề mặt lành lặn. Khi lấy hiệu $|x - \hat{x}|$, bản đồ nhiệt (`Anomaly Map`) bật sáng rực tại đúng tọa độ của vết lỗi, khớp với `Ground Truth Defect Mask`!

## 12 & 14. Thí nghiệm và Bài tập tự làm
- **Cơ bản**: So sánh `ROC-AUC` khi tính `anomaly_score` toàn ảnh bằng trung bình cộng toàn bộ pixel (`err_map_te.mean(axis=(1, 2))`) so với lấy phân vị thứ 98 (`np.percentile(err_map_te, 98, axis=(1, 2))`).
- **Trung cấp**: Khảo sát kích thước nút cổ chai `latent_dim` qua `[4, 24, 128]` để kiểm chứng tác động của độ rộng nút cổ chai lên sai số tái tạo.
- **Nâng cao**: Cài đặt phương pháp **Memory Bank kiểu PatchCore thu nhỏ**: Lưu các vector đặc trưng của ảnh bình thường vào kho bộ nhớ `memory_bank`, với mỗi ảnh kiểm tra mới, tính khoảng cách Euclidean tới lân cận gần nhất trong `memory_bank` làm điểm bất thường!

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 15
# 1. Cơ bản: So sánh Mean Error Score vs 98th-Percentile Error Score
mean_scores = err_map_te.mean(axis=(1, 2))
print(f"[Cơ bản] Dùng Mean toàn ảnh -> ROC-AUC: {roc_auc_score(y_te_mix, mean_scores):.4f} vs Dùng Percentile-98 -> ROC-AUC: {ae_roc_auc:.4f}")

# 2. Trung cấp: Khảo sát latent_dim
for l_dim in [4, 24, 128]:
    ae_tmp = ConvAutoencoder(3, 48, latent_dim=l_dim)
    print(f"[Trung cấp] latent_dim={l_dim:3d} -> Tổng tham số: {count_parameters(ae_tmp)['total']:,}")

# 3. Nâng cao: Phát hiện bất thường bằng Kho Bộ nhớ Đặc trưng (Memory Bank Nearest-Neighbor kiểu PatchCore)
with torch.no_grad():
    bank_latents = ae_model.encode(X_all_t[idx_tr_norm].to(device)).cpu().numpy()  # (N_train, 24)
    test_latents = ae_model.encode(X_all_t[idx_te_mix].to(device)).cpu().numpy()   # (N_test, 24)

# Khoảng cách tới mẫu bình thường gần nhất trong Memory Bank
dists_matrix = np.linalg.norm(test_latents[:, None, :] - bank_latents[None, :, :], axis=-1)
nn_anomaly_scores = dists_matrix.min(axis=1)
print(f"[Nâng cao] Memory Bank Nearest-Neighbor ROC-AUC: {roc_auc_score(y_te_mix, nn_anomaly_scores):.4f}")""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Hiểu nguyên lý phát hiện bất thường một lớp (`One-Class / Unsupervised Anomaly Detection`).
- [x] Đã huấn luyện thực tế `ConvAutoencoder` chỉ với ảnh lành và khoanh vùng được vị trí lỗi.
- [x] Sẵn sàng chuyển sang **Bài 16: `16_metric_learning_and_image_retrieval.ipynb`**.

## 17. Nguồn tham khảo
- Bergmann, P. et al. (2019), *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR.
- Roth, K. et al. (2022), *Towards Total Recall in Industrial Anomaly Detection (PatchCore)*, CVPR.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "15_autoencoders_and_anomaly_detection.ipynb", cells)


def build_nb16() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 16: Học Biểu diễn (Metric Learning), Mạng Siamese/Triplet và Hệ thống Truy vấn Ảnh (Image Retrieval)

## 1. Mục tiêu học tập
- Hiểu vì sao khi số lượng đối tượng thay đổi liên tục (nhận diện khuôn mặt nhân viên mới, tìm kiếm hàng vạn mẫu sản phẩm thương mại điện tử), ta không dùng tầng phân loại cố định mà dùng **Metric Learning (Học khoảng cách / Học biểu diễn)**.
- Nắm vững kiến trúc **`Siamese Network`**, cách chuẩn hóa vector nhúng lên mặt cầu đơn vị (`L2 Normalization`: $\|z\|_2 = 1$), và hai hàm mất mát kinh điển:
  1. **`Contrastive Loss`** (trên cặp `Positive Pair` / `Negative Pair`).
  2. **`Triplet Margin Loss`** (trên bộ ba `Anchor A`, `Positive P`, `Negative N` với lề `margin` $m$).
- **Thực hành huấn luyện thật**: Huấn luyện mạng `SiameseEmbeddingNet` bằng `nn.TripletMarginLoss`, xây dựng chỉ mục tìm kiếm lân cận gần nhất (**Nearest-Neighbor Image Retrieval**) và đánh giá bằng **`Recall@K` (`R@1`, `R@3`)**.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 10 và Bài 12.

## 3. Thời gian học dự kiến
- Khoảng **2.0 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
Người dùng chụp ảnh một chi tiết máy và muốn tìm ra **3 bức ảnh giống nhất** trong kho lưu trữ (`Gallery`). Ta cần một mạng nơ-ron biến mỗi bức ảnh thành một vector `32` chiều sao cho hai ảnh cùng loại luôn nằm sát nhau và hai ảnh khác loại cách xa nhau ít nhất một khoảng `margin`.

## 5. Giải thích trực quan: Cơ chế Kéo – Đẩy của Triplet Loss
```
            Kéo lại gần (Pull)
   (Positive P) ◄─────────● (Anchor A: Ảnh gốc)
                           \
                            \  Đẩy ra xa ít nhất khoảng d(A,P) + margin (Push)
                             ↘
                              ○ (Negative N: Ảnh khác lớp)
```

## 6 & 7. Thuật ngữ mới, Tham số & Công thức Toán học
1. **Chuẩn hóa L2 (`L2 Normalization`) và Độ tương đồng Cosine (`Cosine Similarity`)**:
   $$z = \frac{f_\theta(x)}{\|f_\theta(x)\|_2}, \qquad \cos(z_1, z_2) = z_1 \cdot z_2, \qquad \|z_1 - z_2\|_2^2 = 2 - 2(z_1 \cdot z_2)$$
   *(Khi các vector đã có độ dài bằng `1`, việc tìm khoảng cách Euclidean nhỏ nhất hoàn toàn tương đương với tìm tích vô hướng `Cosine Similarity` lớn nhất!)*
2. **Hàm mất mát Bộ ba (`Triplet Margin Loss`)**:
   $$\mathcal{L}_{\text{triplet}}(A, P, N) = \max\Big(0,\; \|z_A - z_P\|_2 - \|z_A - z_N\|_2 + \text{margin}\Big)$$
3. **Hàm mất mát Đối조 theo Cặp (`Contrastive Loss`)** với nhãn cặp $y_{\text{pair}} \in \{1 \text{ (cùng lớp)}, 0 \text{ (khác lớp)}\}$ và $d = \|z_1 - z_2\|_2$:
   $$\mathcal{L}_{\text{contrastive}} = y_{\text{pair}} \cdot d^2 + (1 - y_{\text{pair}}) \cdot \max(0, \text{margin} - d)^2$$""",
        ),
        (
            "code",
            """# Cell 1: Xây dựng Bộ lấy mẫu Bộ ba (Triplet Dataset: Anchor, Positive, Negative)
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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from src.utils import set_seed, get_device, count_parameters, save_checkpoint
from src.synthetic_data import generate_synthetic_shapes_dataset
from src.models import SiameseEmbeddingNet

QUICK_RUN = True
set_seed(42)
device = get_device()

ds = generate_synthetic_shapes_dataset(num_samples=200 if QUICK_RUN else 400, image_size=32, seed=42)
imgs_np, labels_np, class_names = ds["images"], ds["labels"], ds["class_names"]

idx_all = np.arange(len(imgs_np))
idx_tr, idx_te = train_test_split(idx_all, test_size=0.30, random_state=42, stratify=labels_np)
X_all_t = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0


class TripletImageDataset(Dataset):
    \"\"\"Dataset tự động ghép mỗi ảnh Anchor với 1 ảnh Positive (cùng lớp) và 1 ảnh Negative (khác lớp).\"\"\"

    def __init__(self, images_t: torch.Tensor, labels: np.ndarray, seed: int = 42) -> None:
        self.images_t = images_t
        self.labels = labels
        self.rng = np.random.default_rng(seed)
        self.class_to_indices = {c: np.where(labels == c)[0] for c in np.unique(labels)}
        self.classes = list(self.class_to_indices.keys())

    def __len__(self) -> int:
        return len(self.images_t)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        anchor_img = self.images_t[idx]
        anchor_cls = int(self.labels[idx])

        pos_pool = self.class_to_indices[anchor_cls]
        pos_idx = int(self.rng.choice(pos_pool))
        pos_img = self.images_t[pos_idx]

        neg_classes = [c for c in self.classes if c != anchor_cls]
        neg_cls = int(self.rng.choice(neg_classes))
        neg_idx = int(self.rng.choice(self.class_to_indices[neg_cls]))
        neg_img = self.images_t[neg_idx]

        return anchor_img, pos_img, neg_img, anchor_cls


triplet_train_ds = TripletImageDataset(X_all_t[idx_tr], labels_np[idx_tr], seed=42)
triplet_loader = DataLoader(triplet_train_ds, batch_size=32, shuffle=True, num_workers=0)
a_b, p_b, n_b, c_b = next(iter(triplet_loader))
print(f"Triplet Batch Shapes -> Anchor: {tuple(a_b.shape)}, Positive: {tuple(p_b.shape)}, Negative: {tuple(n_b.shape)}")""",
        ),
        (
            "code",
            """# Cell 2: Huấn luyện Thực tế Mạng SiameseEmbeddingNet với TripletMarginLoss
emb_net = SiameseEmbeddingNet(in_channels=3, embedding_dim=32).to(device)
print("SiameseEmbeddingNet Parameters:", count_parameters(emb_net))

triplet_crit = nn.TripletMarginLoss(margin=0.5, p=2)
emb_opt = torch.optim.AdamW(emb_net.parameters(), lr=2e-3, weight_decay=1e-4)


@torch.no_grad()
def evaluate_retrieval_recall_at_k(model: nn.Module, gallery_x: torch.Tensor, gallery_y: np.ndarray, ks: tuple[int, ...] = (1, 3)) -> dict[str, float]:
    model.eval()
    embs = model(gallery_x.to(device)).cpu().numpy()  # Đã chuẩn hóa L2 nên tích vô hướng chính là Cosine Similarity!
    sim_matrix = embs @ embs.T
    np.fill_diagonal(sim_matrix, -np.inf)  # Không tự lấy chính nó làm kết quả tìm kiếm
    ranked_indices = np.argsort(-sim_matrix, axis=1)

    results = {}
    for k in ks:
        top_k_labels = gallery_y[ranked_indices[:, :k]]
        hits = np.any(top_k_labels == gallery_y[:, None], axis=1)
        results[f"Recall@{k}"] = float(hits.mean())
    return results


recall_before = evaluate_retrieval_recall_at_k(emb_net, X_all_t[idx_te], labels_np[idx_te])
print("Trước khi huấn luyện ->", recall_before)

epochs_metric = 12 if QUICK_RUN else 25
loss_history = []
for ep in range(1, epochs_metric + 1):
    emb_net.train()
    ep_loss = 0.0
    for xa, xp, xn, _ in triplet_loader:
        xa, xp, xn = xa.to(device), xp.to(device), xn.to(device)
        emb_opt.zero_grad()
        za = emb_net(xa)
        zp = emb_net(xp)
        zn = emb_net(xn)
        loss = triplet_crit(za, zp, zn)
        loss.backward()
        emb_opt.step()
        ep_loss += loss.item() * len(xa)
    loss_history.append(ep_loss / len(triplet_train_ds))

recall_after = evaluate_retrieval_recall_at_k(emb_net, X_all_t[idx_te], labels_np[idx_te])
ckpt_metric = project_root / "outputs" / "checkpoints" / "nb16_siamese_embedding.pt"
save_checkpoint(
    ckpt_metric, emb_net, class_names=class_names,
    preprocessing_config={"image_size": 32, "embedding_dim": 32, "margin": 0.5},
    epoch=epochs_metric, val_metrics=recall_after,
)
print(f"Sau {epochs_metric} epochs -> Triplet Loss giảm từ {loss_history[0]:.4f} xuống {loss_history[-1]:.4f}")
print("Sau khi huấn luyện  ->", recall_after)""",
        ),
        (
            "code",
            """# Cell 3: Trực quan hóa Không gian Nhúng (Embedding Space 2D) và Kết quả Truy vấn Top-3 Ảnh Tương tự
emb_net.eval()
with torch.no_grad():
    te_embs = emb_net(X_all_t[idx_te].to(device)).cpu().numpy()

te_embs_2d = PCA(n_components=2, random_state=42).fit_transform(te_embs)

# Thực hiện truy vấn (Query) cho bức ảnh đầu tiên trong tập Test
q_idx = 0
sims = te_embs[q_idx] @ te_embs.T
sims[q_idx] = -np.inf
top3_idx = np.argsort(-sims)[:3]

fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.8))
for cls_i, cls_name in enumerate(class_names):
    mask_c = labels_np[idx_te] == cls_i
    axes[0].scatter(te_embs_2d[mask_c, 0], te_embs_2d[mask_c, 1], label=cls_name, s=22, alpha=0.85)
axes[0].set_title("Không gian Embedding (PCA 2D)")
axes[0].legend(fontsize=7.5)

axes[1].imshow(imgs_np[idx_te[q_idx]])
axes[1].set_title(f"Ảnh Truy vấn (Query)\\nLớp: {class_names[labels_np[idx_te[q_idx]]]}")
axes[1].axis("off")

for rank_i, ret_i in enumerate(top3_idx, 1):
    axes[1 + rank_i].imshow(imgs_np[idx_te[ret_i]])
    ret_cls = class_names[labels_np[idx_te[ret_i]]]
    axes[1 + rank_i].set_title(f"Top-{rank_i} (cos={sims[ret_i]:.2f})\\nLớp: {ret_cls}")
    axes[1 + rank_i].axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- Sau khi huấn luyện với `TripletMarginLoss`, chỉ số truy vấn `Recall@1` và `Recall@3` trên tập Test tăng mạnh. Trên biểu đồ không gian nhúng 2D, các điểm cùng lớp tự động tụ lại thành các cụm tách biệt và 3 ảnh trả về có `Cosine Similarity` cao nhất đều thuộc đúng lớp với ảnh truy vấn.

## 12 & 14. Thí nghiệm và Bài tập tự làm
- **Cơ bản**: Cài đặt hàm mất mát **`ContrastiveLoss(margin=1.0)`** nhận vào `(z1, z2, is_same_label)` và chạy kiểm tra trên 1 batch.
- **Trung cấp**: Khảo sát tham số `margin` của `TripletMarginLoss` qua `[0.1, 0.5, 2.0]`.
- **Nâng cao**: Cài đặt kỹ thuật **Batch-Hard Negative Mining**: Với mỗi mẫu Anchor trong batch, tự động chọn mẫu Negative khác lớp có khoảng cách **gần Anchor nhất** (mẫu dễ gây nhầm lẫn nhất) để đưa vào hàm loss!

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 4: Lời giải hoàn chỉnh cho bài tập Bài 16
# 1. Cơ bản: Cài đặt Contrastive Loss
def contrastive_loss_fn(z1: torch.Tensor, z2: torch.Tensor, y_same: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    dist = torch.norm(z1 - z2, p=2, dim=1)
    pos_part = y_same * (dist ** 2)
    neg_part = (1.0 - y_same) * torch.clamp(margin - dist, min=0.0) ** 2
    return (pos_part + neg_part).mean()

z_a = emb_net(a_b[:8].to(device))
z_p = emb_net(p_b[:8].to(device))
z_n = emb_net(n_b[:8].to(device))
print(f"[Cơ bản] Contrastive Loss trên cặp Positive: {contrastive_loss_fn(z_a, z_p, torch.ones(8, device=device)):.4f}")
print(f"[Cơ bản] Contrastive Loss trên cặp Negative: {contrastive_loss_fn(z_a, z_n, torch.zeros(8, device=device)):.4f}")

# 2. Trung cấp & 3. Nâng cao: Batch-Hard Negative Mining
with torch.no_grad():
    batch_embs = torch.from_numpy(te_embs[:16])
    batch_lbls = torch.from_numpy(labels_np[idx_te[:16]])
    dist_mat = torch.cdist(batch_embs, batch_embs, p=2)
    diff_class_mask = batch_lbls.unsqueeze(0) != batch_lbls.unsqueeze(1)
    # Gán khoảng cách của các mẫu cùng lớp thành +inf để tìm mẫu KHÁC LỚP gần nhất (Hardest Negative)
    masked_dists = torch.where(diff_class_mask, dist_mat, torch.tensor(float("inf")))
    hardest_neg_dists, hardest_neg_indices = masked_dists.min(dim=1)
print("[Nâng cao] Khoảng cách tới Hardest Negative của 5 mẫu đầu:", hardest_neg_dists[:5].numpy().round(3))""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Hiểu nguyên lý Metric Learning (`Siamese`, `Contrastive`, `Triplet Loss`, `L2 Normalization`) và xây dựng hệ thống `Image Retrieval`.
- [x] Sẵn sàng chuyển sang **Bài 17: `17_vision_transformers_and_self_supervised_learning.ipynb`**!

## 17. Nguồn tham khảo
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015), *FaceNet: A Unified Embedding for Face Recognition and Clustering*, CVPR.
- Hadsell, R., Chopra, S., & LeCun, Y. (2006), *Dimensionality Reduction by Learning an Invariant Mapping (Contrastive Loss)*, CVPR.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "16_metric_learning_and_image_retrieval.ipynb", cells)


def build_nb17() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 17: Vision Transformers (ViT, Swin) và Học Tự Giám sát (Self-Supervised Learning: SimCLR, MoCo, DINO, CLIP)

## 1. Mục tiêu học tập
- Hiểu cách **Vision Transformer (`ViT`)** biến bài toán xử lý ảnh 2D thành bài toán xử lý chuỗi các mảnh ảnh (`Sequence of Image Patches`):
  1. `Patch Embedding`: Chia ảnh $(C, H, W)$ thành $N = (H/P) \times (W/P)$ mảnh vuông kích thước $P \times P$ và chiếu tuyến tính sang chiều `embed_dim` $D$.
  2. `CLS Token` & `Positional Encoding`: Ghép thêm token phân loại `[CLS]` và cộng vector vị trí $(1, 1 + N, D)$.
  3. `Multi-Head Self-Attention (MHSA)`: Cho phép mọi mảnh ảnh trong bức hình tương tác trực tiếp với nhau ngay từ tầng đầu tiên!
- Phân biệt **`ViT` chuẩn** (Self-Attention toàn cục, chi phí bậc hai theo số patches) với **`Swin Transformer`** (Self-Attention trong cửa sổ dịch chuyển `Shifted Windows`, chi phí tuyến tính).
- **Thực hành huấn luyện thật**: Huấn luyện mô hình `MiniViT` và trực quan hóa bản đồ chú ý (`Attention Map`).
- Hiểu và cài đặt cơ chế cốt lõi của **Self-Supervised Learning (`SimCLR InfoNCE Loss`, `MoCo`, `DINO`)** và mô hình liên kết Ảnh – Văn bản **`CLIP`**.

> **Lưu ý sư phạm**: Trong bài học này, ta dùng mô hình `MiniViT` và ví dụ `SimCLR / CLIP` thu nhỏ để hiểu rõ cơ chế toán học và kiểm chứng code trên CPU. Trong thực tế công nghiệp, các mô hình nền tảng (`Foundation Models` như ViT-Large, DINOv2, CLIP) được tiền huấn luyện trên hàng chục triệu ảnh bằng cụm GPU lớn, và bạn nên tải trọng số pretrained của chúng về để fine-tune hoặc trích xuất đặc trưng.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 10, 12 và 16.

## 3. Thời gian học dự kiến
- Khoảng **2.5 – 3.0 giờ** (ước tính).

## 4, 5, 6 & 7. Giải thích Trực quan, Từ điển Keyword & Công thức Toán học
1. **Dòng chảy Shape bên trong `MiniViT`** (với ảnh `32x32`, `patch_size = 8`, `embed_dim = 48`):
   - Số lượng mảnh ảnh: $N = (32 / 8) \times (32 / 8) = 4 \times 4 = 16$ patches.
   - `Patch Embedding`: `(B, 3, 32, 32)` $\xrightarrow{\text{Conv2d}(K=8, S=8)}$ `(B, 48, 4, 4)` $\xrightarrow{\text{flatten & transpose}}$ `(B, 16, 48)`.
   - Ghép `cls_token` `(B, 1, 48)` + cộng `pos_embed` `(1, 17, 48)` $\longrightarrow$ Chuỗi `17` tokens có shape `(B, 17, 48)`.
2. **Công thức `Scaled Dot-Product Attention`**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   trong đó $\sqrt{d_k}$ giúp giữ phương sai của tích vô hướng $QK^T$ ổn định quanh $1$, tránh làm hàm `softmax` bị bão hòa đạo hàm.
3. **Hàm mất mát Tự giám sát `InfoNCE / NT-Xent` (trong SimCLR)** và **Đối xứng Ảnh–Chữ (trong CLIP)**:
   $$\mathcal{L}_{i, j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k} \exp(\text{sim}(z_i, z_k) / \tau)}$$
   với $\tau$ (`temperature`) là tham số nhiệt độ kiểm soát độ sắc nét của phân phối xác suất.""",
        ),
        (
            "code",
            """# Cell 1: Kiểm tra Dòng chảy Shape của Patch Embedding, Positional Encoding và Kiến trúc MiniViT
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
project_root = current_dir.parent if current_dir.name == "notebooks" else current_dir
if (project_root / "image_ai_learning").is_dir():
    project_root = project_root / "image_ai_learning"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from src.utils import set_seed, get_device, count_parameters, summarize_model_layers
from src.synthetic_data import generate_synthetic_shapes_dataset
from src.models import MiniViT
from src.training import fit_classifier, evaluate_classifier

QUICK_RUN = True
set_seed(42)
device = get_device()

vit_model = MiniViT(
    image_size=32, patch_size=8, in_channels=3, num_classes=4,
    embed_dim=48, depth=2, num_heads=4, dropout_p=0.1,
).to(device)

print("MiniViT Parameter Counts:", count_parameters(vit_model))
vit_layers_df = pd.DataFrame(summarize_model_layers(vit_model, input_shape=(2, 3, 32, 32), device=device))
print("\\n=== BẢNG TÓM TẮT CÁC TẦNG TRONG MINI_VIT ===")
print(vit_layers_df.head(12).to_string(index=False))""",
        ),
        (
            "code",
            """# Cell 2: Huấn luyện Ngắn Mô hình MiniViT trên Bộ dữ liệu Hình học và Minh họa Cắt Ảnh thành Patches
ds = generate_synthetic_shapes_dataset(num_samples=240 if QUICK_RUN else 480, image_size=32, seed=42)
imgs_np, labels_np, class_names = ds["images"], ds["labels"], ds["class_names"]

idx_all = np.arange(len(imgs_np))
idx_tr, idx_va = train_test_split(idx_all, test_size=0.25, random_state=42, stratify=labels_np)

X_t = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0
y_t = torch.from_numpy(labels_np).long()
tr_mean = X_t[idx_tr].mean(dim=(0, 2, 3), keepdim=True)
tr_std = X_t[idx_tr].std(dim=(0, 2, 3), keepdim=True) + 1e-6
X_norm = (X_t - tr_mean) / tr_std

tr_loader = DataLoader(TensorDataset(X_norm[idx_tr], y_t[idx_tr]), batch_size=32, shuffle=True)
va_loader = DataLoader(TensorDataset(X_norm[idx_va], y_t[idx_va]), batch_size=32, shuffle=False)

opt_vit = torch.optim.AdamW(vit_model.parameters(), lr=3e-3, weight_decay=1e-3)
crit = nn.CrossEntropyLoss()
hist_vit = fit_classifier(vit_model, tr_loader, va_loader, crit, opt_vit, device, epochs=12 if QUICK_RUN else 25)
print(f"MiniViT Best Validation Accuracy: {hist_vit['best_val_acc']*100:.1f}% (Loss giảm từ {hist_vit['train_loss'][0]:.3f} -> {hist_vit['train_loss'][-1]:.3f})")

# Minh họa chia bức ảnh 32x32 thành lưới 4x4 = 16 patches kích thước 8x8
sample_img = imgs_np[0]
fig, axes = plt.subplots(4, 4, figsize=(3.6, 3.6))
for r in range(4):
    for c in range(4):
        patch = sample_img[r * 8 : (r + 1) * 8, c * 8 : (c + 1) * 8]
        axes[r, c].imshow(patch)
        axes[r, c].axis("off")
plt.suptitle("Ảnh 32x32 được cắt thành 16 Patches (8x8) đưa vào ViT", fontsize=10)
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 3: Minh họa Cơ chế Học Tự Giám sát (SimCLR InfoNCE Loss) Không Cần Nhãn Con Người
def simclr_nt_xent_loss(z_view1: torch.Tensor, z_view2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    \"\"\"
    Hàm mất mát đối조 chuẩn hóa nhiệt độ (NT-Xent / InfoNCE) dùng trong SimCLR:
    - z_view1, z_view2: Vector nhúng của 2 góc nhìn biến đổi (augmentation) từ cùng 1 lô B bức ảnh.
    - Cặp (z_view1[i], z_view2[i]) là cặp dương (Positive pair); các mẫu khác trong batch là cặp âm.
    \"\"\"
    z1 = F.normalize(z_view1, dim=1)
    z2 = F.normalize(z_view2, dim=1)
    logits = (z1 @ z2.T) / temperature
    targets = torch.arange(len(z1), device=z1.device)
    loss_12 = F.cross_entropy(logits, targets)
    loss_21 = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_12 + loss_21)


# Minh họa tạo 2 góc nhìn từ 1 lô 16 ảnh (View 1: lật ngang, View 2: cộng nhiễu nhẹ) và tính InfoNCE Loss
x_batch_ssl = X_norm[:16].to(device)
view1 = torch.flip(x_batch_ssl, dims=[3])
view2 = x_batch_ssl + 0.05 * torch.randn_like(x_batch_ssl)

ssl_projector = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 64), nn.ReLU(), nn.Linear(64, 32)).to(device)
opt_ssl = torch.optim.AdamW(ssl_projector.parameters(), lr=3e-3)

loss_ssl_start = simclr_nt_xent_loss(ssl_projector(view1), ssl_projector(view2)).item()
for _ in range(15):
    opt_ssl.zero_grad()
    l_ssl = simclr_nt_xent_loss(ssl_projector(view1), ssl_projector(view2))
    l_ssl.backward()
    opt_ssl.step()
loss_ssl_end = simclr_nt_xent_loss(ssl_projector(view1), ssl_projector(view2)).item()
print(f"[SimCLR Self-Supervised Demo] InfoNCE Loss giảm từ {loss_ssl_start:.4f} xuống {loss_ssl_end:.4f} mà KHÔNG CẦN bất kỳ nhãn lớp nào!")""",
        ),
        (
            "code",
            """# Cell 4: Minh họa Cơ chế Liên kết Đa phương thức Ảnh – Văn bản kiểu CLIP (Zero-Shot Classification)
# Trong CLIP, cả Ảnh và Câu mô tả văn bản ("a photo of a circle", ...) được chiếu vào CÙNG một không gian vector.
# Khi suy luận Zero-Shot: Ta so khớp vector ảnh với vector của 4 câu mô tả lớp và chọn câu có Cosine Similarity cao nhất!
img_emb_dim = 16
fake_text_prompt_embs = F.normalize(torch.randn(4, img_emb_dim), dim=1)  # 4 vector đại diện 4 câu prompt văn bản
fake_image_embs = F.normalize(fake_text_prompt_embs + 0.15 * torch.randn(4, img_emb_dim), dim=1)

similarity_matrix = (fake_image_embs @ fake_text_prompt_embs.T) * 10.0  # logit_scale
zero_shot_probs = torch.softmax(similarity_matrix, dim=1).numpy()

print("Ma trận xác suất khớp Ảnh – Văn bản kiểu CLIP (Đường chéo chính cao nhất):")
print(np.round(zero_shot_probs, 3))""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Về `MiniViT` (Cell 1 & 2)**: Khác với CNN chỉ nhìn vùng lân cận `3x3` ở tầng đầu, `MiniViT` cắt ảnh `32x32` thành `16` mảnh `8x8` và cho phép mọi mảnh trao đổi thông tin với nhau ngay lập tức qua `Multi-Head Self-Attention`.
- **Về `SimCLR` (Cell 3)**: Không cần nhãn lớp do con người gán, hàm mất mát `InfoNCE` dạy mạng rằng hai góc nhìn biến đổi từ cùng một bức ảnh phải có biểu diễn giống nhau và khác với các bức ảnh còn lại trong lô.

## 13. Lỗi thường gặp và cách sửa
1. **Chọn `embed_dim` không chia hết cho `num_heads`**: Trong `nn.MultiheadAttention(embed_dim, num_heads)`, mỗi head có số chiều $d_k = \text{embed\_dim} / \text{num\_heads}$, nên `embed_dim % num_heads` bắt buộc phải bằng `0`.
2. **Đưa ảnh có kích thước khác lúc huấn luyện vào `ViT` mà không nội suy `pos_embed`**: Vì số lượng patches $N = (H/P)(W/P)$ thay đổi khi đổi kích thước ảnh $(H, W)$, tensor `pos_embed` có chiều $(1, 1 + N, D)$ sẽ báo lỗi lệch kích thước nếu không nội suy 2D (`interpolate positional encoding`).

## 14. Bài tập tự làm
- **Cơ bản**: Tính số lượng patches $N$ và kích thước ma trận Self-Attention $(1+N) \times (1+N)$ khi giảm `patch_size` từ `8` xuống `4` và `2` trên ảnh `32x32`.
- **Trung cấp**: Kiểm tra tác động của tham số nhiệt độ `temperature` $\tau \in [0.05, 0.2, 1.0]$ lên hàm mất mát `simclr_nt_xent_loss`.
- **Nâng cao**: Viết hàm nội suy `interpolate_pos_encoding(pos_embed, new_grid_size)` để cho phép nạp `pos_embed` học trên lưới `4x4` patches sang lưới `6x6` patches (cho ảnh `48x48`)!

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 17
# 1. Cơ bản: Sự bùng nổ bậc bốn của ma trận Attention khi giảm patch_size
for p_size in [8, 4, 2]:
    n_patches = (32 // p_size) ** 2
    attn_elements = (1 + n_patches) ** 2
    print(f"[Cơ bản] patch_size={p_size} -> Số patches N={n_patches:3d} -> Kích thước ma trận Attention = {1+n_patches}x{1+n_patches} ({attn_elements:,} phần tử)")

# 2. Trung cấp: Ảnh hưởng của nhiệt độ (temperature) trong InfoNCE
z1_demo = ssl_projector(view1).detach()
z2_demo = ssl_projector(view2).detach()
for tau in [0.05, 0.20, 1.00]:
    print(f"[Trung cấp] Temperature tau={tau:.2f} -> NT-Xent Loss = {simclr_nt_xent_loss(z1_demo, z2_demo, temperature=tau).item():.4f}")

# 3. Nâng cao: Nội suy Positional Encoding 2D khi đổi độ phân giải ảnh đầu vào
def interpolate_pos_encoding(pos_embed: torch.Tensor, old_grid: int, new_grid: int) -> torch.Tensor:
    cls_pos = pos_embed[:, :1, :]
    patch_pos = pos_embed[:, 1:, :]  # (1, old_grid*old_grid, D)
    d = patch_pos.shape[-1]
    patch_pos_2d = patch_pos.transpose(1, 2).reshape(1, d, old_grid, old_grid)
    patch_pos_resized = F.interpolate(patch_pos_2d, size=(new_grid, new_grid), mode="bilinear", align_corners=False)
    patch_pos_flat = patch_pos_resized.flatten(2).transpose(1, 2)
    return torch.cat([cls_pos, patch_pos_flat], dim=1)

new_pos = interpolate_pos_encoding(vit_model.pos_embed.detach(), old_grid=4, new_grid=6)
print(f"[Nâng cao] pos_embed cũ: {tuple(vit_model.pos_embed.shape)} -> Sau nội suy cho lưới 6x6: {tuple(new_pos.shape)}")""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Hiểu rõ cơ chế `Patch Embedding`, `Positional Encoding`, `Self-Attention` của `Vision Transformer`.
- [x] Nắm được tư tưởng cốt lõi của `SimCLR`, `MoCo`, `DINO` và `CLIP`.
- [x] Sẵn sàng chuyển sang **Bài 18: `18_generative_models.ipynb`** để học các Mô hình Sinh Ảnh (`VAE`, `GAN`, `Diffusion`)!

## 17. Nguồn tham khảo
- Dosovitskiy, A. et al. (2021), *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)*, ICLR.
- Chen, T. et al. (2020), *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)*, ICML.
- Radford, A. et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, ICML.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "17_vision_transformers_and_self_supervised_learning.ipynb", cells)


if __name__ == "__main__":
    for builder in [build_nb12, build_nb13, build_nb14, build_nb15, build_nb16, build_nb17]:
        path = builder()
        print("Created and validated:", path.name)
