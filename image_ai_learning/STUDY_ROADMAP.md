# Lộ trình Học tập Chi tiết (`STUDY_ROADMAP.md`)

Tài liệu này giúp bạn theo dõi tiến độ học tập qua 4 giai đoạn từ nền tảng đến đồ án cuối khóa, kèm ước tính thời gian và điều kiện hoàn thành từng chặng.

> **Lưu ý**: Thời gian học bên dưới chỉ là **ước tính tham khảo** cho người mới bắt đầu (kể cả thời gian chạy code, đọc giải thích và tự làm bài tập). Bạn hãy học theo tốc độ hiểu thật sự của mình.

---

## Sơ đồ Tổng quan Lộ trình Học

```
[00: Khởi động]
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN A: NỀN TẢNG PYTHON, NUMPY & XỬ LÝ ẢNH CỔ ĐIỂN (Bài 01–03) │
│ Ma trận số (ndarray) -> Cấu trúc ảnh (RGB/NCHW) -> Lọc, Biên, Hình thái│
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN B: NỀN TẢNG MACHINE LEARNING CHO ẢNH (Bài 04–07)           │
│ Toán & Gradient Descent -> Chia tập & Chống Data Leakage ->          │
│ Đặc trưng HOG/LBP/ColorHist + kNN/SVM/RF -> Phân cụm K-Means & PCA   │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN C: MẠNG NƠ-RON, PYTORCH & KIẾN TRÚC CNN (Bài 08–11)        │
│ Tự code NN bằng NumPy -> PyTorch Autograd & MLP ->                   │
│ Giải phẫu CNN & Residual -> Quy trình Training, Validation & Gỡ lỗi  │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ GIAI ĐOẠN D: CÁC NHÓM MÔ HÌNH COMPUTER VISION & TRIỂN KHAI (12–21)   │
│ Transfer Learning (12) -> Detection (13) -> Segmentation (14) ->     │
│ Anomaly Detection (15) -> Metric Learning & Retrieval (16) ->        │
│ ViT & SSL (17) -> Generative Models (18) -> OCR & Mở rộng (19) ->    │
│ Đánh giá & Tuning (20) -> Inference, Export & Tối ưu hóa (21)        │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
[22: ĐỒ ÁN CUỐI KHÓA (CAPSTONE PROJECT END-TO-END)]
```

---

## Chi tiết Từng Giai đoạn và Điều kiện Qua bài

### Khởi động (`00_start_here.ipynb`) — ~45 phút
- **Mục tiêu**: Xác nhận môi trường chạy đúng, hiểu sự khác biệt giữa Xử lý ảnh bằng quy tắc cố định (Rule-based Image Processing) và Học máy từ dữ liệu (Data-driven ML/DL).
- **Điều kiện qua bài**: Chạy thành công toàn bộ cell trong `00_start_here.ipynb` không báo lỗi import.

---

### Giai đoạn A — Nền tảng Ảnh số & Xử lý Ảnh Cổ điển (~6 – 8 giờ)

| Bài | Notebook | Thời lượng ước tính | Kỹ năng cần nắm vững trước khi qua bài mới |
|---|---|---|---|
| 01 | `01_python_numpy_for_images.ipynb` | 2.0 giờ | Phân biệt `shape`, `dtype`, `axis`; dùng thành thạo `slicing` và `broadcasting` thay cho vòng lặp `for`. |
| 02 | `02_image_fundamentals.ipynb` | 2.5 giờ | Không còn nhầm `RGB` với `BGR`, hiểu rõ `HWC` `(H, W, C)` và `NCHW` `(N, C, H, W)`, chuyển đổi đúng `uint8 [0, 255]` sang `float32 [0.0, 1.0]`. |
| 03 | `03_classical_image_processing.ipynb` | 2.5 giờ | Hiểu khi nào dùng Gaussian blur vs Median filter, chọn đúng Otsu/Adaptive threshold, dùng Morphology và tìm Contour để tách vật thể đơn giản. |

---

### Giai đoạn B — Nền tảng Machine Learning Truyền thống (~8 – 10 giờ)

| Bài | Notebook | Thời lượng ước tính | Kỹ năng cần nắm vững trước khi qua bài mới |
|---|---|---|---|
| 04 | `04_math_and_ml_foundations.ipynb` | 2.0 giờ | Hiểu tích vô hướng (`dot product`), ý nghĩa đạo hàm/gradient là hướng dốc nhất, và cách `learning_rate` quyết định bước nhảy cập nhật tham số. |
| 05 | `05_datasets_and_data_leakage.ipynb` | 2.0 giờ | Biết chia `train/val/test` đúng chuẩn, hiểu vì sao **chỉ được `fit` bộ tiền xử lý trên tập `train`**, và biết dùng `GroupKFold`/`GroupShuffleSplit` cho ảnh chụp liên tiếp. |
| 06 | `06_features_and_classical_ml.ipynb` | 3.0 giờ | Trích xuất được đặc trưng `HOG`, `LBP`, `Color Histogram` và huấn luyện/so sánh công bằng các mô hình `k-NN`, `Logistic Regression`, `Naive Bayes`, `Decision Tree`, `Random Forest`, `SVM`, `Gradient Boosting`. |
| 07 | `07_clustering_and_dimensionality_reduction.ipynb` | 2.0 giờ | Phân biệt rõ `PCA` là giảm chiều dữ liệu (không phải bộ phân loại), ứng dụng `K-Means` để phân cụm đặc trưng/lượng tử hóa màu và `DBSCAN` để tìm cụm mật độ + điểm nhiễu. |

---

### Giai đoạn C — Mạng Nơ-ron Nhân tạo, PyTorch & CNN (~10 – 12 giờ)

| Bài | Notebook | Thời lượng ước tính | Kỹ năng cần nắm vững trước khi qua bài mới |
|---|---|---|---|
| 08 | `08_neural_network_from_scratch.ipynb` | 3.0 giờ | Tự tính được `forward pass`, `softmax`, `cross-entropy loss` và `backpropagation` bằng NumPy; kiểm chứng đạo hàm giải tích khớp với đạo hàm số. |
| 09 | `09_pytorch_and_autograd.ipynb` | 2.5 giờ | Hiểu cơ chế `requires_grad`, `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`; viết được `Dataset`, `DataLoader` và huấn luyện mạng `MLP`. |
| 10 | `10_cnn_from_scratch.ipynb` | 3.0 giờ | Tính nhẩm được kích thước đầu ra của `Conv2d` và `Pooling`; giải thích được vai trò của `BatchNorm2d`, `Dropout`, `Residual Connection` và `Receptive Field`. |
| 11 | `11_training_validation_and_debugging.ipynb` | 3.0 giờ | Không bao giờ đặt `Softmax` trước `nn.CrossEntropyLoss`; phân biệt rạch ròi `model.eval()` và `torch.no_grad()`; biết đọc đồ thị `train/val loss` để phát hiện overfitting/underfitting. |

---

### Giai đoạn D — Các Nhóm Mô hình Computer Vision & Triển khai Thực tế (~18 – 24 giờ)

| Bài | Notebook | Thời lượng ước tính | Kỹ năng cần nắm vững trước khi qua bài mới |
|---|---|---|---|
| 12 | `12_transfer_learning_and_fine_tuning.ipynb` | 2.0 giờ | Phân biệt `Training from scratch`, `Feature extraction` (đóng băng backbone) và `Fine-tuning` (mở khóa tinh chỉnh với `learning_rate` nhỏ). |
| 13 | `13_object_detection.ipynb` | 3.0 giờ | Hiểu định dạng nhãn tọa độ (`xyxy`, `xywh`, `cxcywh`), tính `IoU`, cài đặt thuật toán `NMS`, hiểu `AP/mAP` và quy trình fine-tuning detector. |
| 14 | `14_image_segmentation.ipynb` | 2.5 giờ | Phân biệt `Semantic`, `Instance`, `Panoptic` và `Promptable Segmentation`; huấn luyện được mô hình `MiniUNet` với hàm mất mát kết hợp `BCE + Dice Loss`. |
| 15 | `15_autoencoders_and_anomaly_detection.ipynb` | 2.0 giờ | Huấn luyện `Autoencoder` chỉ trên ảnh bình thường và dùng ngưỡng sai số tái tạo (`reconstruction error`) để phát hiện ảnh lỗi và khoanh vùng vị trí lỗi. |
| 16 | `16_metric_learning_and_image_retrieval.ipynb` | 2.0 giờ | Hiểu `Contrastive Loss`, `Triplet Margin Loss`, chuẩn hóa L2 vector nhúng (`embedding`) và xây dựng hệ thống tìm kiếm ảnh tương tự (Image Retrieval). |
| 17 | `17_vision_transformers_and_self_supervised_learning.ipynb` | 2.5 giờ | Hiểu cách `ViT` biến ảnh thành chuỗi `patches`, cơ chế `Multi-Head Self-Attention`, và nguyên lý học tự giám sát (`SimCLR`, `MoCo`, `DINO`) cùng mô hình đa phương thức `CLIP`. |
| 18 | `18_generative_models.ipynb` | 2.5 giờ | Phân biệt cơ chế sinh ảnh của `VAE` (tối ưu ELBO = Reconstruction + KL), `GAN` (trò chơi đối kháng Generator – Discriminator) và `Diffusion` (thêm nhiễu thuận & khử nhiễu ngược theo `timestep`). |
| 19 | `19_ocr_and_visual_tasks_overview.ipynb` | 2.0 giờ | Nắm được kiến trúc 2 chặng của `OCR` (Text Detection + Text Recognition với CTC), nguyên lý `Pose Estimation` (heatmap), `Tracking`, `Super-Resolution` và `Depth Estimation`. |
| 20 | `20_evaluation_experiment_design_and_tuning.ipynb` | 2.0 giờ | Chọn đúng metric cho từng bài toán, hiểu `ROC-AUC` vs `PR-AUC` khi dữ liệu lệch lớp, kiểm tra độ hiệu chỉnh xác suất (`Calibration`) và thiết kế bảng Ablation Study. |
| 21 | `21_inference_export_and_optimization.ipynb` | 2.0 giờ | Lưu/nạp checkpoint đầy đủ metadata, xuất mô hình sang `TorchScript`, đo `Latency`/`Throughput` chính xác và viết class suy luận `ImagePredictor` cho phần mềm. |

---

### Đồ án Cuối khóa (`22_capstone_project.ipynb`) — ~4 – 6 giờ
- **Mục tiêu**: Kết nối toàn bộ kiến thức từ kiểm định dữ liệu, xây dựng Baseline ML, huấn luyện CNN/Transfer Learning, phân tích lỗi sai (Error Analysis), xuất mô hình và viết hàm dự đoán ảnh thực tế.
- **Bảng tự đánh giá (Self-Assessment Rubric)**: Có sẵn ở cuối notebook `22_capstone_project.ipynb`.

---

## Bảng Theo dõi Tiến độ Học tập Cá nhân

Bạn có thể đánh dấu `[x]` vào ô vuông khi hoàn thành từng bài:

- [ ] `00_start_here.ipynb`
- [ ] `01_python_numpy_for_images.ipynb`
- [ ] `02_image_fundamentals.ipynb`
- [ ] `03_classical_image_processing.ipynb`
- [ ] `04_math_and_ml_foundations.ipynb`
- [ ] `05_datasets_and_data_leakage.ipynb`
- [ ] `06_features_and_classical_ml.ipynb`
- [ ] `07_clustering_and_dimensionality_reduction.ipynb`
- [ ] `08_neural_network_from_scratch.ipynb`
- [ ] `09_pytorch_and_autograd.ipynb`
- [ ] `10_cnn_from_scratch.ipynb`
- [ ] `11_training_validation_and_debugging.ipynb`
- [ ] `12_transfer_learning_and_fine_tuning.ipynb`
- [ ] `13_object_detection.ipynb`
- [ ] `14_image_segmentation.ipynb`
- [ ] `15_autoencoders_and_anomaly_detection.ipynb`
- [ ] `16_metric_learning_and_image_retrieval.ipynb`
- [ ] `17_vision_transformers_and_self_supervised_learning.ipynb`
- [ ] `18_generative_models.ipynb`
- [ ] `19_ocr_and_visual_tasks_overview.ipynb`
- [ ] `20_evaluation_experiment_design_and_tuning.ipynb`
- [ ] `21_inference_export_and_optimization.ipynb`
- [ ] `22_capstone_project.ipynb`
