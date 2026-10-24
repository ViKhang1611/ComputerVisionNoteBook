# Báo cáo Kiểm thử và Xác thực Thực tế (`VALIDATION_REPORT.md`)

- **Ngày kiểm thử**: `2026-10-24`
- **Thư mục dự án**: `E:\programming\JupyterLab\image_ai_learning`

---

## 1. Phần cứng và Hệ điều hành Thực tế

| Hạng mục | Thông số thực tế đã kiểm tra |
|---|---|
| **Hệ điều hành** | Windows 11 (`Windows-11-10.0.26100-SP0`, kiến trúc `AMD64`) |
| **Bộ vi xử lý (CPU)** | `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel` (`20 logical processors`) |
| **Card đồ họa (GPU / CUDA)** | `nvidia-smi` không phát hiện NVIDIA driver đang hoạt động (`torch.cuda.is_available() == False`) |
| **Thiết bị chạy kiểm thử** | **`cpu`** (`torch.device('cpu')`) |

---

## 2. Phiên bản Python và Thư viện Đã Thực sự Sử dụng

> **Giải thích lựa chọn tương thích**: Trên máy có sẵn `Python 3.15.0` và `Python 3.12.10`. Do `Python 3.15` quá mới và chưa có gói cài đặt nhị phân (`prebuilt wheels`) chính thức cho `PyTorch` và `scikit-learn`, dự án sử dụng môi trường ảo độc lập `image_ai_learning/.venv` khởi tạo từ **`Python 3.12.10`** để đảm bảo tương thích 100% mà không tác động tới Python hệ thống.

| Thư viện / Công cụ | Phiên bản thực tế trong `image_ai_learning/.venv` |
|---|---|
| `Python` | `3.12.10` |
| `numpy` | `2.5.2` |
| `matplotlib` | `3.11.2` |
| `pandas` | `3.1.0` |
| `Pillow` (`PIL`) | `12.3.0` |
| `opencv-python-headless` (`cv2`) | `5.0.0.93` (`cv2.__version__ == 5.0.0`) |
| `scikit-learn` (`sklearn`) | `1.9.1` |
| `scipy` | `1.18.1` |
| `torch` | `2.14.1+cpu` |
| `torchvision` | `0.29.1+cpu` |
| `ipykernel` | `7.4.0` |
| `nbformat` | `5.11.1` |
| `nbclient` | `0.11.0` |

---

## 3. Kết quả Kiểm tra Định dạng, Cú pháp và Liên kết (`scripts/validate_project.py`)

| Hạng mục kiểm tra | Kết quả | Chi tiết |
|---|---|---|
| **Kiểm tra schema `nbformat` v4** | **THÀNH CÔNG (23 / 23)** | Toàn bộ 23 file `.ipynb` hợp lệ theo chuẩn `nbformat` v4, có đầy đủ metadata `kernelspec` và `language_info`. |
| **Kiểm tra cú pháp Python (`ast.parse`)** | **THÀNH CÔNG (110 / 110 code cells)** | Tất cả `110` code cells trên `23` notebooks (`178` tổng số cells bao gồm `68` markdown cells) đều parse cú pháp thành công, không có `TODO` hay dấu `...` bỏ trống. |
| **Kiểm tra đường dẫn tương đối** | **THÀNH CÔNG** | Không có bất kỳ đường dẫn tuyệt đối của máy cá nhân nào trong cả 23 notebooks. |
| **Kiểm tra liên kết nội bộ (Internal Links)** | **THÀNH CÔNG** | Mọi liên kết giữa `README.md`, `STUDY_ROADMAP.md`, `MODEL_CATALOG.md`, `KEYWORD_GLOSSARY.md`, `TROUBLESHOOTING.md`, `VALIDATION_REPORT.md` và `notebooks/*.ipynb` đều trỏ tới file tồn tại thực tế. |

---

## 4. Kết quả Thực thi Thực tế 23 Notebooks & Các Pipeline Huấn luyện (`scripts/run_smoke_tests.py`)

- **Chế độ chạy**: `QUICK_RUN = True` trên `CPU`, mỗi notebook được thực thi tuần tự từ cell đầu đến cell cuối trong một không gian biến độc lập (`isolated fresh namespace`) để đảm bảo không phụ thuộc biến từ notebook khác.
- **Dữ liệu kiểm tra**:
  - Bộ dữ liệu tổng hợp có kiểm soát từ `src/synthetic_data.py` (4 lớp phân loại `circle/square/triangle/cross`, dữ liệu `detection` đa vật thể, dữ liệu `segmentation` ảnh + mask, dữ liệu `anomaly` bề mặt bình thường vs vết xước/đốm lỗi).
  - Bộ dữ liệu chữ số viết tay công khai `sklearn.datasets.load_digits` (`1797` mẫu `8x8`, 10 lớp).
- **Tổng thời gian thực thi cả 23 notebooks**: `~76 giây` trên CPU (`23 / 23 PASSED`).

| STT | Notebook | Số Code Cells Đã Chạy | Số Biểu đồ / Hình Sinh ra | Thời gian (giây) | Trạng thái & Nội dung Đã Kiểm chứng Thực tế |
|---|---|---|---|---|---|
| 00 | `00_start_here.ipynb` | 5 / 5 | 2 | 5.70s | **PASSED** — Kiểm tra môi trường, tạo dữ liệu mẫu, thí nghiệm `noise_std`. |
| 01 | `01_python_numpy_for_images.ipynb` | 5 / 5 | 1 | 0.13s | **PASSED** — Kiểm tra `shape`, `dtype`, `slicing`, `broadcasting`, đo tốc độ `vectorization`. |
| 02 | `02_image_fundamentals.ipynb` | 5 / 5 | 3 | 0.71s | **PASSED** — Chuyển đổi `RGB/BGR/Gray/Alpha`, `HWC <-> NCHW`, `Interpolation`, `Histogram/CLAHE`. |
| 03 | `03_classical_image_processing.ipynb` | 5 / 5 | 4 | 0.59s | **PASSED** — Lọc `Box/Gaussian/Median/Sharpen`, `Otsu/Adaptive Threshold`, `Morphology`, `Sobel/Canny/Contour`. |
| 04 | `04_math_and_ml_foundations.ipynb` | 4 / 4 | 2 | 0.54s | **PASSED** — Kiểm chứng đạo hàm số, quỹ đạo `Gradient Descent`, minh họa `Underfitting/Overfitting`. |
| 05 | `05_datasets_and_data_leakage.ipynb` | 5 / 5 | 1 | 1.29s | **PASSED** — `Stratified Split`, `GroupShuffleSplit`, chống `Data Leakage`, `class_weight` & `WeightedRandomSampler`. |
| 06 | `06_features_and_classical_ml.ipynb` | 5 / 5 | 1 | 2.03s | **PASSED** — **[Pipeline Bắt buộc #1: Classical ML]** Trích xuất `Pixel/ColorHist/HOG/LBP` & huấn luyện thật 7 mô hình (`k-NN`, `Logistic`, `Naive Bayes`, `Decision Tree`, `Random Forest`, `SVM`, `HistGradientBoosting`). |
| 07 | `07_clustering_and_dimensionality_reduction.ipynb` | 5 / 5 | 3 | 1.32s | **PASSED** — Phân cụm `K-Means`, `DBSCAN` và giảm chiều/tái tạo ảnh bằng `PCA`. |
| 08 | `08_neural_network_from_scratch.ipynb` | 5 / 5 | 1 | 0.50s | **PASSED** — Tự cài đặt mạng 2 tầng bằng `NumPy`, kiểm tra đạo hàm số (`sai lệch < 1e-9`) & huấn luyện. |
| 09 | `09_pytorch_and_autograd.ipynb` | 5 / 5 | 2 | 0.90s | **PASSED** — **[Pipeline Bắt buộc #2: MLP]** Kiểm tra `autograd`, huấn luyện `SimpleMLP`, lưu & nạp lại `nb09_simple_mlp_best.pt`. |
| 10 | `10_cnn_from_scratch.ipynb` | 5 / 5 | 1 | 3.03s | **PASSED** — **[Pipeline Bắt buộc #3: CNN]** Tính `Receptive Field`, huấn luyện `SimpleCNN` & `ResidualCNN`, lưu `nb10_simple_cnn_best.pt`. |
| 11 | `11_training_validation_and_debugging.ipynb` | 5 / 5 | 1 | 0.35s | **PASSED** — Kiểm chứng lỗi `double Softmax`, `BCEWithLogitsLoss`, `Pre-flight check`, `Overfit 1 Batch (Acc=100%)`, `CosineAnnealingLR + Warmup`. |
| 12 | `12_transfer_learning_and_fine_tuning.ipynb` | 5 / 5 | 1 | 3.29s | **PASSED** — **[Pipeline Bắt buộc #4: Transfer Learning]** So sánh `Scratch` vs `Feature Extraction (Frozen)` vs `Fine-Tuning (Differential LR)`, lưu `nb12_transfer_finetuned.pt`. |
| 13 | `13_object_detection.ipynb` | 5 / 5 | 1 | 39.26s | **PASSED** — **[Detection Pipeline]** Chuyển đổi `VOC/COCO/YOLO`, `IoU`, `NMS`, `AP@0.5`, huấn luyện thử & suy luận với `fasterrcnn_mobilenet_v3_large_320_fpn`. |
| 14 | `14_image_segmentation.ipynb` | 5 / 5 | 1 | 1.87s | **PASSED** — **[Pipeline Bắt buộc #5: Segmentation]** Huấn luyện `MiniUNet` với `BCE + Dice Loss`, lưu & nạp `nb14_mini_unet_best.pt` (`Test Dice > 0.90`). |
| 15 | `15_autoencoders_and_anomaly_detection.ipynb` | 5 / 5 | 1 | 1.30s | **PASSED** — **[Pipeline Bắt buộc #6: Autoencoder Anomaly]** `IsolationForest`, `OneClassSVM` & huấn luyện `ConvAutoencoder` chỉ trên ảnh lành, lưu `nb15_conv_autoencoder.pt`. |
| 16 | `16_metric_learning_and_image_retrieval.ipynb` | 4 / 4 | 1 | 1.82s | **PASSED** — **[Pipeline Bắt buộc #7: Metric Learning]** Huấn luyện `SiameseEmbeddingNet` với `TripletMarginLoss`, truy vấn ảnh `Recall@1, Recall@3`, lưu `nb16_siamese_embedding.pt`. |
| 17 | `17_vision_transformers_and_self_supervised_learning.ipynb` | 5 / 5 | 1 | 1.81s | **PASSED** — Huấn luyện `MiniViT`, minh họa `SimCLR InfoNCE Loss` và ma trận khớp `CLIP`. |
| 18 | `18_generative_models.ipynb` | 5 / 5 | 2 | 2.33s | **PASSED** — Huấn luyện `TinyVAE`, bước đối kháng `GAN` và bộ khử nhiễu `Diffusion Denoiser`. |
| 19 | `19_ocr_and_visual_tasks_overview.ipynb` | 4 / 4 | 1 | 0.31s | **PASSED** — Pipeline `OCR` (`Detection` + `nn.CTCLoss`), `Pose Heatmap`, `IoU Tracking`, `Residual Denoising (PSNR/SSIM)`, `Depth AbsRel`. |
| 20 | `20_evaluation_experiment_design_and_tuning.ipynb` | 4 / 4 | 1 | 0.46s | **PASSED** — Bộ metric đa chiều, `Threshold Tuning`, `Calibration (ECE)`, bảng `Ablation Study` & `Grid Search`. |
| 21 | `21_inference_export_and_optimization.ipynb` | 4 / 4 | 1 | 0.79s | **PASSED** — Đóng gói class `ImagePredictor`, xuất `TorchScript` (`sai lệch < 1e-6`), lượng tử hóa động `INT8` (giảm dung lượng `~3.8 lần`), đo `Latency & Throughput`. |
| 22 | `22_capstone_project.ipynb` | 5 / 5 | 1 | 5.58s | **PASSED** — Đồ án End-to-End (`Data Audit` -> `Baseline SVM` -> `ResidualCNN` -> `Error Analysis` -> xuất `capstone_report.json` & `capstone_residual_cnn_best.pt`). |

---

## 5. Danh sách Checkpoints và Báo cáo Đã Sinh ra Thực tế trên Đĩa

- `outputs/checkpoints/nb09_simple_mlp_best.pt`
- `outputs/checkpoints/nb10_simple_cnn_best.pt`
- `outputs/checkpoints/nb12_transfer_finetuned.pt`
- `outputs/checkpoints/nb14_mini_unet_best.pt`
- `outputs/checkpoints/nb15_conv_autoencoder.pt`
- `outputs/checkpoints/nb16_siamese_embedding.pt`
- `outputs/checkpoints/nb21_deploy_cnn.pt`
- `outputs/checkpoints/nb21_deploy_cnn_torchscript.pt`
- `outputs/checkpoints/nb21_mlp_fp32.pt`
- `outputs/checkpoints/nb21_mlp_int8.pt`
- `outputs/checkpoints/capstone_residual_cnn_best.pt`
- `outputs/reports/capstone_report.json`
- `outputs/reports/execution_summary.json`

---

## 6. Phần Chưa Chạy & Giới hạn Còn lại (Báo cáo Trung thực)

1. **Huấn luyện trên GPU CUDA (`torch.cuda`)**: Chưa chạy thực tế trên GPU vì máy hiện tại không có NVIDIA driver/GPU CUDA hoạt động (`nvidia-smi` không khả dụng). Toàn bộ code đã được viết tương thích hai chiều qua hàm `get_device()`.
2. **Tải trọng số Pretrained lớn từ Internet (ImageNet / COCO / SAM / CLIP gốc)**: Để tuân thủ quy tắc không tự ý tải file dung lượng lớn tốn băng thông và đảm bảo chạy 100% offline, các bài `12` và `13` chạy thực tế bằng trọng số khởi tạo nội bộ (`weights=None` hoặc pretrained từ tập nguồn nội bộ). Hướng dẫn bật `weights="DEFAULT"` khi có mạng Internet đã được ghi chú rõ ràng trong code.
3. **Chế độ huấn luyện dài (`QUICK_RUN = False`) và Dữ liệu tự nhiên quy mô lớn**: Các kết quả thực thi hiện tại được chạy ở `QUICK_RUN = True` trên dữ liệu tổng hợp (`synthetic data`) và `load_digits` nhằm kiểm chứng tính đúng đắn của thuật toán và pipeline, không dùng để kết luận độ chính xác triển khai thương mại.
4. **Xuất file `.onnx` thực tế**: Trong `requirements.txt` nền tảng không ép cài đặt `onnx`/`onnxruntime` để giữ môi trường nhẹ; phần kiểm tra tùy chọn trong Bài 21 đã có cơ chế phát hiện tự động và hướng dẫn cài từ `requirements-optional.txt`.

---

## 7. Lệnh Để Người học Tự Kiểm tra Lại Toàn bộ Dự án

Bạn có thể chạy lại toàn bộ quy trình kiểm tra bất kỳ lúc nào từ Windows PowerShell:

```powershell
# 1. Kiểm tra cấu trúc, định dạng nbformat v4, cú pháp AST và liên kết nội bộ
E:\programming\JupyterLab\image_ai_learning\.venv\Scripts\python.exe E:\programming\JupyterLab\image_ai_learning\scripts\validate_project.py

# 2. Chạy lại toàn bộ 23 notebooks và kiểm thử các pipeline huấn luyện ở chế độ nhanh
E:\programming\JupyterLab\image_ai_learning\.venv\Scripts\python.exe E:\programming\JupyterLab\image_ai_learning\scripts\run_smoke_tests.py
```
