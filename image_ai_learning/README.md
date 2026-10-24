# Dự án Học tập Xử lý Ảnh, Machine Learning & Deep Learning (`image_ai_learning`)

Chào mừng bạn đến với giáo trình thực hành **Xử lý Ảnh (Image Processing), Thị giác Máy tính (Computer Vision), Học máy (Machine Learning) và Học sâu (Deep Learning)** được thiết kế từ con số 0 đến quy trình huấn luyện, kiểm lỗi và triển khai mô hình AI thực tế.

---

## 1. Dành cho Người mới Bắt đầu: Mở file nào đầu tiên?

1. **Tài liệu định hướng**: Đọc nhanh file này (`README.md`) và [`STUDY_ROADMAP.md`](./STUDY_ROADMAP.md).
2. **Notebook mở đầu tiên**: Mở [`notebooks/00_start_here.ipynb`](./notebooks/00_start_here.ipynb) để kiểm tra môi trường Python, thiết bị CPU/GPU và làm quen với cấu trúc bài học.
3. **Tra cứu thường xuyên trong lúc học**:
   - [`MODEL_CATALOG.md`](./MODEL_CATALOG.md): Bản đồ chi tiết các nhóm bài toán và mô hình xử lý ảnh/AI.
   - [`KEYWORD_GLOSSARY.md`](./KEYWORD_GLOSSARY.md): Từ điển giải thích chi tiết từng keyword, tham số (`parameter` vs `hyperparameter`), ý nghĩa khi tăng/giảm và các cặp khái niệm dễ nhầm.
   - [`TROUBLESHOOTING.md`](./TROUBLESHOOTING.md): Cẩm nang chẩn đoán và sửa lỗi khi cài đặt, xử lý dữ liệu và huấn luyện mô hình.
   - [`VALIDATION_REPORT.md`](./VALIDATION_REPORT.md): Báo cáo kiểm thử thực tế trên máy tính hiện tại.

---

## 2. Thiết lập Môi trường Thực hành (Chuẩn & An toàn)

### Vì sao cần môi trường Python 3.12 riêng cho dự án?
Trên máy hiện tại có cài đặt `Python 3.15.0` và `Python 3.12.10`.
- Các thư viện AI cốt lõi như `PyTorch`, `torchvision`, `scikit-learn` thường **chưa phát hành gói cài đặt sẵn (prebuilt wheel)** ngay cho các bản Python vừa ra mắt hoặc đang thử nghiệm (như Python 3.15).
- Ngược lại, **Python 3.12** được hỗ trợ chính thức và ổn định bởi toàn bộ hệ sinh thái AI/Computer Vision.
- Dự án này sử dụng môi trường ảo độc lập tại `image_ai_learning/.venv` (Python 3.12.10), **hoàn toàn không can thiệp hay thay đổi Python hệ thống của bạn**.

### Lệnh tạo môi trường và cài đặt thư viện (Windows PowerShell)

Nếu môi trường `image_ai_learning/.venv` đã được tạo sẵn, bạn có thể dùng ngay. Nếu muốn tự tạo lại từ đầu:

```powershell
# 1. Di chuyển vào thư mục dự án
cd E:\programming\JupyterLab\image_ai_learning

# 2. Tạo môi trường ảo .venv bằng Python 3.12
C:\Users\Dell\AppData\Local\Python\pythoncore-3.12-64\python.exe -m venv .venv

# 3. Cài đặt các thư viện cốt lõi bằng đúng interpreter của môi trường dự án (Bản CPU mặc định)
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 4. Đăng ký kernel cho JupyterLab / VS Code để dễ chọn trong giao diện
.\.venv\Scripts\python.exe -m ipykernel install --user --name image-ai-learning --display-name "Python 3.12 (image_ai_learning)"
```

### Tùy chọn cài đặt khi có GPU NVIDIA (CUDA)
Nếu sau này bạn chạy trên máy có card đồ họa rời NVIDIA và đã cài đặt NVIDIA Driver:

```powershell
# Kiểm tra driver trước
nvidia-smi

# Cài đặt PyTorch hỗ trợ CUDA (ví dụ CUDA 12.4) trong môi trường dự án
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Mọi đoạn code trong 23 notebook đều sử dụng hàm `get_device()` tự động chọn `cuda` nếu có và tự động lùi về `cpu` nếu không có GPU.

---

## 3. Cách Mở và Chọn Kernel trong JupyterLab / VS Code

### Cách 1: Mở bằng JupyterLab
```powershell
cd E:\programming\JupyterLab\image_ai_learning
& "C:\Users\Dell\AppData\Local\Python\pythoncore-3.15-64\Scripts\jupyter-lab.exe"
```
- Khi mở một file `.ipynb`, nhìn góc trên bên phải cửa sổ notebook, bấm vào tên Kernel và chọn **`Python 3.12 (image_ai_learning)`** (hoặc trỏ tới `E:\programming\JupyterLab\image_ai_learning\.venv\Scripts\python.exe`).

### Cách 2: Mở bằng VS Code
1. Mở thư mục `E:\programming\JupyterLab\image_ai_learning` trong VS Code.
2. Mở file `notebooks/00_start_here.ipynb`.
3. Nhấn `Select Kernel` ở góc phải trên cùng -> `Python Environments...` -> Chọn `.venv (Python 3.12.10)` tại `image_ai_learning\.venv\Scripts\python.exe`.

---

## 4. Chế độ `QUICK_RUN` và Chế độ Học Đầy đủ

Mỗi notebook có thực hành huấn luyện đều có biến cấu hình ở đầu bài:
```python
QUICK_RUN = True  # Đặt False khi bạn muốn huấn luyện nhiều epoch hơn hoặc nhiều mẫu hơn
```
- **`QUICK_RUN = True` (Mặc định)**: Sử dụng số lượng mẫu vừa phải và số vòng lặp (`epochs`) ngắn (chạy xong trong vài giây trên CPU) để bạn kiểm tra toàn bộ luồng dữ liệu, kích thước tensor (`shape`), hàm mất mát (`loss`) và biểu đồ.
- **Lưu ý trung thực khoa học**: Kết quả ở chế độ `QUICK_RUN` hoặc trên dữ liệu tổng hợp (`synthetic data`) chỉ dùng để **kiểm chứng pipeline hoạt động đúng**, không dùng để kết luận chất lượng mô hình trong thực tế công nghiệp.

---

## 5. Cấu trúc Thư mục Dự án

```text
image_ai_learning/
├── README.md                     # Hướng dẫn cài đặt, khởi động và tổng quan dự án
├── STUDY_ROADMAP.md              # Lộ trình học 4 giai đoạn (A -> D) và điều kiện qua bài
├── MODEL_CATALOG.md              # Danh mục chi tiết các nhóm mô hình xử lý ảnh & AI
├── KEYWORD_GLOSSARY.md           # Từ điển giải thích chi tiết các keyword & tham số
├── TROUBLESHOOTING.md            # Cẩm nang khắc phục lỗi cài đặt, dữ liệu và training
├── VALIDATION_REPORT.md          # Báo cáo kết quả chạy kiểm thử thực tế
├── requirements.txt              # Thư viện nền tảng bắt buộc
├── requirements-optional.txt     # Thư viện mở rộng tùy chọn (ONNX, timm, OCR...)
├── notebooks/                    # 23 Jupyter Notebooks từ 00 đến 22
├── src/                          # Thư viện hàm dùng chung (utils, data, features, models, training, evaluation)
├── data/                         # Thư mục lưu dữ liệu mẫu và hướng dẫn nạp ảnh cá nhân
├── outputs/
│   ├── figures/                  # Biểu đồ và hình minh họa sinh ra từ code
│   ├── checkpoints/              # Trọng số mô hình (.pt) kèm metadata tiền xử lý
│   └── reports/                  # Báo cáo kết quả đánh giá (.json / .csv)
└── scripts/
    ├── validate_project.py       # Script kiểm tra định dạng nbformat, cú pháp AST và liên kết
    └── run_smoke_tests.py        # Script chạy kiểm thử toàn bộ 7 pipeline huấn luyện bắt buộc
```

---

## 6. Thứ tự Học 23 Notebooks

| Giai đoạn | Notebook | Chủ đề chính |
|---|---|---|
| **Khởi động** | `00_start_here.ipynb` | Kiểm tra môi trường, phần cứng, bức tranh toàn cảnh Xử lý ảnh – CV – ML – DL |
| **Giai đoạn A: Nền tảng Ảnh & Xử lý Ảnh Cổ điển** | `01_python_numpy_for_images.ipynb` | Python & NumPy cho dữ liệu ảnh (`shape`, `dtype`, `axis`, `broadcasting`, `vectorization`) |
| | `02_image_fundamentals.ipynb` | Cấu trúc ảnh số, kênh màu RGB/BGR/Grayscale/Alpha, `HWC` vs `NCHW`, `uint8` vs `float32`, Resize/Crop/Histogram |
| | `03_classical_image_processing.ipynb` | Bộ lọc làm mờ/sắc nét, Thresholding (Otsu/Adaptive), Morphology, phát hiện biên Sobel/Canny & Contour |
| **Giai đoạn B: Nền tảng Machine Learning** | `04_math_and_ml_foundations.ipynb` | Toán trực quan cho ML, phân loại các nhánh học máy, Loss Function & Gradient Descent |
| | `05_datasets_and_data_leakage.ipynb` | Chia tập `train`/`val`/`test`, Group Split, chống rò rỉ dữ liệu (`Data Leakage`), xử lý lệch lớp |
| | `06_features_and_classical_ml.ipynb` | Trích xuất đặc trưng (Pixel, Color Hist, HOG, LBP) & huấn luyện k-NN, Logistic, Naive Bayes, Tree, RF, SVM, Boosting |
| | `07_clustering_and_dimensionality_reduction.ipynb` | Học không giám sát: K-Means, DBSCAN và giảm chiều bằng PCA |
| **Giai đoạn C: Neural Network & CNN** | `08_neural_network_from_scratch.ipynb` | Tự viết mạng nơ-ron bằng NumPy thuần: Forward, Cross-Entropy, Backpropagation |
| | `09_pytorch_and_autograd.ipynb` | PyTorch Tensor, Autograd, `Dataset`/`DataLoader` và huấn luyện mạng `MLP` |
| | `10_cnn_from_scratch.ipynb` | Giải phẫu CNN (`Conv2d`, `BatchNorm2d`, `Pooling`, `Dropout`, `Residual`), tính Receptive Field & huấn luyện CNN |
| | `11_training_validation_and_debugging.ipynb` | Quy trình training chuẩn, phân biệt các khái niệm cốt lõi, lịch giảm LR, Early Stopping & gỡ lỗi training |
| **Giai đoạn D: Các Nhóm Mô hình CV & Triển khai** | `12_transfer_learning_and_fine_tuning.ipynb` | Học chuyển giao (Transfer Learning): Feature Extraction vs Fine-Tuning, đóng băng tầng (`freeze`) |
| | `13_object_detection.ipynb` | Phát hiện vật thể: R-CNN, YOLO, SSD, DETR; Bounding Box, IoU, NMS, mAP, Pretrained Inference & Fine-tuning |
| | `14_image_segmentation.ipynb` | Phân vùng ảnh: FCN, U-Net, DeepLab, Mask R-CNN, SAM; huấn luyện thực tế Mini U-Net với BCE + Dice Loss |
| | `15_autoencoders_and_anomaly_detection.ipynb` | Phát hiện bất thường: Isolation Forest, One-Class SVM & huấn luyện Convolutional Autoencoder |
| | `16_metric_learning_and_image_retrieval.ipynb` | Học biểu diễn & Truy vấn ảnh: Siamese Network, Contrastive/Triplet Loss, SimCLR/MoCo/DINO/CLIP |
| | `17_vision_transformers_and_self_supervised_learning.ipynb` | Kiến trúc Vision Transformer (Patch, Positional Encoding, Self-Attention), huấn luyện Mini-ViT & SSL |
| | `18_generative_models.ipynb` | Mô hình sinh ảnh: VAE, GAN, Diffusion (Noise Schedule, Timestep, Denoising) |
| | `19_ocr_and_visual_tasks_overview.ipynb` | Bài toán mở rộng: OCR, Pose Estimation, Object Tracking, Denoising/Super-Resolution, Depth, VLM |
| | `20_evaluation_experiment_design_and_tuning.ipynb` | Thiết kế thí nghiệm khoa học, hệ thống hóa Metric, Calibration & tinh chỉnh siêu tham số |
| | `21_inference_export_and_optimization.ipynb` | Đóng gói Checkpoint, xuất TorchScript/ONNX, Quantization, đo Latency/Throughput & viết class `ImagePredictor` |
| **Đồ án cuối** | `22_capstone_project.ipynb` | Đồ án End-to-End hoàn chỉnh + hướng dẫn đưa bộ ảnh cá nhân vào huấn luyện & tự đánh giá theo Rubric |

---

## 7. Khắc phục Nhanh Sự cố Cài đặt & Kernel

- **Lỗi `ModuleNotFoundError: No module named 'torch'` trong Jupyter**: Bạn đang chọn nhầm kernel Python hệ thống (ví dụ Python 3.15). Hãy đổi kernel sang `.venv (Python 3.12)` hoặc chạy lệnh đăng ký `ipykernel` ở Mục 2.
- **Lỗi `No module named 'src'` khi chạy cell**: Hãy chạy cell khởi tạo đầu tiên trong mỗi notebook (cell này tự động tìm `project_root` và thêm vào `sys.path`).
- Xem thêm hướng dẫn chi tiết tại [`TROUBLESHOOTING.md`](./TROUBLESHOOTING.md).
