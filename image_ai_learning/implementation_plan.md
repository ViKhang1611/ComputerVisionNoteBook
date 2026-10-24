# Kế hoạch Thực hiện Dự án Học tập `image_ai_learning/`

## 1. Kết quả Kiểm tra Môi trường & Phần cứng Thực tế

| Hạng mục | Kết quả kiểm tra thực tế | Đánh giá & Giải pháp tương thích |
|---|---|---|
| **Hệ điều hành** | Windows 11 (`10.0.26100`, AMD64) | Tương thích tốt, cần lưu ý `num_workers=0` cho PyTorch `DataLoader` trên Windows trong notebook để tránh lỗi multiprocessing. |
| **CPU** | Intel64 Family 6 Model 186 Stepping 2 (`20 logical processors`) | Đủ mạnh để chạy các bài thực hành xử lý ảnh, Machine Learning truyền thống và huấn luyện mạng Deep Learning cỡ nhỏ/trung bình ở chế độ CPU. |
| **GPU / CUDA** | `nvidia-smi` không kết nối được NVIDIA driver hoạt động | Cấu hình mặc định toàn bộ dự án chạy trên **CPU (`device = "cpu"`)**, đồng thời viết code tự động nhận diện `cuda` khi người học chạy trên máy có GPU. |
| **Python hiện có 1** | `Python 3.15.0` (`C:\Users\Dell\AppData\Local\Python\pythoncore-3.15-64\python.exe`) | Đã có `jupyterlab 4.6.4`, `numpy 2.5.4`, `opencv-python 5.0.0.93`, `pillow 13.0.0`, `matplotlib 3.11.2`. Tuy nhiên **Python 3.15 quá mới**, chưa có prebuilt wheel chính thức cho `torch`, `torchvision`, `scikit-learn`. |
| **Python hiện có 2** | `Python 3.12.10` (`C:\Users\Dell\AppData\Local\Python\pythoncore-3.12-64\python.exe`) | **Tương thích chuẩn** với toàn bộ hệ sinh thái AI/ML (`torch`, `torchvision`, `scikit-learn`, `pandas`, `opencv-python-headless`, `matplotlib`, `nbformat`, `nbclient`, `ipykernel`). |
| **Giải pháp môi trường** | Tạo môi trường ảo riêng `image_ai_learning/.venv` từ Python 3.12.10 | Không can thiệp Python hệ thống, hoàn toàn nằm trong thư mục `image_ai_learning/`. |

---

## 2. Cấu trúc Thư mục Dự án

```text
image_ai_learning/
├── README.md
├── STUDY_ROADMAP.md
├── MODEL_CATALOG.md
├── KEYWORD_GLOSSARY.md
├── TROUBLESHOOTING.md
├── VALIDATION_REPORT.md
├── requirements.txt
├── requirements-optional.txt
├── notebooks/
│   ├── 00_start_here.ipynb
│   ├── 01_python_numpy_for_images.ipynb
│   ├── 02_image_fundamentals.ipynb
│   ├── 03_classical_image_processing.ipynb
│   ├── 04_math_and_ml_foundations.ipynb
│   ├── 05_datasets_and_data_leakage.ipynb
│   ├── 06_features_and_classical_ml.ipynb
│   ├── 07_clustering_and_dimensionality_reduction.ipynb
│   ├── 08_neural_network_from_scratch.ipynb
│   ├── 09_pytorch_and_autograd.ipynb
│   ├── 10_cnn_from_scratch.ipynb
│   ├── 11_training_validation_and_debugging.ipynb
│   ├── 12_transfer_learning_and_fine_tuning.ipynb
│   ├── 13_object_detection.ipynb
│   ├── 14_image_segmentation.ipynb
│   ├── 15_autoencoders_and_anomaly_detection.ipynb
│   ├── 16_metric_learning_and_image_retrieval.ipynb
│   ├── 17_vision_transformers_and_self_supervised_learning.ipynb
│   ├── 18_generative_models.ipynb
│   ├── 19_ocr_and_visual_tasks_overview.ipynb
│   ├── 20_evaluation_experiment_design_and_tuning.ipynb
│   ├── 21_inference_export_and_optimization.ipynb
│   └── 22_capstone_project.ipynb
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── synthetic_data.py
│   ├── features.py
│   ├── models.py
│   ├── training.py
│   └── evaluation.py
├── data/
│   └── README.md
├── outputs/
│   ├── figures/
│   ├── checkpoints/
│   └── reports/
└── scripts/
    ├── validate_project.py
    └── run_smoke_tests.py
```

---

## 3. Danh sách 23 Jupyter Notebooks và Nội dung Chính

| STT | Tên file | Giai đoạn | Nội dung trọng tâm & Thực hành |
|---|---|---|---|
| 00 | `00_start_here.ipynb` | Khởi động | Kiểm tra môi trường, phần cứng, cấu hình `QUICK_RUN`, phân biệt Image Processing / CV / ML / DL, quy ước đường dẫn project root. |
| 01 | `01_python_numpy_for_images.ipynb` | Giai đoạn A | Python cơ bản cho AI, NumPy `ndarray`, `shape`, `dtype`, `axis`, `broadcasting`, `slicing`, `vectorization`, tạo ảnh bằng mảng số. |
| 02 | `02_image_fundamentals.ipynb` | Giai đoạn A | Pixel, độ phân giải, kênh màu (Grayscale, RGB, BGR, Alpha), quy ước `HWC` vs `CHW` và `NCHW`, chuyển đổi `uint8` ↔ `float32`, resize/crop/interpolation, histogram, brightness/contrast. |
| 03 | `03_classical_image_processing.ipynb` | Giai đoạn A | Blur, Gaussian filter, Median filter, Sharpening, Thresholding (global/Otsu/adaptive), Morphology (erosion, dilation, opening, closing), Edge detection (Sobel, Canny), Contour. |
| 04 | `04_math_and_ml_foundations.ipynb` | Giai đoạn B | Đại số tuyến tính & giải tích trực quan (dot product, đạo hàm, gradient), xác suất cơ bản, phân loại các nhánh học máy, loss function, Gradient Descent bằng số và đồ thị. |
| 05 | `05_datasets_and_data_leakage.ipynb` | Giai đoạn B | Chia tập `train`/`val`/`test`, Stratified split, Group split (tránh rò rỉ ảnh cùng chuỗi/người), Data Leakage khi fit `StandardScaler`/`PCA`, xử lý mất cân bằng lớp (`class imbalance`). |
| 06 | `06_features_and_classical_ml.ipynb` | Giai đoạn B | Trích xuất đặc trưng (Pixel flattening, Color Histogram, HOG, LBP) + Huấn luyện thực tế k-NN, Logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM, Gradient Boosting. |
| 07 | `07_clustering_and_dimensionality_reduction.ipynb` | Giai đoạn B | Học không giám sát: K-Means (phân cụm ảnh & lượng tử hóa màu), DBSCAN, giảm chiều bằng PCA (eigen-images, phương sai giữ lại, phân biệt PCA với mô hình phân loại). |
| 08 | `08_neural_network_from_scratch.ipynb` | Giai đoạn C | Tự cài đặt mạng nơ-ron 2 tầng bằng NumPy thuần: `Linear`, `ReLU`, `Softmax + CrossEntropy`, `forward`, `backward` (backpropagation), cập nhật trọng số và kiểm tra đạo hàm số. |
| 09 | `09_pytorch_and_autograd.ipynb` | Giai đoạn C | PyTorch `Tensor`, `requires_grad`, đồ thị tính toán (`autograd`), `nn.Module`, `Dataset`, `DataLoader`, xây dựng và huấn luyện `MLP` phân loại ảnh. |
| 10 | `10_cnn_from_scratch.ipynb` | Giai đoạn C | Giải phẫu CNN: `Conv2d` (`kernel_size`, `stride`, `padding`, `dilation`, `groups`), `MaxPool2d`, `AdaptiveAvgPool2d`, `BatchNorm2d`, `Dropout`, `Residual Block`, công thức tính shape và receptive field, huấn luyện CNN thực tế. |
| 11 | `11_training_validation_and_debugging.ipynb` | Giai đoạn C | Quy trình training chuẩn, phân biệt `Softmax` vs `CrossEntropyLoss`, `Sigmoid` vs `BCEWithLogitsLoss`, `model.eval()` vs `torch.no_grad()`, `weight_decay` vs L2, scheduler, early stopping, gradient clipping, bắt lỗi NaN/shape/device. |
| 12 | `12_transfer_learning_and_fine_tuning.ipynb` | Giai đoạn D | Phân biệt Scratch / Feature Extraction / Fine-tuning, dùng `requires_grad=False` (`freeze`) và differential learning rate, thực hành transfer learning hoàn chỉnh (kèm cơ chế fallback offline an toàn). |
| 13 | `13_object_detection.ipynb` | Giai đoạn D | Toàn cảnh Object Detection (R-CNN/Faster R-CNN, SSD/RetinaNet/YOLO, DETR), Bounding box (`xyxy`, `xywh`, `cxcywh`), `IoU`, Anchor, Objectness, `NMS`, `AP`/`mAP`, chuẩn bị annotation, pretrained inference & pipeline fine-tuning thực tế. |
| 14 | `14_image_segmentation.ipynb` | Giai đoạn D | Semantic / Instance / Panoptic / Promptable Segmentation (FCN, U-Net, DeepLab, Mask R-CNN, SAM), hàm loss `BCE + Dice Loss`, chỉ số `IoU` & `Dice`, huấn luyện thực tế mô hình Mini U-Net trên dữ liệu ảnh + mask. |
| 15 | `15_autoencoders_and_anomaly_detection.ipynb` | Giai đoạn D | Phát hiện bất thường (Anomaly Detection) trên ảnh: thống kê, Isolation Forest, One-Class SVM, huấn luyện Convolutional Autoencoder chỉ trên ảnh bình thường và phát hiện lỗi bề mặt qua bản đồ sai số tái tạo (`reconstruction error map`). |
| 16 | `16_metric_learning_and_image_retrieval.ipynb` | Giai đoạn D | Học biểu diễn & Truy vấn ảnh: Siamese Network, Contrastive Loss, Triplet Loss (`anchor`, `positive`, `negative`, `margin`), chuẩn hóa L2 embedding, tìm kiếm lân cận gần nhất (Nearest-Neighbor Image Retrieval). |
| 17 | `17_vision_transformers_and_self_supervised_learning.ipynb` | Giai đoạn D | Cơ chế Patch Embedding, Positional Encoding, Multi-Head Self-Attention, xây dựng và huấn luyện Mini-ViT; nguyên lý Self-Supervised Learning (SimCLR, MoCo, DINO) và mô hình đa phương thức CLIP. |
| 18 | `18_generative_models.ipynb` | Giai đoạn D | Mô hình sinh ảnh: VAE (`latent vector`, `reparameterization trick`, `KL divergence`), GAN (`Generator` vs `Discriminator`), Diffusion (`noise schedule`, `timestep`, `denoising`), huấn luyện thực tế mô hình sinh thu nhỏ. |
| 19 | `19_ocr_and_visual_tasks_overview.ipynb` | Giai đoạn D | Các bài toán thị giác mở rộng: OCR (Text Detection + Recognition với CTC/Attention), Pose Estimation (keypoint heatmap), Object Tracking (IoU tracker/Kalman), Denoising & Super-Resolution (PSNR/SSIM), Depth Estimation, Vision-Language Models. |
| 20 | `20_evaluation_experiment_design_and_tuning.ipynb` | Giai đoạn D | Hệ thống hóa metric (`Accuracy`, `Precision`, `Recall`, `Specificity`, `F1`, `ROC-AUC`, `PR-AUC`, `Calibration`), thiết kế thí nghiệm khoa học, Ablation study, tìm kiếm siêu tham số (Grid/Random search) không làm rò rỉ tập Test. |
| 21 | `21_inference_export_and_optimization.ipynb` | Giai đoạn D | Đóng gói checkpoint chuẩn (kèm `class_to_idx` và cấu hình tiền xử lý), xuất TorchScript / ONNX, lượng tử hóa động (Dynamic Quantization), đo `latency`, `throughput`, `model size`, xây dựng class `ImagePredictor` an toàn cho ứng dụng. |
| 22 | `22_capstone_project.ipynb` | Đồ án cuối | Dự án phân loại ảnh End-to-End hoàn chỉnh từ kiểm định dữ liệu → Baseline ML → CNN/Transfer Learning → Phân tích lỗi (Error Analysis) → Xuất model & chương trình Inference + Hướng dẫn mở rộng sang ảnh cá nhân và nhánh Detection/Segmentation/Anomaly. |

---

## 4. Tiêu chí Kiểm tra và Bàn giao

1. **Xác thực định dạng Notebook (`nbformat` v4)**: Tất cả 23 file `.ipynb` hợp lệ theo schema `nbformat` v4, có metadata kernel chuẩn, không có cell rỗng hoặc `TODO`/`...`.
2. **Kiểm tra cú pháp (AST Syntax Check)**: Mọi cell code trong cả 23 notebook đều parse thành công bằng `ast.parse`.
3. **Thực thi thực tế (Execution & Smoke Test)**:
   - Chạy thực tế các notebook nền tảng và script smoke test cho toàn bộ 7 bài bắt buộc huấn luyện thật:
     1. Baseline ML truyền thống (HOG/Color Histogram + SVM/RandomForest)
     2. MLP (PyTorch)
     3. CNN (PyTorch)
     4. Transfer Learning / Fine-tuning
     5. Image Segmentation (Mini U-Net)
     6. Autoencoder Anomaly Detection
     7. Metric Learning (Siamese / Triplet Network)
   - Kiểm tra lưu/nạp checkpoint và chạy inference thực tế.
4. **Kiểm tra liên kết nội bộ**: Mọi file Markdown và đường dẫn tham chiếu trong dự án đều tồn tại.
5. **Báo cáo trung thực trong `VALIDATION_REPORT.md`**: Ghi rõ phiên bản Python, thư viện, phần cứng CPU, danh sách các notebook đã chạy thực tế bằng `nbclient`, thời gian chạy, kết quả đạt được và các phần cần GPU/internet tùy chọn.
