# Bản đồ Các Nhóm Bài toán và Mô hình Xử lý Ảnh & Computer Vision (`MODEL_CATALOG.md`)

- **Ngày tổng hợp**: `2026-10-24`
- **Tuyên bố phạm vi**: Tài liệu này **không tuyên bố liệt kê toàn bộ mọi mô hình AI đang tồn tại trên thế giới**. Thay vào đó, đây là **bản đồ hệ thống hóa các nhóm bài toán chính và những mô hình đại diện quan trọng nhất**, được phân loại rõ theo 3 vai trò:
  - 🏛️ **Mô hình lịch sử (Historical)**: Đặt nền móng tư tưởng nhưng hiện nay ít dùng trực tiếp trong sản phẩm mới.
  - 🧱 **Mô hình nền tảng (Foundational)**: Kiến trúc chuẩn mực để học nguyên lý, làm đường cơ sở (baseline) hoặc làm khung xương sống (`backbone`).
  - 🚀 **Hướng đang được sử dụng rộng rãi (Actively Used)**: Các dòng mô hình hiện đại đang được dùng phổ biến trong nghiên cứu và công nghiệp.

---

## Phần I. Phân biệt 6 Chế độ Huấn luyện và Suy luận (Cực kỳ Quan trọng)

Người mới học rất dễ nhầm lẫn giữa việc "chạy thử một mô hình có sẵn" với việc "huấn luyện mô hình". Dưới đây là bảng phân biệt rạch ròi:

| Thuật ngữ tiếng Anh | Nghĩa tiếng Việt | Trọng số khởi tạo (`Weights`) | Những tầng nào cập nhật gradient (`requires_grad=True`)? | Khi nào sử dụng? |
|---|---|---|---|---|
| **1. Training from scratch** | Huấn luyện từ đầu | Ngẫu nhiên (Kaiming / Xavier initialization) | Toàn bộ mạng (`100%` tham số) | Khi học nguyên lý kiến trúc, hoặc khi có tập dữ liệu rất lớn / miền dữ liệu đặc thù hoàn toàn khác ảnh tự nhiên (kính hiển vi, radar) và đủ tài nguyên tính toán. |
| **2. Pretrained Inference** | Suy luận bằng mô hình huấn luyện sẵn | Nạp trọng số đã học sẵn (ví dụ ImageNet, COCO) | **Không tầng nào cả** (`0%`, không có bước `backward` hay `optimizer.step`) | Khi bài toán của bạn trùng khớp với các lớp có sẵn của mô hình. **Lưu ý: Chạy pretrained inference KHÔNG phải là huấn luyện (training)!** |
| **3. Feature Extraction (Linear Probing)** | Trích xuất đặc trưng (Đóng băng backbone) | Backbone nạp trọng số pretrained; tầng phân loại cuối (`head`) khởi tạo mới | Chỉ cập nhật tầng `head` cuối cùng; đóng băng (`freeze`) toàn bộ `backbone` (`requires_grad=False`) | Khi tập dữ liệu mới của bạn **rất nhỏ** (vài chục đến vài trăm ảnh/lớp) hoặc chỉ có CPU/GPU yếu. Tránh overfitting rất tốt. |
| **4. Fine-tuning** | Tinh chỉnh mô hình | Backbone nạp trọng số pretrained; tầng `head` khởi tạo mới | Mở khóa (`unfreeze`) một phần hoặc toàn bộ `backbone` cùng tầng `head`, thường dùng `learning_rate` nhỏ cho backbone | Khi tập dữ liệu mới có quy mô vừa đến lớn và bạn muốn mô hình thích nghi sâu với đặc trưng riêng của bài toán. |
| **5. Transfer Learning** | Học chuyển giao | Thuật ngữ bao trùm cho cả **Feature Extraction** và **Fine-tuning** | Tùy chiến lược (đóng băng hoặc mở khóa dần từ tầng sâu ra tầng nông) | Hầu hết các bài toán thị giác thực tế trong công nghiệp đều bắt đầu bằng Transfer Learning thay vì huấn luyện từ đầu. |
| **6. Zero-shot / Few-shot Learning** | Học không mẫu / Học cực ít mẫu | Nạp từ mô hình nền tảng đa phương thức hoặc tự giám sát lớn (như CLIP, SAM, DINOv2) | Thường không cập nhật trọng số (Zero-shot qua prompt văn bản/điểm click) hoặc chỉ đối sánh vector nhúng của `1..5` ảnh mẫu (Few-shot) | Khi cần phân loại hoặc tách vật thể mới ngay lập tức mà chưa kịp thu thập hàng nghìn nhãn dữ liệu. |

---

## Phần II. Phân biệt Trích xuất Đặc trưng, Giảm chiều và Mô hình Học máy

Trước khi đi vào các nhóm mô hình, cần tránh nhầm lẫn rất phổ biến sau:

| Tên công cụ / thuật ngữ | Bản chất thực sự | Đầu vào → Đầu ra | Có phải là mô hình tự đưa ra nhãn phân loại không? |
|---|---|---|---|
| **Pixel Flattening** | Biểu diễn dữ liệu thô (Baseline representation) | Ảnh `(H, W, C)` → Vector `(H*W*C,)` | **Không**. Chỉ là bước duỗi mảng 3D thành 1D. |
| **Color Histogram** | Phương pháp trích xuất đặc trưng màu sắc (Feature Extraction) | Ảnh `(H, W, C)` → Vector tần suất màu `(B*C,)` | **Không**. Cần đưa vector này vào k-NN, SVM, Random Forest... để phân loại. |
| **HOG (Histogram of Oriented Gradients)** | Phương pháp trích xuất đặc trưng hình dáng/cạnh (Feature Extraction) | Ảnh `(H, W)` → Vector hướng gradient cục bộ | **Không**. HOG chỉ biến ảnh thành vector đặc trưng hình học. |
| **LBP (Local Binary Patterns)** | Phương pháp trích xuất đặc trưng kết cấu bề mặt (Texture Feature Extraction) | Ảnh `(H, W)` → Vector histogram mã nhị phân cục bộ | **Không**. LBP chỉ mô tả vân bề mặt (nhẵn, sần, kẻ sọc...). |
| **PCA (Principal Component Analysis)** | Thuật toán giảm chiều dữ liệu không giám sát (Dimensionality Reduction) | Ma trận đặc trưng `(N, D)` → Ma trận chiều thấp `(N, K)` với `K < D` | **Không**. PCA tìm các trục phương sai lớn nhất để nén dữ liệu hoặc khử nhiễu, không dự đoán nhãn lớp. |

---

## Phần III. Chi tiết 7 Nhóm Bài toán và Mô hình Computer Vision

---

### Nhóm 1: Phân loại Ảnh (Image Classification)

#### 1.1 Tổng quan kỹ thuật
- **Bài toán giải quyết**: Gán cho toàn bộ bức ảnh một nhãn lớp duy nhất (Multi-class Classification) hoặc một tập nhiều nhãn (Multi-label Classification).
- **Định dạng dữ liệu và nhãn**:
  - Cấu trúc thư mục `ImageFolder` (`train/class_name/img.jpg`) hoặc bảng CSV `(filepath, label_id)`.
  - Nhãn đơn lớp: số nguyên `y ∈ {0, 1, ..., C-1}` (kiểu `torch.long`).
  - Nhãn đa nhãn: vector nhị phân `y ∈ {0, 1}^C` (kiểu `torch.float32`).
- **Đầu vào / Đầu ra**:
  - Với ML truyền thống: Đầu vào là vector đặc trưng `(N, D)` (từ HOG/LBP/Color Histogram) → Đầu ra là nhãn lớp `(N,)` hoặc xác suất `(N, C)`.
  - Với Deep Learning: Đầu vào là tensor ảnh chuẩn hóa `(N, C_in, H, W)` → Đầu ra là `logits` `(N, num_classes)` → Qua `Softmax` (đơn lớp) hoặc `Sigmoid` (đa nhãn) để lấy xác suất.
- **Loss và Metric thường dùng**:
  - Loss: `nn.CrossEntropyLoss` (đơn lớp, nhận trực tiếp `logits`), `nn.BCEWithLogitsLoss` (nhị phân/đa nhãn, nhận trực tiếp `logits`), Focal Loss (khi mất cân bằng lớp nặng).
  - Metric: `Accuracy`, `Precision`, `Recall`, `Specificity`, `F1-score` (`Macro`, `Micro`, `Weighted`), `Confusion Matrix`, `ROC-AUC`, `PR-AUC`, `Top-k Accuracy`.
- **Lỗi thường gặp**:
  1. Đặt `nn.Softmax` ở cuối model rồi đưa vào `nn.CrossEntropyLoss` (tính hai lần softmax làm bão hòa đạo hàm).
  2. Không chuẩn hóa ảnh (`mean`, `std`) giống lúc huấn luyện khi đưa vào suy luận.
  3. Chỉ nhìn vào `Accuracy` trên tập dữ liệu lệch lớp (ví dụ 95% ảnh bình thường, 5% ảnh lỗi).
- **Bài học liên quan**: `06_features_and_classical_ml.ipynb`, `08_neural_network_from_scratch.ipynb`, `09_pytorch_and_autograd.ipynb`, `10_cnn_from_scratch.ipynb`, `12_transfer_learning_and_fine_tuning.ipynb`, `17_vision_transformers_and_self_supervised_learning.ipynb`.

#### 1.2 Phân nhóm A: Các mô hình Machine Learning truyền thống (Classical ML)

| Mô hình | Nguyên lý hoạt động cốt lõi | Ưu điểm | Hạn chế | Chi phí tính toán | Khi nên dùng / Không nên dùng |
|---|---|---|---|---|---|
| **k-Nearest Neighbors (k-NN)** 🧱 | Không học hàm tham số; lưu toàn bộ tập train và bầu chọn đa số dựa trên `k` điểm gần nhất theo khoảng cách (Euclidean / Cosine). | Cực kỳ dễ hiểu, không tốn thời gian train, làm baseline kiểm tra chất lượng đặc trưng rất tốt. | Suy luận (inference) chậm khi tập train lớn (`O(N*D)`), nhạy cảm với chiều cao (curse of dimensionality) và thang đo đặc trưng. | Train: Rất thấp; Inference: Cao theo số mẫu `N`. | Dùng làm baseline nhanh hoặc bài toán truy vấn vector nhúng; không dùng trực tiếp trên pixel ảnh lớn. |
| **Logistic Regression** 🧱 | Tổ hợp tuyến tính đặc trưng `z = Wx + b` rồi đưa qua hàm `Sigmoid` / `Softmax`, tối ưu bằng Cross-Entropy. | Nhanh, xuất ra xác suất có hiệu chỉnh tốt, trọng số dễ giải thích. | Chỉ tạo được biên quyết định tuyến tính (phẳng) trong không gian đặc trưng. | Rất thấp. | Dùng làm baseline tuyến tính chuẩn mực hoặc làm tầng phân loại trên đỉnh đặc trưng đã trích xuất. |
| **Naive Bayes (GaussianNB)** 🧱 | Áp dụng định lý Bayes với giả định các đặc trưng độc lập có điều kiện khi biết nhãn lớp. | Cực nhanh, hoạt động được ngay cả khi rất ít dữ liệu. | Giả định độc lập giữa các pixel/đặc trưng ảnh hầu như luôn bị vi phạm trong thực tế. | Cực thấp. | Dùng làm baseline tốc độ cao cho đặc trưng thống kê ngắn; không dùng cho bài toán ảnh phức tạp. |
| **Decision Tree** 🧱 | Chia không gian đặc trưng đệ quy bằng các ngưỡng `x_j <= threshold` để giảm độ tạp (`Gini` hoặc `Entropy`). | Không cần chuẩn hóa thang đo đặc trưng, dễ vẽ cây giải thích luật. | Rất dễ học vẹt (overfitting), biên quyết định bậc thang kém bền vững với nhiễu ảnh. | Thấp. | Dùng để học nguyên lý cây và làm thành phần cơ sở cho Random Forest / Boosting. |
| **Random Forest** 🚀 | Kết hợp (Bagging) hàng trăm cây quyết định huấn luyện trên các tập con mẫu (bootstrap) và tập con đặc trưng ngẫu nhiên. | Chống overfitting tốt hơn cây đơn, chạy tốt trên đặc trưng bảng/kết hợp (HOG + Color + Hình học). | Mô hình nặng hơn, không tự học được đặc trưng phân cấp từ pixel thô như CNN. | Trung bình. | Rất mạnh khi kết hợp các đặc trưng đo đạc thủ công trong kiểm tra công nghiệp/y sinh nhỏ. |
| **Support Vector Machine (SVM)** 🚀 | Tìm siêu phẳng phân cách có lề (`margin`) cực đại; dùng `Kernel trick` (như `RBF`) để ánh xạ ngầm lên không gian chiều cao. | Hiệu quả cao trên tập dữ liệu vừa và nhỏ với đặc trưng HOG/LBP chuẩn hóa tốt. | Huấn luyện chậm khi số mẫu `N > 50,000`; bắt buộc phải chuẩn hóa đặc trưng (`StandardScaler`). | Trung bình – Cao theo `N^2`. | Cặp bài trùng kinh điển `HOG + Linear/RBF SVM` khi dữ liệu nhỏ và không có GPU. |
| **Gradient Boosting (HistGradientBoosting / XGBoost / LightGBM / CatBoost)** 🚀 | Xây dựng chuỗi cây quyết định nối tiếp nhau, mỗi cây sau học cách sửa phần dư (gradient của loss) của tổng các cây trước. | Độ chính xác hàng đầu trên dữ liệu dạng bảng và vector đặc trưng tổng hợp. | Dễ overfit nếu chỉnh siêu tham số sai; không khai thác trực tiếp cấu trúc không gian 2D của ảnh thô. | Trung bình. | Dùng khi đầu vào là bảng đo đạc đặc trưng vùng ảnh (diện tích, độ tròn, màu sắc, kết cấu). |

#### 1.3 Phân nhóm B: Các kiến trúc CNN & Vision Transformer cho Phân loại Ảnh

| Mô hình & Năm | Phân loại vai trò | Đóng góp kiến trúc cốt lõi | Chi phí tính toán | Khi nên dùng / Không nên dùng | Nguồn / Paper gốc |
|---|---|---|---|---|---|
| **LeNet-5** (1998) | 🏛️ Lịch sử / Giáo dục | Chuỗi `Conv -> Pool -> Conv -> Pool -> FC` đầu tiên đọc chữ số viết tay. | Cực thấp | Chỉ dùng để học nguyên lý CNN cơ bản hoặc ảnh cực nhỏ (`28x28`). | LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, 1998. |
| **AlexNet** (2012) | 🏛️ Lịch sử | Đưa `ReLU`, `Dropout`, huấn luyện GPU và Data Augmentation vào CNN quy mô lớn trên ImageNet. | Trung bình | Giá trị lịch sử; hiện nay thay bằng ResNet/MobileNet. | Krizhevsky et al., *ImageNet Classification with Deep CNNs*, NeurIPS 2012. |
| **VGG-16 / VGG-19** (2014) | 🏛️/🧱 Lịch sử & Nền tảng | Chuẩn hóa việc xếp chồng các bộ lọc nhỏ `3x3` thay vì bộ lọc lớn `7x7` hay `11x11`. | Rất cao (do 3 tầng FC cuối chứa >100M tham số) | Không nên dùng cho thiết bị yếu; đôi khi còn dùng trích xuất đặc trưng cảm nhận (Perceptual Loss). | Simonyan & Zisserman, *Very Deep Convolutional Networks*, ICLR 2015. |
| **Inception / GoogLeNet** (2014) | 🧱 Nền tảng | Khối Inception xử lý đa tỉ lệ song song (`1x1`, `3x3`, `5x5`, `Pool`) và dùng tích chập `1x1` để giảm số kênh. | Trung bình | Minh họa xuất sắc cho tư duy đa tỉ lệ và tiết kiệm tham số bằng `1x1 Conv`. | Szegedy et al., *Going Deeper with Convolutions*, CVPR 2015. |
| **ResNet (ResNet-18/34/50/101)** (2015) | 🧱/🚀 Nền tảng & Đang dùng mạnh | **Residual / Skip Connection** (`y = F(x) + x`) giải quyết triệt để hiện tượng suy thoái đạo hàm khi mạng rất sâu, dùng `Global Average Pooling` thay tầng FC khổng lồ. | Thấp (`ResNet-18`) đến Trung bình (`ResNet-50`) | **Lựa chọn mặc định số 1** khi bắt đầu một dự án phân loại ảnh hoặc làm backbone cho Detection/Segmentation. | He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016. |
| **DenseNet (DenseNet-121)** (2017) | 🧱/🚀 Nền tảng | Nối kênh (`concatenation`) đầu ra của mọi tầng trước vào tất cả các tầng sau trong cùng một khối (`DenseBlock`), tái sử dụng đặc trưng tối đa. | Trung bình (tiết kiệm tham số nhưng tốn bộ nhớ trung gian) | Phù hợp cho ảnh y tế (X-quang) nhờ giữ lại cả đặc trưng mức thấp lẫn mức cao. | Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017. |
| **MobileNet (v2, v3)** (2018–2019) | 🚀 Đang dùng rộng rãi | **Depthwise Separable Convolution** (tách tích chập không gian và tích chập kênh `1x1`) + **Inverted Residuals** & Linear Bottleneck. | Rất thấp (Tối ưu cho CPU, điện thoại, thiết bị nhúng) | Dùng khi cần chạy thời gian thực trên CPU, Raspberry Pi, thiết bị di động. | Sandler et al., *MobileNetV2*, CVPR 2018; Howard et al., *MobileNetV3*, ICCV 2019. |
| **EfficientNet (B0–B7, V2)** (2019–2021) | 🚀 Đang dùng rộng rãi | **Compound Scaling** (cân bằng đồng thời độ sâu `depth`, độ rộng kênh `width` và độ phân giải ảnh `resolution`) + khối `MBConv` + `Squeeze-and-Excitation`. | Thấp (`B0`) đến Cao (`B7`) | Dùng khi cần tỷ lệ Độ chính xác / Số phép tính (FLOPs) cao trong bài toán phân loại ảnh. | Tan & Le, *EfficientNet*, ICML 2019; *EfficientNetV2*, ICML 2021. |
| **ConvNeXt** (2022) | 🚀 Đang dùng rộng rãi | Hiện đại hóa CNN thuần túy theo thiết kế của Vision Transformer (kernel lớn `7x7` depthwise, `GELU`, `LayerNorm`, inverted bottleneck). | Trung bình – Cao | Khi muốn hiệu năng ngang ngửa Transformer nhưng vẫn giữ sự đơn giản, hiệu quả của tích chập. | Liu et al., *A ConvNet for the 2020s*, CVPR 2022. |
| **Vision Transformer (ViT)** (2020) | 🧱/🚀 Nền tảng & Đang dùng | Cắt ảnh thành chuỗi mảnh (`patches` `16x16`), gắn `Positional Encoding` và đưa vào `Transformer Encoder` thuần túy với cơ chế `Multi-Head Self-Attention`. | Cao (`O(N^2)` theo số lượng patches) | Rất mạnh khi dùng trọng số **Pretrained** trên tập lớn (ImageNet-21k, CLIP, DINOv2); **không nên** train từ đầu trên tập dữ liệu nhỏ vì thiếu thiên kiến quy nạp không gian (inductive bias) như CNN. | Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021. |
| **Swin Transformer** (2021) | 🚀 Đang dùng rộng rãi | Tính Self-Attention cục bộ trong các cửa sổ dịch chuyển (**Shifted Windows**) và giảm độ phân giải phân cấp như CNN. | Trung bình – Cao (Tuyến tính theo kích thước ảnh) | Làm backbone đa năng cho cả Classification, Object Detection và Segmentation độ phân giải cao. | Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, ICCV 2021. |

---

### Nhóm 2: Phát hiện Vật thể (Object Detection)

#### 2.1 Tổng quan kỹ thuật
- **Bài toán giải quyết**: Tìm vị trí của tất cả các vật thể quan tâm trong ảnh (bằng khung chữ nhật bao quanh – `bounding box`) đồng thời phân loại từng vật thể đó thuộc lớp nào.
- **Định dạng dữ liệu và nhãn**:
  - Mỗi ảnh đi kèm danh sách `M` vật thể (`M` thay đổi tùy ảnh, có thể bằng `0`).
  - Ba quy ước tọa độ phổ biến (cực kỳ dễ nhầm nếu không kiểm tra):
    1. **Pascal VOC (`xyxy`)**: `[x_min, y_min, x_max, y_max]` theo đơn vị pixel tuyệt đối. (Được `torchvision.models.detection` sử dụng).
    2. **COCO (`xywh`)**: `[x_min, y_min, width, height]` theo đơn vị pixel tuyệt đối.
    3. **YOLO (`cxcywh` chuẩn hóa)**: `[x_center / W, y_center / H, width / W, height / H]` nằm trong đoạn `[0.0, 1.0]`.
- **Đầu vào / Đầu ra**:
  - Đầu vào: Ảnh `(C, H, W)` (thường giữ kích thước linh hoạt hoặc resize kèm padding – letterbox).
  - Đầu ra khi suy luận: Danh sách các dự đoán sau bước lọc `NMS`, mỗi dự đoán gồm `(box [x1, y1, x2, y2], class_label, confidence_score)`.
- **Loss và Metric thường dùng**:
  - Loss: Tổng hợp của **Classification Loss** (`CrossEntropy` / `Focal Loss` cho nhãn lớp & `objectness`) + **Box Regression Loss** (`Smooth L1` / `GIoU` / `DIoU` / `CIoU Loss`).
  - Metric: `IoU` (Intersection over Union), `Precision–Recall Curve`, `AP` (Average Precision – diện tích dưới đường PR của 1 lớp), `mAP@0.5` (trung bình AP của các lớp tại ngưỡng `IoU = 0.5`), `mAP@[0.5:0.95]` (chuẩn COCO trung bình từ `IoU 0.50` đến `0.95` bước `0.05`).
- **Lỗi thường gặp**:
  1. Nhầm định dạng khung `xywh` sang `xyxy` hoặc quên nhân/chia kích thước ảnh khi resize.
  2. Quên lớp nền (`background = 0`) trong các mô hình họ Faster R-CNN / SSD của `torchvision`.
  3. Đặt ngưỡng `conf_threshold` hoặc `nms_iou_threshold` không phù hợp dẫn đến giữ lại nhiều khung trùng lặp hoặc bỏ sót vật thể đứng gần nhau.
- **Bài học liên quan**: `13_object_detection.ipynb`.

#### 2.2 Các dòng mô hình Object Detection tiêu biểu

| Nhóm kiến trúc | Mô hình đại diện | Vai trò | Nguyên lý hoạt động & Thành phần kiến trúc | Ưu điểm & Hạn chế | Nguồn / Paper gốc |
|---|---|---|---|---|---|
| **Two-Stage (Hai giai đoạn)** | **R-CNN (2014), Fast R-CNN (2015)** | 🏛️ Lịch sử | Dùng thuật toán Selective Search ngoài mạng để đề xuất ~2000 vùng ứng viên rồi mới chạy CNN phân loại. | Đặt nền móng cho Detection hiện đại nhưng rất chậm vì bước đề xuất vùng tách rời. | Girshick et al., CVPR 2014 / ICCV 2015. |
| **Two-Stage (Hai giai đoạn)** | **Faster R-CNN (2015) + FPN (2017)** | 🧱/🚀 Nền tảng & Đang dùng | Giai đoạn 1: **RPN (Region Proposal Network)** quét các khung mẫu (`anchors`) trên tháp đặc trưng **FPN** để đề xuất vùng có vật thể (`objectness`). Giai đoạn 2: **RoI Align / RoI Pooling** cắt đặc trưng từng vùng để phân loại và tinh chỉnh tọa độ hộp. | **Ưu**: Độ chính xác định vị cao, xử lý vật thể nhỏ tốt nhờ FPN, hỗ trợ chính thức trong `torchvision`. **Hạn chế**: Tốc độ chậm hơn dòng One-stage trên CPU/thiết bị yếu. | Ren et al., *Faster R-CNN*, NeurIPS 2015; Lin et al., *Feature Pyramid Networks*, CVPR 2017. |
| **One-Stage (Một giai đoạn)** | **SSD (Single Shot MultiBox Detector, 2016)** | 🧱/🚀 Nền tảng & Nhúng | Dự đoán trực tiếp nhãn lớp và độ lệch khung (`anchor offsets`) ngay trên nhiều tầng đặc trưng có độ phân giải khác nhau trong một lượt chạy duy nhất. | **Ưu**: Rất nhanh, bản `SSDLite + MobileNetV3` chạy tốt trên CPU. **Hạn chế**: Kém chính xác hơn trên vật thể rất nhỏ so với kiến trúc có FPN sâu. | Liu et al., *SSD: Single Shot MultiBox Detector*, ECCV 2016. |
| **One-Stage (Một giai đoạn)** | **RetinaNet (2017)** | 🧱/🚀 Nền tảng | Kết hợp FPN với hàm mất mát **Focal Loss** (`-(1 - p_t)^gamma * log(p_t)`) để giải quyết sự mất cân bằng cực lớn giữa hàng chục nghìn anchor nền (easy negatives) và vài anchor chứa vật thể. | **Ưu**: Độ chính xác tiệm cận Two-stage nhưng giữ tốc độ One-stage. | Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017. |
| **One-Stage (Một giai đoạn)** | **Họ YOLO (YOLOv1 -> YOLOv3/v4/v5/v8/v10/v11)** | 🚀 Đang dùng rộng rãi | Chia ảnh thành lưới ô (`grid`), dự đoán trực tiếp tọa độ tâm, kích thước, độ tự tin và xác suất lớp. Các thế hệ mới sử dụng kiến trúc `CSPNet`, `PANet`, `Decoupled Head`, thiết kế `Anchor-free` và hướng tới bỏ cả `NMS` (như YOLOv10). | **Ưu**: Tốc độ thời gian thực xuất sắc, dễ triển khai công nghiệp. **Hạn chế**: Các phiên bản thuộc nhiều nhóm tác giả/giấy phép khác nhau (ví dụ GPL/AGPL vs Apache-2.0) – **cần kiểm tra kỹ giấy phép trước khi dùng thương mại**. | Redmon et al., *You Only Look Once*, CVPR 2016; Bochkovskiy et al., 2020; Wang et al., NeurIPS 2024. |
| **Transformer-based** | **DETR (2020), Deformable DETR (2021), RT-DETR (2023)** | 🚀 Đang dùng rộng rãi | Coi phát hiện vật thể là bài toán dự đoán tập hợp trực tiếp (Set Prediction) bằng bộ truy vấn vật thể (`Object Queries`) trong Transformer Decoder và ghép cặp nghiệm tối ưu bằng thuật toán **Hungarian Matching** — loại bỏ hoàn toàn `Anchor` thủ công và `NMS`. | **Ưu**: Pipeline đầu-cuối (end-to-end) gọn đẹp, RT-DETR đạt tốc độ thời gian thực. **Hạn chế**: DETR gốc hội tụ rất chậm (cần nhiều epoch) và tốn bộ nhớ hơn CNN nhỏ. | Carion et al., *End-to-End Object Detection with Transformers (DETR)*, ECCV 2020. |

---

### Nhóm 3: Phân vùng Ảnh (Image Segmentation)

#### 3.1 Tổng quan kỹ thuật và 4 bài toán con
1. **Semantic Segmentation (Phân vùng ngữ nghĩa)**: Gán nhãn lớp cho **mọi điểm ảnh (pixel)** trong ảnh (ví dụ: pixel thuộc đường, xe, cây, nền), nhưng **không tách biệt** các cá thể cùng lớp đứng cạnh nhau (2 chiếc xe chạm nhau đều mang nhãn `car`).
2. **Instance Segmentation (Phân vùng thực thể)**: Chỉ tìm các vật thể đếm được (`things`), vừa đóng khung vừa tạo mặt nạ (`mask`) riêng biệt cho từng cá thể (`car_1`, `car_2`).
3. **Panoptic Segmentation (Phân vùng toàn cảnh)**: Hợp nhất cả hai bài toán trên — mọi pixel trong ảnh đều có cặp nhãn `(class_id, instance_id)`, vừa phân vùng nền vô định hình (`stuff`: bầu trời, mặt đường) vừa tách riêng từng vật thể đếm được (`things`: từng người, từng xe).
4. **Promptable Segmentation (Phân vùng theo gợi ý - tương tác)**: Mô hình nhận ảnh cùng một tín hiệu gợi ý (`prompt`: điểm click chuột, khung hộp, hoặc văn bản) và trả về mặt nạ vật thể tương ứng ngay lập tức (tiêu biểu là họ **SAM – Segment Anything Model**).

- **Định dạng dữ liệu và nhãn**:
  - Semantic: Ảnh mặt nạ 2D `(H, W)` kiểu số nguyên, trong đó giá trị tại `(y, x)` là ID của lớp (`0..C-1`).
  - Instance: Danh sách các mặt nạ nhị phân `(M, H, W)` hoặc đa giác (`polygon` chuẩn COCO JSON) kèm nhãn lớp cho từng cá thể.
- **Loss và Metric thường dùng**:
  - Loss: Pixel-wise `CrossEntropyLoss` / `BCEWithLogitsLoss`, kết hợp với **Dice Loss** hoặc **Focal Loss** để xử lý trường hợp vật thể chiếm diện tích rất nhỏ so với nền.
  - Metric: **IoU (Intersection over Union / Jaccard Index)**, **Dice Coefficient** ($2|A \cap B| / (|A| + |B|)$), **mIoU** (trung bình IoU qua các lớp), **Pixel Accuracy** (dễ gây ảo tưởng nếu nền chiếm 95% ảnh), **PQ (Panoptic Quality)**.
- **Bài học liên quan**: `14_image_segmentation.ipynb`.

#### 3.2 Các mô hình Segmentation đại diện

| Mô hình | Nhánh bài toán & Vai trò | Nguyên lý kiến trúc cốt lõi | Khi nên dùng / Hạn chế | Nguồn / Paper gốc |
|---|---|---|---|---|
| **FCN (Fully Convolutional Network, 2015)** | Semantic 🏛️/🧱 | Thay toàn bộ tầng `Linear` cuối của mạng phân loại bằng tầng tích chập `1x1` và phóng đại lại kích thước gốc bằng `Transposed Convolution`. | Nền tảng lịch sử; biên vật thể còn thô nếu thiếu nhiều nhánh kết nối chi tiết. | Long et al., *Fully Convolutional Networks for Semantic Segmentation*, CVPR 2015. |
| **U-Net (2015)** | Semantic 🧱/🚀 | Kiến trúc đối xứng hình chữ U gồm **Encoder** (thu nhỏ không gian để bắt ngữ cảnh) và **Decoder** (phóng đại không gian) nối trực tiếp với nhau qua các **Skip Connections** (`concatenation`) ở từng độ phân giải. | **Lựa chọn số 1** cho ảnh y sinh, kiểm tra lỗi bề mặt công nghiệp, ảnh vệ tinh, đặc biệt hiệu quả ngay cả khi số lượng ảnh huấn luyện không lớn. | Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015. |
| **DeepLabv3 / DeepLabv3+ (2017–2018)** | Semantic 🚀 | Sử dụng **Atrous (Dilated) Convolution** (tích chập giãn nở để mở rộng vùng nhìn `receptive field` mà không làm giảm độ phân giải) và mô-đun **ASPP (Atrous Spatial Pyramid Pooling)** đa tỉ lệ. | Rất mạnh cho ảnh cảnh vật tự nhiên phức tạp có vật thể ở nhiều kích thước khác nhau (hỗ trợ sẵn trong `torchvision.models.segmentation`). | Chen et al., *Encoder-Decoder with Atrous Separable Convolution (DeepLabv3+)*, ECCV 2018. |
| **Mask R-CNN (2017)** | Instance 🧱/🚀 | Mở rộng từ Faster R-CNN bằng cách thêm nhánh thứ 3 (song song với nhánh phân loại và nhánh khung hộp) dùng `FCN` nhỏ trên từng vùng **RoIAlign** để dự đoán mặt nạ nhị phân cho từng vật thể. | Chuẩn mực cho Instance Segmentation; hỗ trợ sẵn trong `torchvision`. | He et al., *Mask R-CNN*, ICCV 2017. |
| **Mask2Former (2022)** | Semantic + Instance + Panoptic 🚀 | Dùng kiến trúc Transformer Decoder với cơ chế **Masked Attention** để dự đoán trực tiếp một tập hợp các mặt nạ kèm nhãn lớp cho cả 3 bài toán phân vùng. | Độ chính xác rất cao trên các bộ chuẩn hiện đại; chi phí tính toán nặng hơn U-Net/DeepLab. | Cheng et al., *Masked-attention Mask Transformer for Universal Image Segmentation*, CVPR 2022. |
| **SAM (Segment Anything, 2023) & SAM 2 (2024)** | Promptable Foundation Model 🚀 | Gồm `Image Encoder` lớn (ViT), `Prompt Encoder` (nhận điểm click, hộp, mask) và `Mask Decoder` gọn nhẹ, được huấn luyện trên hơn 1 tỷ mặt nạ (SA-1B). | Tuyệt vời để **gán nhãn bán tự động (auto-labeling)** hoặc tách vật thể zero-shot theo tương tác người dùng; cần fine-tune hoặc kết hợp detector nếu muốn phân loại chuyên biệt tự động hoàn toàn. | Kirillov et al., *Segment Anything*, ICCV 2023; Ravi et al., *SAM 2*, 2024. |

---

### Nhóm 4: Phát hiện Bất thường trên Ảnh (Visual Anomaly Detection)

#### 4.1 Tổng quan kỹ thuật
- **Bài toán giải quyết**: Phát hiện ảnh có lỗi/dị vật (Image-level Anomaly Classification) và khoanh vùng đúng những pixel bị lỗi như vết xước, vết nứt, đốm bẩn (Pixel-level Anomaly Localization) khi **trong lúc huấn luyện chỉ có (hoặc chủ yếu có) ảnh bình thường (Normal / Good samples)**.
- **Vì sao không dùng phân loại 2 lớp thông thường?**: Trong dây chuyền sản xuất, 99.9% sản phẩm là bình thường, còn lỗi có thể xuất hiện dưới muôn hình vạn trạng chưa từng thấy trước đó.
- **Định dạng dữ liệu**:
  - Tập `train`: Chỉ gồm các ảnh mẫu chuẩn không lỗi (`label = 0`).
  - Tập `val / test`: Gồm cả ảnh bình thường (`label = 0`) và ảnh lỗi (`label = 1`) kèm mặt nạ vị trí lỗi `(H, W)` để đánh giá.
- **Đầu vào / Đầu ra**:
  - Đầu vào: Ảnh `(C, H, W)`.
  - Đầu ra: Điểm số bất thường toàn ảnh (`anomaly_score ∈ ℝ`) và bản đồ nhiệt bất thường mức điểm ảnh (`anomaly_map` kích thước `(H, W)`). Nếu `anomaly_score > threshold` thì kết luận bất thường.
- **Loss và Metric thường dùng**:
  - Loss (với hướng tái tạo): `MSELoss` ($L_2$), `L1Loss`, hoặc `SSIM Loss` giữa ảnh gốc $x$ và ảnh tái tạo $\hat{x}$.
  - Metric: `Image-level ROC-AUC`, `PR-AUC`, `F1-score` tại ngưỡng tối ưu, và `Pixel-level AUROC` / `IoU` trên bản đồ lỗi.
- **Bài học liên quan**: `15_autoencoders_and_anomaly_detection.ipynb`.

#### 4.2 Các phương pháp và mô hình đại diện

| Phương pháp / Mô hình | Vai trò | Nguyên lý hoạt động | Ưu điểm & Hạn chế | Nguồn tham khảo |
|---|---|---|---|---|
| **Sai lệch mẫu chuẩn & Khoảng cách Mahalanobis** | 🧱 Nền tảng thống kê | Căn chỉnh ảnh về đúng vị trí chuẩn rồi tính trung bình $\mu$ và ma trận hiệp phương sai $\Sigma$ của đặc trưng; điểm xa phân phối chuẩn bị coi là bất thường. | Rất nhanh, dễ hiểu; nhưng thất bại ngay nếu sản phẩm bị lệch vị trí/xoay nhẹ. | Kỹ thuật kiểm tra quang học tự động (AOI) cổ điển. |
| **Isolation Forest & One-Class SVM** | 🧱/🚀 ML truyền thống | Trích xuất vector đặc trưng (HOG/LBP hoặc vector từ CNN pretrained) rồi cô lập điểm ngoại lai bằng cây ngẫu nhiên (`IsolationForest`) hoặc mặt cầu bao quanh dữ liệu bình thường (`OneClassSVM`). | Huấn luyện cực nhanh trên CPU, không cần GPU; nhưng khó chỉ ra chính xác pixel nào trên ảnh bị lỗi nếu chỉ dùng vector toàn ảnh. | Liu et al., *Isolation Forest*, ICDM 2008; Schölkopf et al., 2001. |
| **Convolutional Autoencoder (CAE) / VAE** | 🧱/🚀 Nền tảng DL | Nén ảnh bình thường qua nút cổ chai (`latent bottleneck`) rồi giải nén lại. Vì chỉ học tái tạo cấu trúc bình thường, khi gặp vết xước lạ mô hình sẽ tái tạo thành vùng lành lặn → hiệu số $|x - \hat{x}|$ sẽ sáng rực tại đúng vết xước. | Vừa phát hiện được ảnh lỗi vừa tạo được bản đồ nhiệt vị trí lỗi; hạn chế là đôi khi nếu nút cổ chai quá rộng, Autoencoder "sao chép" luôn cả vết lỗi làm giảm độ nhạy. | Bergmann et al., *MVTec AD*, CVPR 2019. |
| **PaDiM (2020) & PatchCore (2022)** | 🚀 Đang dùng rộng rãi trong công nghiệp | Dùng mạng CNN pretrained (như WideResNet) trích xuất đặc trưng của từng vùng nhỏ (`patch`) trên ảnh lành, lưu vào bộ nhớ đối sánh (**Memory Bank**). Khi kiểm tra, tính khoảng cách từ từng patch của ảnh mới tới lân cận gần nhất trong kho mẫu lành. | Đạt độ chính xác rất cao trên chuẩn công nghiệp MVTec AD mà không cần huấn luyện lại mạng bằng gradient descent lâu dài. | Defard et al., *PaDiM*, ICPR 2020; Roth et al., *Towards Total Recall in Industrial Anomaly Detection (PatchCore)*, CVPR 2022. |

---

### Nhóm 5: Học Biểu diễn, Metric Learning và Truy vấn Ảnh (Representation Learning & Image Retrieval)

#### 5.1 Tổng quan kỹ thuật
- **Bài toán giải quyết**: Thay vì huấn luyện một tầng phân loại cố định `C` lớp, ta huấn luyện mô hình ánh xạ mỗi bức ảnh $x$ thành một **vector nhúng (`embedding`)** $z = f_\theta(x) \in \mathbb{R}^D$ (thường chuẩn hóa độ dài $\|z\|_2 = 1$) sao cho:
  - Hai ảnh cùng đối tượng / cùng ngữ nghĩa có khoảng cách nhỏ (hoặc `Cosine Similarity` lớn).
  - Hai ảnh khác đối tượng có khoảng cách lớn.
- **Ứng dụng thực tế**: Tìm kiếm ảnh sản phẩm tương tự (Visual Search / Image Retrieval), nhận diện khuôn mặt / định danh lại đối tượng (Re-Identification) nơi có hàng nghìn lớp mới phát sinh mà không thể huấn luyện lại mạng mỗi lần thêm 1 người/sản phẩm mới.
- **Định dạng dữ liệu**:
  - Theo cặp (`Pair`): `(img_1, img_2, is_same ∈ {0, 1})` (`Positive pair` nếu cùng lớp, `Negative pair` nếu khác lớp).
  - Theo bộ ba (`Triplet`): `(Anchor A, Positive P, Negative N)` trong đó `A` và `P` cùng lớp, `N` khác lớp.
- **Loss và Metric thường dùng**:
  - Loss: **Contrastive Loss**, **Triplet Margin Loss** ($\max(0, d(A, P) - d(A, N) + \text{margin})$), **InfoNCE / NT-Xent Loss** (dùng trong tự giám sát và CLIP), **ArcFace** (nhận diện khuôn mặt).
  - Metric: `Recall@K` (`R@1`, `R@5` — xác suất ảnh cùng lớp xuất hiện trong `K` kết quả đầu tiên), `mAP@K`, `Silhouette Score` của không gian nhúng.
- **Bài học liên quan**: `16_metric_learning_and_image_retrieval.ipynb`, `17_vision_transformers_and_self_supervised_learning.ipynb`.

#### 5.2 Các mô hình và hướng tiếp cận đại diện

| Mô hình / Khung huấn luyện | Nhánh & Vai trò | Nguyên lý hoạt động cốt lõi | Khi nên dùng / Lưu ý | Nguồn / Paper gốc |
|---|---|---|---|---|
| **Siamese Network + Contrastive / Triplet Loss** | 🧱/🚀 Nền tảng Metric Learning | Dùng hai (hoặc ba) nhánh mạng chia sẻ cùng bộ trọng số $\theta$ để trích xuất vector nhúng cho từng ảnh trong cặp/bộ ba, sau đó tối ưu khoảng cách trực tiếp trong không gian Euclidean/Cosine. | Rất hiệu quả cho bài toán đối sánh chữ ký, linh kiện, tìm ảnh tương tự. Cần lưu ý kỹ thuật chọn mẫu khó (**Hard Negative Mining**) để mạng không học những bộ ba quá dễ. | Bromley et al., 1993; Schroff et al., *FaceNet: A Unified Embedding for Face Recognition and Clustering*, CVPR 2015. |
| **SimCLR (2020) & MoCo (2020)** | 🧱/🚀 Self-Supervised Learning (Tự giám sát) | Không cần nhãn của con người: từ 1 ảnh gốc tạo ra 2 góc nhìn biến đổi mạnh (`augmentation`) làm `positive pair`, coi các ảnh khác trong batch (SimCLR) hoặc trong hàng đợi động lượng (MoCo) là `negative pairs`, tối ưu bằng hàm mất mát đối조 `InfoNCE`. | Dùng để tiền huấn luyện (pretrain) trên kho ảnh lớn chưa gán nhãn trước khi fine-tune trên tập nhỏ có nhãn. | Chen et al., *SimCLR*, ICML 2020; He et al., *Momentum Contrast (MoCo)*, CVPR 2020. |
| **DINO (2021) & DINOv2 (2023)** | 🚀 Self-Supervised Foundation Model | Tự chưng cất tri thức (Self-distillation with no labels) giữa mạng `Student` và mạng `Teacher` (cập nhật bằng trung bình trượt EMA) trên kiến trúc Vision Transformer. | Đặc trưng trích xuất từ DINOv2 cực kỳ giàu thông tin hình học, dùng làm backbone đa năng cho phân loại, truy vấn ảnh, ước lượng độ sâu mà không cần fine-tune phức tạp. | Caron et al., *DINO*, ICCV 2021; Oquab et al., *DINOv2*, TMLR 2024. |
| **CLIP (Contrastive Language–Image Pre-training, 2021)** | 🚀 Vision–Language Representation | Huấn luyện đồng thời `Image Encoder` và `Text Encoder` trên hàng trăm triệu cặp `(ảnh, câu mô tả)` sao cho tích vô hướng của cặp đúng đạt cực đại trên ma trận tương đồng $N \times N$. | Cho phép **Zero-shot Classification** (phân loại bằng câu lệnh văn bản mà không cần mẫu huấn luyện) và tìm kiếm ảnh bằng câu chữ tự nhiên. | Radford et al., *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, ICML 2021. |

---

### Nhóm 6: Các Mô hình Sinh Ảnh (Generative Models)

#### 6.1 Tổng quan kỹ thuật
- **Bài toán giải quyết**: Học phân phối xác suất $p(x)$ của dữ liệu ảnh để sinh ra các bức ảnh mới hợp lý, chỉnh sửa ảnh, khôi phục vùng ảnh bị khuyết (inpainting) hoặc chuyển đổi phong cách ảnh (image-to-image translation).
- **Đầu vào / Đầu ra**:
  - Sinh không điều kiện (Unconditional): Vector nhiễu ngẫu nhiên $z \sim \mathcal{N}(0, I)$ → Ảnh sinh ra $\hat{x} \in \mathbb{R}^{C \times H \times W}$.
  - Sinh có điều kiện (Conditional): Vector nhiễu $z$ + Điều kiện $c$ (nhãn lớp, câu văn bản, ảnh phác thảo) → Ảnh sinh ra $\hat{x}$.
- **Bài học liên quan**: `18_generative_models.ipynb`.

#### 6.2 So sánh 5 họ mô hình sinh ảnh chính

| Họ mô hình | Vai trò | Nguyên lý toán học & Kiến trúc | Hàm mất mát (Loss) | Ưu điểm & Hạn chế thực tế | Nguồn / Paper gốc |
|---|---|---|---|---|---|
| **Autoencoder (AE)** | 🧱 Nền tảng nén & tái tạo | Nén $x \to z$ rồi giải nén $z \to \hat{x}$ một cách tất định. | `MSE` hoặc `BCE` tái tạo | Tốt cho nén và phát hiện bất thường, nhưng không gian ẩn $z$ bị rời rạc nên lấy mẫu $z$ ngẫu nhiên thường không sinh ra ảnh có nghĩa. | Hinton & Salakhutdinov, *Science*, 2006. |
| **Variational Autoencoder (VAE, 2013)** | 🧱/🚀 Nền tảng xác suất & Latent space | Encoder dự đoán trung bình $\mu(x)$ và phương sai $\sigma^2(x)$ của phân phối ẩn; dùng **Reparameterization Trick** ($z = \mu + \sigma \odot \epsilon$) để truyền đạo hàm qua bước lấy mẫu ngẫu nhiên. | $\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, I))$ | **Ưu**: Huấn luyện rất ổn định, không gian ẩn liên tục dễ nội suy. **Hạn chế**: Ảnh sinh trực tiếp từ VAE cổ điển thường hơi mờ do tính trung bình của hàm loss tái tạo; hiện nay VAE còn đóng vai trò nén ảnh vào không gian ẩn cho **Latent Diffusion**. | Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014. |
| **GAN (Generative Adversarial Networks, 2014)** | 🧱/🚀 Đối kháng | Trò chơi hai người giữa **Generator $G$** (tạo ảnh giả từ nhiễu $z$ để đánh lừa) và **Discriminator $D$** (phân biệt ảnh thật từ tập dữ liệu với ảnh giả của $G$). | Minimax / Non-saturating Adversarial Loss (BCE hoặc Hinge/Wasserstein Loss) | **Ưu**: Sinh ảnh sắc nét chỉ trong **1 lượt forward** duy nhất. **Hạn chế**: Huấn luyện khó ổn định, dễ gặp **Mode Collapse** ($G$ chỉ sinh lặp đi lặp lại một vài mẫu giống nhau) hoặc mất cân bằng giữa $G$ và $D$. | Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014; Radford et al., *DCGAN*, ICLR 2016. |
| **Diffusion Models (DDPM 2020, Latent Diffusion / Stable Diffusion 2022)** | 🚀 Hướng chủ đạo hiện nay | **Quá trình thuận (Forward)**: Cộng nhiễu Gaussian dần dần vào ảnh qua $T$ bước thời gian (`timesteps`) theo lịch nhiễu (`noise schedule`) cho đến khi thành nhiễu trắng. **Quá trình ngược (Reverse)**: Huấn luyện mạng (thường là U-Net hoặc Diffusion Transformer – DiT) dự đoán lượng nhiễu $\epsilon_\theta(x_t, t)$ để khử nhiễu từng bước từ $t=T$ về $t=0$. | $\mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$ | **Ưu**: Chất lượng ảnh rất cao, đa dạng mẫu tốt, huấn luyện ổn định hơn GAN. **Hạn chế**: Quá trình lấy mẫu (inference) cần lặp qua nhiều bước khử nhiễu nên tốn thời gian và tài nguyên GPU hơn. | Ho et al., *Denoising Diffusion Probabilistic Models (DDPM)*, NeurIPS 2020; Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022. |
| **Autoregressive & Flow-based Models (PixelCNN, VQ-VAE + Transformer, Normalizing Flows)** | 🧱/🚀 Định hướng bổ trợ | **Autoregressive**: Rời rạc hóa ảnh thành chuỗi token (qua bộ mã hóa lượng tử `VQ-VAE` / `VQGAN`) rồi dự đoán từng token tiếp theo giống mô hình ngôn ngữ. **Flow-based**: Dùng chuỗi phép biến đổi khả nghịch (invertible) có định thức Jacobian tính được chính xác. | Negative Log-Likelihood (NLL) chính xác | Cho phép tính mật độ xác suất tường minh và kết hợp tự nhiên với kiến trúc Multimodal LLM hiện đại. | van den Oord et al., *PixelCNN*, 2016; *VQ-VAE*, NeurIPS 2017; Dinh et al., *RealNVP*, ICLR 2017. |

---

### Nhóm 7: Các Bài toán Thị giác Mở rộng (Extended Computer Vision Tasks)

Chi tiết thực hành và minh họa cơ chế của toàn bộ nhóm này nằm trong `notebooks/19_ocr_and_visual_tasks_overview.ipynb`:

| Bài toán mở rộng | Mục tiêu & Đầu vào / Đầu ra | Kiến trúc & Phương pháp tiêu biểu | Loss & Metric đặc trưng | Lưu ý thực tế khi triển khai |
|---|---|---|---|---|
| **1. OCR (Optical Character Recognition)** | Chuyển chữ viết trong ảnh thành chuỗi văn bản số hóa. Thường chia làm 2 bước: **Text Detection** (tìm đa giác/khung chứa dòng chữ) → **Text Recognition** (đọc chuỗi ký tự trong khung đã cắt). | - Detection: **EAST**, **CRAFT**, **DBNet** (Differentiable Binarization).<br>- Recognition: **CRNN** (`CNN + BiLSTM + CTC Loss`), hoặc kiến trúc Transformer (**TrOCR**, **SVTR**).<br>- Hệ thống tích hợp: Tesseract, PaddleOCR, EasyOCR. | - Loss: `Dice/BCE` (Detection) + **`nn.CTCLoss`** (Connectionist Temporal Classification – căn chỉnh chuỗi không cần nhãn vị trí từng chữ cái) hoặc `CrossEntropy`.<br>- Metric: **CER** (Character Error Rate), **WER** (Word Error Rate). | Tiền xử lý ảnh (xoay thẳng dòng chữ bị nghiêng – deskew, tăng tương phản) quyết định tới 50% chất lượng OCR thực tế. |
| **2. Pose Estimation (Ước lượng tư thế / Điểm mốc)** | Tìm tọa độ $(x_k, y_k)$ và độ tin cậy $v_k$ của $K$ khớp xương cơ thể người hoặc điểm đặc trưng linh kiện (Keypoints / Landmarks). | - **Top-down**: Tìm khung người trước rồi dự đoán bản đồ nhiệt (**Gaussian Heatmaps**) cho từng khớp (**HRNet**, **ViTPose**, **YOLOv8-Pose**).<br>- **Bottom-up**: Tìm mọi khớp trong ảnh trước rồi nối thành từng người (**OpenPose**). | - Loss: `MSELoss` trên các bản đồ nhiệt 2D Gaussian quanh tọa độ thật.<br>- Metric: **PCK** (Percentage of Correct Keypoints), **OKS** (Object Keypoint Similarity). | Dự đoán bản đồ nhiệt 2D (`heatmap` rồi lấy `argmax`) thường chính xác và dễ học hơn nhiều so với hồi quy trực tiếp tọa độ $(x, y)$ bằng tầng `Linear`. |
| **3. Object Tracking (Theo vết đối tượng qua chuỗi khung hình)** | Duy trì ID nhất quán (`track_id`) cho từng vật thể di chuyển qua các khung hình video liên tiếp. | - **Tracking-by-Detection**: Phát hiện vật thể ở mỗi frame rồi ghép nối qua bộ lọc **Kalman Filter** + thuật toán **Hungarian** dựa trên `IoU` (**SORT**, **ByteTrack**) và kết hợp thêm khoảng cách vector nhúng ngoại hình (**DeepSORT**). | - Metric: **MOTA** (Multi-Object Tracking Accuracy), **IDF1** (độ ổn định giữ đúng ID), **HOTA**. | **ByteTrack** chứng minh rằng việc giữ lại cả các khung phát hiện có điểm tự tin thấp (do bị che khuất tạm thời) ở vòng ghép nối thứ hai giúp giảm đứt gãy quỹ đạo rất hiệu quả. |
| **4. Denoising, Super-Resolution & Image Restoration** | Khôi phục ảnh sạch / độ phân giải cao $\hat{I}_{HR}$ từ ảnh nhiễu, mờ hoặc độ phân giải thấp $I_{LR}$. | **DnCNN** (học phần nhiễu dư `residual noise`), **SRCNN**, **ESRGAN**, **SwinIR**, **Restormer**. | - Loss: `L1` / `Charbonnier Loss` + `Perceptual Loss` (VGG).<br>- Metric: **PSNR** (Peak Signal-to-Noise Ratio, đơn vị dB) và **SSIM** (Structural Similarity Index). | Mô hình siêu phân giải có yếu tố sinh (GAN/Diffusion) có thể tự "vẽ thêm" chi tiết trông đẹp mắt nhưng **không được dùng làm bằng chứng pháp y hay chẩn đoán y khoa** vì có nguy cơ tạo chi tiết giả (hallucination). |
| **5. Monocular Depth Estimation (Ước lượng độ sâu từ 1 ảnh)** | Từ 1 ảnh RGB duy nhất, dự đoán bản đồ khoảng cách tương đối hoặc tuyệt đối $D(y, x)$ từ camera tới từng điểm ảnh. | Kiến trúc Encoder-Decoder đa tỉ lệ (**MiDaS**, **DPT – Dense Prediction Transformer**, **Depth Anything v1/v2**). | - Loss: Scale-invariant log loss.<br>- Metric: `AbsRel` (sai số tương đối tuyệt đối), `RMSE`, $\delta < 1.25$. | Ước lượng độ sâu từ 1 ảnh đơn lẻ có tính mơ hồ về tỉ lệ thực (scale ambiguity); nếu cần đo kích thước milimét chính xác trong công nghiệp, nên ưu tiên Camera Stereo hoặc cảm biến ánh sáng có cấu trúc / LiDAR. |
| **6. Vision–Language Models (VLM)** | Kết hợp hiểu hình ảnh và suy luận ngôn ngữ tự nhiên: Hỏi đáp trên ảnh (VQA), mô tả ảnh (Image Captioning), đọc biểu đồ/tài liệu. | Kết nối một `Vision Encoder` (như ViT từ CLIP/SigLIP) qua một bộ chuyển đổi (`Projector` / `Q-Former`) vào một `Large Language Model` (như họ LLaVA, Qwen-VL, PaliGemma, Gemini). | Cross-Entropy sinh chuỗi token tiếp theo (Autoregressive Next-Token Prediction). | VLM rất mạnh cho hiểu ngữ cảnh tổng quát và giao tiếp tự nhiên, nhưng chi phí suy luận cao và độ trễ lớn hơn nhiều so với một mô hình CNN/YOLO chuyên biệt. |

---

## Phụ lục: Bảng Đối chiếu Khái niệm giữa PyTorch, TensorFlow/Keras và scikit-learn

Khóa học này sử dụng **PyTorch** cho Deep Learning và **scikit-learn** cho Machine Learning truyền thống để đảm bảo sự nhất quán và minh bạch. Nếu bạn đọc tài liệu có dùng TensorFlow/Keras, bảng dưới đây giúp bạn chuyển đổi khái niệm ngay lập tức:

| Khái niệm / Thao tác | **PyTorch** (Chính trong khóa học) | **TensorFlow / Keras** (Đối chiếu) | **scikit-learn** (ML truyền thống) |
|---|---|---|---|
| **Cấu trúc mảng dữ liệu** | `torch.Tensor` | `tf.Tensor` / `np.ndarray` | `np.ndarray` / `pd.DataFrame` |
| **Quy ước thứ tự chiều ảnh mặc định** | **Channels-First (`NCHW`)**: `(batch, C, H, W)` | **Channels-Last (`NHWC`)**: `(batch, H, W, C)` | Ma trận 2D `(n_samples, n_features)` |
| **Lớp cơ sở định nghĩa mô hình** | Kế thừa `torch.nn.Module`, viết hàm `forward(self, x)` | Kế thừa `tf.keras.Model` hoặc dùng `Sequential` / Functional API | Các class Estimator (`SVC`, `RandomForestClassifier`...) |
| **Tầng kết nối đầy đủ (Fully Connected)** | `nn.Linear(in_features, out_features)` | `layers.Dense(units)` | `MLPClassifier` |
| **Tầng tích chập 2D** | `nn.Conv2d(in_channels, out_channels, kernel_size)` | `layers.Conv2D(filters, kernel_size)` | Không áp dụng |
| **Cơ chế tính đạo hàm tự động** | `loss.backward()` trên các tensor có `requires_grad=True` | `with tf.GradientTape() as tape: ... tape.gradient(...)` | Giải tích đóng hoặc tối ưu hóa bên trong `model.fit(X, y)` |
| **Vòng lặp huấn luyện** | Tường minh (`zero_grad()` → `forward` → `loss` → `backward()` → `step()`) | Thường ẩn sau `model.compile(...)` + `model.fit(...)` (hoặc viết `train_step` tùy chỉnh) | Gọi 1 dòng `model.fit(X_train, y_train)` |
| **Chuyển chế độ Huấn luyện / Suy luận** | `model.train()` và `model.eval()` + `with torch.no_grad():` | Tham số `training=True / False` khi gọi `model(x, training=False)` | Dùng hàm `model.predict(X_test)` hoặc `model.predict_proba(X_test)` |
| **Lưu và nạp trọng số** | `torch.save(model.state_dict(), 'ckpt.pt')` / `model.load_state_dict(...)` | `model.save_weights('ckpt.weights.h5')` / `model.save(...)` | `joblib.dump(model, 'model.joblib')` |
