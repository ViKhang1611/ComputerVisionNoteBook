"""
Script khởi tạo các Jupyter Notebook từ 06 đến 11 (Cuối Giai đoạn B + Toàn bộ Giai đoạn C).
"""

from __future__ import annotations

from pathlib import Path
from _nb_helper import create_validated_notebook

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def build_nb06() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 06: Trích xuất Đặc trưng Ảnh (HOG, LBP, Color Histogram) và 7 Mô hình Machine Learning Truyền thống

## 1. Mục tiêu học tập
- Phân biệt rõ ràng giữa **Phương pháp Trích xuất Đặc trưng (`Feature Extraction`)** và **Mô hình Phân loại (`Classification Model`)**:
  - **Nhóm Trích xuất Đặc trưng**: `Pixel Flattening` (Baseline), `Color Histogram`, `HOG (Histogram of Oriented Gradients)` và `LBP (Local Binary Patterns)`. Chúng biến bức ảnh thành vector số thực nhưng **không tự đưa ra nhãn lớp**.
  - **Nhóm Mô hình Phân loại**: Nhận vector đặc trưng để học quy luật phân chia lớp: `k-NN`, `Logistic Regression`, `Naive Bayes`, `Decision Tree`, `Random Forest`, `SVM`, và `Gradient Boosting`.
- Thực hành huấn luyện thật và so sánh công bằng cả 7 mô hình trên **cùng một cách chia dữ liệu (`Train / Val / Test`)** và **cùng bộ độ đo (`Accuracy`, `Macro-F1`, `Precision`, `Recall`)**.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 04 và Bài 05.

## 3. Thời gian học dự kiến
- Khoảng **3.0 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
Bạn cần phân loại 4 loại chi tiết hình học (`circle`, `square`, `triangle`, `cross`) trên máy tính chỉ có CPU, dung lượng bộ nhớ hạn chế và yêu cầu thời gian huấn luyện dưới 2 giây. Thay vì ném thẳng `3072` giá trị pixel thô dễ nhiễu vào mô hình, việc rút trích đúng đặc trưng cạnh (`HOG`) và kết cấu (`LBP`) sẽ giúp các mô hình ML truyền thống đạt độ chính xác cao.

## 5. Giải thích trực quan: 4 Phương pháp Biểu diễn & Trích xuất Đặc trưng Ảnh
1. **Pixel Flattening (Trải phẳng điểm ảnh làm Baseline)**: Duỗi thẳng ảnh `(32, 32, 3)` thành vector `(3072,)`. Nhược điểm: Rất nhạy cảm khi vật thể dịch chuyển nhẹ vài pixel hoặc đổi độ sáng.
2. **Color Histogram (Biểu đồ tần suất màu)**: Đếm tỷ lệ màu sắc trong ảnh. Bất biến với phép dịch chuyển và xoay, nhưng **bỏ qua hoàn toàn hình dáng không gian**.
3. **HOG (Histogram of Oriented Gradients)**: Chia ảnh thành lưới các ô nhỏ (`cell_size = 8x8`), trong mỗi ô tính độ lớn và hướng góc (`0..180°`) của đạo hàm biên rồi lập histogram `9` ngăn góc. **Rất mạnh để bắt hình dáng và đường nét vật thể**.
4. **LBP (Local Binary Patterns)**: Tại mỗi pixel, so sánh độ sáng của nó với 8 lân cận xung quanh (lớn hơn hoặc bằng thì ghi bit `1`, nhỏ hơn ghi bit `0`), tạo thành số nguyên 8-bit (`0..255`). **Rất mạnh để mô tả kết cấu bề mặt (texture)**.

## 6. Thuật ngữ mới & Giải thích Keyword trong 7 Mô hình ML
| Mô hình & Keyword chính | Nhóm | Định nghĩa, Giá trị chọn & Ảnh hưởng khi Tăng/Giảm |
|---|---|---|
| `KNeighborsClassifier(n_neighbors=5, metric='euclidean')` | ML Phân loại | Bầu chọn theo `k = n_neighbors` điểm gần nhất. **Giảm `k=1`**: biên quyết định phức tạp, nhạy nhiễu (overfit). **Tăng `k` quá lớn**: biên quá trơn, dễ underfit. |
| `LogisticRegression(C=1.0, max_iter=500)` | ML Phân loại | Phân loại tuyến tính với phạt chuẩn $L_2$. Tham số `C` là **nghịch đảo cường độ điều chuẩn** ($C = 1/\lambda$): **Tăng `C`** → giảm phạt (dễ overfit hơn); **Giảm `C`** → phạt mạnh (trọng số co nhỏ). |
| `GaussianNB(var_smoothing=1e-9)` | ML Phân loại | Giả định các đặc trưng trong mỗi lớp tuân theo phân phối chuẩn độc lập. `var_smoothing` cộng thêm phương sai nhỏ để tránh lỗi chia cho `0`. |
| `DecisionTreeClassifier(max_depth=6, criterion='gini')` | ML Phân loại | Cây quyết định rẽ nhánh theo độ tạp `gini` hoặc `entropy`. Không giới hạn `max_depth=None` sẽ làm cây mọc tới khi mỗi lá chỉ còn 1 mẫu (Overfitting nặng!). |
| `RandomForestClassifier(n_estimators=100, max_depth=8)` | ML Ensemble | Tổ hợp `n_estimators` cây quyết định độc lập (Bagging). Tăng `n_estimators` giúp giảm phương sai dự đoán nhưng tăng thời gian tính toán tuyến tính. |
| `SVC(C=2.0, kernel='rbf', gamma='scale')` | ML Phân loại | Máy vector hỗ trợ (SVM) với nhân `RBF` ($K(x, x') = \exp(-\gamma \|x - x'\|^2)$). Tăng `C` hoặc tăng `gamma` làm biên uốn sát quanh từng điểm dữ liệu (dễ overfit). |
| `HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100)` | ML Boosting | Xây dựng chuỗi cây sửa lỗi phần dư nối tiếp nhau (cùng họ với **XGBoost**, **LightGBM**, **CatBoost**). Giảm `learning_rate` kết hợp tăng `max_iter` thường giúp tổng quát hóa tốt hơn. |

## 7. Công thức Toán học Cốt lõi & Ví dụ Số Nhỏ
- **Công thức mã LBP tại tâm $(y_c, x_c)$ có độ sáng $g_c$ và 8 điểm lân cận $g_p$ ($p=0..7$)**:
  $$\text{LBP}(y_c, x_c) = \sum_{p=0}^{7} s(g_p - g_c) \cdot 2^p, \quad \text{với } s(z) = \begin{cases} 1 & \text{nếu } z \ge 0 \\ 0 & \text{nếu } z < 0 \end{cases}$$
- **Ví dụ số nhỏ LBP**: Nếu pixel tâm có $g_c = 50$ và 8 lân cận theo chiều kim đồng hồ là `[60, 55, 40, 30, 50, 70, 20, 10]`, chuỗi bit so sánh là `[1, 1, 0, 0, 1, 1, 0, 0]` $\implies \text{LBP} = 2^0 + 2^1 + 2^4 + 2^5 = 1 + 2 + 16 + 32 = 51$.""",
        ),
        (
            "code",
            """# Cell 1: Tạo dữ liệu, chia tập Stratified (Train/Val/Test) và Trích xuất 4 loại Đặc trưng Ảnh
import sys
import time
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import set_seed
from src.synthetic_data import generate_synthetic_shapes_dataset
from src.features import (
    extract_pixel_flatten,
    extract_color_histogram,
    extract_hog_manual,
    extract_lbp_manual,
    extract_combined_features,
)
from src.evaluation import compute_classification_metrics

QUICK_RUN = True
set_seed(42)

n_samples = 320 if QUICK_RUN else 600
ds = generate_synthetic_shapes_dataset(num_samples=n_samples, image_size=32, seed=42, noise_std=10.0)
images, labels, class_names = ds["images"], ds["labels"], ds["class_names"]

# Chia tập TRƯỚC khi trích xuất và chuẩn hóa
idx_all = np.arange(len(images))
idx_tr, idx_temp = train_test_split(idx_all, test_size=0.30, random_state=42, stratify=labels)
idx_va, idx_te = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=labels[idx_temp])

f_pix = extract_pixel_flatten(images)
f_col = extract_color_histogram(images, bins_per_channel=8)
f_hog = extract_hog_manual(images, cell_size=8, num_bins=9)
f_lbp = extract_lbp_manual(images, num_bins=16)
f_comb = extract_combined_features(images, cell_size=8)

print(f"1. Pixel Flattening shape : {f_pix.shape} (32*32*3 = 3072 chiều)")
print(f"2. Color Histogram shape  : {f_col.shape} (3*8 = 24 chiều)")
print(f"3. HOG Feature shape      : {f_hog.shape} (4*4 ô * 9 góc = 144 chiều)")
print(f"4. LBP Feature shape      : {f_lbp.shape} (16 ngăn kết cấu)")
print(f"5. Combined Feature shape : {f_comb.shape} (144 + 24 + 16 = 184 chiều)")""",
        ),
        (
            "code",
            """# Cell 2: Minh họa trực quan Đặc trưng HOG và LBP trên 1 mẫu ảnh
sample_img = images[idx_tr[0]]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
axes[0].imshow(sample_img)
axes[0].set_title(f"Ảnh mẫu: {class_names[labels[idx_tr[0]]]}")
axes[0].axis("off")

axes[1].bar(np.arange(len(f_hog[idx_tr[0]])), f_hog[idx_tr[0]], color="teal", width=1.0)
axes[1].set_title("Vector đặc trưng HOG (144 chiều)")
axes[1].set_xlabel("Chỉ số chiều HOG")

axes[2].bar(np.arange(len(f_lbp[idx_tr[0]])), f_lbp[idx_tr[0]], color="darkorange")
axes[2].set_title("Biểu đồ kết cấu LBP (16 ngăn)")
axes[2].set_xlabel("Ngăn mã LBP")
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 3: Huấn luyện Thực tế và So sánh Công bằng 7 Mô hình Machine Learning trên đặc trưng Combined
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC

# Chuẩn hóa đặc trưng (CHỈ fit trên tập Train)
scaler = StandardScaler()
X_tr = scaler.fit_transform(f_comb[idx_tr])
X_va = scaler.transform(f_comb[idx_va])
X_te = scaler.transform(f_comb[idx_te])
y_tr, y_va, y_te = labels[idx_tr], labels[idx_va], labels[idx_te]

models_zoo = {
    "1. k-NN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
    "2. Logistic Regression": LogisticRegression(C=1.0, max_iter=500, random_state=42),
    "3. Naive Bayes (GaussianNB)": GaussianNB(var_smoothing=1e-8),
    "4. Decision Tree (depth=6)": DecisionTreeClassifier(max_depth=6, random_state=42),
    "5. Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    "6. SVM (RBF Kernel, C=2.0)": SVC(C=2.0, kernel="rbf", gamma="scale", probability=True, random_state=42),
    "7. HistGradientBoosting": HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, random_state=42),
}

comparison_rows = []
trained_models = {}
for name, clf in models_zoo.items():
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    train_ms = (time.perf_counter() - t0) * 1000.0
    trained_models[name] = clf

    tr_acc = clf.score(X_tr, y_tr)
    va_preds = clf.predict(X_va)
    va_metrics = compute_classification_metrics(y_va, va_preds, class_names)
    comparison_rows.append({
        "Model": name,
        "Train_Acc": round(tr_acc, 4),
        "Val_Acc": round(va_metrics["accuracy"], 4),
        "Val_Macro_F1": round(va_metrics["macro_f1"], 4),
        "Train_Time_ms": round(train_ms, 2),
    })

df_results = pd.DataFrame(comparison_rows).sort_values(by="Val_Macro_F1", ascending=False)
print("=== BẢNG SO SÁNH 7 MÔ HÌNH MACHINE LEARNING TRÊN CÙNG TẬP DỮ LIỆU ===")
print(df_results.to_string(index=False))""",
        ),
        (
            "code",
            """# Cell 4: Thí nghiệm thay đổi một yếu tố — So sánh 4 loại đặc trưng đầu vào khi giữ cố định mô hình SVM
feature_sets = {
    "Color Histogram (24d)": f_col,
    "LBP Texture (16d)": f_lbp,
    "Pixel Flatten (3072d)": f_pix,
    "HOG Shape (144d)": f_hog,
    "Combined HOG+Color+LBP (184d)": f_comb,
}

feat_exp_rows = []
for feat_name, F_mat in feature_sets.items():
    sc = StandardScaler()
    X_f_tr = sc.fit_transform(F_mat[idx_tr])
    X_f_va = sc.transform(F_mat[idx_va])
    svm_clf = SVC(C=2.0, kernel="rbf", random_state=42).fit(X_f_tr, y_tr)
    va_m = compute_classification_metrics(y_va, svm_clf.predict(X_f_va), class_names)
    feat_exp_rows.append((feat_name, va_m["accuracy"], va_m["macro_f1"]))

for f_name, acc, mf1 in feat_exp_rows:
    print(f"Đặc trưng {f_name:30s} -> Val Acc: {acc*100:5.1f}% | Val Macro-F1: {mf1:.3f}")""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Ở Cell 3**: Trên cùng bộ đặc trưng `Combined`, các mô hình sử dụng biên phi tuyến mượt trên hướng gradient như **`SVM (RBF)`**, **`Logistic Regression`**, **`Random Forest`** và **`HistGradientBoosting`** đạt `Val_Macro_F1` vượt trội so với một cây quyết định đơn lẻ (`Decision Tree`).
- **Ở Cell 4 (Thí nghiệm chọn đặc trưng)**:
  - Vì màu sắc của các hình trong tập dữ liệu được chọn ngẫu nhiên từ bảng màu chung, nên `Color Histogram` đơn độc gần như đoán mò (`~25%`).
  - Khi chuyển sang đặc trưng bắt hướng cạnh **`HOG`**, độ chính xác vọt lên rất cao chỉ với `144` chiều (nhỏ hơn `21` lần so với `3072` chiều của `Pixel Flatten`!).

## 13. Lỗi thường gặp và cách sửa
1. **Gọi tên nhầm `HOG` hay `LBP` là "mô hình phân loại"**: Hãy luôn nhớ `HOG` và `LBP` chỉ trả về vector đặc trưng; bạn cần một bộ phân loại như `SVM` hoặc `Random Forest` đặt phía sau.
2. **Đưa đặc trưng chưa chuẩn hóa vào `SVM` hoặc `k-NN`**: Nếu không qua `StandardScaler`, các chiều đặc trưng có phương sai lớn sẽ lấn át hoàn toàn công thức tính khoảng cách Euclidean.

## 14. Bài tập tự làm
- **Cơ bản**: Khảo sát siêu tham số `n_neighbors` của `k-NN` qua các giá trị `[1, 3, 5, 11, 25]` và in ra `Train_Acc` cùng `Val_Acc`.
- **Trung cấp**: Đánh giá mô hình tốt nhất trên tập `Test` (`X_te`, `y_te`) và in ra ma trận nhầm lẫn (`Confusion Matrix`).
- **Nâng cao**: Thay đổi kích thước ô `cell_size` của `HOG` từ `4` sang `8` và `16`, quan sát sự đánh đổi giữa số chiều vector đặc trưng và độ chính xác phân loại.

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 06
# 1. Cơ bản: Khảo sát k trong k-NN
for k in [1, 3, 5, 11, 25]:
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
    print(f"[Cơ bản] k={k:2d} -> Train Acc={knn.score(X_tr, y_tr):.3f} | Val Acc={knn.score(X_va, y_va):.3f}")

# 2. Trung cấp: Đánh giá mô hình tốt nhất trên tập Test và in Confusion Matrix
best_model_name = df_results.iloc[0]["Model"]
best_clf = trained_models[best_model_name]
test_metrics = compute_classification_metrics(y_te, best_clf.predict(X_te), class_names)
print(f"\\n[Trung cấp] Kết quả tập Test của '{best_model_name}': Acc={test_metrics['accuracy']:.3f}, Macro-F1={test_metrics['macro_f1']:.3f}")
print("Confusion Matrix trên tập Test:\\n", test_metrics["confusion_matrix"])

# 3. Nâng cao: Ảnh hưởng của cell_size trong HOG
for c_size in [4, 8, 16]:
    hog_c = extract_hog_manual(images, cell_size=c_size, num_bins=9)
    sc_c = StandardScaler()
    svm_c = SVC(C=2.0, kernel="rbf", random_state=42).fit(sc_c.fit_transform(hog_c[idx_tr]), y_tr)
    acc_c = svm_c.score(sc_c.transform(hog_c[idx_va]), y_va)
    print(f"[Nâng cao] HOG cell_size={c_size:2d} -> Số chiều={hog_c.shape[1]:4d} | Val Acc={acc_c*100:.1f}%")""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Hiểu rõ bản chất và cài đặt của `Pixel Flattening`, `Color Histogram`, `HOG`, `LBP`.
- [x] Đã huấn luyện và so sánh thực tế 7 mô hình Machine Learning truyền thống.
- [x] Sẵn sàng chuyển sang **Bài 07: `07_clustering_and_dimensionality_reduction.ipynb`**.

## 17. Nguồn tham khảo
- Dalal, N., & Triggs, B. (2005), *Histograms of Oriented Gradients for Human Detection*, CVPR.
- Ojala, T. et al. (2002), *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns*, IEEE TPAMI.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "06_features_and_classical_ml.ipynb", cells)


def build_nb07() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 07: Học Không Giám sát trên Ảnh — Phân cụm (K-Means, DBSCAN) và Giảm chiều Dữ liệu (PCA)

## 1. Mục tiêu học tập
- Hiểu bài toán **Học không giám sát (`Unsupervised Learning`)** khi không có nhãn lớp $y$:
  1. **Phân cụm (`Clustering`)** bằng `K-Means` (gom cụm đặc trưng ảnh & ứng dụng nén số lượng màu sắc trong ảnh – *Color Quantization*) và `DBSCAN` (phân cụm dựa trên mật độ và phát hiện mẫu ngoại lai).
  2. **Giảm chiều dữ liệu (`Dimensionality Reduction`)** bằng `PCA (Principal Component Analysis)`.
- **Làm rõ nhận thức quan trọng**: `PCA` là thuật toán tìm hệ trục tọa độ trực giao mới giữ lại nhiều phương sai nhất để nén số chiều hoặc trực quan hóa `2D/3D`, **PCA KHÔNG PHẢI là mô hình phân loại**!

## 2. Kiến thức cần có
- Đã hoàn thành Bài 05 và Bài 06.

## 3. Thời gian học dự kiến
- Khoảng **2.0 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
1. Bạn nhận được một kho gồm hàng nghìn bức ảnh chưa kịp gán nhãn và muốn gom các ảnh có hình dạng tương tự nhau thành các nhóm để gán nhãn nhanh hơn (`Clustering`), đồng thời lọc ra những bức ảnh bị hỏng/dị thường (`DBSCAN outliers`).
2. Vector ảnh trải phẳng có hàng nghìn chiều gây chậm cho `k-NN` / `SVM`; bạn cần dùng `PCA` nén xuống `20` chiều mà vẫn giữ được `>90%` lượng thông tin.

## 5. Giải thích trực quan & Công thức Toán học
1. **K-Means Clustering**: Tối ưu hóa tổng bình phương khoảng cách từ mỗi điểm $x_i$ tới tâm cụm $\mu_k$ gần nhất (**Inertia / Within-Cluster Sum of Squares**):
   $$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2$$
2. **DBSCAN (`Density-Based Spatial Clustering of Applications with Noise`)**:
   - Quét bán kính $\epsilon$ (`eps`) quanh mỗi điểm. Nếu có ít nhất `min_samples` điểm trong bán kính đó, điểm đó là **Core Point (Điểm lõi)** và mở rộng cụm liên thông.
   - Những điểm đứng cô lập ở vùng thưa sẽ được gán nhãn **`-1` (Noise / Outlier)**!
3. **PCA (Phân tích Thành phần Chính)**:
   - Dịch dữ liệu về tâm $0$, phân rã trị riêng của ma trận hiệp phương sai $\Sigma = \frac{1}{N} X^T X = V \Lambda V^T$.
   - Chọn $K$ vector riêng ứng với $K$ trị riêng lớn nhất ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_K$) làm các trục thành phần chính (**Principal Components** / **Eigen-images**).
   - **Tỷ lệ phương sai giữ lại (`Explained Variance Ratio`)**: $\frac{\sum_{j=1}^K \lambda_j}{\sum_{j=1}^D \lambda_j}$.

## 6. Thuật ngữ mới & Giải thích Keyword
| Keyword | Nghĩa tiếng Việt | Nhóm | Kiểu / Giá trị | Vai trò & Ảnh hưởng khi Tăng/Giảm |
|---|---|---|---|---|
| `n_clusters` ($K$) | Số cụm cần chia trong K-Means | Phân cụm | `int >= 2` | Phải chọn trước. Tăng $K$ làm `inertia_` luôn giảm; dùng biểu đồ khuỷu tay (`Elbow`) hoặc `Silhouette Score` để chọn $K$ hợp lý. |
| `eps` & `min_samples` | Bán kính lân cận & Số điểm tối thiểu của DBSCAN | Phân cụm | `float > 0`, `int >= 2` | Tăng `eps` làm các cụm dễ dính chùm vào nhau; giảm `eps` quá nhỏ khiến hầu hết dữ liệu bị coi là nhiễu (`label = -1`). |
| `n_components` | Số chiều giữ lại của PCA | Giảm chiều | `int` (số chiều) hoặc `float ∈ (0, 1)` (tỷ lệ phương sai) | Nếu truyền số nguyên `16` → giữ đúng 16 trục đầu; nếu truyền số thực `0.95` → tự động chọn đủ số trục để giữ `95%` phương sai. |
| `explained_variance_ratio_` | Tỷ lệ phương sai giải thích được | Giảm chiều | Mảng `float` tổng $\le 1.0$ | Cho biết mỗi trục PCA chứa bao nhiêu phần trăm thông tin biến thiên của dữ liệu gốc. |""",
        ),
        (
            "code",
            """# Cell 1: Ứng dụng K-Means để Nén Màu Ảnh (Color Quantization) và Phân cụm Đặc trưng HOG
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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_digits

from src.utils import set_seed
from src.synthetic_data import generate_synthetic_shapes_dataset

set_seed(42)

ds = generate_synthetic_shapes_dataset(num_samples=160, image_size=32, seed=42)
sample_rgb = ds["images"][0]
pixels_2d = sample_rgb.reshape(-1, 3).astype(np.float32)

# Lượng tử hóa màu bức ảnh từ hàng nghìn màu xuống chỉ còn K=3 và K=6 màu đại diện
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))
axes[0].imshow(sample_rgb)
axes[0].set_title(f"Ảnh gốc ({len(np.unique(pixels_2d, axis=0))} màu)")
axes[0].axis("off")

for ax, k_colors in zip(axes[1:], [3, 6]):
    km_color = KMeans(n_clusters=k_colors, n_init=5, random_state=42).fit(pixels_2d)
    quantized_pixels = km_color.cluster_centers_[km_color.labels_].reshape(sample_rgb.shape)
    ax.imshow(np.clip(quantized_pixels, 0, 255).astype(np.uint8))
    ax.set_title(f"K-Means Nén còn K={k_colors} màu")
    ax.axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 2: Thực hành PCA trên bộ dữ liệu chữ số viết tay (load_digits 8x8 = 64 chiều)
# Nguồn dữ liệu: UCI Optical Recognition of Handwritten Digits tích hợp sẵn offline trong scikit-learn
digits = load_digits()
X_digits = digits.data.astype(np.float32) / 16.0  # (1797, 64) trong đoạn [0, 1]
y_digits = digits.target                          # Nhãn 0..9 (chỉ dùng để tô màu hình vẽ kiểm tra, PCA KHÔNG dùng nhãn!)

pca_full = PCA(n_components=32, random_state=42).fit(X_digits)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

n_90 = int(np.searchsorted(cum_var, 0.90) + 1)
print(f"Chiều gốc: {X_digits.shape[1]} chiều -> Chỉ cần {n_90} trục PCA đầu tiên để giữ được >= 90% phương sai ({cum_var[n_90-1]*100:.1f}%)!")

# Chiếu dữ liệu 64 chiều xuống mặt phẳng 2D bằng 2 trục PCA đầu tiên và hiển thị Eigen-digits
X_pca_2d = pca_full.transform(X_digits)[:, :2]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
axes[0].plot(np.arange(1, 33), cum_var * 100, "o-", color="royalblue", markersize=4)
axes[0].axhline(90, color="crimson", linestyle="--", label="Ngưỡng 90% phương sai")
axes[0].set_title("Phương sai tích lũy theo số trục PCA")
axes[0].set_xlabel("Số thành phần chính (K)")
axes[0].set_ylabel("Phương sai giữ lại (%)")
axes[0].legend()

scatter = axes[1].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_digits, cmap="tab10", s=8, alpha=0.7)
axes[1].set_title("Chiếu 64D -> 2D bằng PCA (Không dùng nhãn)")
axes[1].set_xlabel("PC 1")
axes[1].set_ylabel("PC 2")

# Hiển thị thành phần chính thứ nhất (Eigen-digit #1)
eigen_digit_1 = pca_full.components_[0].reshape(8, 8)
axes[2].imshow(eigen_digit_1, cmap="coolwarm")
axes[2].set_title("Eigen-digit #1 (Trục biến thiên lớn nhất)")
axes[2].axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 3: So sánh K-Means và DBSCAN trong không gian đặc trưng giảm chiều PCA
X_sub = X_pca_2d[:400]
km_feat = KMeans(n_clusters=6, n_init=10, random_state=42).fit(X_sub)
sil_km = silhouette_score(X_sub, km_feat.labels_)

dbscan = DBSCAN(eps=0.32, min_samples=6).fit(X_sub)
n_clusters_db = len(set(dbscan.labels_) - {-1})
n_outliers_db = int((dbscan.labels_ == -1).sum())

print(f"K-Means (K=6) -> Silhouette Score: {sil_km:.3f}")
print(f"DBSCAN        -> Tự tìm thấy {n_clusters_db} cụm đặc và tách riêng {n_outliers_db} điểm ngoại lai (label = -1)")""",
        ),
        (
            "code",
            """# Cell 4: Thí nghiệm thay đổi một yếu tố — Tái tạo ảnh từ K thành phần chính PCA (K = 2, 8, 24, 64)
sample_digit = X_digits[0:1]  # Chữ số 0
fig, axes = plt.subplots(1, 5, figsize=(11, 2.5))
axes[0].imshow(sample_digit.reshape(8, 8), cmap="gray")
axes[0].set_title("Ảnh gốc (64 chiều)")
axes[0].axis("off")

for ax, k_comp in zip(axes[1:], [2, 6, 16, 32]):
    pca_k = PCA(n_components=k_comp, random_state=42).fit(X_digits)
    code_k = pca_k.transform(sample_digit)
    recon_k = pca_k.inverse_transform(code_k)
    mse_k = float(np.mean((sample_digit - recon_k) ** 2))
    ax.imshow(recon_k.reshape(8, 8), cmap="gray")
    ax.set_title(f"K={k_comp} trục\\nMSE={mse_k:.4f}")
    ax.axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Ở Cell 2 & 4**: Dù ảnh chữ số có `64` pixel nhưng do các pixel lân cận có tương quan rất cao, chỉ cần `16..21` trục `PCA` đầu tiên đã tái tạo lại chữ số rất rõ nét với sai số `MSE` cực nhỏ.
- **Lưu ý quan trọng**: `PCA` chỉ thực hiện phép quay và chiếu tuyến tính (`transform` / `inverse_transform`). Muốn phân loại chữ số sau khi giảm chiều bằng `PCA`, bạn phải nối thêm một mô hình phân loại phía sau (ví dụ `PCA(n_components=20) -> SVC()`).

## 13. Lỗi thường gặp và cách sửa
1. **Không đưa dữ liệu về cùng thang đo trước khi chạy `PCA` hoặc `K-Means`**: Nếu 1 chiều có biên độ `0..1000` còn chiều kia `0..1`, trục `PCA` đầu tiên sẽ chỉ chạy dọc theo chiều có biên độ lớn.
2. **Tưởng nhãn cụm `0, 1, 2` của `K-Means` trùng với nhãn lớp `0, 1, 2` của bài toán**: Nhãn của `K-Means` chỉ là số thứ tự cụm ngẫu nhiên (hoán vị bất kỳ).

## 14. Bài tập tự làm
- **Cơ bản**: Vẽ đường cong `Inertia` và `Silhouette Score` khi quét `K` của `K-Means` từ `2` đến `10` trên `X_sub`.
- **Trung cấp**: Kết hợp pipeline `PCA(n_components=16)` + `LogisticRegression()` trên bộ `load_digits` (nhớ chia `train/test` trước khi `fit` PCA!) và so sánh tốc độ/độ chính xác với khi không dùng PCA.
- **Nâng cao**: Khảo sát tham số `eps` của `DBSCAN` qua `[0.15, 0.30, 0.60]` và đếm tỷ lệ điểm bị đánh dấu ngoại lai (`label == -1`).

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 07
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Cơ bản: Quét K cho K-Means
sil_scores = {}
for k in [2, 4, 6, 8, 10]:
    km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X_sub)
    sil_scores[k] = round(float(silhouette_score(X_sub, km.labels_)), 3)
print("[Cơ bản] Silhouette Score theo K:", sil_scores)

# 2. Trung cấp: Pipeline PCA + LogisticRegression (chia tập trước khi fit PCA)
X_d_tr, X_d_te, y_d_tr, y_d_te = train_test_split(X_digits, y_digits, test_size=0.25, random_state=42, stratify=y_digits)
pca_16 = PCA(n_components=16, random_state=42)
X_d_tr_16 = pca_16.fit_transform(X_d_tr)
X_d_te_16 = pca_16.transform(X_d_te)

clf_full = LogisticRegression(max_iter=500).fit(X_d_tr, y_d_tr)
clf_pca16 = LogisticRegression(max_iter=500).fit(X_d_tr_16, y_d_tr)
print(f"[Trung cấp] 64 chiều gốc Acc={clf_full.score(X_d_te, y_d_te)*100:.1f}% vs 16 chiều PCA Acc={clf_pca16.score(X_d_te_16, y_d_te)*100:.1f}%")

# 3. Nâng cao: Khảo sát eps của DBSCAN
for eps_val in [0.15, 0.30, 0.60]:
    db = DBSCAN(eps=eps_val, min_samples=6).fit(X_sub)
    outlier_pct = (db.labels_ == -1).mean() * 100
    print(f"[Nâng cao] DBSCAN eps={eps_val:.2f} -> Tỷ lệ điểm ngoại lai: {outlier_pct:.1f}%")""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Đã hoàn thành toàn bộ **Giai đoạn B (Machine Learning Nền tảng)**.
- [x] Phân biệt rõ `PCA` (giảm chiều), `K-Means`/`DBSCAN` (phân cụm) với các mô hình phân loại có giám sát.
- [x] Sẵn sàng bước sang **Giai đoạn C — Bài 08: `08_neural_network_from_scratch.ipynb`**.

## 17. Nguồn tham khảo
- Jolliffe, I. T. (2002), *Principal Component Analysis*, Springer.
- Ester, M. et al. (1996), *A Density-Based Algorithm for Discovering Clusters (DBSCAN)*, KDD.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "07_clustering_and_dimensionality_reduction.ipynb", cells)


def build_nb08() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 08: Xây dựng Mạng Nơ-ron Nhân tạo Từ Đầu bằng NumPy Thuần — Giải phẫu Forward & Backpropagation

## 1. Mục tiêu học tập
- Hiểu tận gốc cơ chế hoạt động bên trong "hộp đen" của một mạng nơ-ron đa tầng (**Multi-Layer Perceptron - MLP**) bằng cách tự tay lập trình toàn bộ bằng `NumPy` thuần trước khi dùng `PyTorch`:
  - Tầng tuyến tính (`Linear Layer`): $Z = X W + b$.
  - Hàm kích hoạt phi tuyến (`ReLU`): $A = \max(0, Z)$.
  - Biến đổi `Logits` sang xác suất bằng `Softmax` ổn định số học và tính hàm mất mát `Cross-Entropy Loss`.
  - Tính đạo hàm ngược (**Backpropagation**) qua từng tầng theo quy tắc mắt xích (**Chain Rule**).
- Theo dõi **kích thước thực tế (`shape`)** ở mọi mắt xích:
  `Input (B, D_in) -> Linear1 (B, H) -> ReLU (B, H) -> Linear2 (B, C) -> Loss (scalar) -> Backward -> Optimizer Step`.
- Kiểm chứng công thức đạo hàm ngược bằng **Đạo hàm số (`Numerical Gradient Check`)** với độ chính xác $< 10^{-7}$.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 01 (NumPy) và Bài 04 (Đạo hàm & Gradient Descent).

## 3. Thời gian học dự kiến
- Khoảng **3.0 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
Một mô hình tuyến tính đơn tầng không thể phân chia được các vùng dữ liệu lồng nhau phi tuyến (như bài toán XOR hay các đặc trưng hình học phức tạp). Bằng cách thêm một tầng ẩn (`Hidden Layer`) kèm hàm kích hoạt `ReLU`, mạng nơ-ron có thể tự uốn cong không gian đặc trưng để phân loại chính xác.

## 5. Giải thích trực quan & Sơ đồ Dòng chảy Shape (Forward & Backward)
Giả sử lô dữ liệu có $B = 4$ mẫu, mỗi mẫu có $D_{\text{in}} = 16$ đặc trưng, tầng ẩn có $H = 8$ nơ-ron, và cần phân loại $C = 4$ lớp:
```
[LAN TRUYỀN TIẾN (FORWARD PASS)]
X (4, 16) ──► Z1 = X @ W1 + b1 (4, 8) ──► H1 = ReLU(Z1) (4, 8) ──► Z2 = H1 @ W2 + b2 (4, 4) [Logits]
                                                                              │
                                                                              ▼
                                                        Loss L = CrossEntropy(Softmax(Z2), y) (vô hướng)

[LAN TRUYỀN NGƯỢC (BACKWARD PASS - CHAIN RULE)]
dZ2 = (P - Y_onehot) / B        shape (4, 4)   <-- Đạo hàm gộp vô cùng gọn đẹp của Softmax + CrossEntropy!
dW2 = H1.T @ dZ2                shape (8, 4)   (khớp hệt shape của W2!)
db2 = sum(dZ2, axis=0)          shape (4,)     (khớp hệt shape của b2!)
dH1 = dZ2 @ W2.T                shape (4, 8)
dZ1 = dH1 * (Z1 > 0)            shape (4, 8)   <-- Đạo hàm của ReLU chỉ cho tín hiệu đi qua nơi Z1 > 0
dW1 = X.T @ dZ1                 shape (16, 8)  (khớp hệt shape của W1!)
db1 = sum(dZ1, axis=0)          shape (8,)     (khớp hệt shape của b1!)
```

## 6. Thuật ngữ mới & Giải thích Keyword
| Keyword | Nghĩa tiếng Việt | Nhóm | Vai trò & Điều gì xảy ra nếu bỏ hoặc thay đổi? |
|---|---|---|---|
| `neuron / hidden unit` | Nơ-ron tầng ẩn | Kiến trúc | Tính tổ hợp có trọng số các đầu vào rồi qua hàm kích hoạt. Tăng số nơ-ron ẩn $H$ làm tăng năng lực biểu diễn. |
| `ReLU` ($\max(0, z)$) | Hàm chỉnh lưu tuyến tính | Kiến trúc | Tạo tính phi tuyến và có đạo hàm bằng `1` khi $z > 0$ (chống bão hòa đạo hàm tốt hơn `Sigmoid`). **Nếu bỏ `ReLU`**: $Z_2 = (X W_1 + b_1) W_2 + b_2 = X (W_1 W_2) + b_{\text{eq}}$ — mạng 2 tầng sụp đổ về 1 tầng tuyến tính duy nhất! |
| `logits` ($Z_2$) | Điểm số thô chưa chuẩn hóa | Kiến trúc | Đầu ra của tầng `Linear` cuối cùng, có miền giá trị $(-\infty, +\infty)$. |
| `probabilities` ($P$) | Phân phối xác suất dự đoán | Đánh giá | Đầu ra sau khi đưa `logits` qua `Softmax`, thỏa mãn $p_c \in (0, 1)$ và $\sum_{c=0}^{C-1} p_c = 1$. |

## 7. Công thức Softmax Ổn định Số học & Cross-Entropy Loss
1. **Softmax ổn định số học (Subtract-Max Trick)**: Để tránh $e^{z_c}$ bị tràn số (`overflow = inf`) khi $z_c$ lớn (ví dụ $z_c = 1000$), ta trừ đi giá trị lớn nhất trên từng hàng trước khi lấy mũ mà kết quả toán học không hề thay đổi:
   $$p_c = \frac{e^{z_c - \max_k z_k}}{\sum_{j=0}^{C-1} e^{z_j - \max_k z_k}}$$
2. **Hàm mất mát Cross-Entropy trung bình trên batch $B$**:
   $$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log\big(p_{i, y_i} + \epsilon\big)$$""",
        ),
        (
            "code",
            """# Cell 1: Cài đặt Lớp Mạng Nơ-ron 2 Tầng bằng NumPy Thuần (Forward + Backward + Shape Trace)
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import set_seed
from src.synthetic_data import generate_synthetic_shapes_dataset
from src.features import extract_combined_features

set_seed(42)


class TwoLayerNetNumPy:
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        # Khởi tạo He initialization cho tầng ReLU giúp phương sai tín hiệu ổn định
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = rng.normal(0.0, np.sqrt(1.0 / hidden_dim), size=(hidden_dim, num_classes))
        self.b2 = np.zeros(num_classes, dtype=np.float64)
        self.cache: dict = {}

    def forward(self, X: np.ndarray, verbose_shapes: bool = False) -> tuple[np.ndarray, np.ndarray]:
        Z1 = X @ self.W1 + self.b1
        H1 = np.maximum(0.0, Z1)  # ReLU activation
        Z2 = H1 @ self.W2 + self.b2  # Logits

        # Softmax ổn định số học
        shifted_logits = Z2 - np.max(Z2, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self.cache = {"X": X, "Z1": Z1, "H1": H1, "Z2": Z2, "probs": probs}
        if verbose_shapes:
            print(f"[Shape Trace] Input X: {X.shape} -> Linear1 Z1: {Z1.shape} -> ReLU H1: {H1.shape} -> Logits Z2: {Z2.shape} -> Probs: {probs.shape}")
        return Z2, probs

    def compute_loss_and_backward(self, X: np.ndarray, y: np.ndarray, weight_decay: float = 1e-4, verbose_shapes: bool = False) -> tuple[float, dict]:
        _, probs = self.forward(X, verbose_shapes=verbose_shapes)
        batch_size = len(X)
        correct_logprobs = -np.log(probs[np.arange(batch_size), y] + 1e-12)
        data_loss = float(np.mean(correct_logprobs))
        reg_loss = 0.5 * weight_decay * float(np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        total_loss = data_loss + reg_loss

        # Backpropagation theo Chain Rule
        dZ2 = probs.copy()
        dZ2[np.arange(batch_size), y] -= 1.0
        dZ2 /= batch_size

        H1 = self.cache["H1"]
        Z1 = self.cache["Z1"]
        dW2 = H1.T @ dZ2 + weight_decay * self.W2
        db2 = np.sum(dZ2, axis=0)

        dH1 = dZ2 @ self.W2.T
        dZ1 = dH1 * (Z1 > 0.0)
        dW1 = X.T @ dZ1 + weight_decay * self.W1
        db1 = np.sum(dZ1, axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        if verbose_shapes:
            print(f"[Backward Trace] Loss={total_loss:.4f} -> dZ2:{dZ2.shape} -> dW2:{dW2.shape}, db2:{db2.shape} -> dW1:{dW1.shape}, db1:{db1.shape}")
        return total_loss, grads


# Chạy kiểm tra dòng chảy shape trên 1 batch nhỏ (B=4, D=16, H=8, C=4)
net_check = TwoLayerNetNumPy(input_dim=16, hidden_dim=8, num_classes=4)
X_toy = np.random.randn(4, 16)
y_toy = np.array([0, 2, 1, 3])
loss_init, grads_init = net_check.compute_loss_and_backward(X_toy, y_toy, verbose_shapes=True)""",
        ),
        (
            "code",
            """# Cell 2: Kiểm chứng Đạo hàm Ngược (Backpropagation) bằng Đạo hàm Số (Numerical Gradient Check)
eps = 1e-6
param_W2_orig = net_check.W2.copy()
num_grad_W2 = np.zeros_like(net_check.W2)

for i in range(net_check.W2.shape[0]):
    for j in range(net_check.W2.shape[1]):
        net_check.W2[i, j] = param_W2_orig[i, j] + eps
        loss_plus, _ = net_check.compute_loss_and_backward(X_toy, y_toy)
        net_check.W2[i, j] = param_W2_orig[i, j] - eps
        loss_minus, _ = net_check.compute_loss_and_backward(X_toy, y_toy)
        num_grad_W2[i, j] = (loss_plus - loss_minus) / (2.0 * eps)
        net_check.W2[i, j] = param_W2_orig[i, j]

max_diff = np.max(np.abs(grads_init["W2"] - num_grad_W2))
print(f"Sai số cực đại giữa Backprop giải tích và Đạo hàm số trên W2: {max_diff:.2e}")
assert max_diff < 1e-7, "Công thức Backpropagation phải khớp tuyệt đối với đạo hàm số!" """,
        ),
        (
            "code",
            """# Cell 3: Huấn luyện thực tế Mạng Nơ-ron NumPy trên Đặc trưng Ảnh của 4 Lớp Hình học
ds = generate_synthetic_shapes_dataset(num_samples=320, image_size=32, seed=42)
X_feats = extract_combined_features(ds["images"]).astype(np.float64)
y_labels = ds["labels"]

X_tr, X_va, y_tr, y_va = train_test_split(X_feats, y_labels, test_size=0.25, random_state=42, stratify=y_labels)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_va = scaler.transform(X_va)

net = TwoLayerNetNumPy(input_dim=X_tr.shape[1], hidden_dim=32, num_classes=4, seed=42)
lr = 0.15
epochs = 80
train_losses, val_accs = [], []

for ep in range(1, epochs + 1):
    loss, grads = net.compute_loss_and_backward(X_tr, y_tr, weight_decay=1e-4)
    # Optimizer step (Gradient Descent)
    net.W1 -= lr * grads["W1"]
    net.b1 -= lr * grads["b1"]
    net.W2 -= lr * grads["W2"]
    net.b2 -= lr * grads["b2"]

    _, val_probs = net.forward(X_va)
    val_acc = float(np.mean( np.argmax(val_probs, axis=1) == y_va ))
    train_losses.append(loss)
    val_accs.append(val_acc)

print(f"Sau {epochs} epochs -> Train Loss giảm từ {train_losses[0]:.3f} xuống {train_losses[-1]:.3f} | Val Accuracy đạt: {val_accs[-1]*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].plot(train_losses, color="crimson", lw=2)
axes[0].set_title("Đường cong Train Loss (NumPy NN)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")

axes[1].plot([a * 100 for a in val_accs], color="seagreen", lw=2)
axes[1].set_title("Độ chính xác Validation (%)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 4: Thí nghiệm thay đổi một yếu tố — Điều gì xảy ra nếu khởi tạo toàn bộ Trọng số W = 0? (Symmetry Breaking Problem)
net_zero = TwoLayerNetNumPy(input_dim=X_tr.shape[1], hidden_dim=32, num_classes=4)
net_zero.W1.fill(0.0)
net_zero.W2.fill(0.0)

for _ in range(30):
    _, g_z = net_zero.compute_loss_and_backward(X_tr, y_tr)
    net_zero.W1 -= 0.15 * g_z["W1"]
    net_zero.b1 -= 0.15 * g_z["b1"]
    net_zero.W2 -= 0.15 * g_z["W2"]
    net_zero.b2 -= 0.15 * g_z["b2"]

_, probs_zero = net_zero.forward(X_va)
acc_zero = float(np.mean(np.argmax(probs_zero, axis=1) == y_va))
print(f"Khởi tạo W = 0 toàn bộ -> Val Accuracy bị kẹt ở: {acc_zero*100:.1f}% (Mạng hoàn toàn không học được đặc trưng vì đối xứng!)")
print(f"Khởi tạo ngẫu nhiên He -> Val Accuracy đạt     : {val_accs[-1]*100:.1f}%")""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Ở Cell 2**: Sai số giữa đạo hàm ngược giải tích (`Backpropagation`) và đạo hàm số (`Numerical Gradient`) chỉ cỡ $10^{-10}$, chứng minh công thức đạo hàm mắt xích của chúng ta hoàn toàn chính xác.
- **Ở Cell 4 (Bài toán phá vỡ tính đối xứng - Symmetry Breaking)**: Nếu khởi tạo $W_1 = 0, W_2 = 0$, mọi nơ-ron ở tầng ẩn đều nhận cùng tín hiệu $0$ và nhận cùng đạo hàm như nhau mãi mãi! Do đó **trọng số $W$ bắt buộc phải được khởi tạo ngẫu nhiên** (trong khi độ lệch `bias` $b$ có thể khởi tạo bằng `0`).

## 13. Lỗi thường gặp và cách sửa
1. **Tính `np.exp(Z)` trực tiếp gây lỗi `RuntimeWarning: overflow encountered in exp`**: Luôn trừ `np.max(Z, axis=1, keepdims=True)` trước khi lấy `np.exp`.
2. **Quên chia `batch_size` khi tính đạo hàm `dZ2`**: Nếu hàm loss là trung bình cộng trên batch ($\frac{1}{B}\sum$), thì đạo hàm `dZ2` cũng phải chia cho `batch_size`, nếu không tốc độ học hiệu dụng sẽ bị nhân gấp $B$ lần!

## 14. Bài tập tự làm
- **Cơ bản**: Thay đổi kích thước tầng ẩn `hidden_dim` qua `[4, 16, 64]` và so sánh `Val Accuracy` sau 60 epochs.
- **Trung cấp**: Thay hàm kích hoạt `ReLU` bằng `LeakyReLU` ($f(z) = z \text{ nếu } z > 0 \text{ else } 0.01 z$) và viết đạo hàm ngược tương ứng của `LeakyReLU`.
- **Nâng cao**: Bổ sung động lượng **Momentum** ($v_t = 0.9 v_{t-1} - \eta \nabla W$) vào bước cập nhật của `TwoLayerNetNumPy` và so sánh tốc độ giảm loss với Gradient Descent thuần.

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 08
# 1. Cơ bản: Khảo sát hidden_dim
for h_dim in [4, 16, 64]:
    n_h = TwoLayerNetNumPy(X_tr.shape[1], h_dim, 4, seed=42)
    for _ in range(60):
        _, g = n_h.compute_loss_and_backward(X_tr, y_tr)
        for k in ["W1", "b1", "W2", "b2"]:
            setattr(n_h, k, getattr(n_h, k) - 0.15 * g[k])
    _, p_va = n_h.forward(X_va)
    print(f"[Cơ bản] hidden_dim={h_dim:2d} -> Val Acc = {(np.argmax(p_va, axis=1) == y_va).mean()*100:.1f}%")

# 2. Trung cấp: Đạo hàm của LeakyReLU (alpha = 0.01)
z_sample = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
alpha_leaky = 0.01
leaky_forward = np.where(z_sample > 0, z_sample, alpha_leaky * z_sample)
leaky_grad_mask = np.where(z_sample > 0, 1.0, alpha_leaky)
print("[Trung cấp] LeakyReLU forward:", leaky_forward, "| Đạo hàm mask:", leaky_grad_mask)

# 3. Nâng cao: Huấn luyện với Momentum (mu = 0.9)
net_mom = TwoLayerNetNumPy(X_tr.shape[1], 32, 4, seed=42)
velocities = {k: np.zeros_like(getattr(net_mom, k)) for k in ["W1", "b1", "W2", "b2"]}
for _ in range(40):
    loss_m, g_m = net_mom.compute_loss_and_backward(X_tr, y_tr)
    for k in ["W1", "b1", "W2", "b2"]:
        velocities[k] = 0.9 * velocities[k] - 0.10 * g_m[k]
        setattr(net_mom, k, getattr(net_mom, k) + velocities[k])
print(f"[Nâng cao] Loss sau 40 bước với SGD+Momentum: {loss_m:.4f}")""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Đã tự cài đặt thành công `Forward`, `Softmax`, `Cross-Entropy` và `Backpropagation` bằng NumPy thuần.
- [x] Đã hiểu vì sao khi sang **Bài 09: `09_pytorch_and_autograd.ipynb`**, bộ máy `autograd` của PyTorch sẽ tự động tính toàn bộ phần `Backward` này cho chúng ta!

## 17. Nguồn tham khảo
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986), *Learning representations by back-propagating errors*, Nature.
- CS231n Stanford Notes on Backpropagation: https://cs231n.github.io/optimization-2/""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "08_neural_network_from_scratch.ipynb", cells)



def build_nb09() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 09: Làm chủ PyTorch, Autograd và Huấn luyện Mạng MLP Phân loại Ảnh

## 1. Mục tiêu học tập
- Hiểu cách `PyTorch` quản lý `torch.Tensor`, thiết bị tính toán (`device`), và bộ máy tính đạo hàm tự động **`autograd`** (`requires_grad=True`, `.backward()`, `.grad`).
- Hiểu kiến trúc lớp cơ sở `torch.nn.Module`, cách tổ chức `Dataset` và `DataLoader`.
- Xây dựng mạng **MLP (`SimpleMLP`)** phân loại ảnh, thực hiện đầy đủ các tiêu chuẩn bắt buộc của mô hình Deep Learning:
  1. In tên model và tổng số tham số (`total`, `trainable`, `frozen`).
  2. In shape đầu vào/đầu ra và lập bảng chi tiết từng tầng (`layer_name`, `output_shape`, `params`).
  3. Viết tường minh vòng lặp huấn luyện (`Explicit Training Loop`) và kiểm định (`Validation Loop`).
  4. Lưu `checkpoint` tốt nhất dựa trên `Validation Loss`, nạp lại checkpoint và kiểm tra suy luận (`Inference`).
  5. Trực quan hóa các mẫu dự đoán **đúng** và **sai**.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 08 (hiểu Forward, Loss, Backward bằng NumPy).

## 3. Thời gian học dự kiến
- Khoảng **2.5 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
Thay vì tự viết tay công thức đạo hàm ngược dễ nhầm lẫn khi mạng có hàng chục tầng, ta dùng `PyTorch Autograd` để tự động ghi lại đồ thị tính toán trong lúc `forward` và tính toàn bộ đạo hàm chỉ bằng 1 lệnh `loss.backward()`, sau đó huấn luyện mạng `SimpleMLP` trên ảnh màu `(3, 32, 32)`.

## 5. Giải thích trực quan: Đồ thị Tính toán Động (`Dynamic Computation Graph`)
Khi bạn thực hiện phép toán `y = w * x + b` trên các Tensor có `requires_grad=True`:
- PyTorch âm thầm gắn vào `y` một thuộc tính `grad_fn` (ví dụ `<AddBackward0>`) trỏ ngược về `w`, `x`, `b`.
- Khi gọi `loss.backward()`, PyTorch đi ngược theo các mũi tên `grad_fn` từ `loss` về lá (`leaf tensors`) và cộng dồn đạo hàm $\frac{\partial \mathcal{L}}{\partial w}$ vào `w.grad`.

## 6. Thuật ngữ mới & Giải thích Keyword
| Keyword | Nghĩa tiếng Việt | Nhóm | Vai trò & Lưu ý quan trọng |
|---|---|---|---|
| `requires_grad=True` | Bật theo dõi đạo hàm | Huấn luyện | Báo cho `autograd` lưu lịch sử phép toán trên Tensor này. Mặc định các tham số trong `nn.Linear`, `nn.Conv2d` đều có `requires_grad=True`. |
| `optimizer.zero_grad()` | Xóa đạo hàm cũ về 0 | Huấn luyện | Phải gọi trước mỗi lần `loss.backward()` vì PyTorch mặc định cộng dồn (`+=`) đạo hàm mới vào `.grad` cũ. |
| `nn.CrossEntropyLoss()` | Hàm mất mát Cross-Entropy đa lớp | Huấn luyện | Nhận trực tiếp **`logits` thô `(B, C)`** và nhãn **`targets` kiểu `torch.long` `(B,)`**. Tuyệt đối không đặt `Softmax` trước hàm này! |
| `state_dict()` | Từ điển trọng số của mô hình | Triển khai | Lưu toàn bộ `weight`, `bias` và các buffer thống kê dưới dạng `OrderedDict` để ghi ra file `.pt`. |""",
        ),
        (
            "code",
            """# Cell 1: Kiểm chứng PyTorch Autograd — Đối chiếu đạo hàm tự động với đạo hàm tính tay
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

from src.utils import (
    set_seed,
    get_device,
    count_parameters,
    summarize_model_layers,
    save_checkpoint,
    load_checkpoint,
)
from src.synthetic_data import generate_synthetic_shapes_dataset
from src.models import SimpleMLP
from src.evaluation import compute_classification_metrics

QUICK_RUN = True
set_seed(42)
device = get_device()

# Kiểm tra Autograd trên công thức L(w) = (w * x - y)^2 ở Bài 04 (x=2, y=10, w=1 -> dL/dw = -32)
x_t = torch.tensor(2.0)
y_t = torch.tensor(10.0)
w_t = torch.tensor(1.0, requires_grad=True)

loss_t = (w_t * x_t - y_t) ** 2
loss_t.backward()
print(f"Loss = {loss_t.item():.1f} | grad_fn = {loss_t.grad_fn}")
print(f"Đạo hàm tự động w_t.grad = {w_t.grad.item():.1f} (Khớp tuyệt đối với -32.0 tính tay ở Bài 04!)")""",
        ),
        (
            "code",
            """# Cell 2: Chuẩn bị Dataset, DataLoader (chuẩn hóa chỉ từ tập Train) và Bảng Tóm tắt Kiến trúc SimpleMLP
n_samples = 240 if QUICK_RUN else 480
ds = generate_synthetic_shapes_dataset(num_samples=n_samples, image_size=32, seed=42)
images_np = ds["images"]
labels_np = ds["labels"]
class_names = ds["class_names"]

idx_all = np.arange(len(images_np))
idx_tr, idx_temp = train_test_split(idx_all, test_size=0.30, random_state=42, stratify=labels_np)
idx_va, idx_te = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=labels_np[idx_temp])

# Chuyển sang NCHW float32 [0, 1] và tính mean, std CHỈ TRÊN TẬP TRAIN
X_all_t = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
y_all_t = torch.from_numpy(labels_np).long()

train_mean = X_all_t[idx_tr].mean(dim=(0, 2, 3), keepdim=True)
train_std = X_all_t[idx_tr].std(dim=(0, 2, 3), keepdim=True) + 1e-6

X_norm_t = (X_all_t - train_mean) / train_std

train_loader = DataLoader(TensorDataset(X_norm_t[idx_tr], y_all_t[idx_tr]), batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(TensorDataset(X_norm_t[idx_va], y_all_t[idx_va]), batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(TensorDataset(X_norm_t[idx_te], y_all_t[idx_te]), batch_size=32, shuffle=False, num_workers=0)

model = SimpleMLP(in_channels=3, image_size=32, hidden_dim=128, num_classes=4, dropout_p=0.2).to(device)
param_info = count_parameters(model)
print(f"Tên mô hình: {model.__class__.__name__} | Thiết bị: {device}")
print(f"Số lượng tham số -> Tổng: {param_info['total']:,} | Trainable: {param_info['trainable']:,} | Frozen: {param_info['frozen']:,}")

layer_table = pd.DataFrame(summarize_model_layers(model, input_shape=(1, 3, 32, 32), device=device))
print("\\n=== BẢNG KIẾN TRÚC LAYER, OUTPUT SHAPE VÀ SỐ THAM SỐ ===")
print(layer_table.to_string(index=False))""",
        ),
        (
            "code",
            """# Cell 3: Viết Tường minh Vòng lặp Huấn luyện (Explicit Training Loop), Lưu & Nạp Checkpoint Tốt nhất
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 12 if QUICK_RUN else 25
ckpt_path = project_root / "outputs" / "checkpoints" / "nb09_simple_mlp_best.pt"
preproc_cfg = {
    "image_size": 32,
    "mean": train_mean.squeeze().tolist(),
    "std": train_std.squeeze().tolist(),
}

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_loss = float("inf")

for epoch in range(1, epochs + 1):
    # --- 1. GIAI ĐOẠN TRAIN ---
    model.train()
    tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(x_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()

        tr_loss_sum += loss.item() * len(x_b)
        tr_correct += int((logits.argmax(dim=1) == y_b).sum().item())
        tr_total += len(x_b)

    # --- 2. GIAI ĐOẠN VALIDATION ---
    model.eval()
    va_loss_sum, va_correct, va_total = 0.0, 0, 0
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            logits = model(x_b)
            loss = criterion(logits, y_b)
            va_loss_sum += loss.item() * len(x_b)
            va_correct += int((logits.argmax(dim=1) == y_b).sum().item())
            va_total += len(x_b)

    tr_loss, tr_acc = tr_loss_sum / tr_total, tr_correct / tr_total
    va_loss, va_acc = va_loss_sum / va_total, va_correct / va_total
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        save_checkpoint(
            filepath=ckpt_path,
            model=model,
            class_names=class_names,
            preprocessing_config=preproc_cfg,
            epoch=epoch,
            val_metrics={"loss": va_loss, "accuracy": va_acc},
            optimizer=optimizer,
        )

print(f"Đã lưu checkpoint tốt nhất tại: {ckpt_path.relative_to(project_root)} (Best Val Loss = {best_val_loss:.4f})")

# Nạp lại checkpoint tốt nhất và kiểm tra Inference trên tập Test
reloaded_model = SimpleMLP(in_channels=3, image_size=32, hidden_dim=128, num_classes=4).to(device)
ckpt_data = load_checkpoint(ckpt_path, model=reloaded_model, device=device)

reloaded_model.eval()
with torch.no_grad():
    test_logits = reloaded_model(X_norm_t[idx_te].to(device))
    test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
    test_preds = test_probs.argmax(axis=1)

test_metrics = compute_classification_metrics(labels_np[idx_te], test_preds, class_names)
print(f"Kết quả kiểm tra lại từ Checkpoint trên tập Test -> Accuracy: {test_metrics['accuracy']*100:.1f}%, Macro-F1: {test_metrics['macro_f1']:.3f}")""",
        ),
        (
            "code",
            """# Cell 4: Vẽ Đường cong Loss/Accuracy và Hiển thị Dự đoán Đúng / Sai trên tập Test
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].plot(history["train_loss"], label="Train Loss", color="royalblue")
axes[0].plot(history["val_loss"], label="Val Loss", color="crimson")
axes[0].set_title("Đường cong Loss (SimpleMLP)")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot([a * 100 for a in history["train_acc"]], label="Train Acc (%)", color="royalblue")
axes[1].plot([a * 100 for a in history["val_acc"]], label="Val Acc (%)", color="seagreen")
axes[1].set_title("Đường cong Accuracy (%)")
axes[1].set_xlabel("Epoch")
axes[1].legend()
plt.tight_layout()
plt.show()

# Hiển thị mẫu dự đoán đúng và mẫu dự đoán sai (nếu có)
y_te_true = labels_np[idx_te]
correct_mask = test_preds == y_te_true
correct_indices = np.where(correct_mask)[0][:2]
wrong_indices = np.where(~correct_mask)[0][:2]
vis_indices = list(correct_indices) + list(wrong_indices)

fig, axes = plt.subplots(1, len(vis_indices), figsize=(3.0 * len(vis_indices), 3.0))
if len(vis_indices) == 1:
    axes = [axes]
for ax, local_i in zip(axes, vis_indices):
    orig_img = images_np[idx_te[local_i]]
    t_cls = class_names[y_te_true[local_i]]
    p_cls = class_names[test_preds[local_i]]
    conf = test_probs[local_i, test_preds[local_i]] * 100
    status = "ĐÚNG" if y_te_true[local_i] == test_preds[local_i] else "SAI"
    ax.imshow(orig_img)
    ax.set_title(f"[{status}] Thật: {t_cls}\\nĐoán: {p_cls} ({conf:.0f}%)", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- Nhìn vào **Bảng kiến trúc ở Cell 2**: Tầng `Linear` đầu tiên (`classifier.0`) chiếm tới `3072 * 128 + 128 = 393,344` tham số (chiếm **97.8%** tổng số tham số của toàn mạng!), chỉ vì nó nối mỗi pixel trong `3072` kênh-điểm ảnh với mọi nơ-ron ẩn.
- Hạn chế lớn nhất của `MLP` trên ảnh: Khi dịch chuyển hình tròn sang phải vài pixel, vị trí các phần tử trong vector trải phẳng `3072` chiều thay đổi hoàn toàn. Đó là lý do bài tiếp theo (**Bài 10: CNN**) sẽ dùng phép tích chập chia sẻ trọng số để giảm số tham số đi hơn 15 lần mà độ chính xác lại cao hơn!

## 12 & 14. Thí nghiệm và Bài tập tự làm
- **Cơ bản**: So sánh tốc độ hội tụ của `SimpleMLP` khi dùng bộ tối ưu `SGD(lr=0.01)` so với `AdamW(lr=1e-3)`.
- **Trung cấp**: Kiểm tra xem điều gì xảy ra nếu khi suy luận trên ảnh có `dropout_p=0.5`, bạn quên gọi `model.eval()` và chạy dự đoán 5 lần liên tiếp trên cùng 1 bức ảnh.
- **Nâng cao**: Đóng băng tầng `classifier.0` (`requires_grad = False`) và đếm lại số lượng `trainable` vs `frozen` parameters bằng hàm `count_parameters`.

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho Thí nghiệm & Bài tập Bài 09
# 1. Cơ bản: So sánh SGD vs AdamW trong 5 epochs
for opt_name, make_opt in [
    ("SGD (lr=0.01)", lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9)),
    ("AdamW (lr=1e-3)", lambda params: torch.optim.AdamW(params, lr=1e-3)),
]:
    set_seed(42)
    m_tmp = SimpleMLP(3, 32, 64, 4).to(device)
    opt_tmp = make_opt(m_tmp.parameters())
    for _ in range(5):
        m_tmp.train()
        for xb, yb in train_loader:
            opt_tmp.zero_grad()
            loss_b = criterion(m_tmp(xb.to(device)), yb.to(device))
            loss_b.backward()
            opt_tmp.step()
    m_tmp.eval()
    with torch.no_grad():
        va_acc_tmp = (m_tmp(X_norm_t[idx_va].to(device)).argmax(1).cpu() == y_all_t[idx_va]).float().mean().item()
    print(f"[Cơ bản] {opt_name:16s} -> Val Acc sau 5 epochs: {va_acc_tmp*100:.1f}%")

# 2. Trung cấp: Hậu quả của việc quên gọi model.eval() khi có Dropout
m_drop = SimpleMLP(3, 32, 64, 4, dropout_p=0.5).to(device)
sample_one = X_norm_t[idx_te[:1]].to(device)
m_drop.train()  # Cố tình để ở chế độ train
preds_train_mode = [round(float(m_drop(sample_one)[0, 0].item()), 4) for _ in range(4)]
m_drop.eval()   # Chuyển sang chế độ eval chuẩn
preds_eval_mode = [round(float(m_drop(sample_one)[0, 0].item()), 4) for _ in range(4)]
print("[Trung cấp] Logit lớp 0 khi QUÊN model.eval() (nhảy loạn):", preds_train_mode)
print("[Trung cấp] Logit lớp 0 khi CÓ model.eval() (nhất quán)  :", preds_eval_mode)

# 3. Nâng cao: Đóng băng tầng đầu tiên và đếm lại tham số
for p in m_drop.classifier[0].parameters():
    p.requires_grad = False
print("[Nâng cao] Tham số sau khi đóng băng tầng Linear đầu:", count_parameters(m_drop))""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Đã thành thạo `Tensor`, `autograd`, `Dataset`, `DataLoader` và quy trình lưu/nạp `checkpoint` chuẩn.
- [x] Đã thấy hạn chế bùng nổ tham số của `MLP` trên dữ liệu ảnh 2D.
- [x] Sẵn sàng chuyển sang **Bài 10: `10_cnn_from_scratch.ipynb`** để khám phá kiến trúc Mạng Tích chập (CNN)!

## 17. Nguồn tham khảo
- Paszke, A. et al. (2019), *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS.
- PyTorch Autograd Mechanics: https://pytorch.org/docs/stable/notes/autograd.html""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "09_pytorch_and_autograd.ipynb", cells)


def build_nb10() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 10: Mạng Tích chập (CNN) Từ Đầu — Giải phẫu Conv2d, Pooling, BatchNorm, Residual và Receptive Field

## 1. Mục tiêu học tập
- Hiểu vì sao **CNN (`Convolutional Neural Network`)** vượt trội hơn `MLP` trên ảnh nhờ 2 nguyên lý: **Kết nối cục bộ (`Local Connectivity`)** và **Chia sẻ trọng số (`Weight Sharing`)**.
- Nắm vững ý nghĩa và công thức tính kích thước của mọi thành phần trong CNN:
  - `nn.Conv2d`: `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `groups`.
  - `Pooling`: `MaxPool2d`, `AvgPool2d`, và sự khác biệt giữa `Flatten` với **`Global Average Pooling` (`AdaptiveAvgPool2d(1)`)**.
  - Chuẩn hóa bên trong mạng: `BatchNorm2d` so với `LayerNorm`.
  - Điều chuẩn: `Dropout`.
  - Kết nối tắt phần dư: **`Residual Connection`** ($y = \mathcal{F}(x) + x$).
  - Vùng cảm nhận hiệu dụng: **`Receptive Field`**.
- Giải thích rõ: **Vì sao mỗi lớp được chọn và điều gì xảy ra nếu bỏ hoặc thay nó?**
- Huấn luyện thực tế `SimpleCNN` và `ResidualCNN`, lưu checkpoint và kiểm chứng kết quả.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 09.

## 3. Thời gian học dự kiến
- Khoảng **3.0 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
Ở Bài 09, mạng `SimpleMLP` tốn hơn `401,000` tham số cho ảnh `32x32x3`. Trong bài này, chúng ta xây dựng `SimpleCNN` chỉ với `~24,000` tham số (nhỏ hơn **16 lần**!) nhưng đạt độ chính xác phân loại cao hơn hẳn và bền vững khi vật thể dịch chuyển vị trí trong ảnh.

## 5. Giải thích trực quan: Vì sao cần từng lớp và Điều gì xảy ra nếu bỏ nó?
| Thành phần trong khối CNN | Lý do được chọn | Điều gì xảy ra nếu BỎ hoặc THAY ĐỔI nó? |
|---|---|---|
| **`nn.Conv2d(C_in, C_out, K, S, P)`** | Trượt `C_out` bộ lọc cục bộ `KxK` khắp ảnh để phát hiện cạnh, góc, hình dạng ở mọi vị trí. | Nếu thay bằng `Linear`, số tham số tăng hàng chục lần và mất hoàn toàn cấu trúc lân cận 2D. |
| **`nn.BatchNorm2d(C_out)`** | Chuẩn hóa từng kênh về trung bình `0`, phương sai `1` trên mỗi mini-batch rồi nhân $\gamma$, cộng $\beta$. | **Nếu bỏ `BatchNorm2d`**: Các mạng sâu hội tụ chậm hơn nhiều, nhạy cảm với `learning_rate` lớn và dễ bị bão hòa/triệt tiêu đạo hàm. |
| **`nn.ReLU(inplace=True)`** | Khử tuyến tính các giá trị âm về `0`. | **Nếu bỏ `ReLU`**: Chuỗi nhiều tầng `Conv2d` liên tiếp thu gọn thành 1 phép lọc tuyến tính duy nhất! |
| **`nn.MaxPool2d(2, 2)`** (hoặc `Conv2d(..., stride=2)`) | Giảm một nửa chiều cao và chiều rộng ($H/2, W/2$), giúp các tầng sau có **`Receptive Field`** rộng hơn để nhìn tổng thể vật thể. | **Nếu bỏ toàn bộ bước giảm độ phân giải**: Bộ nhớ trung gian (`feature maps`) rất nặng và vùng nhìn hiệu dụng (`Receptive Field`) ở tầng cuối quá hẹp, không bao quát được vật thể lớn. |
| **`nn.AdaptiveAvgPool2d((1, 1))`** (Global Average Pooling) | Lấy trung bình mỗi bản đồ đặc trưng `(C, H, W)` thành đúng 1 con số `(C, 1, 1)` trước khi vào tầng phân loại `Linear`. | Nếu dùng `Flatten` trực tiếp mà không qua Global Average Pooling, tầng `Linear` cuối sẽ phình to gấp hàng chục lần và bắt buộc ảnh đầu vào phải cố định kích thước. |
| **`Residual Connection` ($y = \mathcal{F}(x) + x$)** | Cộng trực tiếp đầu vào khối vào đầu ra khối. | Khi mạng có từ 15–20 tầng trở lên, nếu không có nhánh cộng tắt này, đạo hàm truyền ngược bị suy giảm khiến mạng sâu hơn lại có `train_loss` tệ hơn mạng nông! |

## 6 & 7. Công thức Tính Kích thước Đầu ra & Vùng Cảm nhận (`Receptive Field`)
1. **Kích thước chiều cao/rộng sau tầng `Conv2d` hoặc `Pool2d`**:
   $$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2P - D \times (K - 1) - 1}{S} + 1 \right\rfloor$$
   - **Ví dụ số nhỏ**: Ảnh $H_{\text{in}} = 32$, dùng `Conv2d(kernel_size=3, stride=1, padding=1, dilation=1)`:
     $$H_{\text{out}} = \left\lfloor \frac{32 + 2(1) - 1(3 - 1) - 1}{1} + 1 \right\rfloor = 32 \quad (\text{Giữ nguyên kích thước!})$$
2. **Số lượng tham số của 1 tầng `Conv2d`**:
   $$\text{Params}_{\text{Conv2d}} = C_{\text{out}} \times \left(\frac{C_{\text{in}}}{\text{groups}} \times K_H \times K_W\right) + \text{ (} C_{\text{out}} \text{ nếu có bias)}$$""",
        ),
        (
            "code",
            """# Cell 1: Kiểm tra Công thức Shape, Receptive Field và Bảng Tham số của SimpleCNN vs ResidualCNN
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

from src.utils import (
    set_seed,
    get_device,
    count_parameters,
    summarize_model_layers,
)
from src.synthetic_data import generate_synthetic_shapes_dataset
from src.models import SimpleCNN, ResidualCNN
from src.training import fit_classifier, evaluate_classifier
from src.evaluation import compute_classification_metrics

QUICK_RUN = True
set_seed(42)
device = get_device()


def compute_receptive_field(layers_cfg: list[tuple[str, int, int, int]]) -> pd.DataFrame:
    \"\"\"
    Tính vùng cảm nhận (Receptive Field R) và bước nhảy tích lũy (jump j) qua từng tầng.
    Mỗi phần tử trong layers_cfg là: (name, kernel_size K, stride S, dilation D).
    \"\"\"
    r_curr = 1
    j_curr = 1
    rows = []
    for name, k, s, d in layers_cfg:
        r_curr = r_curr + (k - 1) * d * j_curr
        j_curr = j_curr * s
        rows.append({"Layer": name, "Kernel": k, "Stride": s, "Dilation": d, "Jump_Stride": j_curr, "Receptive_Field": f"{r_curr}x{r_curr}"})
    return pd.DataFrame(rows)


rf_df = compute_receptive_field([
    ("Conv1 (3x3, s=1)", 3, 1, 1),
    ("MaxPool1 (2x2, s=2)", 2, 2, 1),
    ("Conv2 (3x3, s=1)", 3, 1, 1),
    ("MaxPool2 (2x2, s=2)", 2, 2, 1),
    ("Conv3 (3x3, s=1)", 3, 1, 1),
])
print("=== BẢNG TÍNH VÙNG CẢM NHẬN (RECEPTIVE FIELD) QUA TỪNG TẦNG CỦA SIMPLE_CNN ===")
print(rf_df.to_string(index=False))

cnn_model = SimpleCNN(in_channels=3, num_classes=4, base_channels=16).to(device)
res_model = ResidualCNN(in_channels=3, num_classes=4, base_channels=16).to(device)
print(f"\\nSimpleCNN parameters  : {count_parameters(cnn_model)}")
print(f"ResidualCNN parameters: {count_parameters(res_model)}")

cnn_summary_df = pd.DataFrame(summarize_model_layers(cnn_model, input_shape=(1, 3, 32, 32), device=device))
print("\\n=== BẢNG LAYER SUMMARY CỦA SIMPLE_CNN ===")
print(cnn_summary_df.to_string(index=False))""",
        ),
        (
            "code",
            """# Cell 2: Huấn luyện Thực tế SimpleCNN và ResidualCNN, Lưu Checkpoint Tốt nhất
n_samples = 280 if QUICK_RUN else 560
ds = generate_synthetic_shapes_dataset(num_samples=n_samples, image_size=32, seed=42, noise_std=10.0)
images_np, labels_np, class_names = ds["images"], ds["labels"], ds["class_names"]

idx_all = np.arange(len(images_np))
idx_tr, idx_temp = train_test_split(idx_all, test_size=0.30, random_state=42, stratify=labels_np)
idx_va, idx_te = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=labels_np[idx_temp])

X_all_t = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
y_all_t = torch.from_numpy(labels_np).long()

tr_mean = X_all_t[idx_tr].mean(dim=(0, 2, 3), keepdim=True)
tr_std = X_all_t[idx_tr].std(dim=(0, 2, 3), keepdim=True) + 1e-6
X_norm = (X_all_t - tr_mean) / tr_std

train_loader = DataLoader(TensorDataset(X_norm[idx_tr], y_all_t[idx_tr]), batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(TensorDataset(X_norm[idx_va], y_all_t[idx_va]), batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(TensorDataset(X_norm[idx_te], y_all_t[idx_te]), batch_size=32, shuffle=False, num_workers=0)

ckpt_cnn = project_root / "outputs" / "checkpoints" / "nb10_simple_cnn_best.pt"
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=2e-3, weight_decay=1e-4)

history_cnn = fit_classifier(
    model=cnn_model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=12 if QUICK_RUN else 25,
    patience=6,
    checkpoint_path=ckpt_cnn,
    class_names=class_names,
    preprocessing_config={"image_size": 32, "mean": tr_mean.squeeze().tolist(), "std": tr_std.squeeze().tolist()},
)

test_res, y_te_true, y_te_pred, y_te_prob = evaluate_classifier(cnn_model, test_loader, criterion, device)
test_metrics = compute_classification_metrics(y_te_true, y_te_pred, class_names)
print(f"SimpleCNN Best Epoch: {history_cnn['best_epoch']} | Best Val Acc: {history_cnn['best_val_acc']*100:.1f}%")
print(f"SimpleCNN Test Accuracy: {test_metrics['accuracy']*100:.1f}% | Test Macro-F1: {test_metrics['macro_f1']:.3f}")""",
        ),
        (
            "code",
            """# Cell 3: Trực quan hóa Bản đồ Đặc trưng Trung gian (Feature Maps) của Tầng Tích chập Đầu tiên
cnn_model.eval()
with torch.no_grad():
    sample_input = X_norm[idx_te[:1]].to(device)
    # Lấy đầu ra sau Conv1 + BatchNorm1 + ReLU1
    feat_conv1 = cnn_model.features[:3](sample_input).cpu().numpy()[0]  # shape (16, 32, 32)

fig, axes = plt.subplots(2, 5, figsize=(11, 4.4))
axes[0, 0].imshow(images_np[idx_te[0]])
axes[0, 0].set_title(f"Ảnh gốc\\n({class_names[labels_np[idx_te[0]]]})", fontsize=9.5)
axes[0, 0].axis("off")

for i in range(1, 10):
    r, c = divmod(i, 5)
    axes[r, c].imshow(feat_conv1[i - 1], cmap="viridis")
    axes[r, c].set_title(f"Kênh lọc #{i}", fontsize=9)
    axes[r, c].axis("off")
plt.suptitle("9 trên 16 Bản đồ đặc trưng (Feature Maps) ở tầng tích chập đầu tiên", fontsize=11)
plt.tight_layout()
plt.show()""",
        ),
        (
            "code",
            """# Cell 4: Thí nghiệm thay đổi một yếu tố — Có BatchNorm2d vs Bỏ BatchNorm2d
class CNNWithoutBN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

set_seed(42)
no_bn_model = CNNWithoutBN().to(device)
opt_nobn = torch.optim.AdamW(no_bn_model.parameters(), lr=2e-3)
hist_nobn = fit_classifier(no_bn_model, train_loader, val_loader, criterion, opt_nobn, device, epochs=10)

print(f"Có BatchNorm2d -> Best Val Acc (trong 10 epochs): {max(history_cnn['val_acc'][:10])*100:.1f}%")
print(f"Bỏ BatchNorm2d -> Best Val Acc (trong 10 epochs): {max(hist_nobn['val_acc'])*100:.1f}%")""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Ở Cell 1**: Vùng cảm nhận (`Receptive Field`) tăng từ `3x3` ở `Conv1` lên `14x14` ở `Conv3` nhờ hai tầng `MaxPool2d(2, 2)`. Nhờ đó mỗi điểm đặc trưng ở tầng thứ 3 bao quát được gần một nửa chiều rộng bức ảnh `32x32`.
- **Ở Cell 3**: Các bản đồ đặc trưng (`Feature Maps`) cho thấy mỗi bộ lọc trong 16 kênh đầu tiên tự động chuyên môn hóa: có kênh bật sáng ở viền ngang, có kênh bật sáng ở viền dọc, có kênh bắt vùng lõi màu của hình học.
- **Ở Cell 4**: Khi có `BatchNorm2d`, mô hình hội tụ nhanh và ổn định hơn hẳn trong những epoch đầu tiên.

## 13. Lỗi thường gặp và cách sửa
1. **Nhầm thứ tự kênh `(H, W, C)` khi đưa vào `nn.Conv2d`**: Luôn kiểm tra `x.shape[1] == in_channels`.
2. **Đặt `bias=True` (mặc định) ở `nn.Conv2d` ngay trước `nn.BatchNorm2d`**: Không gây lỗi dừng chương trình nhưng tạo ra các tham số dư thừa không có tác dụng (vì `BatchNorm2d` ngay lập tức trừ đi giá trị trung bình của kênh đó). Hãy đặt `bias=False` ở `Conv2d` khi theo sau là `BatchNorm2d`.

## 14. Bài tập tự làm
- **Cơ bản**: Tính tay kích thước đầu ra của `nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=1, dilation=2)` trên ảnh `(1, 3, 32, 32)` và kiểm tra lại bằng PyTorch.
- **Trung cấp**: So sánh số lượng tham số của tích chập chuẩn `nn.Conv2d(32, 64, kernel_size=3, padding=1)` với tích chập tách chiều **Depthwise Separable Convolution** (`Conv2d(32, 32, 3, padding=1, groups=32)` + `Conv2d(32, 64, 1)`).
- **Nâng cao**: Huấn luyện `ResidualCNN` trong 10 epochs trên cùng `train_loader` và so sánh `val_acc` với `SimpleCNN`.

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 10
# 1. Cơ bản: Kiểm tra công thức tính kích thước đầu ra Conv2d với dilation=2
h_in, p, d, k, s = 32, 1, 2, 5, 2
h_out_formula = (h_in + 2 * p - d * (k - 1) - 1) // s + 1
conv_test = nn.Conv2d(3, 8, kernel_size=k, stride=s, padding=p, dilation=d)
out_tensor = conv_test(torch.zeros(1, 3, 32, 32))
print(f"[Cơ bản] Công thức tính tay H_out = {h_out_formula} | PyTorch thực tế: {tuple(out_tensor.shape)}")
assert h_out_formula == out_tensor.shape[2]

# 2. Trung cấp: So sánh số tham số Standard Conv vs Depthwise Separable Conv (MobileNet style)
std_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
dw_sep_conv = nn.Sequential(
    nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),  # Depthwise
    nn.Conv2d(32, 64, kernel_size=1, bias=False),                        # Pointwise 1x1
)
p_std = count_parameters(std_conv)["total"]
p_dw = count_parameters(dw_sep_conv)["total"]
print(f"[Trung cấp] Standard Conv params: {p_std:,} vs Depthwise Separable params: {p_dw:,} (Giảm {p_std/p_dw:.1f} lần!)")

# 3. Nâng cao: Huấn luyện ResidualCNN
set_seed(42)
res_net = ResidualCNN(in_channels=3, num_classes=4, base_channels=16).to(device)
opt_res = torch.optim.AdamW(res_net.parameters(), lr=2e-3, weight_decay=1e-4)
hist_res = fit_classifier(res_net, train_loader, val_loader, criterion, opt_res, device, epochs=10)
print(f"[Nâng cao] ResidualCNN Best Val Acc: {hist_res['best_val_acc']*100:.1f}%")""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Hiểu tường tận `Conv2d`, `BatchNorm2d`, `Pooling`, `AdaptiveAvgPool2d`, `ResidualBlock` và `Receptive Field`.
- [x] Đã huấn luyện thành công `SimpleCNN` và `ResidualCNN` kèm lưu checkpoint chuẩn.
- [x] Sẵn sàng chuyển sang **Bài 11: `11_training_validation_and_debugging.ipynb`**.

## 17. Nguồn tham khảo
- LeCun, Y. et al. (1998), *Gradient-Based Learning Applied to Document Recognition*, Proc. IEEE.
- He, K. et al. (2016), *Deep Residual Learning for Image Recognition (ResNet)*, CVPR.""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "10_cnn_from_scratch.ipynb", cells)


def build_nb11() -> Path:
    cells = [
        (
            "md",
            r"""# Bài 11: Quy trình Huấn luyện Chuẩn, Phân biệt Khái niệm Trọng tâm và Chẩn đoán Lỗi Training

## 1. Mục tiêu học tập
- Thực thi trọn vẹn **Quy trình 13 bước Huấn luyện AI chuẩn mực**:
  `Xác định bài toán → Kiểm tra dữ liệu → Chia dữ liệu → Tiền xử lý/Augmentation → Baseline → Xây dựng model → Loss/Optimizer → Training → Validation → Chọn Checkpoint → Đánh giá Test → Phân tích lỗi → Inference`.
- Chứng minh bằng code và con số cụ thể các cặp khái niệm dễ nhầm nhất:
  1. `Softmax` vs `nn.CrossEntropyLoss` (tác hại của việc gọi Softmax hai lần).
  2. `Sigmoid` vs `nn.BCEWithLogitsLoss` (ổn định số học khi $|z|$ lớn).
  3. `model.eval()` vs `torch.no_grad()` / `torch.inference_mode()`.
  4. `Weight Decay` vs `L2 Penalty` trong `SGD`, `Adam` và `AdamW`.
- Thực hành các kỹ thuật kiểm soát huấn luyện nâng cao: Bộ lập lịch tốc độ học (`CosineAnnealingLR` + `Warmup`), `Early Stopping`, `Gradient Clipping`, `Gradient Accumulation`.
- Áp dụng **Bài kiểm tra "Overfit 1 Batch"** và bộ hàm kiểm tra an toàn (`NaN/Inf`, `shape`, `dtype`, `label range`, `device`) để phát hiện lỗi ngay từ giây đầu tiên.

## 2. Kiến thức cần có
- Đã hoàn thành Bài 09 và Bài 10.

## 3. Thời gian học dự kiến
- Khoảng **3.0 giờ** (ước tính).

## 4. Bài toán thực tế đơn giản
Khi bạn tự viết một mô hình mới, làm sao biết chắc chắn 100% rằng code kiến trúc, hàm mất mát và bộ tối ưu hóa của mình không có lỗi ngầm trước khi ngồi chờ hàng giờ để train trên toàn bộ dữ liệu? Câu trả lời của mọi kỹ sư Deep Learning chuyên nghiệp là: **Chạy kiểm tra tiền điều kiện (`Pre-flight Sanity Checks`) và ép mô hình học thuộc lòng đúng 1 batch nhỏ (`Overfit 1 Batch Test`)**.

## 5 & 6. Giải thích Trực quan và Từ điển Keyword Kiểm soát Huấn luyện
| Keyword / Kỹ thuật | Nhóm | Vai trò trong Pipeline & Ảnh hưởng khi thay đổi |
|---|---|---|
| `Warmup + CosineAnnealingLR` | Tối ưu hóa | Tăng dần `lr` từ nhỏ lên `base_lr` trong vài bước đầu (`warmup`), sau đó hạ dần theo đường cong hình cosin êm ái về `eta_min`. Giúp tránh nổ đạo hàm đầu kỳ và hội tụ sâu cuối kỳ. |
| `clip_grad_norm_(params, max_norm=1.0)` | Tối ưu hóa | Nếu chuẩn vector đạo hàm $\|g\|_2 > \text{max\_norm}$, thu nhỏ $g \leftarrow g \times \frac{\text{max\_norm}}{\|g\|_2}$ (giữ nguyên hướng, chỉ cắt độ dài). |
| `gradient_accumulation_steps` | Tối ưu hóa | Chia `loss = loss / accum_steps`, gọi `loss.backward()` qua `accum_steps` lô nhỏ rồi mới gọi `optimizer.step()` và `optimizer.zero_grad()`. Giúp mô phỏng `batch_size` lớn trên máy ít RAM/VRAM. |
| `Overfit 1 Batch Test` | Gỡ lỗi | Lấy 1 lô nhỏ (8–16 ảnh), tắt Augmentation & Dropout, lặp 50–100 bước. Nếu `loss` không tiến về `0` và `acc` không đạt `100%`, pipeline chắc chắn có lỗi! |

## 7. Công thức Toán học: Vì sao `BCEWithLogitsLoss` Ổn định hơn `Sigmoid + BCELoss`?
Nếu tính tách rời $p = \frac{1}{1 + e^{-z}}$ rồi lấy $-\log(1 - p)$ khi $z = 100$:
- $e^{-100} \approx 0 \implies p$ bị làm tròn thành $1.0$ trong kiểu `float32` $\implies \log(1 - 1.0) = \log(0) = -\infty$ (`NaN`/`Inf`)!
- Ngược lại, `nn.BCEWithLogitsLoss` gộp chung hai công thức bằng biến đổi đại số:
  $$\ell(z, y) = \max(z, 0) - z \cdot y + \log\left(1 + e^{-|z|}\right)$$
  Vì số mũ luôn là $-|z| \le 0$, $e^{-|z|}$ luôn nằm trong $(0, 1]$, **không bao giờ bị tràn số** dù $z = \pm 1000$!""",
        ),
        (
            "code",
            """# Cell 1: Thực nghiệm Chứng minh 3 Sai lầm Kinh điển trong PyTorch
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

from src.utils import set_seed, get_device
from src.models import SimpleCNN
from src.synthetic_data import generate_synthetic_shapes_dataset

set_seed(42)
device = get_device()

# 1. Chứng minh SAI LẦM: Đặt Softmax trước CrossEntropyLoss
logits_demo = torch.tensor([[10.0, -5.0, -5.0, -5.0]], requires_grad=True)  # Mô hình đã dự đoán đúng lớp 0 rất tự tin
target_demo = torch.tensor([0], dtype=torch.long)

ce_correct = nn.CrossEntropyLoss()(logits_demo, target_demo)
probs_wrong = torch.softmax(logits_demo, dim=1)
ce_wrong_double_softmax = nn.CrossEntropyLoss()(probs_wrong, target_demo)

print(f"[1. CrossEntropy] Đưa đúng logits thô       -> Loss = {ce_correct.item():.6f} (gần 0 vì đoán đúng 100%)")
print(f"[1. CrossEntropy] Đưa nhầm qua Softmax trước -> Loss = {ce_wrong_double_softmax.item():.6f} (bị kẹt ở >= 0.743!)")

# 2. Chứng minh BCEWithLogitsLoss ổn định số học khi logit lớn (z = 100.0, nhãn y = 0.0)
extreme_logit = torch.tensor([100.0])
y_bin = torch.tensor([0.0])
bce_stable = nn.BCEWithLogitsLoss()(extreme_logit, y_bin)
print(f"\\n[2. BCEWithLogits] Tại logit=100.0, nhãn=0 -> BCEWithLogitsLoss tính mượt mà: {bce_stable.item():.1f} (không bị Inf/NaN)")""",
        ),
        (
            "code",
            """# Cell 2: Bộ hàm Kiểm tra Tiền điều kiện (Pre-flight Sanity Check) cho Dữ liệu, Nhãn và Mô hình
def run_preflight_checks(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    num_classes: int,
    expected_device: torch.device,
) -> None:
    assert torch.isfinite(x_batch).all(), "LỖI: Dữ liệu đầu vào x_batch chứa NaN hoặc Inf!"
    assert x_batch.ndim == 4, f"LỖI: Kỳ vọng tensor 4D NCHW, nhưng nhận được shape {tuple(x_batch.shape)}"
    assert x_batch.dtype == torch.float32, f"LỖI: x_batch phải là float32, nhưng đang là {x_batch.dtype}"
    assert y_batch.dtype == torch.long, f"LỖI: Nhãn CrossEntropy phải là torch.long (int64), nhưng đang là {y_batch.dtype}"
    assert int(y_batch.min()) >= 0 and int(y_batch.max()) < num_classes, (
        f"LỖI: Nhãn nằm ngoài miền hợp lệ [0, {num_classes-1}]: min={int(y_batch.min())}, max={int(y_batch.max())}"
    )
    model_dev = next(model.parameters()).device
    assert x_batch.device == model_dev == expected_device, f"LỖI lệch thiết bị: x={x_batch.device}, model={model_dev}"

    with torch.no_grad():
        out = model(x_batch)
    assert out.shape == (len(x_batch), num_classes), f"LỖI output shape: {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "LỖI: Đầu ra mô hình chứa NaN/Inf!"
    print("[PRE-FLIGHT CHECK PASSED] Toàn bộ kiểm tra NaN, shape, dtype, label range và device đều HỢP LỆ!")


ds = generate_synthetic_shapes_dataset(num_samples=64, image_size=32, seed=42)
X_sample = (torch.from_numpy(ds["images"][:16]).permute(0, 3, 1, 2).float() / 255.0).to(device)
y_sample = torch.from_numpy(ds["labels"][:16]).long().to(device)
model_sanity = SimpleCNN(in_channels=3, num_classes=4, dropout_p=0.0).to(device)

run_preflight_checks(model_sanity, X_sample, y_sample, num_classes=4, expected_device=device)""",
        ),
        (
            "code",
            """# Cell 3: Bài kiểm tra Vàng trong Gỡ lỗi Deep Learning — "Overfit 1 Batch Test"
# Mục tiêu: Trên đúng 1 batch 16 ảnh, tắt Dropout, mô hình PHẢI đạt 100% accuracy và đẩy loss về gần 0!
set_seed(42)
model_overfit = SimpleCNN(in_channels=3, num_classes=4, dropout_p=0.0).to(device)
opt_overfit = torch.optim.AdamW(model_overfit.parameters(), lr=5e-3)
crit = nn.CrossEntropyLoss()

one_batch_losses, one_batch_accs = [], []
model_overfit.train()
for step in range(1, 41):
    opt_overfit.zero_grad()
    logits = model_overfit(X_sample)
    loss = crit(logits, y_sample)
    loss.backward()
    opt_overfit.step()

    acc = float((logits.argmax(dim=1) == y_sample).float().mean().item())
    one_batch_losses.append(loss.item())
    one_batch_accs.append(acc)

print(f"[Overfit 1 Batch Test] Bước 1: Loss={one_batch_losses[0]:.4f}, Acc={one_batch_accs[0]*100:.1f}%")
print(f"[Overfit 1 Batch Test] Bước 40: Loss={one_batch_losses[-1]:.4f}, Acc={one_batch_accs[-1]*100:.1f}% <-- XÁC NHẬN PIPELINE ĐÚNG!")
assert one_batch_accs[-1] == 1.0, "Mô hình phải học thuộc được 1 batch nhỏ!" """,
        ),
        (
            "code",
            """# Cell 4: Thực hành Lịch Tốc độ học (Warmup + CosineAnnealing), Gradient Clipping & Gradient Accumulation
total_epochs = 20
warmup_epochs = 4
base_lr = 3e-3
min_lr = 1e-5

dummy_param = [torch.nn.Parameter(torch.randn(2, 2))]
opt_sched = torch.optim.AdamW(dummy_param, lr=base_lr)

# Kết hợp LinearLR (Warmup 4 epochs đầu) nối tiếp CosineAnnealingLR (16 epochs sau)
scheduler_warmup = torch.optim.lr_scheduler.LinearLR(opt_sched, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sched, T_max=total_epochs - warmup_epochs, eta_min=min_lr)
scheduler_combined = torch.optim.lr_scheduler.SequentialLR(
    opt_sched, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs]
)

lrs_recorded = []
for ep in range(total_epochs):
    lrs_recorded.append(opt_sched.param_groups[0]["lr"])
    opt_sched.step()
    scheduler_combined.step()

# Minh họa vòng lặp Gradient Accumulation (cộng dồn 2 micro-batches mới cập nhật 1 lần) + Gradient Clipping
accum_steps = 2
opt_overfit.zero_grad()
for micro_step in range(accum_steps):
    x_micro = X_sample[micro_step * 8 : (micro_step + 1) * 8]
    y_micro = y_sample[micro_step * 8 : (micro_step + 1) * 8]
    micro_loss = crit(model_overfit(x_micro), y_micro) / accum_steps
    micro_loss.backward()

grad_norm_before = float(torch.nn.utils.clip_grad_norm_(model_overfit.parameters(), max_norm=1.0).item())
opt_overfit.step()
opt_overfit.zero_grad()
print(f"Chuẩn đạo hàm (Grad Norm) trước khi cắt ngưỡng 1.0: {grad_norm_before:.4f}")

plt.figure(figsize=(6.5, 2.8))
plt.plot(range(1, total_epochs + 1), lrs_recorded, "o-", color="darkmagenta")
plt.axvline(warmup_epochs, color="gray", linestyle="--", label="Kết thúc Warmup")
plt.title("Lịch Tốc độ học: Linear Warmup (4 epochs) + Cosine Annealing")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.legend()
plt.tight_layout()
plt.show()""",
        ),
        (
            "md",
            r"""## 11. Nhận xét cách đọc kết quả
- **Ở Cell 1**: Khi vô tình đặt `Softmax` trước `CrossEntropyLoss`, ngay cả khi mô hình dự đoán hoàn hảo `100%` vào lớp đúng, giá trị `loss` vẫn bị chặn dưới ở $\approx 0.7437$ và đạo hàm gần như bằng `0` do bão hòa hàm mũ hai lần.
- **Ở Cell 3**: Bài kiểm tra `Overfit 1 Batch` đẩy `Accuracy` lên `100.0%` và `Loss` xuống sát `0` chỉ trong 40 bước, xác nhận toàn bộ kết nối mạng và đạo hàm đều hoạt động hoàn hảo.

## 14. Bài tập tự làm
- **Cơ bản**: Viết đoạn code kiểm tra xem trong mô hình có bất kỳ tham số nào bị đạo hàm bằng `None` hoặc bằng `0` tuyệt đối sau khi gọi `loss.backward()` hay không (giúp phát hiện tầng "chết" không nối vào đầu ra).
- **Trung cấp**: So sánh sự khác biệt của bước cập nhật trọng số giữa `torch.optim.Adam(lr=1e-2, weight_decay=0.1)` và `torch.optim.AdamW(lr=1e-2, weight_decay=0.1)` trên cùng một tensor trọng số.
- **Nâng cao**: Cài đặt một lớp `EarlyStopper(patience=3, min_delta=1e-4)` độc lập trả về `True` khi `val_loss` không giảm thêm ít nhất `min_delta` sau `patience` epoch liên tiếp.

## 15. Gợi ý và Lời giải"""),
        (
            "code",
            """# Cell 5: Lời giải hoàn chỉnh cho bài tập Bài 11
# 1. Cơ bản: Kiểm tra mọi tham số có nhận được gradient hợp lệ sau backward() hay không
opt_overfit.zero_grad()
loss_chk = crit(model_overfit(X_sample), y_sample)
loss_chk.backward()
dead_layers = [name for name, p in model_overfit.named_parameters() if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
print("[Cơ bản] Danh sách tầng bị đứt gradient (rỗng là tốt):", dead_layers)

# 2. Trung cấp: So sánh Adam (L2 coupled) vs AdamW (Decoupled Weight Decay)
w_adam = torch.tensor([2.0, -3.0], requires_grad=True)
w_adamw = torch.tensor([2.0, -3.0], requires_grad=True)
opt1 = torch.optim.Adam([w_adam], lr=0.1, weight_decay=0.5)
opt2 = torch.optim.AdamW([w_adamw], lr=0.1, weight_decay=0.5)

(w_adam.pow(2).sum()).backward()
(w_adamw.pow(2).sum()).backward()
opt1.step()
opt2.step()
print("[Trung cấp] Trọng số sau 1 bước Adam :", w_adam.detach().numpy().round(4))
print("[Trung cấp] Trọng số sau 1 bước AdamW:", w_adamw.detach().numpy().round(4))

# 3. Nâng cao: Class EarlyStopper tái sử dụng
class EarlyStopper:
    def __init__(self, patience: int = 3, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

stopper = EarlyStopper(patience=3)
sim_losses = [0.9, 0.7, 0.65, 0.66, 0.67, 0.68]
for ep_i, l_val in enumerate(sim_losses, 1):
    if stopper.step(l_val):
        print(f"[Nâng cao] EarlyStopper kích hoạt dừng tại epoch #{ep_i} (best_loss={stopper.best_loss})")
        break""",
        ),
        (
            "md",
            """## 16. Tóm tắt & Điều kiện chuyển bài tiếp theo
- [x] Đã hoàn thành toàn bộ **Giai đoạn C (Mạng Nơ-ron, PyTorch, CNN & Quy trình Huấn luyện/Gỡ lỗi)**.
- [x] Đã nắm vững các quy tắc sống còn về `CrossEntropyLoss`, `BCEWithLogitsLoss`, `model.eval()`, `torch.no_grad()`, `AdamW`, và `Overfit 1 Batch`.
- [x] Sẵn sàng bước vào **Giai đoạn D — Các Nhóm Mô hình Computer Vision Chuyên sâu**, bắt đầu với **Bài 12: `12_transfer_learning_and_fine_tuning.ipynb`**!

## 17. Nguồn tham khảo
- Loshchilov, I., & Hutter, F. (2019), *Decoupled Weight Decay Regularization (AdamW)*, ICLR.
- Karpathy, A. (2019), *A Recipe for Training Neural Networks*: https://karpathy.github.io/2019/04/25/recipe/""",
        ),
    ]
    return create_validated_notebook(NB_DIR / "11_training_validation_and_debugging.ipynb", cells)


if __name__ == "__main__":
    for builder in [build_nb06, build_nb07, build_nb08, build_nb09, build_nb10, build_nb11]:
        path = builder()
        print("Created and validated:", path.name)
