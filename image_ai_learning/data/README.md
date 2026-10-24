# Hướng dẫn Thư mục Dữ liệu (`data/README.md`)

Thư mục `data/` dùng để lưu trữ các tập dữ liệu mẫu hoặc bộ ảnh cá nhân của bạn khi thực hành.

## 1. Dữ liệu Mặc định trong Khóa học (Hoàn toàn Offline & Có Giấy phép Rõ ràng)

Để đảm bảo bạn có thể chạy toàn bộ 23 notebook **ngay cả khi không có kết nối Internet hoặc không có GPU**, khóa học cung cấp hai nguồn dữ liệu an toàn:

1. **Dữ liệu ảnh tổng hợp có kiểm soát (`src/synthetic_data.py`)**:
   - Tạo trực tiếp bằng code NumPy/OpenCV với hạt giống ngẫu nhiên cố định (`seed=42`).
   - Bao gồm 4 bộ dữ liệu chuyên biệt:
     - Phân loại ảnh 4 lớp (`circle`, `square`, `triangle`, `cross`) kèm thông tin `group_ids`.
     - Phát hiện vật thể đa đối tượng (`boxes` định dạng `[x_min, y_min, x_max, y_max]` + `labels`).
     - Phân vùng ảnh ngữ nghĩa (`images` + `masks` mức điểm ảnh).
     - Kiểm tra lỗi bề mặt công nghiệp (`normal` vs `anomalous` kèm mặt nạ vết xước/đốm lỗi).
   - **Giới hạn**: Dữ liệu tổng hợp có phân phối hình học đơn giản hơn ảnh chụp tự nhiên; chỉ dùng để học thuật toán và kiểm chứng pipeline, không dùng để khẳng định độ chính xác trên dữ liệu thực tế phức tạp.

2. **Bộ dữ liệu chữ số viết tay `load_digits` từ `scikit-learn`**:
   - Nguồn gốc: UCI Machine Learning Repository (Optical Recognition of Handwritten Digits), tích hợp sẵn offline trong `scikit-learn` (Giấy phép BSD-3-Clause).
   - Kích thước: `1797` ảnh thang xám `8x8` thuộc 10 lớp chữ số `0..9`.

## 2. Cách Đưa Bộ Ảnh Cá nhân của Bạn vào Học và Huấn luyện

Khi thực hành ở `notebooks/12_transfer_learning_and_fine_tuning.ipynb` hoặc `notebooks/22_capstone_project.ipynb`, bạn có thể đặt ảnh của mình theo cấu trúc thư mục chuẩn `ImageFolder`:

```text
image_ai_learning/data/my_custom_dataset/
├── train/
│   ├── class_a/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── class_b/
│       ├── img_101.jpg
│       └── img_102.jpg
├── val/
│   ├── class_a/
│   └── class_b/
└── test/
    ├── class_a/
    └── class_b/
```

Lưu ý quan trọng:
- Kiểm tra bản quyền và quyền riêng tư của ảnh trước khi sử dụng.
- Không trộn ảnh chụp liên tiếp của cùng một vật thể vào cả `train` và `test` (xem bài `05_datasets_and_data_leakage.ipynb`).
