# ♻️ HỆ THỐNG PHÂN LOẠI RÁC THẢI THÔNG MINH SỬ DỤNG CNN

## Giới thiệu

Dự án xây dựng hệ thống phân loại rác thải bằng Deep Learning nhằm nhận diện ảnh rác thành 2 nhóm:

* 🥬 Rác hữu cơ (Organic Waste)
* ♻️ Rác tái chế (Recyclable Waste)

Mô hình được xây dựng bằng PyTorch theo kiến trúc ResNet Custom và triển khai thành Web App bằng FastAPI.

---

## Giao diện hệ thống

### Trang chủ

![Trang chủ](Trang%20chủ%20test%201%20ảnh.png)

### Nhận diện ảnh đơn - Rác hữu cơ

![Organic](Kết%20quả%20nhận%20diện%20rác%20hữu%20cơ.png)

### Nhận diện ảnh đơn - Rác tái chế

![Recyclable](Kết%20quả%20nhận%20diện%20rác%20tái%20chế.png)

---

## Nhận diện hàng loạt

### Chọn nhiều ảnh

![Batch Upload](Giao%20diện%20chọn%20nhiều%20ảnh.png)

### Kết quả nhận diện hàng loạt

![Batch Result](Kết%20quả%20nhận%20diện%20hàng%20loạt.png)

---

## Bộ dữ liệu

Nguồn dữ liệu:

**Waste Classification Dataset (Kaggle)**

Thông tin dữ liệu:

* Tổng số ảnh ban đầu: 25.077
* Rác hữu cơ (Organic): 13.966 ảnh
* Rác tái chế (Recyclable): 11.111 ảnh
* Sau khi loại bỏ ảnh trùng bằng MD5: 24.751 ảnh

Chia dữ liệu:

* Train: 19.800 ảnh
* Validation: 2.475 ảnh
* Test: 2.476 ảnh

---

## Tiền xử lý dữ liệu

Các bước tiền xử lý:

* Loại bỏ ảnh trùng bằng MD5 Hash
* Resize ảnh về kích thước 128×128
* Chuẩn hóa theo ImageNet:

  * Mean = [0.485, 0.456, 0.406]
  * Std = [0.229, 0.224, 0.225]

Data Augmentation áp dụng cho tập Train:

* Random Horizontal Flip (p=0.5)
* Random Rotation ±15°
* Color Jitter (Brightness, Contrast, Saturation ±10%)

---

## Kiến trúc mô hình

Mô hình sử dụng kiến trúc **AdvancedGarbageCNN (ResNet Custom)** được xây dựng hoàn toàn bằng PyTorch.

### Cấu trúc mạng

* Conv2D + BatchNorm + ReLU
* Residual Block 1: 32 → 64
* Residual Block 2: 64 → 128
* Residual Block 3: 128 → 256
* Residual Block 4: 256 → 512
* Global Average Pooling (GAP)
* Dropout (0.4)
* Fully Connected Layer
* Softmax Output (2 lớp)

### Tổng số tham số

**4.881.954 parameters**

---

## Cấu hình huấn luyện

* Epochs: 50
* Batch Size: 64
* Optimizer: AdamW
* Learning Rate: 2e-4
* Weight Decay: 1e-3
* Loss Function: CrossEntropyLoss + Label Smoothing (0.1)
* Scheduler: CosineAnnealingLR
* Warmup: 3 Epoch đầu
* Early Stopping: patience = 7
* Gradient Clipping: max_norm = 1.0

---

## Kết quả huấn luyện

* Train Accuracy: 95.13%
* Validation Accuracy: 92.20%
* Best Validation Loss: 0.3383
* Early Stopping tại Epoch 33

### Biểu đồ Loss và Accuracy

![Training Curve](Biểu%20đồ%20training.png)

---

## Kết quả đánh giá

| Chỉ số    | Giá trị |
| --------- | ------- |
| Accuracy  | 92.20%  |
| Precision | 92%     |
| Recall    | 92%     |
| F1-score  | 92%     |

### Ma trận nhầm lẫn

![Confusion Matrix](Confusion%20Matrix.png)

---

## Triển khai Web Application

Ứng dụng được phát triển bằng FastAPI.

### Chức năng 1: Nhận diện ảnh đơn

* Upload 1 ảnh
* Trả về nhãn dự đoán
* Hiển thị độ tin cậy (%)

### Chức năng 2: Nhận diện hàng loạt

* Upload nhiều ảnh cùng lúc
* Thống kê số lượng ảnh theo từng lớp
* Hiển thị xác suất dự đoán cho từng ảnh

### Chức năng 3: Xuất báo cáo PDF

* Tự động tạo báo cáo kết quả
* Xuất file PDF bằng wkhtmltopdf

---

## Công nghệ sử dụng

* Python
* PyTorch
* TorchVision
* FastAPI
* HTML/CSS/JavaScript
* Matplotlib
* Scikit-learn
* wkhtmltopdf

---

## Kết quả đạt được

* Xây dựng thành công mô hình phân loại rác thải 2 lớp.
* Đạt Validation Accuracy 92.20%.
* Triển khai Web App phục vụ nhận diện ảnh đơn và hàng loạt.
* Hỗ trợ xuất báo cáo PDF tự động.

---

## Tác giả

**Lê Thị Trà My**

MSSV: 2045230063

Ngành: Khoa học Dữ liệu

Trường Đại học Công Thương Thành phố Hồ Chí Minh (HUIT)

Giảng viên hướng dẫn: **ThS. Trần Đình Toàn**

---

## Tài liệu tham khảo

1. He et al. (2016) - Deep Residual Learning for Image Recognition
2. LeCun et al. (2015) - Deep Learning
3. Ioffe & Szegedy (2015) - Batch Normalization
4. Srivastava et al. (2014) - Dropout
5. Loshchilov & Hutter (2019) - AdamW
6. Paszke et al. (2019) - PyTorch
7. FastAPI Documentation
8. Kaggle Waste Classification Dataset
