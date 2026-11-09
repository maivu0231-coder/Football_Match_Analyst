# ⚽ Football Analyst AI - Hệ Thống Phân Tích Video Bóng Đá Tự Động

Dự án áp dụng **Computer Vision** và **Deep Learning** để tự động theo dõi, phân tích chiến thuật và đo lường thể lực của cầu thủ cũng như chuyển động của bóng trong các trận đấu bóng đá từ video quay phát sóng truyền hình.

---

## 🌟 Các Tính Năng Nổi Bật

1. **Phát hiện & Theo dõi Đối tượng (Object Detection & Multi-Object Tracking)**
   - Nhận diện 4 lớp đối tượng chính: Cầu thủ (`player`), Thủ môn (`goalkeeper`), Trọng tài (`referee`) và Bóng (`ball`) bằng mô hình **YOLOv8** tùy chỉnh.
   - Duy trì ID cầu thủ ổn định qua từng khung hình với thuật toán **ByteTrack** (thông qua thư viện `supervision`).
   - Tích hợp trích xuất khung xương, khớp nối cơ thể (Keypoints) với **YOLOv8x-pose**.

2. **Tự động Phân Biệt Màu Áo & Chia Đội (Team Assignment & Color Clustering)**
   - Cắt vùng nửa thân trên (áo đấu) của từng cầu thủ và lọc bỏ màu nền sân cỏ.
   - Ứng dụng giải thuật gom cụm **K-Means Clustering** trên không gian màu RGB để tự động xác định 2 màu áo chủ đạo của 2 đội và gán màu đại diện cho từng cầu thủ.

3. **Bù Trừ Chuyển Động Camera (Camera Movement Estimation)**
   - Sử dụng giải thuật **Lucas-Kanade Optical Flow** (`cv2.calcOpticalFlowPyrLK`) trên các đặc trưng ổn định (`goodFeaturesToTrack`) ở hai bên rìa khung hình.
   - Tính toán vector dịch chuyển góc quay camera tích lũy qua từng khung hình để bù trừ tọa độ cầu thủ, phản ánh đúng chuyển động thực tế trên sân.

4. **Chuyển Đổi Góc Nhìn Phối Cảnh (Perspective Transform / Homography)**
   - Ánh xạ tọa độ pixel từ góc nhìn camera phối cảnh (2D perspective frame) sang tọa độ phẳng chuẩn tính theo mét trên sân đấu thực tế (2D Pitch Dimensions) bằng ma trận Homography (`cv2.getPerspectiveTransform`).

5. **Đo Lường Khoảng Cách & Vận Tốc Cầu Thủ (Distance & Velocity Measurement)**
   - Tính toán tổng quãng đường di chuyển (mét) và vận tốc tức thời (km/h) của mỗi cầu thủ theo thời gian thực dựa trên vị trí đã hiệu chỉnh góc nhìn.
   - Sử dụng kỹ thuật cửa sổ trượt (sliding window) theo frame-rate để làm mượt chỉ số đo đạc.

6. **Khôi Phục Quỹ Đạo Bóng (Ball Interpolation)**
   - Xử lý tình trạng bóng bị che khuất hoặc bay nhanh mất dấu (occlusion) bằng phương pháp nội suy chuỗi thời gian (`pandas interpolation` & `bfill`).
7. **Tối Ưu Hóa & Caching (Stubs & Generator Stream)**
   - Hỗ trợ lưu cache trung gian dạng pickle (`.pkl` trong thư mục `Stubs/`) giúp quá trình tinh chỉnh thuật toán hoặc vẽ lại video diễn ra tức thì mà không cần inference lại toàn bộ mạng deep learning.
   - Cơ chế đọc và ghi video dạng **Frame Generator Streaming** giúp tiết kiệm tối đa bộ nhớ RAM khi xử lý video dung lượng lớn và độ phân giải cao.

---

## 🏗️ Kiến Trúc Luồng Xử Lý (Pipeline Architecture)

```
[Video Đầu Vào (.mp4)]
         │
         ▼
[1. YOLOv8 Detection + ByteTrack + YOLO-Pose] ──► Lưu Stubs (stub.pkl)
         │
         ▼
[2. K-Means Jersey Color Clustering] ──────────► Phân chia Đội 1 & Đội 2
         │
         ▼
[3. Ball Trajectory Interpolation / Kalman] ───► Khôi phục quỹ đạo bóng
         │
         ▼
[4. Optical Flow Camera Movement Tracker] ─────► Bù trừ rung lắc/lia máy
         │
         ▼
[5. Homography Perspective Transform] ─────────► Tọa độ mét trên mặt sân
         │
         ▼
[6. Measurement Engine] ───────────────────────► Vận tốc (km/h) & Quãng đường (m)
         │
         ▼
[7. Frame Annotation & Video Streamer] ────────► [Video Kết Quả (.mp4)]
```

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
Football_Analyst_Project/
├── Camera_movement/               # Xử lý chuyển động camera
│   ├── __init__.py
│   └── camera_movement.py         # Optical flow Lucas-Kanade & bù sai số dời tọa độ
├── Color_Analyst/                 # Phân tích màu sắc & thử nghiệm K-Means
│   └── color_analyst.ipynb
├── Distance_And_Velocity/         # Tính toán chỉ số vật lý của cầu thủ
│   ├── __init__.py
│   └── distance_and_velocity.py   # Đo tốc độ (km/h), quãng đường (m) & vẽ overlay
├── Inputs/                        # Video đầu vào (Match_Video.mp4, ...)
├── Models/                        # Chứa trọng số các mô hình Deep Learning
│   ├── best.pt                    # Trọng số YOLOv8 custom cho phát hiện bóng/cầu thủ
│   ├── last.pt
│   └── yolov8x-pose.pt            # Trọng số YOLOv8 pose cho phát hiện khung xương
├── Outputs/                       # Video và hình ảnh sau khi phân tích hoàn tất
├── Stubs/                         # Dữ liệu cache pickle để tăng tốc thử nghiệm
│   ├── camera_movement_stub.pkl
│   └── stub.pkl
├── Team_Assigner/                 # Phân loại đội theo màu áo thi đấu
│   ├── __init__.py
│   └── team_assigner.py
├── Tracker/                       # Tracking đa đối tượng & trực quan hóa
│   ├── __init__.py
│   ├── Tracker.py                 # Tích hợp ByteTrack, YOLO và vẽ annotations
│   └── fix.py
├── Training/                      # Huấn luyện mô hình YOLO
│   ├── Dataset/                   # Dữ liệu huấn luyện (Roboflow YOLO format)
│   ├── model_training.ipynb
│   ├── training.py                # Script huấn luyện với Ultralytics
│   └── yolov8m.pt
├── Utils/                         # Các hàm tiện ích hỗ trợ
│   ├── __init__.py
│   ├── compute_ultils.py          # Tính tọa độ tâm, vị trí chân cầu thủ
│   └── video_ultils.py            # Đọc, ghi và stream video bằng OpenCV
├── main.py                        # Điểm khởi chạy chính toàn bộ quy trình phân tích
├── requirements.txt               # Danh sách thư viện phụ thuộc
├── .gitignore                     # Cấu hình bỏ qua tệp nhị phân, video, weights
└── README.md                      # Tài liệu hướng dẫn dự án
```

---

## ⚙️ Yêu Cầu Môi Trường & Cài Đặt

### 1. Yêu cầu hệ thống
- **Hệ điều hành**: Windows 10/11, Ubuntu 20.04+ hoặc macOS.
- **Python**: Phiên bản 3.10 - 3.12 (khuyến nghị `Python 3.12`).
- **GPU**: NVIDIA GPU với CUDA 11.8 / 12.x (khuyến nghị để tăng tốc độ inference mô hình).

### 2. Cài đặt môi trường

1. **Khởi tạo và kích hoạt môi trường ảo (Virtual Environment):**
   ```bash
   # Tạo virtual environment
   python -m venv venv

   # Kích hoạt trên Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # Hoặc trên Linux/macOS:
   source venv/bin/activate
   ```

2. **Cài đặt PyTorch phù hợp với phần cứng:**
   - Nếu có GPU NVIDIA (CUDA 12.1):
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     ```
   - Nếu chỉ dùng CPU:
     ```bash
     pip install torch torchvision
     ```

3. **Cài đặt các gói phụ thuộc từ `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Chạy phân tích video đầy đủ

Mở tệp `main.py` và kiểm tra lại đường dẫn video đầu vào và đầu ra nếu cần:

```python
input_video_path = "Personal_Project/Football_Analyst_Project/Inputs/Match_Video.mp4"
output_video_path = "Personal_Project/Football_Analyst_Project/Outputs/Match_Video_R.mp4"
```

Sau đó chạy script:
```bash
python main.py
```

*Lưu ý về Stub Cache*: 
- Trong `main.py`, tham số `use_stub=True` được kích hoạt mặc định.
- Nếu bạn thay đổi video đầu vào hoặc muốn nhận diện lại từ đầu, hãy đặt `use_stub=False` hoặc xóa các tệp `.pkl` trong thư mục `Stubs/`.

### 2. Huấn luyện mô hình YOLO phát hiện đối tượng mới

Để tự huấn luyện mô hình với dataset riêng hoặc dataset từ Roboflow:
```bash
python Training/training.py
```
Cấu hình tập dữ liệu nằm tại `Training/Dataset/data.yaml`.

---

## 🎨 Minh Họa Trực Quan Hóa (Visualization Elements)

Khi video được xuất ra tại thư mục `Outputs/`, các thông tin sau sẽ được vẽ trực tiếp lên khung hình:
- **Hình Elip dưới chân cầu thủ**: Được tô viền theo màu sắc đại diện của đội bóng (Đội 1 hoặc Đội 2), đi kèm số ID của cầu thủ.
- **Hình tam giác ngược màu vàng**: Đánh dấu vị trí bóng trên sân.
- **Chỉ số Vận Tốc & Quãng Đường**: Hiển thị dạng chữ số ngay bên dưới cầu thủ (ví dụ: `21.45 km/h`, `45.20 m`).
- **Bảng thông tin Camera Movement (Góc trên bên trái)**: Hiển thị độ dịch chuyển pixel `X` và `Y` của camera trong khung hình hiện tại.

---

## 📜 Giấy Phép & Nguồn Tham Khảo

- **Dataset**: Cung cấp theo giấy phép CC BY 4.0 từ Roboflow Universe (`football-players-detection-3zvbc`).
- **YOLOv8**: Bản quyền thuộc về [Ultralytics](https://github.com/ultralytics/ultralytics).
- **ByteTrack & Supervision**: [Roboflow Supervision](https://github.com/roboflow/supervision).
