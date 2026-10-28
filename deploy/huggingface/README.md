---
title: Smart Parking
emoji: 🅿️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.1
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Phát hiện chỗ đỗ trống realtime bằng MobileNetV3 trên CPU
---

# Smart Parking – hệ thống bãi đỗ xe thông minh

Bài tập lớn môn Thị giác máy tính – Trường Đại học Sư phạm Kỹ thuật Hưng Yên.

- **Dashboard realtime** (`/`): hai camera demo phát lặp video bãi xe thật (PKLot, CNR-EXT); ô trống viền xanh, có xe viền đỏ.
- **Tải video lên** (`/upload`): hệ thống tự đề xuất vị trí ô đỗ bằng YOLO, sửa lại rồi bắt đầu theo dõi.
  Bản public giới hạn video ≤ 50 MB, ≤ 2 phút, theo dõi tối đa 15 phút.
- **API** (`/docs`).

Mô hình chính: MobileNetV3-Small (ONNX Runtime trên CPU) + làm mượt theo thời gian 4/5 khung.
Mã nguồn, notebook huấn luyện và kết quả đánh giá: xem repo GitHub của dự án.

Dữ liệu demo: PKLot (CC BY 4.0, de Almeida et al. 2015) và CNRPark+EXT (ODbL v1.0, Amato et al. 2017).
Mô hình YOLO (Ultralytics) theo giấy phép AGPL-3.0.
