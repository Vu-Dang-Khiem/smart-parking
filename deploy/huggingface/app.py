"""Chạy Smart Parking trên Hugging Face Spaces (SDK Gradio, CPU miễn phí): Space chạy `python app.py` ở cổng 7860.

Không dùng giao diện Gradio – file này khởi động đúng backend FastAPI + dashboard của dự án, kèm hai camera demo
và các giới hạn cho chức năng tải video (ai có link cũng dùng được).
"""
import os
import secrets

PORT = 7860
SETTINGS = {
    "SMART_PARKING_INTERNAL_API": f"http://127.0.0.1:{PORT}",
    "SMART_PARKING_DEMO_CAMERAS": "configs/pklot_pucpr.json=data/pklot_demo/video.mp4;"
                                  "configs/cnr_camera8.json=data/cnr_demo/camera8/video.mp4",
    "SMART_PARKING_MAX_UPLOAD_MB": "50",
    "SMART_PARKING_MAX_UPLOAD_SECONDS": "120",
    "SMART_PARKING_MAX_RUNNING_UPLOADS": "2",
    "SMART_PARKING_UPLOAD_MINUTES": "15",
    "SMART_PARKING_KEEP_UPLOADS": "5",
}
for key, value in SETTINGS.items():
    os.environ.setdefault(key, value)
# Mã ngẫu nhiên mỗi lần khởi động: chỉ worker trong Space (kế thừa biến môi trường) được gửi dữ liệu lên dashboard
os.environ["SMART_PARKING_WORKER_TOKEN"] = secrets.token_hex(16)

import uvicorn  # noqa: E402

from src.backend.app import create_app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=PORT)
