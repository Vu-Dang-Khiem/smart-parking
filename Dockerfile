# Demo public trên Hugging Face Spaces (Docker, CPU). Chạy thử trên máy:
#   docker build -t smart-parking . && docker run -p 7860:7860 smart-parking  → http://localhost:7860
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces chạy container bằng user uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir --user -r requirements-deploy.txt
COPY --chown=user . .

# Hai camera demo tự chạy; tải video lên có giới hạn vì ai có link cũng dùng được
ENV SMART_PARKING_INTERNAL_API=http://127.0.0.1:7860 \
    SMART_PARKING_DEMO_CAMERAS="configs/pklot_pucpr.json=data/pklot_demo/video.mp4;configs/cnr_camera8.json=data/cnr_demo/camera8/video.mp4" \
    SMART_PARKING_MAX_UPLOAD_MB=50 \
    SMART_PARKING_MAX_UPLOAD_SECONDS=120 \
    SMART_PARKING_MAX_RUNNING_UPLOADS=2 \
    SMART_PARKING_UPLOAD_MINUTES=15 \
    SMART_PARKING_KEEP_UPLOADS=5

EXPOSE 7860
# Mã worker sinh ngẫu nhiên mỗi lần khởi động: chỉ worker trong container được gửi dữ liệu lên dashboard
CMD ["sh", "-c", "export SMART_PARKING_WORKER_TOKEN=$(python -c 'import secrets; print(secrets.token_hex(16))') && exec uvicorn src.backend.main:app --host 0.0.0.0 --port 7860"]
