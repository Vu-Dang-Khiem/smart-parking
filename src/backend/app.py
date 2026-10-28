import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import sessionmaker

from .db import DEFAULT_DB, Base, make_engine
from .jobs import WorkerManager
from .live import FrameStore, Hub
from .routers import cameras, events, stats, videos
from .routers.cameras import register_camera
from .schemas import CameraIn, SlotIn
from ..vision.slot_cropper import CameraConfig

STATIC_DIR = Path(__file__).parent / "static"
ROOT = Path(__file__).resolve().parents[2]   # thư mục smart-parking


def first_existing(*paths: Path) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def demo_cameras() -> list[tuple[Path, Path]]:
    """SMART_PARKING_DEMO_CAMERAS="cấu_hình.json=video.mp4;…" → các camera demo tự chạy khi backend khởi động."""
    pairs = [item.split("=", 1) for item in os.environ.get("SMART_PARKING_DEMO_CAMERAS", "").split(";") if "=" in item]
    return [(ROOT / c.strip(), ROOT / v.strip()) for c, v in pairs]


def create_app(db_url: str | None = None, upload_dir: Path | None = None, config_dir: Path | None = None) -> FastAPI:
    engine = make_engine(db_url or os.environ.get("SMART_PARKING_DB", DEFAULT_DB))
    Base.metadata.create_all(engine)

    app = FastAPI(title="Smart Parking", description="API cho hệ thống bãi đỗ xe thông minh")
    app.state.sessionmaker = sessionmaker(engine, expire_on_commit=False)
    app.state.hub = Hub()
    app.state.frames = FrameStore()
    # Video tải lên: lưu ở data/uploads, cấu hình ô ở configs/, mỗi video một tiến trình worker
    app.state.upload_dir = Path(upload_dir or ROOT / "data" / "uploads")
    app.state.config_dir = Path(config_dir or ROOT / "configs")
    app.state.upload_dir.mkdir(parents=True, exist_ok=True)
    app.state.config_dir.mkdir(parents=True, exist_ok=True)
    app.state.workers = WorkerManager(ROOT)
    app.state.classifier_model = Path(os.environ.get("SMART_PARKING_MODEL", ROOT / "models" / "classifier.onnx"))
    # Khi chạy public: mã riêng cho worker và giới hạn tải video (0 = không giới hạn)
    app.state.worker_token = os.environ.get("SMART_PARKING_WORKER_TOKEN") or None
    app.state.limits = {"max_upload_mb": env_int("SMART_PARKING_MAX_UPLOAD_MB", 2048),
                        "max_upload_seconds": env_int("SMART_PARKING_MAX_UPLOAD_SECONDS", 0),
                        "max_running_uploads": env_int("SMART_PARKING_MAX_RUNNING_UPLOADS", 0),
                        "upload_minutes": env_int("SMART_PARKING_UPLOAD_MINUTES", 0),
                        "keep_uploads": env_int("SMART_PARKING_KEEP_UPLOADS", 0)}
    internal_api = os.environ.get("SMART_PARKING_INTERNAL_API")   # sau reverse proxy, cổng ngoài ≠ cổng uvicorn
    app.state.api_url = lambda request: internal_api or f"http://127.0.0.1:{request.url.port or 8000}"

    @lru_cache(maxsize=1)
    def get_proposer():
        # Nạp YOLO lần đầu cần tự tìm ô; COCO tìm xe đỗ ở bãi bất kỳ, PKLot tìm ô ở bãi giống PKLot
        from ..vision.slot_proposer import SlotProposer
        yolo = ROOT / "models" / "yolo"
        return SlotProposer(first_existing(yolo / "yolo11n_coco_1280.onnx", yolo / "yolov8n_coco_1280.onnx"),
                            first_existing(yolo / "yolo11n_pklot.onnx", yolo / "yolov8n_pklot.onnx"))

    app.state.get_proposer = get_proposer

    def start_demo_cameras():
        # Đăng ký trước để dashboard có camera ngay, rồi chạy worker phát lặp video demo
        for config_path, video in demo_cameras():
            config = CameraConfig.load(config_path)
            with app.state.sessionmaker() as db:
                register_camera(CameraIn(camera_id=config.camera_id, frame_size=tuple(config.frame_size),
                                         slots=[SlotIn(id=s.id, polygon=s.polygon.tolist()) for s in config.slots]), db)
            app.state.workers.start(f"demo-{config.camera_id}",
                                    ["--video", str(video), "--config", str(config_path),
                                     "--model", str(app.state.classifier_model),
                                     "--api", internal_api or "http://127.0.0.1:8000", "--loop"],
                                    app.state.upload_dir / f"demo-{config.camera_id}.log")

    app.add_event_handler("startup", start_demo_cameras)
    app.add_event_handler("shutdown", app.state.workers.stop_all)

    for module in (cameras, events, stats, videos):
        app.include_router(module.router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def revalidate_static(request, call_next):
        # Trình duyệt luôn hỏi lại server (ETag) nên sửa dashboard xong chỉ cần tải lại trang
        response = await call_next(request)
        if request.url.path in ("/", "/upload") or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/upload", include_in_schema=False)
    def upload_page():
        return FileResponse(STATIC_DIR / "upload.html")

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        await app.state.hub.connect(ws)
        try:
            while True:
                await ws.receive_text()  # dashboard không gửi gì; chỉ giữ kết nối
        except WebSocketDisconnect:
            app.state.hub.disconnect(ws)

    return app
