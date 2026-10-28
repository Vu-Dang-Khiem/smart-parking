"""Tải video lên → tự tìm vị trí ô đỗ → người dùng sửa → chạy worker theo dõi video đó."""
import json
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ...vision.slot_cropper import CameraConfig, Slot
from ..db import Camera, SlotEvent
from ..schemas import CameraIn, SlotIn
from .cameras import register_camera

router = APIRouter(prefix="/api/videos", tags=["videos"])
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
WORKER_FPS = 2   # số khung worker xử lý mỗi giây (mặc định của worker)


class StartIn(BaseModel):
    slots: list[SlotIn] = Field(min_length=1)


def remove_upload(request: Request, folder: Path) -> None:
    """Xoá video tải lên cùng cấu hình ô và camera của nó trên dashboard."""
    camera_id = f"upload-{folder.name}"
    request.app.state.workers.stop(folder.name)
    (request.app.state.config_dir / f"{camera_id}.json").unlink(missing_ok=True)
    with request.app.state.sessionmaker() as db:
        db.query(SlotEvent).filter(SlotEvent.camera_id == camera_id).delete()
        if camera := db.get(Camera, camera_id):
            db.delete(camera)
        db.commit()
    shutil.rmtree(folder, ignore_errors=True)


def prune_uploads(request: Request, keep: int) -> None:
    """Chỉ giữ `keep` video mới nhất (video đang chạy luôn được giữ)."""
    folders = sorted((p.parent for p in request.app.state.upload_dir.glob("*/meta.json")),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for folder in folders[keep:]:
        if not request.app.state.workers.running(folder.name):
            remove_upload(request, folder)


@router.get("/limits")
def upload_limits(request: Request):
    """Giới hạn tải lên (0 = không giới hạn) để trang tải video hiển thị cho người dùng."""
    return request.app.state.limits


def load_meta(request: Request, video_id: str) -> tuple[Path, dict]:
    folder = request.app.state.upload_dir / video_id
    meta_path = folder / "meta.json"
    if not video_id.isalnum() or not meta_path.exists():
        raise HTTPException(404, f"Không có video '{video_id}'")
    return folder, json.loads(meta_path.read_text(encoding="utf-8"))


@router.post("")
async def upload_video(request: Request, filename: str = Query(..., min_length=1)):
    """Nhận file video gửi thẳng trong body (không cần multipart) và kiểm tra đọc được."""
    ext = Path(filename).suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(415, f"Định dạng {ext or '(không có đuôi)'} không hỗ trợ; dùng {', '.join(sorted(VIDEO_EXT))}")
    limits = request.app.state.limits
    max_bytes = limits["max_upload_mb"] * 1024 ** 2
    video_id = uuid.uuid4().hex[:8]
    folder = request.app.state.upload_dir / video_id
    folder.mkdir(parents=True)
    path, size = folder / f"video{ext}", 0
    with open(path, "wb") as f:
        async for chunk in request.stream():
            size += len(chunk)
            if size > max_bytes:
                f.close()
                shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(413, f"Video lớn hơn {limits['max_upload_mb']} MB")
            f.write(chunk)
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, "Không đọc được video – file hỏng hoặc codec không hỗ trợ")
    if limits["max_upload_seconds"] and fps > 0 and frames / fps > limits["max_upload_seconds"]:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(413, f"Video dài {frames / fps:.0f} giây, tối đa {limits['max_upload_seconds']} giây")
    meta = {"video_id": video_id, "name": Path(filename).name, "file": path.name, "size_mb": round(size / 1e6, 1),
            "frame_size": [frame.shape[1], frame.shape[0]], "fps": round(fps, 2), "frames": frames,
            "camera_id": f"upload-{video_id}"}
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    if limits["keep_uploads"]:
        prune_uploads(request, limits["keep_uploads"])
    return meta


@router.get("")
def list_videos(request: Request):
    items = []
    for meta_path in sorted(request.app.state.upload_dir.glob("*/meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["running"] = request.app.state.workers.running(meta["video_id"])
        items.append(meta)
    return items


@router.get("/{video_id}/frame.jpg")
def video_frame(video_id: str, request: Request, position: float = Query(0.5, ge=0, le=1)):
    """Một khung hình của video (position = 0 → đầu, 1 → cuối) để hiển thị khi sửa ô."""
    folder, meta = load_meta(request, video_id)
    cap = cv2.VideoCapture(str(folder / meta["file"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(position * max(meta["frames"] - 1, 0)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(500, "Không đọc được khung hình")
    return Response(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), media_type="image/jpeg")


@router.get("/{video_id}/slots")
def saved_slots(video_id: str, request: Request):
    """Ô đã lưu ở lần bắt đầu theo dõi trước (rỗng nếu chưa có)."""
    _, meta = load_meta(request, video_id)
    path = request.app.state.config_dir / f"{meta['camera_id']}.json"
    if not path.exists():
        return {"frame_size": meta["frame_size"], "slots": []}
    config = CameraConfig.load(path)
    return {"frame_size": list(config.frame_size), "slots": [{"id": s.id, "polygon": s.polygon.tolist()} for s in config.slots]}


@router.post("/{video_id}/propose")
async def propose_slots(video_id: str, request: Request):
    """Tự tìm vị trí ô đỗ bằng YOLO (mất khoảng 20–40 giây)."""
    folder, meta = load_meta(request, video_id)
    proposer = request.app.state.get_proposer()
    config = await run_in_threadpool(proposer.propose, folder / meta["file"], 24, meta["camera_id"])
    return {"frame_size": list(config.frame_size),
            "slots": [{"id": s.id, "polygon": np.round(s.polygon, 1).tolist()} for s in config.slots]}


@router.post("/{video_id}/start")
def start_monitoring(video_id: str, body: StartIn, request: Request):
    """Lưu cấu hình ô, đăng ký camera rồi chạy worker phát lặp video theo thời gian thực."""
    folder, meta = load_meta(request, video_id)
    camera_id = meta["camera_id"]
    config = CameraConfig(camera_id, tuple(meta["frame_size"]),
                          [Slot(s.id, np.asarray(s.polygon, dtype=np.float32)) for s in body.slots])
    config_path = request.app.state.config_dir / f"{camera_id}.json"
    config.save(config_path)
    with request.app.state.sessionmaker() as db:   # đăng ký trước để dashboard mở ra là thấy camera ngay
        register_camera(CameraIn(camera_id=camera_id, frame_size=tuple(meta["frame_size"]),
                                 slots=[SlotIn(id=s.id, polygon=s.polygon) for s in body.slots]), db)
    workers, limits = request.app.state.workers, request.app.state.limits
    if limits["max_running_uploads"]:   # quá số video chạy cùng lúc → dừng video tải lên đã chạy lâu nhất
        uploads = [k for k in workers.running_keys() if not k.startswith("demo-") and k != video_id]
        for key in uploads[:max(0, len(uploads) - limits["max_running_uploads"] + 1)]:
            workers.stop(key)
    args = ["--video", str(folder / meta["file"]), "--config", str(config_path),
            "--model", str(request.app.state.classifier_model), "--api", request.app.state.api_url(request), "--loop"]
    if limits["upload_minutes"]:        # worker tự dừng sau số phút cho phép
        args += ["--fps", str(WORKER_FPS), "--max-frames", str(limits["upload_minutes"] * 60 * WORKER_FPS)]
    workers.start(video_id, args, folder / "worker.log")
    return {"camera_id": camera_id, "slots": len(body.slots)}


@router.post("/{video_id}/stop")
def stop_monitoring(video_id: str, request: Request):
    load_meta(request, video_id)
    request.app.state.workers.stop(video_id)
    return {"stopped": video_id}
