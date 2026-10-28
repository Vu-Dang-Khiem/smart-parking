import secrets

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from ..db import Camera
from ..schemas import iso


def get_db(request: Request):
    with request.app.state.sessionmaker() as session:
        yield session


def require_worker(request: Request) -> None:
    """Khi chạy public (đặt SMART_PARKING_WORKER_TOKEN), chỉ worker của hệ thống được gửi dữ liệu lên dashboard."""
    token = request.app.state.worker_token
    if token and not secrets.compare_digest(request.headers.get("X-Worker-Token", ""), token):
        raise HTTPException(403, "Chỉ worker của hệ thống được gửi dữ liệu")


def get_camera(db: Session, camera_id: str) -> Camera:
    camera = db.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(404, f"Không có camera '{camera_id}'")
    return camera


def counts(camera: Camera) -> dict:
    statuses = [s.status for s in camera.slots]
    return {"total": len(statuses), "vacant": statuses.count("vacant"),
            "occupied": statuses.count("occupied"), "unknown": statuses.count(None)}


def camera_out(camera: Camera, with_slots: bool = True) -> dict:
    out = {"camera_id": camera.id, "frame_size": [camera.frame_w, camera.frame_h], "counts": counts(camera)}
    if with_slots:
        out["slots"] = [{"id": s.slot_id, "polygon": s.polygon, "status": s.status, "confidence": s.confidence,
                         "updated_at": iso(s.updated_at)} for s in camera.slots]
    return out
