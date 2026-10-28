from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import Camera, Slot
from ..schemas import CameraIn
from .common import camera_out, get_camera, get_db, require_worker

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.post("", dependencies=[Depends(require_worker)])
def register_camera(body: CameraIn, db: Session = Depends(get_db)):
    """Worker gọi khi khởi động: tạo mới hoặc cập nhật camera và danh sách ô (giữ trạng thái ô cũ còn tồn tại)."""
    camera = db.get(Camera, body.camera_id)
    if camera is None:
        camera = Camera(id=body.camera_id, frame_w=body.frame_size[0], frame_h=body.frame_size[1])
        db.add(camera)
    camera.frame_w, camera.frame_h = body.frame_size
    existing = {s.slot_id: s for s in camera.slots}
    slots = []
    for i, s in enumerate(body.slots):
        slot = existing.get(s.id) or Slot(slot_id=s.id)
        slot.polygon, slot.position = s.polygon, i
        slots.append(slot)
    camera.slots = slots  # ô không còn trong cấu hình bị xoá (delete-orphan)
    db.commit()
    return camera_out(camera)


@router.get("")
def list_cameras(db: Session = Depends(get_db)):
    return [camera_out(c, with_slots=False) for c in db.scalars(select(Camera).order_by(Camera.id))]


@router.get("/{camera_id}")
def camera_detail(camera_id: str, db: Session = Depends(get_db)):
    return camera_out(get_camera(db, camera_id))


@router.post("/{camera_id}/frame", status_code=204, dependencies=[Depends(require_worker)])
async def upload_frame(camera_id: str, request: Request):
    data = await request.body()
    if not data.startswith(b"\xff\xd8"):
        raise HTTPException(415, "Cần ảnh JPEG")
    request.app.state.frames.put(camera_id, data)


@router.get("/{camera_id}/snapshot.jpg")
def snapshot(camera_id: str, request: Request):
    frame = request.app.state.frames.get(camera_id)
    if frame is None:
        raise HTTPException(404, "Chưa có hình ảnh từ worker")
    return Response(frame.data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/{camera_id}/stream")
def stream(camera_id: str, request: Request):
    """Luồng MJPEG – mở trực tiếp trên trình duyệt hoặc VLC."""
    return StreamingResponse(request.app.state.frames.mjpeg(camera_id, request.is_disconnected),
                             media_type="multipart/x-mixed-replace; boundary=frame")
