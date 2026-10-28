import logging

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import SlotEvent
from ..schemas import EventIn, iso, to_utc
from .common import counts, get_camera, get_db, require_worker

router = APIRouter(prefix="/api/events", tags=["events"])
log = logging.getLogger("backend")


@router.post("", dependencies=[Depends(require_worker)])
async def post_events(body: EventIn, request: Request, db: Session = Depends(get_db)):
    """Worker gửi các ô vừa đổi trạng thái → cập nhật ô, ghi lịch sử, phát cho dashboard qua WebSocket."""
    camera = get_camera(db, body.camera_id)
    slots = {s.slot_id: s for s in camera.slots}
    ts = to_utc(body.ts)
    applied = []
    for c in body.changes:
        slot = slots.get(c.slot_id)
        if slot is None:
            log.warning("Bỏ qua ô không có trong cấu hình: %s/%s", body.camera_id, c.slot_id)
            continue
        slot.status, slot.confidence, slot.updated_at = c.status, c.confidence, ts
        db.add(SlotEvent(camera_id=camera.id, slot_id=c.slot_id, status=c.status, confidence=c.confidence, ts=ts))
        applied.append(c.model_dump())
    db.commit()
    await request.app.state.hub.broadcast(
        {"type": "slots", "camera_id": camera.id, "ts": iso(ts), "changes": applied, "counts": counts(camera)})
    return {"applied": len(applied)}


@router.get("/{camera_id}")
def recent_events(camera_id: str, limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    get_camera(db, camera_id)
    rows = db.scalars(select(SlotEvent).where(SlotEvent.camera_id == camera_id)
                      .order_by(SlotEvent.ts.desc(), SlotEvent.id.desc()).limit(limit))
    return [{"slot_id": e.slot_id, "status": e.status, "confidence": e.confidence, "ts": iso(e.ts)} for e in rows]
