from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import SlotEvent
from ..schemas import iso
from .common import get_camera, get_db

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/{camera_id}/occupancy")
def occupancy(camera_id: str, hours: float = Query(24, gt=0, le=24 * 31),
              bucket_minutes: int = Query(10, ge=1, le=24 * 60), db: Session = Depends(get_db)):
    """Tỷ lệ lấp đầy theo thời gian, dựng lại từ lịch sử sự kiện: mỗi mốc lấy trạng thái cuối cùng của từng ô."""
    camera = get_camera(db, camera_id)
    total = len(camera.slots)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    start = now - timedelta(hours=hours)
    step = timedelta(minutes=bucket_minutes)
    events = list(db.scalars(select(SlotEvent).where(SlotEvent.camera_id == camera_id, SlotEvent.ts <= now)
                             .order_by(SlotEvent.ts, SlotEvent.id)))
    state: dict[str, str] = {}
    points, i = [], 0
    t = start + step
    while t < now + step:
        while i < len(events) and events[i].ts <= t:
            state[events[i].slot_id] = events[i].status
            i += 1
        known = len(state)
        if known:
            occ = sum(v == "occupied" for v in state.values())
            points.append({"t": iso(min(t, now)), "occupied": occ, "known": known,
                           "rate": round(occ / total, 4) if total else 0.0})
        t += step
    return {"camera_id": camera_id, "total": total, "hours": hours, "points": points}
