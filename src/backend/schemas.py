from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class SlotIn(BaseModel):
    id: str
    polygon: list[list[float]] = Field(min_length=3)


class CameraIn(BaseModel):
    camera_id: str
    frame_size: tuple[int, int]
    slots: list[SlotIn]


class ChangeIn(BaseModel):
    slot_id: str
    status: Literal["vacant", "occupied"]
    confidence: float = Field(ge=0, le=1)


class EventIn(BaseModel):
    camera_id: str
    ts: datetime
    changes: list[ChangeIn]


def to_utc(ts: datetime) -> datetime:
    """Datetime bất kỳ → UTC không kèm múi giờ (cách lưu trong SQLite). Không có múi giờ thì coi là giờ máy."""
    if ts.tzinfo is None:
        ts = ts.astimezone()
    return ts.astimezone(timezone.utc).replace(tzinfo=None)


def iso(ts: datetime | None) -> str | None:
    return ts.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z") if ts else None
