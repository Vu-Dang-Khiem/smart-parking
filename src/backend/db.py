from datetime import datetime
from pathlib import Path

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

DEFAULT_DB = "sqlite:///data/smart_parking.db"


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    frame_w: Mapped[int]
    frame_h: Mapped[int]
    slots: Mapped[list["Slot"]] = relationship(back_populates="camera", cascade="all, delete-orphan",
                                               order_by="Slot.position")


class Slot(Base):
    __tablename__ = "slots"
    __table_args__ = (UniqueConstraint("camera_id", "slot_id"),)

    pk: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"))
    slot_id: Mapped[str] = mapped_column(String(64))
    position: Mapped[int] = mapped_column(default=0)
    polygon: Mapped[list] = mapped_column(JSON)
    status: Mapped[str | None] = mapped_column(String(16), default=None)  # vacant / occupied / None = chưa rõ
    confidence: Mapped[float | None] = mapped_column(Float, default=None)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    camera: Mapped[Camera] = relationship(back_populates="slots")


class SlotEvent(Base):
    """Lịch sử đổi trạng thái; thời gian lưu theo UTC."""
    __tablename__ = "slot_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    slot_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)


def make_engine(url: str):
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
