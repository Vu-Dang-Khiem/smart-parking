"""Đọc cấu hình ô đỗ của camera và cắt ảnh từng ô từ frame.

Cách cắt phải giống hệt lúc train (notebooks/02_make_crops + CenterCrop trong 03):
lấy hình chữ nhật bao polygon, nới `pad` mỗi phía, nới viền ảnh bằng cách lặp pixel mép
nếu vùng cắt vượt ra ngoài frame, rồi resize về `size` × `size`.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Slot:
    id: str
    polygon: np.ndarray  # (N, 2) toạ độ pixel theo frame_size của cấu hình


@dataclass
class CameraConfig:
    camera_id: str
    frame_size: tuple[int, int]  # (width, height)
    slots: list[Slot]

    @classmethod
    def load(cls, path: str | Path) -> "CameraConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        slots = [Slot(s["id"], np.asarray(s["polygon"], dtype=np.float32)) for s in data["slots"]]
        ids = [s.id for s in slots]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path}: trùng id ô đỗ")
        return cls(data["camera_id"], tuple(data["frame_size"]), slots)

    def save(self, path: str | Path) -> None:
        data = {
            "camera_id": self.camera_id,
            "frame_size": list(self.frame_size),
            "slots": [{"id": s.id, "polygon": np.round(s.polygon).astype(int).tolist()} for s in self.slots],
        }
        Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "frame_size": list(self.frame_size),
            "slots": [{"id": s.id, "polygon": s.polygon.tolist()} for s in self.slots],
        }


def scale_polygons(config: CameraConfig, frame_shape: tuple[int, int]) -> list[np.ndarray]:
    """Polygon theo kích thước frame thật (frame có thể khác kích thước lúc vẽ ô)."""
    h, w = frame_shape
    cw, ch = config.frame_size
    scale = np.array([w / cw, h / ch], dtype=np.float32)
    return [s.polygon * scale for s in config.slots]


def crop_box(polygon: np.ndarray, pad: float) -> tuple[int, int, int, int]:
    """Hình chữ nhật bao polygon, nới `pad` mỗi phía → (x0, y0, x1, y1), có thể vượt ra ngoài frame."""
    x, y = polygon[:, 0].min(), polygon[:, 1].min()
    w, h = polygon[:, 0].max() - x, polygon[:, 1].max() - y
    return round(x - pad * w), round(y - pad * h), round(x + w * (1 + pad)), round(y + h * (1 + pad))


class SlotCropper:
    def __init__(self, config: CameraConfig, pad: float = 0.10, size: int = 128):
        self.config, self.pad, self.size = config, pad, size
        self._shape = None

    def _prepare(self, frame_shape: tuple[int, int]) -> None:
        # Scale polygon nếu frame thật khác kích thước lúc vẽ ô, rồi tính sẵn vùng cắt
        h, w = frame_shape
        self.boxes = np.array([crop_box(p, self.pad) for p in scale_polygons(self.config, frame_shape)])
        self.border = int(max(0, -self.boxes[:, :2].min(), (self.boxes[:, 2] - w).max(), (self.boxes[:, 3] - h).max()))
        self._shape = frame_shape

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Frame BGR (H, W, 3) → mảng (số ô, size, size, 3) BGR uint8, theo thứ tự config.slots."""
        if self._shape != frame.shape[:2]:
            self._prepare(frame.shape[:2])
        b = self.border
        if b:
            frame = cv2.copyMakeBorder(frame, b, b, b, b, cv2.BORDER_REPLICATE)
        out = np.empty((len(self.boxes), self.size, self.size, 3), np.uint8)
        for i, (x0, y0, x1, y1) in enumerate(self.boxes + b):
            out[i] = cv2.resize(frame[y0:y1, x0:x1], (self.size, self.size), interpolation=cv2.INTER_AREA)
        return out
