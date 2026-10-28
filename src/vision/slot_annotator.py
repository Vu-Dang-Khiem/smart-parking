"""Tool vẽ ô đỗ cho một camera, lưu thành file JSON dùng cho worker.

    python -m src.vision.slot_annotator --video data/sample.mp4 --out configs/camera_01.json --camera-id cam01

Chuột trái: thêm góc (đủ 4 góc = 1 ô) | u: xoá góc / ô vừa vẽ | s: lưu | q: lưu và thoát | Esc: thoát không lưu.
Nếu file --out đã có thì mở ra để vẽ tiếp.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from .slot_cropper import CameraConfig, Slot

HELP = "Chuot trai: them goc | u: xoa | s: luu | q: luu & thoat | Esc: thoat"


class Annotator:
    def __init__(self, camera_id: str, frame_size: tuple[int, int], prefix: str = "S",
                 slots: list[Slot] | None = None, corners: int = 4):
        self.camera_id, self.frame_size, self.prefix, self.corners = camera_id, frame_size, prefix, corners
        self.slots = list(slots or [])
        self.pending: list[tuple[float, float]] = []

    def next_id(self) -> str:
        used = {s.id for s in self.slots}
        n = len(self.slots) + 1
        while f"{self.prefix}{n:02d}" in used:
            n += 1
        return f"{self.prefix}{n:02d}"

    def add_point(self, x: float, y: float) -> Slot | None:
        self.pending.append((x, y))
        if len(self.pending) < self.corners:
            return None
        slot = Slot(self.next_id(), np.array(self.pending, dtype=np.float32))
        self.slots.append(slot)
        self.pending = []
        return slot

    def undo(self) -> None:
        if self.pending:
            self.pending.pop()
        elif self.slots:
            self.slots.pop()

    def to_config(self) -> CameraConfig:
        return CameraConfig(self.camera_id, self.frame_size, self.slots)


def render(frame: np.ndarray, ann: Annotator, scale: float) -> np.ndarray:
    img = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1 else frame.copy()
    for s in ann.slots:
        pts = np.round(s.polygon * scale).astype(np.int32)
        cv2.polylines(img, [pts], True, (0, 220, 255), 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(img, s.id, (cx - 14, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    for x, y in ann.pending:
        cv2.circle(img, (round(x * scale), round(y * scale)), 4, (0, 0, 255), -1)
    cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]), (30, 30, 30), -1)
    cv2.putText(img, f"{HELP}   | {len(ann.slots)} o", (8, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def read_frame(source: str, index: int) -> np.ndarray:
    if Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        frame = cv2.imread(source)
    else:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        cap.release()
        frame = frame if ok else None
    if frame is None:
        raise SystemExit(f"Không đọc được frame từ {source}")
    return frame


def main(argv=None):
    p = argparse.ArgumentParser(description="Vẽ polygon ô đỗ cho camera")
    p.add_argument("--video", required=True, help="video, ảnh hoặc số thứ tự webcam")
    p.add_argument("--out", required=True, help="file JSON đầu ra")
    p.add_argument("--camera-id", default="cam01")
    p.add_argument("--frame", type=int, default=0, help="frame dùng để vẽ")
    p.add_argument("--prefix", default="S", help="tiền tố id ô, ví dụ A → A01, A02...")
    p.add_argument("--max-width", type=int, default=1600, help="thu nhỏ cửa sổ nếu frame rộng hơn")
    args = p.parse_args(argv)

    frame = read_frame(args.video, args.frame)
    h, w = frame.shape[:2]
    out = Path(args.out)
    if out.exists():
        cfg = CameraConfig.load(out)
        sx, sy = w / cfg.frame_size[0], h / cfg.frame_size[1]
        slots = [Slot(s.id, s.polygon * np.array([sx, sy], dtype=np.float32)) for s in cfg.slots]
        ann = Annotator(cfg.camera_id, (w, h), args.prefix, slots)
        print(f"Mở {out}: {len(slots)} ô")
    else:
        ann = Annotator(args.camera_id, (w, h), args.prefix)
    scale = min(1.0, args.max_width / w)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            slot = ann.add_point(x / scale, y / scale)
            if slot:
                print("Thêm ô", slot.id)

    win = "Slot annotator"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        cv2.imshow(win, render(frame, ann, scale))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("u"):
            ann.undo()
        elif key in (ord("s"), ord("q")):
            out.parent.mkdir(parents=True, exist_ok=True)
            ann.to_config().save(out)
            print(f"Đã lưu {len(ann.slots)} ô vào {out}")
            if key == ord("q"):
                break
        elif key == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
