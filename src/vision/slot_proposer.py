"""Tự đề xuất vị trí ô đỗ cho một video mới (thay cho việc vẽ tay bằng slot_annotator).

Quét các khung hình trải đều cả video bằng YOLO:
- YOLO train trên PKLot phát hiện thẳng ô đỗ (cả ô trống và ô có xe) – tốt khi bãi giống PKLot;
- YOLO pretrained COCO phát hiện xe – vị trí có xe ĐỨNG YÊN ở nhiều khung hình là một ô đỗ.
Box xuất hiện lặp lại ở cùng chỗ được gom thành cụm; cụm có đủ số khung hình ủng hộ trở thành một ô.
Xe đang chạy chỉ xuất hiện thoáng qua nên bị loại. Vì quét cả video, ô mà về sau mới có xe vào cũng
được tìm thấy ngay từ đầu (lúc đầu hiện trống, khi xe vào chuyển sang có xe).
Ô không bao giờ có xe và không được YOLO-PKLot nhận ra sẽ bị thiếu → người dùng vẽ thêm trên giao diện.
"""
from pathlib import Path

import cv2
import numpy as np

from .slot_cropper import CameraConfig, Slot
from .yolo_detector import COCO_VEHICLES, YoloOnnx, nms, pairwise_overlap


def sample_frames(video: str | Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    indices = np.linspace(0, max(total - 1, 0), num=min(n, total) if total else n).astype(int)
    frames = []
    for idx in sorted(set(indices.tolist())):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"Không đọc được khung hình nào từ {video}")
    return frames


def cluster_boxes(boxes: list[np.ndarray], n_frames: int, min_support: float, iou_thr: float = 0.5) -> np.ndarray:
    """Gom box của nhiều khung hình (mỗi phần tử: box xyxy của một khung) thành ô; giữ cụm xuất hiện ở ≥ min_support khung."""
    if not any(len(b) for b in boxes):
        return np.zeros((0, 4), np.float32)
    all_boxes = np.concatenate([b for b in boxes if len(b)])
    frame_of = np.concatenate([np.full(len(b), i) for i, b in enumerate(boxes) if len(b)])
    inter, a1, a2 = pairwise_overlap(all_boxes, all_boxes)
    iou = inter / (a1 + a2 - inter + 1e-9)
    support = np.array([len(set(frame_of[iou[i] >= iou_thr])) for i in range(len(all_boxes))])
    assigned = np.zeros(len(all_boxes), bool)
    slots, scores = [], []
    for i in np.argsort(-support):     # box có nhiều khung hình ủng hộ nhất làm tâm cụm
        if assigned[i]:
            continue
        members = np.where((iou[i] >= iou_thr) & ~assigned)[0]
        assigned[members] = True
        frames_hit = len(set(frame_of[members]))
        if frames_hit >= max(2, min_support * n_frames):
            slots.append(np.median(all_boxes[members], axis=0))
            scores.append(frames_hit)
    if not slots:
        return np.zeros((0, 4), np.float32)
    slots, scores = np.array(slots, np.float32), np.array(scores, np.float32)
    keep = nms(slots, scores, 0.3)   # bỏ ô chồng lên nhau, giữ ô được ủng hộ nhiều hơn
    slots = slots[keep]
    # IoU bỏ sót ô nhỏ nằm lọt trong ô lớn (IoU thấp). Hai ô lồng nhau (≥ 60% diện tích ô nhỏ hơn nằm trong ô kia)
    # → giữ ô được ủng hộ nhiều hơn (đứng trước sau NMS), dù nó to hay nhỏ
    inter, a1, a2 = pairwise_overlap(slots, slots)
    nested = inter / (np.minimum(a1, a2) + 1e-9)
    drop: set[int] = set()
    for i in range(len(slots)):
        if i not in drop:
            drop.update(j for j in range(i + 1, len(slots)) if nested[i, j] >= 0.6)
    return np.delete(slots, sorted(drop), axis=0)


class SlotProposer:
    def __init__(self, coco_model: str | Path | None = None, pklot_model: str | Path | None = None,
                 conf: float = 0.25, min_support: float = 0.25):
        self.coco = YoloOnnx(coco_model, conf=conf) if coco_model and Path(coco_model).exists() else None
        self.pklot = YoloOnnx(pklot_model, conf=conf) if pklot_model and Path(pklot_model).exists() else None
        if not (self.coco or self.pklot):
            raise FileNotFoundError("Cần ít nhất một mô hình YOLO (COCO hoặc PKLot) để tự tìm ô")
        self.min_support = min_support

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        boxes = []
        if self.pklot:
            b, _, _ = self.pklot.detect(frame)
            boxes.append(b)
        if self.coco:
            b, _, c = self.coco.detect(frame)
            vehicles = [k for k, v in self.coco.names.items() if v in COCO_VEHICLES]
            boxes.append(b[np.isin(c, vehicles)])
        merged = np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32)
        return merged

    def propose(self, video: str | Path, n_frames: int = 24, camera_id: str = "upload") -> CameraConfig:
        frames = sample_frames(video, n_frames)
        boxes = cluster_boxes([self._detect(f) for f in frames], len(frames), self.min_support)
        order = np.lexsort((boxes[:, 0], np.round(boxes[:, 1] / 40)))   # đánh số theo hàng (trên → dưới), trái → phải
        slots = [Slot(f"S{i + 1:03d}", np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32))
                 for i, (x0, y0, x1, y1) in enumerate(boxes[order])]
        h, w = frames[0].shape[:2]
        return CameraConfig(camera_id, (w, h), slots)
