"""Pipeline so sánh: YOLOv8 (file ONNX xuất từ notebooks/04_train_yolo) → trạng thái từng ô.

Chạy bằng ONNX Runtime, tự làm letterbox / giải mã box / NMS nên không cần cài ultralytics.
Hai loại mô hình, nhận biết qua tên lớp lưu trong metadata của file ONNX:
- PKLot (lớp vacant / occupied): mỗi ô lấy box có IoU lớn nhất với ô (≥ match_iou).
- COCO (car, motorcycle, bus, truck…): ô có xe nếu một box xe che ≥ cover_thr diện tích ô.
"""
import ast
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .slot_cropper import CameraConfig, scale_polygons

COCO_VEHICLES = ("car", "motorcycle", "bus", "truck")


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Non-maximum suppression trên box xyxy, trả về chỉ số giữ lại (giảm dần theo score)."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        lt = np.maximum(boxes[i, :2], boxes[rest, :2])
        rb = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        inter = np.prod(np.clip(rb - lt, 0, None), axis=1)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        iou = inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-9)
        order = rest[iou < iou_thr]
    return np.array(keep, dtype=int)


def pairwise_overlap(a: np.ndarray, b: np.ndarray):
    """(diện tích giao, diện tích a, diện tích b) cho mọi cặp box xyxy, shape (len(a), len(b))."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter, area(a)[:, None], area(b)[None, :]


class YoloOnnx:
    def __init__(self, model_path: str | Path, conf: float = 0.25, iou: float = 0.5, threads: int = 0):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = int(inp.shape[2])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = {int(k): v for k, v in ast.literal_eval(meta["names"]).items()} if "names" in meta else {}
        self.conf, self.iou = conf, iou

    def letterbox(self, frame: np.ndarray):
        # Giữ tỷ lệ, đệm viền xám 114 cho đủ imgsz × imgsz (giống ultralytics)
        h, w = frame.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = round(w * r), round(h * r)
        px, py = (self.imgsz - nw) / 2, (self.imgsz - nh) / 2
        img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, round(py - 0.1), round(py + 0.1), round(px - 0.1), round(px + 0.1),
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (px, py)

    def detect(self, frame: np.ndarray):
        """Frame BGR → (boxes xyxy theo pixel frame, scores, class ids)."""
        img, r, (px, py) = self.letterbox(frame)
        x = img[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: np.ascontiguousarray(x)})[0][0].T  # (anchors, 4 + nc)
        scores_all = out[:, 4:]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(cls)), cls]
        mask = scores >= self.conf
        boxes, scores, cls = out[mask, :4], scores[mask], cls[mask]
        xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                         boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)
        xyxy = (xyxy - [px, py, px, py]) / r
        # NMS theo từng lớp: dịch box mỗi lớp ra xa nhau để chạy NMS một lần
        keep = nms(xyxy + cls[:, None] * 10000.0, scores, self.iou) if len(scores) else np.zeros(0, int)
        h, w = frame.shape[:2]
        xyxy = np.clip(xyxy[keep], 0, [w, h, w, h])
        return xyxy.astype(np.float32), scores[keep].astype(np.float32), cls[keep]


class YoloSlotPredictor:
    """Quy kết quả YOLO về xác suất occupied của từng ô (cùng giao diện với ClassifierPredictor)."""

    def __init__(self, config: CameraConfig, detector: YoloOnnx, match_iou: float = 0.3, cover_thr: float = 0.3):
        self.config, self.detector = config, detector
        self.match_iou, self.cover_thr = match_iou, cover_thr
        names = {v: k for k, v in detector.names.items()}
        if "occupied" in names:
            self.mode, self.occupied_id = "pklot", names["occupied"]
        else:
            self.mode = "coco"
            self.vehicle_ids = np.array([names[n] for n in COCO_VEHICLES if n in names])
        self.last_detections = None

    def slot_boxes(self, frame_shape) -> np.ndarray:
        return np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]
                         for p in scale_polygons(self.config, frame_shape)], dtype=np.float32)

    def probs_from_detections(self, slot_boxes, boxes, scores, cls) -> np.ndarray:
        # Box đã qua ngưỡng conf là một quyết định: "có xe" → 0.5 + score/2 (luôn > 0.5), "trống" → 0.5 − score/2.
        # Không dùng thẳng score làm xác suất, vì box xe score 0.3 sẽ bị hiểu nhầm là ô trống ở ngưỡng 0.5.
        probs = np.zeros(len(slot_boxes), dtype=np.float32)  # không có box phù hợp → coi là trống
        if self.mode == "pklot":
            if len(boxes):
                inter, a, b = pairwise_overlap(slot_boxes, boxes)
                iou = inter / (a + b - inter + 1e-9)
                best = iou.argmax(1)
                matched = iou.max(1) >= self.match_iou
                half = scores[best] / 2
                p = np.where(cls[best] == self.occupied_id, 0.5 + half, 0.5 - half)
                probs[matched] = p[matched]
        else:
            vehicle = np.isin(cls, self.vehicle_ids)
            if vehicle.any():
                inter, a, _ = pairwise_overlap(slot_boxes, boxes[vehicle])
                covered = inter / (a + 1e-9) >= self.cover_thr
                best = np.where(covered, scores[vehicle][None, :], -1).max(1)
                probs = np.where(best >= 0, 0.5 + best / 2, 0.0).astype(np.float32)
        return probs

    def predict(self, frame: np.ndarray) -> np.ndarray:
        boxes, scores, cls = self.detector.detect(frame)
        self.last_detections = (boxes, scores, cls)
        return self.probs_from_detections(self.slot_boxes(frame.shape[:2]), boxes, scores, cls)
