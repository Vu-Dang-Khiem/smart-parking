"""Vòng lặp chính: đọc video → dự đoán trạng thái từng ô → làm mượt → gửi lên backend.

Ví dụ:
    python -m src.vision.worker --video data/sample.mp4 --config configs/camera_01.json \
        --model models/classifier.onnx --api http://localhost:8000 --show

Pipeline so sánh YOLO (file ONNX từ notebooks/04_train_yolo):
    python -m src.vision.worker ... --method yolo --model models/yolo/yolov8n_coco.onnx
"""
import argparse
import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

from .classifier import SlotClassifier
from .slot_cropper import CameraConfig, SlotCropper, scale_polygons
from .smoothing import OCCUPIED, VACANT, Change, SlotSmoother
from .yolo_detector import YoloOnnx, YoloSlotPredictor

log = logging.getLogger("worker")

COLORS = {VACANT: (60, 190, 60), OCCUPIED: (50, 50, 230), None: (160, 160, 160)}  # BGR


class ClassifierPredictor:
    """Pipeline chính: cắt từng ô → CNN phân loại → xác suất occupied của từng ô."""

    def __init__(self, config: CameraConfig, model_path: str):
        self.classifier = SlotClassifier(model_path)
        self.cropper = SlotCropper(config, pad=self.classifier.runtime_pad, size=self.classifier.input_size)
        self.last_detections = None

    def predict(self, frame: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.cropper.crop(frame))


def build_predictor(args, config: CameraConfig):
    if args.method == "yolo":
        return YoloSlotPredictor(config, YoloOnnx(args.model, conf=args.conf), match_iou=args.match_iou,
                                 cover_thr=args.cover_thr)
    return ClassifierPredictor(config, args.model)


def draw_overlay(frame: np.ndarray, polygons, slot_ids, states: dict, counts: dict, detections=None) -> np.ndarray:
    out = frame.copy()
    if detections is not None:  # box YOLO (vàng) để thấy mô hình phát hiện được gì
        for x0, y0, x1, y1 in detections[0].astype(int):
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 215, 255), 1)
    fill = out.copy()
    for poly, sid in zip(polygons, slot_ids):
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(fill, [pts], COLORS[states[sid]])
        cv2.polylines(out, [pts], True, COLORS[states[sid]], 2)
    out = cv2.addWeighted(fill, 0.25, out, 0.75, 0)
    total = len(slot_ids)
    text = f"Trong: {counts['vacant']}/{total}   Co xe: {counts['occupied']}"
    cv2.rectangle(out, (0, 0), (20 + 14 * len(text), 40), (30, 30, 30), -1)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


class BackendClient:
    """Gửi dữ liệu lên backend; lỗi mạng chỉ ghi cảnh báo để worker không dừng."""

    def __init__(self, base_url: str, camera: CameraConfig, timeout: float = 2.0):
        self.base_url, self.camera, self.timeout = base_url.rstrip("/"), camera, timeout
        self.session = requests.Session()
        if token := os.environ.get("SMART_PARKING_WORKER_TOKEN"):   # backend public chỉ nhận dữ liệu có mã này
            self.session.headers["X-Worker-Token"] = token
        self._failing = False

    def _post(self, path: str, **kwargs) -> bool:
        try:
            self.session.post(self.base_url + path, timeout=self.timeout, **kwargs).raise_for_status()
        except requests.RequestException as e:
            if not self._failing:
                log.warning("Không gửi được lên backend (%s): %s", path, e)
            self._failing = True
            return False
        if self._failing:
            log.info("Đã kết nối lại backend")
        self._failing = False
        return True

    def register(self) -> bool:
        return self._post("/api/cameras", json=self.camera.to_dict())

    def send_events(self, changes: list[Change], ts: str) -> bool:
        payload = {"camera_id": self.camera.camera_id, "ts": ts,
                   "changes": [{"slot_id": c.slot_id, "status": c.status, "confidence": c.confidence} for c in changes]}
        return self._post("/api/events", json=payload)

    def send_frame(self, frame: np.ndarray) -> bool:
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return ok and self._post(f"/api/cameras/{self.camera.camera_id}/frame", data=jpg.tobytes(),
                                 headers={"Content-Type": "image/jpeg"})


def open_source(video: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(video) if video.isdigit() else video)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được nguồn video: {video}")
    return cap


def run(args) -> dict:
    config = CameraConfig.load(args.config)
    predictor = build_predictor(args, config)
    slot_ids = [s.id for s in config.slots]
    smoother = SlotSmoother(slot_ids, window=args.window, k=args.k, threshold=args.threshold)
    client = BackendClient(args.api, config) if args.api else None
    if client:
        client.register()

    cap = open_source(args.video)
    is_file = not args.video.isdigit() and Path(args.video).exists()
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 25) / args.fps))
    log_file = open(args.log_csv, "w", newline="", encoding="utf-8") if args.log_csv else None
    log_writer = csv.writer(log_file) if log_file else None
    if log_writer:
        log_writer.writerow(["frame", "slot_id", "prob", "state"])

    frame_idx, processed, infer_ms = -1, 0, []
    polygons = None
    try:
        while args.max_frames is None or processed < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                if args.loop and is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = -1
                    continue
                break
            frame_idx += 1
            if frame_idx % step:
                continue
            t0 = time.perf_counter()
            probs = predictor.predict(frame)
            changes = smoother.update(probs)
            infer_ms.append((time.perf_counter() - t0) * 1000)
            processed += 1

            if log_writer:
                log_writer.writerows([frame_idx, sid, f"{p:.4f}", smoother.state[sid]] for sid, p in zip(slot_ids, probs))
            if polygons is None:
                polygons = scale_polygons(config, frame.shape[:2])
            counts = smoother.counts()
            overlay = draw_overlay(frame, polygons, slot_ids, smoother.state, counts, predictor.last_detections)
            if client:
                if changes:
                    client.send_events(changes, datetime.now().astimezone().isoformat(timespec="seconds"))
                client.send_frame(overlay)
            if changes:
                log.info("frame %d: %s", frame_idx, ", ".join(f"{c.slot_id}→{c.status}" for c in changes))
            if processed % 50 == 0:
                log.info("%d frame, %.1f ms/frame, trống %d/%d", processed, np.mean(infer_ms[-50:]), counts["vacant"], len(slot_ids))
            if args.show:
                cv2.imshow(f"Smart Parking – {config.camera_id}", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.pace and is_file:
                time.sleep(max(0.0, 1 / args.fps - (time.perf_counter() - t0)))
    finally:
        cap.release()
        if log_file:
            log_file.close()
        if args.show:
            cv2.destroyAllWindows()

    stats = {"frames": processed, "slots": len(slot_ids),
             "mean_ms": float(np.mean(infer_ms)) if infer_ms else 0.0, "state": dict(smoother.state)}
    if infer_ms:
        log.info("Xong: %d frame, %d ô, trung bình %.1f ms/frame (%.0f FPS)",
                 processed, len(slot_ids), stats["mean_ms"], 1000 / max(stats["mean_ms"], 1e-6))
    return stats


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Smart Parking – CV worker")
    p.add_argument("--video", required=True, help="file video, URL RTSP hoặc số thứ tự webcam (0, 1, ...)")
    p.add_argument("--config", required=True, help="file JSON polygon ô đỗ (tạo bằng slot_annotator)")
    p.add_argument("--method", choices=["classifier", "yolo"], default="classifier",
                   help="classifier = cắt ô + CNN (chính); yolo = YOLOv8 ONNX (so sánh)")
    p.add_argument("--model", default="models/classifier.onnx", help="file ONNX của phương pháp đã chọn")
    p.add_argument("--conf", type=float, default=0.25, help="[yolo] ngưỡng tin cậy của box")
    p.add_argument("--match-iou", type=float, default=0.3, help="[yolo PKLot] IoU tối thiểu giữa box và ô")
    p.add_argument("--cover-thr", type=float, default=0.3, help="[yolo COCO] tỷ lệ diện tích ô bị xe che để coi là có xe")
    p.add_argument("--api", default=None, help="địa chỉ backend, ví dụ http://localhost:8000")
    p.add_argument("--fps", type=float, default=2.0, help="số frame xử lý mỗi giây video")
    p.add_argument("--window", type=int, default=5, help="số frame trong cửa sổ làm mượt")
    p.add_argument("--k", type=int, default=4, help="số phiếu tối thiểu để đổi trạng thái")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--show", action="store_true", help="mở cửa sổ xem overlay (nhấn q để thoát)")
    p.add_argument("--loop", action="store_true", help="phát lại video khi hết")
    p.add_argument("--no-pace", dest="pace", action="store_false", help="xử lý nhanh nhất có thể thay vì theo thời gian thực")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--log-csv", default=None, help="ghi xác suất/trạng thái từng ô mỗi frame (để đánh giá)")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run(parse_args())
