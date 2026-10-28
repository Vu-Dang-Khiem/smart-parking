import numpy as np
import pytest

from src.vision.slot_cropper import CameraConfig, Slot
from src.vision.yolo_detector import YoloOnnx, YoloSlotPredictor, nms


def rect(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


class FakeSession:
    def __init__(self, output):
        self.output = output

    def run(self, _, __):
        return [self.output]


def fake_yolo(output, names, imgsz=640):
    det = YoloOnnx.__new__(YoloOnnx)  # bỏ qua việc nạp file ONNX
    det.session, det.input_name, det.imgsz = FakeSession(output), "images", imgsz
    det.names, det.conf, det.iou = names, 0.25, 0.5
    return det


class FakeDetector:
    def __init__(self, names, boxes, scores, cls):
        self.names = names
        self.result = (np.array(boxes, np.float32).reshape(-1, 4), np.array(scores, np.float32), np.array(cls, int))

    def detect(self, frame):
        return self.result


def test_nms_keeps_highest_of_overlapping():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    keep = nms(boxes, np.array([0.6, 0.9, 0.7]), 0.5)
    assert keep.tolist() == [1, 2]


def test_detect_maps_boxes_back_through_letterbox():
    # Frame 1280×720 → letterbox 640: scale 0.5, đệm 140 px trên/dưới. Một box xe ở (100,200)-(300,400) trên frame
    # tương ứng tâm (100, 290), kích thước (100, 100) trong ảnh 640.
    out = np.zeros((1, 4 + 3, 3), np.float32)
    out[0, :, 0] = [100, 290, 100, 100, 0.0, 0.0, 0.9]   # lớp 2, score 0.9
    out[0, :, 1] = [102, 291, 100, 100, 0.0, 0.0, 0.6]   # trùng box trên → bị NMS loại
    out[0, :, 2] = [500, 500, 20, 20, 0.1, 0.0, 0.0]     # dưới ngưỡng conf
    boxes, scores, cls = fake_yolo(out, {0: "a", 1: "b", 2: "car"}).detect(np.zeros((720, 1280, 3), np.uint8))
    assert cls.tolist() == [2] and scores[0] == pytest.approx(0.9)
    np.testing.assert_allclose(boxes[0], [100, 200, 300, 400], atol=1e-3)


def test_letterbox_output_is_square():
    img, r, (px, py) = fake_yolo(np.zeros((1, 6, 0)), {}).letterbox(np.zeros((720, 1280, 3), np.uint8))
    assert img.shape == (640, 640, 3) and r == 0.5 and (px, py) == (0, 140)


def test_pklot_mode_uses_best_matching_box():
    cfg = CameraConfig("c", (1280, 720), [Slot("A", rect(0, 0, 100, 100)), Slot("B", rect(200, 0, 100, 100)),
                                          Slot("C", rect(400, 0, 100, 100))])
    det = FakeDetector({0: "vacant", 1: "occupied"}, [[0, 0, 100, 100], [205, 0, 300, 100]], [0.9, 0.8], [1, 0])
    pred = YoloSlotPredictor(cfg, det)
    assert pred.mode == "pklot"
    probs = pred.predict(np.zeros((720, 1280, 3), np.uint8))
    np.testing.assert_allclose(probs, [0.95, 0.1, 0.0], atol=1e-6)  # C không khớp box nào → coi là trống


def test_low_score_detection_still_counts_as_decision():
    # Box "occupied" score 0.3 (đã qua ngưỡng conf 0.25) phải cho xác suất > 0.5, không bị hiểu thành trống
    cfg = CameraConfig("c", (1280, 720), [Slot("A", rect(0, 0, 100, 100))])
    det = FakeDetector({0: "vacant", 1: "occupied"}, [[0, 0, 100, 100]], [0.3], [1])
    assert YoloSlotPredictor(cfg, det).predict(np.zeros((720, 1280, 3), np.uint8))[0] == pytest.approx(0.65)
    car = FakeDetector({2: "car"}, [[0, 0, 100, 100]], [0.3], [2])
    assert YoloSlotPredictor(cfg, car).predict(np.zeros((720, 1280, 3), np.uint8))[0] == pytest.approx(0.65)


def test_coco_mode_uses_vehicle_coverage():
    cfg = CameraConfig("c", (1280, 720), [Slot("A", rect(0, 0, 100, 100)), Slot("B", rect(200, 0, 100, 100)),
                                          Slot("C", rect(400, 0, 100, 100))])
    names = {0: "person", 2: "car", 7: "truck"}
    det = FakeDetector(names, [[10, 10, 90, 90], [280, 0, 400, 100], [400, 0, 500, 100]], [0.8, 0.9, 0.95], [2, 7, 0])
    probs = YoloSlotPredictor(cfg, det, cover_thr=0.3).predict(np.zeros((720, 1280, 3), np.uint8))
    # A bị xe che 64% → có xe; B chỉ bị che 20% → trống; C bị người che → không tính
    np.testing.assert_allclose(probs, [0.9, 0.0, 0.0], atol=1e-6)
