import cv2
import numpy as np
import pytest

from src.vision.slot_cropper import CameraConfig, Slot, SlotCropper, crop_box


def make_frame(w=1280, h=720):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(frame, (0, 0), 3)  # ảnh có kết cấu mịn để so sánh crop có ý nghĩa


def rect(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def training_crop(frame, x, y, w, h, save_pad=0.25, save_size=160, input_size=128):
    # Cách notebooks/02_make_crops lưu crop + CenterCrop(128) trong notebooks/03_train_classifier
    b = 200
    big = cv2.copyMakeBorder(frame, b, b, b, b, cv2.BORDER_REPLICATE)
    x0, x1 = round(x - save_pad * w) + b, round(x + w * (1 + save_pad)) + b
    y0, y1 = round(y - save_pad * h) + b, round(y + h * (1 + save_pad)) + b
    saved = cv2.resize(big[y0:y1, x0:x1], (save_size, save_size), interpolation=cv2.INTER_AREA)
    c = (save_size - input_size) // 2
    return saved[c:c + input_size, c:c + input_size]


@pytest.mark.parametrize("box", [(400, 300, 80, 120), (0, 0, 60, 90), (1230, 650, 50, 70)])
def test_runtime_crop_matches_training_preprocessing(box):
    frame = make_frame()
    cfg = CameraConfig("cam", (1280, 720), [Slot("A01", rect(*box))])
    got = SlotCropper(cfg, pad=0.10, size=128).crop(frame)[0].astype(int)
    want = training_crop(frame, *box).astype(int)
    assert got.shape == (128, 128, 3)
    assert np.abs(got - want).mean() < 3  # chỉ khác do nội suy hai lần


def test_crop_box_pads_each_side():
    assert crop_box(rect(100, 200, 50, 100), 0.10) == (95, 190, 155, 310)


def test_polygons_scale_with_frame_size():
    frame = make_frame()
    small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    cfg = CameraConfig("cam", (1280, 720), [Slot("A01", rect(400, 300, 160, 200))])
    full = SlotCropper(cfg).crop(frame)[0].astype(int)
    half = SlotCropper(cfg).crop(small)[0].astype(int)
    assert np.abs(full - half).mean() < 4


def test_config_roundtrip(tmp_path):
    cfg = CameraConfig("cam01", (1280, 720), [Slot("A01", rect(10, 20, 30, 40)), Slot("A02", rect(50, 60, 30, 40))])
    cfg.save(tmp_path / "c.json")
    loaded = CameraConfig.load(tmp_path / "c.json")
    assert loaded.camera_id == "cam01" and loaded.frame_size == (1280, 720)
    assert [s.id for s in loaded.slots] == ["A01", "A02"]
    np.testing.assert_allclose(loaded.slots[1].polygon, rect(50, 60, 30, 40))


def test_duplicate_slot_ids_rejected(tmp_path):
    CameraConfig("c", (100, 100), [Slot("A", rect(0, 0, 5, 5)), Slot("A", rect(9, 9, 5, 5))]).save(tmp_path / "d.json")
    with pytest.raises(ValueError):
        CameraConfig.load(tmp_path / "d.json")
