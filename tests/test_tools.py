"""Test các script trong tools/ bằng dữ liệu nhỏ dựng trong thư mục tạm."""
import json
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import eval_cnrpark  # noqa: E402
import export_onnx  # noqa: E402
import finetune_cnrpark  # noqa: E402
import make_summary  # noqa: E402
from src.vision.slot_cropper import CameraConfig, Slot, SlotCropper  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "red_green.onnx"  # đỏ → occupied, xanh → vacant


@pytest.fixture
def fake_cnr(tmp_path, monkeypatch):
    """CNR-EXT thu nhỏ: camera 1 có 2 ô (ô 10 có xe đỏ, ô 11 trống xanh), 1 khung hình; 8 camera còn lại không có ảnh."""
    monkeypatch.setattr(eval_cnrpark, "DATA", tmp_path)
    full = tmp_path / "FULL_IMAGE_1000x750"
    frame_dir = full / "SUNNY" / "2015-11-12" / "camera1"
    frame_dir.mkdir(parents=True)
    frame = np.full((750, 1000, 3), 90, np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 200), (0, 0, 230), -1)    # ô 10: có xe
    cv2.rectangle(frame, (400, 100), (500, 200), (0, 220, 0), -1)    # ô 11: trống
    cv2.imwrite(str(frame_dir / "2015-11-12_0709.jpg"), frame)
    s = 2592 / 1000   # toạ độ trong CSV theo ảnh gốc 2592×1944
    for cam in range(1, 10):
        rows = [(10, 100 * s, 100 * 1944 / 750, 100 * s, 100 * 1944 / 750),
                (11, 400 * s, 100 * 1944 / 750, 100 * s, 100 * 1944 / 750)] if cam == 1 else [(99, 0, 0, 10, 10)]
        pd.DataFrame(rows, columns=["SlotId", "X", "Y", "W", "H"]).to_csv(tmp_path / f"camera{cam}.csv", index=False)
    meta = pd.DataFrame({
        "camera": ["01", "01", "A"], "datetime": ["2015-11-12_07.09"] * 2 + ["20150703_0805"],
        "image_url": ["CNR-EXT/PATCHES/SUNNY/2015-11-12/camera1/S_2015-11-12_07.09_C01_10.jpg",
                      "CNR-EXT/PATCHES/SUNNY/2015-11-12/camera1/S_2015-11-12_07.09_C01_11.jpg",
                      "CNRPark/A/free/20150703_0805_1.jpg"],
        "occupancy": [1, 0, 0], "slot_id": [10, 11, 1], "weather": ["S", "S", "S"]})
    meta.to_csv(tmp_path / "CNRPark+EXT.csv", index=False)
    return tmp_path


def test_load_labels_keeps_cnr_ext_and_parses_time(fake_cnr):
    labels = eval_cnrpark.load_labels()
    assert len(labels) == 2  # bỏ dòng thuộc phần CNRPark (camera A)
    assert labels["cam"].tolist() == [1, 1] and labels["date"].iloc[0] == "2015-11-12" and labels["time"].iloc[0] == "0709"


def test_camera_boxes_rescaled_to_1000x750(fake_cnr):
    boxes = eval_cnrpark.read_camera_boxes(fake_cnr / "FULL_IMAGE_1000x750")
    assert sorted(boxes) == list(range(1, 10))
    np.testing.assert_allclose(boxes[1].iloc[0][["x", "y", "w", "h"]].astype(float), [100, 100, 100, 100], atol=1e-6)


def test_eval_ours_end_to_end(fake_cnr):
    args = Namespace(method="classifier", model=str(FIXTURE), limit_frames=None)
    res = eval_cnrpark.eval_ours(args, eval_cnrpark.load_labels())
    by_slot = res.set_index("slot_id")
    assert by_slot.loc[10, "prob"] > 0.9 and by_slot.loc[11, "prob"] < 0.1
    out = eval_cnrpark.report(res, "test")
    assert out["overall"]["accuracy"] == 1.0 and out["by_camera"]["camera1"]["n"] == 2 and "SUNNY" in out["by_weather"]


def test_metrics_counts_positive_class():
    m = eval_cnrpark.metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert m == {"n": 4, "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_finetune_eval_crop_matches_runtime_crop():
    # Crop lưu (nới 25%, 160) + CenterCrop(128) phải gần bằng crop lúc chạy thật (nới 10%, 128)
    rng = np.random.default_rng(0)
    frame = cv2.GaussianBlur(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8), (0, 0), 3)
    poly = np.array([[200, 100], [300, 100], [300, 240], [200, 240]], np.float32)
    cfg = CameraConfig("c", (640, 360), [Slot("A", poly)])
    saved = SlotCropper(cfg, pad=0.25, size=160).crop(frame)[:, :, :, ::-1]
    runtime = SlotCropper(cfg, pad=0.10, size=128).crop(frame)[:, :, :, ::-1]
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2)
    got = finetune_cnrpark.eval_batch(to_t(saved))
    want = finetune_cnrpark.normalize(to_t(runtime).float() / 255)
    assert got.shape == (1, 3, 128, 128)
    assert (got - want).abs().mean() < 0.06


def test_finetune_runs_and_train_augment_shapes():
    x = torch.randint(0, 255, (6, 3, 160, 160), dtype=torch.uint8)
    aug = finetune_cnrpark.train_batch(x)
    assert aug.shape == (6, 3, 128, 128) and torch.isfinite(aug).all()
    from torchvision import models
    base = models.mobilenet_v3_small(weights=None)
    base.classifier[3] = torch.nn.Linear(base.classifier[3].in_features, 2)
    model = finetune_cnrpark.finetune(base, x, np.array([0, 1, 0, 1, 0, 1]), steps=2, lr=1e-3, batch=4)
    probs = finetune_cnrpark.predict(model, x)
    assert probs.shape == (6,) and ((probs >= 0) & (probs <= 1)).all()
    assert not torch.equal(next(model.parameters()), next(base.parameters()))  # base không bị sửa


def test_make_summary_combines_sources(tmp_path, monkeypatch):
    res, demo = tmp_path / "results", tmp_path / "demo"
    (res / "classifier").mkdir(parents=True)
    (res / "cnrpark").mkdir()
    demo.mkdir()
    pd.DataFrame([
        {"experiment": "in-domain", "model": "mobilenet_v3", "test_camera": "all", "accuracy": 0.99, "f1": 0.98},
        {"experiment": "LOCO", "model": "mobilenet_v3", "test_camera": "A", "accuracy": 0.9, "f1": 0.9},
        {"experiment": "LOCO", "model": "mobilenet_v3", "test_camera": "B", "accuracy": 0.8, "f1": 0.8},
    ]).to_csv(res / "classifier" / "results.csv", index=False)
    (res / "cnrpark" / "results_classifier.json").write_text(json.dumps({"ours": {"overall": {"accuracy": 0.94, "recall": 0.9}}}))
    pd.DataFrame({"frame": [0, 1], "slot_id": ["s", "s"], "prob": [0.9, 0.2], "state": ["occupied", "vacant"]}).to_csv(demo / "log_models.csv", index=False)
    pd.DataFrame({"frame": [0, 1], "slot_id": ["s", "s"], "status": ["occupied", "occupied"]}).to_csv(demo / "ground_truth.csv", index=False)
    monkeypatch.setattr(make_summary, "RES", res)
    monkeypatch.setattr(make_summary, "DEMO", demo)
    make_summary.main()
    table = pd.read_csv(res / "summary_all.csv")
    row = table.set_index("phương pháp").loc["MobileNetV3-Small (chính)"]
    assert row["PKLot test – acc"] == 0.99 and row["PKLot bỏ một camera – acc TB"] == pytest.approx(0.85)
    assert row["CNR-EXT – acc"] == 0.94 and row["video demo – acc"] == 0.5
    assert len(table) == 1  # các phương pháp chưa có kết quả bị bỏ qua


@pytest.mark.parametrize("arch", ["small_cnn", "mobilenet_v3"])
def test_export_onnx_matches_pytorch(arch, tmp_path):
    pytest.importorskip("onnx")   # torch.onnx cần gói onnx (có trong requirements.txt)
    weights = tmp_path / f"{arch}.pt"
    torch.save(export_onnx.build(arch).state_dict(), weights)
    out = tmp_path / f"{arch}.onnx"
    assert export_onnx.export(arch, weights, out) < 1e-3
    cfg = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert cfg["model"] == arch and cfg["input_size"] == 128 and "test_accuracy" not in cfg
