"""Bản web (web/vision.js) phải cắt ô và làm mượt giống hệt bản Python."""
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.slot_cropper import CameraConfig, Slot, SlotCropper
from src.vision.slot_proposer import cluster_boxes
from src.vision.smoothing import SlotSmoother

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="cần Node.js để chạy web/vision.js")


def run_node(tmp_path: Path, script: str) -> dict:
    js = tmp_path / "check.js"
    js.write_text(f"const V = require({json.dumps(str(ROOT / 'web' / 'vision.js'))});\n" + script, encoding="utf-8")
    out = subprocess.run(["node", str(js)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_crop_resize_matches_opencv(tmp_path):
    rng = np.random.default_rng(0)
    frame = cv2.GaussianBlur(rng.integers(0, 256, (240, 320, 3), dtype=np.uint8), (5, 5), 0)
    polys = [[[-12, -6], [40, -6], [40, 50], [-12, 50]],          # vượt mép trên-trái → lặp điểm ảnh mép
             [[300, 200], [330, 200], [330, 250], [300, 250]],    # vượt mép dưới-phải
             [[50, 60], [250, 60], [250, 230], [50, 230]],        # lớn hơn 128 → thu nhỏ theo diện tích
             [[100, 100], [260, 100], [260, 118], [100, 118]],    # một chiều to, một chiều nhỏ
             [[10.4, 20.5], [53.5, 20.5], [53.5, 61.5], [10.4, 61.5]]]
    slots = [Slot(str(i), np.array(p, np.float32)) for i, p in enumerate(polys)]
    ref = SlotCropper(CameraConfig("t", (320, 240), slots)).crop(frame)[..., ::-1]   # → RGB
    rgba = np.dstack([frame[..., ::-1], np.full(frame.shape[:2], 255, np.uint8)])
    (tmp_path / "frame.bin").write_bytes(rgba.tobytes())
    out = run_node(tmp_path, f"""
const rgba = new Uint8ClampedArray(require("fs").readFileSync({json.dumps(str(tmp_path / 'frame.bin'))}));
const polys = {json.dumps(polys)};
console.log(JSON.stringify(polys.map((p) => Array.from(V.cropResize(rgba, 320, 240, V.cropBox(p, 0.1), 128)))));
""")
    diff = np.abs(np.array(out, np.int16).reshape(ref.shape) - ref.astype(np.int16))
    assert diff.max() <= 1          # OpenCV tính bằng số nguyên → lệch làm tròn tối đa 1 mức xám


def test_smoother_and_clustering_match_python(tmp_path):
    rng = np.random.default_rng(1)
    probs = rng.random((30, 4)).round(3).tolist()
    py = SlotSmoother(["a", "b", "c", "d"])
    expected = [[(c.slot_id, c.status) for c in py.update(p)] for p in probs]
    frames = [[[100 + i % 3, 100, 150, 180], [300, 100, 350, 180]] if i >= 4 else [[100, 100, 150, 180]] for i in range(10)]
    frames = [f + [[50 + 60 * i, 400, 100 + 60 * i, 480]] for i, f in enumerate(frames)]   # xe đang chạy
    slots = cluster_boxes([np.array(f, np.float32) for f in frames], 10, 0.25)
    out = run_node(tmp_path, f"""
const s = new V.SlotSmoother(["a", "b", "c", "d"]);
const changes = {json.dumps(probs)}.map((p) => s.update(p).map((c) => [c.slot_id, c.status]));
const slots = V.clusterBoxes({json.dumps(frames)}, 10, 0.25);
console.log(JSON.stringify({{changes, slots}}));
""")
    assert [[tuple(c) for c in step] for step in out["changes"]] == expected
    assert np.allclose(np.array(out["slots"]), slots, atol=1e-4)
