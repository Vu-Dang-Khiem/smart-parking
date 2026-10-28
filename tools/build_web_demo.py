"""Dựng bản demo chạy hoàn toàn trong trình duyệt (GitHub Pages) vào một thư mục tĩnh.

    python tools/build_web_demo.py _site          → mở bằng: python -m http.server -d _site 8080

Mỗi video demo time-lapse lặp một ảnh trong nhiều khung liên tiếp, nên chỉ cần lấy một khung cho mỗi ảnh; trang web
đưa kết quả của ảnh đó vào bộ làm mượt đúng số khung như worker. Kèm nhãn thật từng ô để dashboard hiện độ chính xác.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.vision.slot_cropper import CameraConfig  # noqa: E402

DEMOS = [
    {"id": "pklot-pucpr", "name": "PKLot – PUCPR (Brazil)", "video": "data/pklot_demo/video.mp4",
     "truth": "data/pklot_demo/ground_truth.csv", "info": "data/pklot_demo/info.json", "config": "configs/pklot_pucpr.json"},
    {"id": "cnr-camera8", "name": "CNR-EXT camera 8 (Ý, bãi chưa từng học)", "video": "data/cnr_demo/camera8/video.mp4",
     "truth": "data/cnr_demo/camera8/ground_truth.csv", "info": "data/cnr_demo/camera8/info.json",
     "config": "configs/cnr_camera8.json"},
]
MODELS = ["models/classifier.onnx", "models/classifier.json", "models/yolo/yolo11n_coco_1280.onnx"]


def photo_labels(demo: dict, info: dict, n: int) -> list[str]:
    if "photos" in info:      # PKLot: PUCPR_2012-09-20_06_34_15.jpg → 06:34
        return [p.rsplit("_", 3)[-3] + ":" + p.rsplit("_", 3)[-2] for p in info["photos"]][:n]
    return [f"{t} · {w}" for t, w in zip(info["times"], info["weather"])][:n]


def build_demo(demo: dict, out: Path) -> dict:
    info = json.loads((ROOT / demo["info"]).read_text(encoding="utf-8"))
    hold = int(info.get("frames_per_photo", info.get("hold_frames", 6)))
    config = CameraConfig.load(ROOT / demo["config"])
    truth: dict[int, dict[str, int]] = {}
    with open(ROOT / demo["truth"], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] in ("vacant", "occupied"):
                truth.setdefault(int(row["frame"]), {})[row["slot_id"]] = int(row["status"] == "occupied")
    folder = out / "demo" / demo["id"]
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(ROOT / demo["video"]))
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % hold == 0:   # khung đầu của mỗi ảnh
            name = f"{len(frames):03d}.jpg"
            cv2.imwrite(str(folder / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            frames.append({"src": f"demo/{demo['id']}/{name}", "truth": truth.get(idx, {})})
        idx += 1
    cap.release()
    for f, label in zip(frames, photo_labels(demo, info, len(frames))):
        f["label"] = label
    return {"id": demo["id"], "name": demo["name"], "hold": hold, "frame_size": list(config.frame_size),
            "slots": [{"id": s.id, "polygon": s.polygon.round(1).tolist()} for s in config.slots], "frames": frames}


def reference_probs(out: Path, camera: dict) -> list[float]:
    """Xác suất có xe của ảnh đầu tiên theo đúng pipeline Python – để kiểm tra bản JavaScript."""
    from src.vision.classifier import SlotClassifier
    from src.vision.slot_cropper import SlotCropper
    config = CameraConfig.load(ROOT / DEMOS[0]["config"])
    frame = cv2.imread(str(out / camera["frames"][0]["src"]))
    probs = SlotClassifier(ROOT / "models/classifier.onnx").predict_proba(SlotCropper(config).crop(frame))
    return [round(float(p), 5) for p in probs]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--reference", action="store_true", help="ghi thêm xác suất tham chiếu từ ONNX Runtime Python")
    args = p.parse_args(argv)
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web", out)
    shutil.copy(ROOT / "src/backend/static/style.css", out / "style.css")
    for m in MODELS:
        (out / m).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / m, out / m)
    cameras = [build_demo(d, out) for d in DEMOS]
    (out / "demo" / "cameras.json").write_text(json.dumps(cameras, ensure_ascii=False), encoding="utf-8")
    if args.reference:
        (out / "demo" / "reference.json").write_text(json.dumps(reference_probs(out, cameras[0])), encoding="utf-8")
    (out / ".nojekyll").touch()
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"{out}: {sum(len(c['frames']) for c in cameras)} ảnh demo, {size:.1f} MB")


if __name__ == "__main__":
    main()
