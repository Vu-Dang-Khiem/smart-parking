"""Đánh giá mô hình (train trên PKLot) trên CNR-EXT – bãi xe khác hoàn toàn (Ý, 9 camera, nắng/mây/mưa).

Dùng đúng các module của hệ thống chạy thật nên kết quả phản ánh đúng những gì demo sẽ làm.

    python tools/eval_cnrpark.py --model models/classifier.onnx
    python tools/eval_cnrpark.py --method yolo --model models/yolo/yolov8n_coco_1280.onnx --cover-thr 0.3

Hai cách đánh giá (--protocol):
  ours     cắt ô từ ảnh toàn cảnh 1000×750 theo toạ độ ô của CNR-EXT, đúng cách hệ thống làm (bbox + 10%).
  patches  dùng ảnh ô 150×150 do CNRPark cắt sẵn (cách cắt khác) – để so với các bài báo dùng CNR-EXT.

Dữ liệu (http://cnrpark.it, license ODbL) đặt trong data/cnrpark/: CNRPark+EXT.csv,
CNR-EXT_FULL_IMAGE_1000x750.tar, CNR-EXT-Patches-150x150.zip – script tự giải nén lần đầu.
"""
import argparse
import json
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.vision.classifier import SlotClassifier  # noqa: E402
from src.vision.slot_cropper import CameraConfig, Slot  # noqa: E402
from src.vision.worker import ClassifierPredictor  # noqa: E402
from src.vision.yolo_detector import YoloOnnx, YoloSlotPredictor  # noqa: E402

DATA = ROOT / "data" / "cnrpark"
WEATHER = {"S": "SUNNY", "O": "OVERCAST", "R": "RAINY"}
ORIG_W, ORIG_H = 2592, 1944    # toạ độ ô trong CSV của CNR-EXT theo ảnh gốc, cần scale về 1000×750


def extract_once(archive: Path, marker: str) -> Path:
    out = DATA / marker
    if not out.exists():
        print(f"Giải nén {archive.name} ...")
        t0 = time.time()
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(DATA)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(DATA)
        print(f"  xong sau {time.time() - t0:.0f}s")
    return out


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(DATA / "CNRPark+EXT.csv", dtype={"camera": str})
    df = df[df["image_url"].str.startswith("CNR-EXT/")].copy()
    df["cam"] = df["camera"].astype(int)
    df["date"], df["time"] = df["datetime"].str.split("_").str[0], df["datetime"].str.split("_").str[1].str.replace(".", "")
    return df


def read_camera_boxes(full_root: Path) -> dict[int, pd.DataFrame]:
    """CSV toạ độ ô của từng camera (cột: slot id, x, y, w, h theo ảnh 2592×1944) → scale về 1000×750."""
    boxes = {}
    for csv in sorted(full_root.parent.rglob("camera*.csv")):
        cam = int("".join(ch for ch in csv.stem if ch.isdigit()))
        t = pd.read_csv(csv)
        cols = {c.lower().strip(): c for c in t.columns}
        sid = next(cols[c] for c in cols if "slot" in c or c == "id")
        x, y, w, h = (cols[k] for k in ("x", "y", "w", "h"))
        sx, sy = 1000 / ORIG_W, 750 / ORIG_H
        boxes[cam] = pd.DataFrame({"slot_id": t[sid].astype(int), "x": t[x] * sx, "y": t[y] * sy,
                                   "w": t[w] * sx, "h": t[h] * sy})
    if len(boxes) != 9:
        raise RuntimeError(f"Cần 9 file camera*.csv trong {full_root.parent}, tìm thấy {sorted(boxes)}")
    return boxes


def make_predictor(args, config: CameraConfig):
    if args.method == "yolo":
        return YoloSlotPredictor(config, YoloOnnx(args.model, conf=args.conf), match_iou=args.match_iou,
                                 cover_thr=args.cover_thr)
    return ClassifierPredictor(config, args.model)


def eval_ours(args, labels: pd.DataFrame) -> pd.DataFrame:
    full_root = extract_once(DATA / "CNR-EXT_FULL_IMAGE_1000x750.tar", "FULL_IMAGE_1000x750")
    boxes = read_camera_boxes(full_root)
    predictors = {}
    rows, missing = [], 0
    frames = list(labels.groupby(["cam", "weather", "date", "time"]))
    if args.limit_frames:
        frames = frames[:: max(1, len(frames) // args.limit_frames)][: args.limit_frames]
    t0 = time.time()
    for i, ((cam, weather, date, hhmm), g) in enumerate(frames, 1):
        path = full_root / WEATHER[weather] / date / f"camera{cam}" / f"{date}_{hhmm}.jpg"
        frame = cv2.imread(str(path))
        if frame is None:
            missing += 1
            continue
        if cam not in predictors:
            b = boxes[cam]
            slots = [Slot(str(r.slot_id), np.array([[r.x, r.y], [r.x + r.w, r.y], [r.x + r.w, r.y + r.h], [r.x, r.y + r.h]],
                                                   dtype=np.float32)) for r in b.itertuples()]
            predictors[cam] = (make_predictor(args, CameraConfig(f"cnr{cam}", (1000, 750), slots)), [s.id for s in slots])
        pred, slot_ids = predictors[cam]
        probs = dict(zip(slot_ids, pred.predict(frame)))
        for r in g.itertuples():
            p = probs.get(str(r.slot_id))
            if p is not None:
                rows.append((cam, weather, date, hhmm, r.slot_id, r.occupancy, float(p)))
        if i % 500 == 0:
            print(f"  {i}/{len(frames)} khung hình, {(time.time() - t0) / i * 1000:.0f} ms/khung hình")
    if missing:
        print(f"Bỏ qua {missing} khung hình không tìm thấy ảnh")
    return pd.DataFrame(rows, columns=["cam", "weather", "date", "time", "slot_id", "y", "prob"])


def eval_patches(args, labels: pd.DataFrame) -> pd.DataFrame:
    if args.method == "yolo":
        raise SystemExit("--protocol patches chỉ dùng cho --method classifier (YOLO cần ảnh toàn cảnh)")
    extract_once(DATA / "CNR-EXT-Patches-150x150.zip", "PATCHES")
    clf = SlotClassifier(args.model)
    df = labels if not args.limit_frames else labels.sample(min(len(labels), args.limit_frames * 35), random_state=0)
    paths = [DATA / u.removeprefix("CNR-EXT/") for u in df["image_url"]]
    probs, t0 = [], time.time()
    for start in range(0, len(paths), 512):
        size = (clf.input_size, clf.input_size)   # vài patch không đúng 150×150 → resize từng ảnh trước khi ghép batch
        imgs = [cv2.resize(cv2.imread(str(p)), size, interpolation=cv2.INTER_AREA) for p in paths[start:start + 512]]
        probs.append(clf.predict_proba(np.stack(imgs)))
        if start // 512 % 50 == 0:
            print(f"  {start + len(imgs)}/{len(paths)} ô, {time.time() - t0:.0f}s")
    return pd.DataFrame({"cam": df["cam"].to_numpy(), "weather": df["weather"].to_numpy(), "date": df["date"].to_numpy(),
                         "time": df["time"].to_numpy(), "slot_id": df["slot_id"].to_numpy(),
                         "y": df["occupancy"].to_numpy(), "prob": np.concatenate(probs)})


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"n": int(len(y)), "accuracy": float((y == pred).mean()), "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0}


def report(res: pd.DataFrame, name: str) -> dict:
    res = res.assign(pred=(res["prob"] >= 0.5).astype(int))
    out = {"overall": metrics(res["y"].to_numpy(), res["pred"].to_numpy()),
           "by_camera": {f"camera{c}": metrics(g["y"].to_numpy(), g["pred"].to_numpy()) for c, g in res.groupby("cam")},
           "by_weather": {WEATHER[w]: metrics(g["y"].to_numpy(), g["pred"].to_numpy()) for w, g in res.groupby("weather")}}
    o = out["overall"]
    print(f"\n=== {name}: {o['n']:,} ô | accuracy {o['accuracy']:.4f} | precision {o['precision']:.4f} | "
          f"recall {o['recall']:.4f} | F1 {o['f1']:.4f}")
    for group in ("by_camera", "by_weather"):
        print("  " + "  ".join(f"{k}: {v['accuracy']:.3f}" for k, v in out[group].items()))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--method", choices=["classifier", "yolo"], default="classifier")
    p.add_argument("--model", default=str(ROOT / "models" / "classifier.onnx"))
    p.add_argument("--protocol", choices=["ours", "patches", "both"], default="both")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--match-iou", type=float, default=0.3)
    p.add_argument("--cover-thr", type=float, default=0.3)
    p.add_argument("--limit-frames", type=int, default=None, help="chỉ chạy một phần để thử nhanh")
    p.add_argument("--out", default=None, help="file JSON kết quả (mặc định data/cnrpark/results_<model>.json)")
    args = p.parse_args(argv)
    if args.method == "yolo" and args.protocol == "both":
        args.protocol = "ours"

    labels = load_labels()
    print(f"CNR-EXT: {len(labels):,} ô có nhãn, {labels.groupby(['cam', 'datetime']).ngroups:,} khung hình")
    results = {"model": args.model, "method": args.method}
    for proto in (["ours", "patches"] if args.protocol == "both" else [args.protocol]):
        res = eval_ours(args, labels) if proto == "ours" else eval_patches(args, labels)
        results[proto] = report(res, f"{Path(args.model).stem} – {proto}")
        res.to_csv(DATA / f"pred_{Path(args.model).stem}_{proto}.csv", index=False)
    out = Path(args.out) if args.out else DATA / f"results_{Path(args.model).stem}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu {out}")


if __name__ == "__main__":
    main()
