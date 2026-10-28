"""Tạo video demo time-lapse "bãi xe mới" từ ảnh CNR-EXT (Ý) – bãi mà mô hình chưa từng thấy khi train.

    python tools/make_cnr_demo.py --camera 8 --day 2015-11-27

Kết quả: data/cnr_demo/camera<N>/video.mp4, ground_truth.csv và configs/cnr_camera<N>.json (toạ độ ô của CNR-EXT).
Cần dữ liệu CNR-EXT trong data/cnrpark/ (xem tools/eval_cnrpark.py).
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from eval_cnrpark import DATA, WEATHER, extract_once, load_labels, read_camera_boxes  # noqa: E402
from make_pklot_demo import write_video  # noqa: E402
from src.vision.slot_cropper import CameraConfig, Slot  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--camera", type=int, default=8, choices=range(1, 10))
    p.add_argument("--day", default=None, help="YYYY-MM-DD; mặc định chọn ngày có tỷ lệ lấp đầy thay đổi nhiều nhất")
    p.add_argument("--hold", type=float, default=3.0, help="số giây giữ mỗi ảnh")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--out", default=None, help="thư mục kết quả; mặc định data/cnr_demo/camera<N>")
    args = p.parse_args(argv)

    labels = load_labels()
    full = extract_once(DATA / "CNR-EXT_FULL_IMAGE_1000x750.tar", "FULL_IMAGE_1000x750")
    cam = labels[labels["cam"] == args.camera]
    if args.day is None:   # ngày mà bãi đầy/vơi rõ nhất → demo dễ thấy ô đổi màu
        occ = cam.groupby(["date", "time"])["occupancy"].mean().groupby("date")
        args.day = (occ.max() - occ.min()).idxmax()
    day = cam[cam["date"] == args.day].sort_values("time")
    if day.empty:
        raise SystemExit(f"Camera {args.camera} không có ảnh ngày {args.day}")

    boxes = read_camera_boxes(full)[args.camera]
    slots = [Slot(str(r.slot_id), np.array([[r.x, r.y], [r.x + r.w, r.y], [r.x + r.w, r.y + r.h], [r.x, r.y + r.h]],
                                           dtype=np.float32)) for r in boxes.itertuples()]
    per_photo = max(1, round(args.hold * args.fps))
    frames, gt_rows = [], []
    for (weather, hhmm), g in day.groupby(["weather", "time"], sort=False):
        img = cv2.imread(str(full / WEATHER[weather] / args.day / f"camera{args.camera}" / f"{args.day}_{hhmm}.jpg"))
        if img is None:
            continue
        status = {str(r.slot_id): ("occupied" if r.occupancy else "vacant") for r in g.itertuples()}
        for _ in range(per_photo):
            gt_rows += [(len(frames), s.id, status.get(s.id, "")) for s in slots]
            frames.append(img)

    out = Path(args.out) if args.out else ROOT / "data" / "cnr_demo" / f"camera{args.camera}"
    out.mkdir(parents=True, exist_ok=True)
    video = write_video(out / "video", frames, args.fps)
    config_path = ROOT / "configs" / f"cnr_camera{args.camera}.json"
    CameraConfig(f"cnr-camera{args.camera}", (1000, 750), slots).save(config_path)
    with open(out / "ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("frame", "slot_id", "status"), *gt_rows])
    print(f"Ngày {args.day}: {len(frames) // per_photo} ảnh → {video} ({len(frames)} frame)")
    print(f"Cấu hình ô: {config_path} ({len(slots)} ô) | nhãn thật: {out / 'ground_truth.csv'}")


if __name__ == "__main__":
    main()
