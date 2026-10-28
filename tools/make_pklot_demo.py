"""Tạo video demo time-lapse từ ảnh PKLot thật (chưa cần video tự quay).

Ghép các ảnh liên tiếp (cách nhau ~5 phút) của một camera thành video, mỗi ảnh giữ `--hold` giây,
kèm file cấu hình ô đỗ và nhãn thật từng ô theo từng frame video để đánh giá cả hệ thống.

    python tools/make_pklot_demo.py --camera PUCPR --day 2012-09-20 --start 06:30 --photos 48

Kết quả trong --out (mặc định data/pklot_demo):
    video.mp4, ground_truth.csv (frame, slot_id, status) và configs/pklot_<camera>.json
Đặt biến môi trường HF_TOKEN nếu bị Hugging Face giới hạn tải (HTTP 429).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.vision.slot_cropper import CameraConfig, Slot  # noqa: E402

HF_REPO = "dronefreak/PKLot"
STATUS = {0: "vacant", 1: "occupied"}


def hf_file(path: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_REPO, path, repo_type="dataset", local_dir=cache, token=os.environ.get("HF_TOKEN")))


def write_video(path: Path, frames, fps: float) -> Path:
    h, w = frames[0].shape[:2]
    for fourcc, suffix in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        out_path = path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if writer.isOpened():
            for f in frames:
                writer.write(f)
            writer.release()
            if cv2.VideoCapture(str(out_path)).read()[0]:
                return out_path
        out_path.unlink(missing_ok=True)
    raise RuntimeError("OpenCV không ghi được video (mp4v/MJPG)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--camera", default="PUCPR", choices=["PUCPR", "UFPR04", "UFPR05"])
    p.add_argument("--day", default="2012-09-20", help="ngày chụp, xem tên file trong tập test")
    p.add_argument("--start", default="06:30", help="giờ bắt đầu HH:MM")
    p.add_argument("--photos", type=int, default=48, help="số ảnh liên tiếp (mỗi ảnh cách ~5 phút)")
    p.add_argument("--hold", type=float, default=3.0, help="số giây giữ mỗi ảnh trong video")
    p.add_argument("--fps", type=float, default=2.0, help="fps của video (nên bằng --fps của worker)")
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--out", default=str(ROOT / "data" / "pklot_demo"))
    args = p.parse_args(argv)

    out, cache = Path(args.out), ROOT / "data" / "pklot_cache"
    out.mkdir(parents=True, exist_ok=True)
    meta = hf_file(f"data/images/{args.split}/metadata.jsonl", cache)
    start = args.start.replace(":", "_")
    recs = [json.loads(line) for line in meta.read_text().splitlines()]
    recs = sorted((r for r in recs if r["file_name"].startswith(f"{args.camera}_{args.day}_")
                   and r["file_name"][len(args.camera) + 12:] >= start), key=lambda r: r["file_name"])[:args.photos]
    if not recs:
        raise SystemExit(f"Không có ảnh {args.camera} ngày {args.day} từ {args.start} trong tập {args.split}")

    # Ô đỗ: hợp các bbox (toạ độ giữ nguyên giữa các ảnh; vài ảnh thiếu ô chưa được gán nhãn ở bản gốc)
    boxes: dict[tuple, int] = {}
    for r in recs:
        for b in r["objects"]["bbox"]:
            boxes.setdefault(tuple(b), len(boxes))
    prefix = args.camera[0] + args.camera[-1]
    slot_ids = [f"{prefix}{i + 1:03d}" for i in range(len(boxes))]
    slots = [Slot(sid, np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32))
             for sid, (x, y, w, h) in zip(slot_ids, boxes)]
    camera_id = f"pklot-{args.camera.lower()}"

    per_photo = max(1, round(args.hold * args.fps))
    frames, gt_rows = [], []
    for i, r in enumerate(recs):
        print(f"[{i + 1}/{len(recs)}] {r['file_name']}")
        img = cv2.imread(str(hf_file(f"data/images/{args.split}/{r['file_name']}", cache)))
        status = {boxes[tuple(b)]: STATUS[c] for b, c in zip(r["objects"]["bbox"], r["objects"]["categories"])}
        for _ in range(per_photo):
            frame_idx = len(frames)
            frames.append(img)
            gt_rows += [(frame_idx, sid, status.get(j, "")) for j, sid in enumerate(slot_ids)]

    video = write_video(out / "video", frames, args.fps)
    config_path = ROOT / "configs" / f"pklot_{args.camera.lower()}.json"
    config_path.parent.mkdir(exist_ok=True)
    CameraConfig(camera_id, (frames[0].shape[1], frames[0].shape[0]), slots).save(config_path)
    with open(out / "ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("frame", "slot_id", "status"), *gt_rows])
    (out / "info.json").write_text(json.dumps({
        "camera": args.camera, "day": args.day, "split": args.split, "photos": [r["file_name"] for r in recs],
        "frames_per_photo": per_photo, "fps": args.fps, "slots": len(slots), "config": str(config_path.relative_to(ROOT)),
    }, indent=1), encoding="utf-8")
    print(f"\nVideo: {video} ({len(frames)} frame, {len(recs)} ảnh × {per_photo} frame)")
    print(f"Cấu hình ô đỗ: {config_path} ({len(slots)} ô)")
    print(f"Nhãn thật: {out / 'ground_truth.csv'}")


if __name__ == "__main__":
    main()
