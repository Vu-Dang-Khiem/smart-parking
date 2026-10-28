import csv
from pathlib import Path

import cv2
import numpy as np

from src.vision.slot_annotator import Annotator
from src.vision.slot_cropper import CameraConfig, Slot
from src.vision.worker import BackendClient, parse_args, run

FIXTURE = Path(__file__).parent / "fixtures" / "red_green.onnx"
SLOTS = {"A01": (100, 100), "A02": (300, 100), "A03": (500, 100)}  # góc trái trên, ô 120×160


def rect(x, y, w=120, h=160):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def write_video(path, n_frames=20):
    # A01 luôn trống (xanh), A02 luôn có xe (đỏ), A03 có xe vào từ frame 10
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
    for i in range(n_frames):
        frame = np.full((360, 640, 3), 90, np.uint8)
        for sid, (x, y) in SLOTS.items():
            occupied = sid == "A02" or (sid == "A03" and i >= 10)
            cv2.rectangle(frame, (x, y), (x + 120, y + 160), (0, 0, 230) if occupied else (0, 220, 0), -1)
        out.write(frame)
    out.release()


def test_worker_end_to_end(tmp_path):
    video, config, log = tmp_path / "v.avi", tmp_path / "cam.json", tmp_path / "log.csv"
    write_video(video)
    CameraConfig("cam01", (640, 360), [Slot(sid, rect(x, y)) for sid, (x, y) in SLOTS.items()]).save(config)
    args = parse_args(["--video", str(video), "--config", str(config), "--model", str(FIXTURE),
                       "--fps", "10", "--no-pace", "--log-csv", str(log)])
    stats = run(args)
    assert stats["frames"] == 20
    assert stats["state"] == {"A01": "vacant", "A02": "occupied", "A03": "occupied"}
    rows = list(csv.DictReader(open(log, encoding="utf-8")))
    assert len(rows) == 20 * 3
    a03 = [r["state"] for r in rows if r["slot_id"] == "A03"]
    assert a03[12] == "vacant" and a03[13] == "occupied"  # xe vào frame 10, đủ 4/5 phiếu ở frame 13


def test_annotator_builds_slots_and_undo():
    ann = Annotator("cam01", (640, 360), prefix="A")
    for x, y in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        slot = ann.add_point(x, y)
    assert slot.id == "A01" and len(ann.slots) == 1
    ann.add_point(20, 20)
    ann.undo()  # xoá góc đang vẽ dở
    assert ann.pending == [] and len(ann.slots) == 1
    ann.undo()  # xoá ô vừa vẽ
    assert ann.slots == []


def test_backend_client_sends_worker_token(monkeypatch):
    monkeypatch.setenv("SMART_PARKING_WORKER_TOKEN", "s3cret")
    client = BackendClient("http://127.0.0.1:1", CameraConfig("cam", (10, 10), []))
    assert client.session.headers["X-Worker-Token"] == "s3cret"
