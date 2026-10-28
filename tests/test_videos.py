import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.backend.app import create_app
from src.vision.slot_cropper import CameraConfig, Slot
from src.vision.slot_proposer import cluster_boxes


def box(x, y, w=50, h=80, jitter=0.0):
    return np.array([x + jitter, y + jitter, x + w + jitter, y + h + jitter], np.float32)


def test_cluster_keeps_parked_cars_and_drops_passing_cars():
    frames = []
    for i in range(10):
        b = [box(100, 100, jitter=i % 3)]           # xe đỗ suốt video (lệch vài pixel giữa các khung)
        if i >= 6:
            b.append(box(300, 100))                  # xe đỗ vào từ khung 6 (4/10 khung)
        b.append(box(50 + 60 * i, 400))              # xe đang chạy: mỗi khung một chỗ
        frames.append(np.stack(b))
    slots = cluster_boxes(frames, n_frames=10, min_support=0.25)
    assert len(slots) == 2
    xs = sorted(s[0] for s in slots)
    assert xs[0] == pytest.approx(101, abs=2) and xs[1] == pytest.approx(300, abs=1)


def test_cluster_drops_slot_nested_inside_bigger_slot():
    # Ô to (car COCO) và box nhỏ nằm lọt bên trong (IoU thấp) → chỉ giữ ô được ủng hộ nhiều hơn
    frames = [np.stack([box(100, 100, 120, 120), box(110, 110, 40, 50)]) if i < 8 else np.stack([box(100, 100, 120, 120)])
              for i in range(10)]
    slots = cluster_boxes(frames, n_frames=10, min_support=0.25)
    assert len(slots) == 1 and slots[0][2] - slots[0][0] == pytest.approx(120)


def test_cluster_empty():
    assert cluster_boxes([np.zeros((0, 4))] * 5, 5, 0.25).shape == (0, 4)


class FakeProposer:
    def propose(self, video, n_frames, camera_id):
        slot = Slot("S001", np.array([[10, 10], [60, 10], [60, 90], [10, 90]], np.float32))
        return CameraConfig(camera_id, (320, 240), [slot])


@pytest.fixture
def client(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'db.sqlite'}", upload_dir=tmp_path / "uploads", config_dir=tmp_path / "configs")
    started = []
    app.state.workers.start = lambda key, args, log: started.append((key, args))   # không chạy tiến trình thật
    app.state.get_proposer = lambda: FakeProposer()
    with TestClient(app) as c:
        c.started = started
        yield c


def make_video(path, n=12):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for i in range(n):
        out.write(np.full((240, 320, 3), 20 * i % 255, np.uint8))
    out.release()
    return path.read_bytes()


def test_upload_propose_start_flow(client, tmp_path):
    data = make_video(tmp_path / "lot.avi")
    meta = client.post("/api/videos?filename=lot.avi", content=data).json()
    assert meta["frame_size"] == [320, 240] and meta["frames"] == 12 and meta["camera_id"].startswith("upload-")
    vid = meta["video_id"]

    frame = client.get(f"/api/videos/{vid}/frame.jpg?position=0.5")
    assert frame.status_code == 200 and frame.content[:2] == b"\xff\xd8"
    assert client.get(f"/api/videos/{vid}/slots").json()["slots"] == []

    proposed = client.post(f"/api/videos/{vid}/propose").json()
    assert [s["id"] for s in proposed["slots"]] == ["S001"]

    res = client.post(f"/api/videos/{vid}/start", json={"slots": proposed["slots"]}).json()
    assert res == {"camera_id": meta["camera_id"], "slots": 1}
    key, args = client.started[0]
    assert key == vid and "--loop" in args and args[args.index("--api") + 1].startswith("http://127.0.0.1:")
    config = json.loads((tmp_path / "configs" / f"{meta['camera_id']}.json").read_text(encoding="utf-8"))
    assert config["slots"][0]["id"] == "S001"
    assert client.get(f"/api/cameras/{meta['camera_id']}").json()["counts"]["total"] == 1   # camera đã đăng ký
    assert len(client.get(f"/api/videos/{vid}/slots").json()["slots"]) == 1
    assert client.get("/api/videos").json()[0]["video_id"] == vid
    assert client.post(f"/api/videos/{vid}/stop").json() == {"stopped": vid}


def test_upload_rejects_bad_files(client):
    assert client.post("/api/videos?filename=notes.txt", content=b"hello").status_code == 415
    assert client.post("/api/videos?filename=broken.mp4", content=b"not a video").status_code == 400
    assert client.get("/api/videos").json() == []   # file hỏng đã bị xoá


def test_start_requires_slots_and_known_video(client, tmp_path):
    vid = client.post("/api/videos?filename=lot.avi", content=make_video(tmp_path / "lot.avi")).json()["video_id"]
    assert client.post(f"/api/videos/{vid}/start", json={"slots": []}).status_code == 422
    assert client.post("/api/videos/nope/start", json={"slots": []}).status_code in (404, 422)
    assert client.get("/api/videos/../x/frame.jpg").status_code == 404


def test_upload_page_served(client):
    r = client.get("/upload")
    assert r.status_code == 200 and "Tải video" in r.text


def test_worker_token_protects_write_endpoints(client):
    client.app.state.worker_token = "s3cret"
    camera = {"camera_id": "cam", "frame_size": [320, 240], "slots": [{"id": "A", "polygon": [[0, 0], [9, 0], [9, 9]]}]}
    assert client.post("/api/cameras", json=camera).status_code == 403
    assert client.post("/api/cameras", json=camera, headers={"X-Worker-Token": "s3cret"}).status_code == 200
    jpg = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
    assert client.post("/api/cameras/cam/frame", content=jpg).status_code == 403
    assert client.get("/api/cameras").status_code == 200        # đọc vẫn công khai


def test_upload_limits(client, tmp_path):
    limits = client.app.state.limits
    limits.update(max_upload_mb=0, max_upload_seconds=0)
    data = make_video(tmp_path / "lot.avi")
    assert client.post("/api/videos?filename=lot.avi", content=data).status_code == 413     # quá dung lượng
    limits.update(max_upload_mb=50, max_upload_seconds=1)                                  # video 12 khung / 10 fps
    assert client.post("/api/videos?filename=lot.avi", content=data).status_code == 413     # quá thời lượng
    assert client.get("/api/videos").json() == []
    assert client.get("/api/videos/limits").json()["max_upload_mb"] == 50


def test_running_uploads_capped_and_old_uploads_pruned(client, tmp_path):
    app = client.app
    app.state.limits.update(max_running_uploads=1, upload_minutes=15, keep_uploads=2)
    running, stopped = {}, []
    app.state.workers.start = lambda key, args, log: running.__setitem__(key, args)
    app.state.workers.running_keys = lambda: list(running)
    app.state.workers.running = lambda key: key in running
    app.state.workers.stop = lambda key: (stopped.append(key), running.pop(key, None))
    data = make_video(tmp_path / "lot.avi")
    ids = [client.post("/api/videos?filename=lot.avi", content=data).json()["video_id"] for _ in range(2)]
    slot = {"id": "S1", "polygon": [[0, 0], [9, 0], [9, 9], [0, 9]]}
    for vid in ids:
        client.post(f"/api/videos/{vid}/start", json={"slots": [slot]})
    assert list(running) == [ids[1]] and ids[0] in stopped           # chỉ 1 video chạy, video cũ bị dừng
    assert running[ids[1]][-2:] == ["--max-frames", str(15 * 60 * 2)]
    client.post("/api/videos?filename=lot.avi", content=data)
    remaining = {v["video_id"] for v in client.get("/api/videos").json()}
    assert len(remaining) == 2 and ids[1] in remaining                # giữ 2 video, video đang chạy không bị xoá


def test_demo_cameras_start_on_startup(tmp_path, monkeypatch):
    config = tmp_path / "demo.json"
    CameraConfig("demo-cam", (320, 240), [Slot("S1", np.array([[0, 0], [9, 0], [9, 9]], np.float32))]).save(config)
    monkeypatch.setenv("SMART_PARKING_DEMO_CAMERAS", f"{config}={tmp_path / 'v.mp4'}")
    app = create_app(f"sqlite:///{tmp_path / 'db.sqlite'}", upload_dir=tmp_path / "up", config_dir=tmp_path / "cfg")
    started = []
    app.state.workers.start = lambda key, args, log: started.append((key, args))
    with TestClient(app) as c:
        assert c.get("/api/cameras/demo-cam").json()["counts"]["total"] == 1
    assert started[0][0] == "demo-demo-cam" and "--loop" in started[0][1]
