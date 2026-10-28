from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.backend.app import create_app

CAMERA = {
    "camera_id": "cam01",
    "frame_size": [1280, 720],
    "slots": [{"id": f"A0{i}", "polygon": [[i * 100, 0], [i * 100 + 80, 0], [i * 100 + 80, 120], [i * 100, 120]]}
              for i in range(1, 4)],
}


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(f"sqlite:///{tmp_path / 'test.db'}")) as c:
        assert c.post("/api/cameras", json=CAMERA).status_code == 200
        yield c


def event(changes, ts=None):
    ts = ts or datetime.now(timezone.utc)
    return {"camera_id": "cam01", "ts": ts.isoformat(),
            "changes": [{"slot_id": s, "status": st, "confidence": 0.9} for s, st in changes]}


def test_register_and_detail(client):
    cam = client.get("/api/cameras/cam01").json()
    assert cam["counts"] == {"total": 3, "vacant": 0, "occupied": 0, "unknown": 3}
    assert [s["id"] for s in cam["slots"]] == ["A01", "A02", "A03"]
    assert client.get("/api/cameras").json()[0]["camera_id"] == "cam01"
    assert client.get("/api/cameras/nope").status_code == 404


def test_reregister_keeps_status_and_drops_removed_slots(client):
    client.post("/api/events", json=event([("A01", "occupied")]))
    client.post("/api/cameras", json={**CAMERA, "slots": CAMERA["slots"][:2]})
    cam = client.get("/api/cameras/cam01").json()
    assert [s["id"] for s in cam["slots"]] == ["A01", "A02"]
    assert cam["slots"][0]["status"] == "occupied"


def test_events_update_slots_and_broadcast(client):
    with client.websocket_connect("/ws") as ws:
        r = client.post("/api/events", json=event([("A01", "occupied"), ("A02", "vacant"), ("ZZ", "vacant")]))
        assert r.json() == {"applied": 2}  # ô không tồn tại bị bỏ qua
        msg = ws.receive_json()
    assert msg["type"] == "slots" and [c["slot_id"] for c in msg["changes"]] == ["A01", "A02"]
    assert msg["counts"] == {"total": 3, "vacant": 1, "occupied": 1, "unknown": 1}
    events = client.get("/api/events/cam01").json()
    assert {e["slot_id"] for e in events} == {"A01", "A02"}


def test_invalid_status_rejected(client):
    assert client.post("/api/events", json=event([("A01", "parked")])).status_code == 422


def test_occupancy_series_follows_events(client):
    now = datetime.now(timezone.utc)
    client.post("/api/events", json=event([("A01", "occupied"), ("A02", "vacant"), ("A03", "vacant")], now - timedelta(minutes=50)))
    client.post("/api/events", json=event([("A02", "occupied")], now - timedelta(minutes=20)))
    pts = client.get("/api/stats/cam01/occupancy?hours=1&bucket_minutes=10").json()["points"]
    rates = [p["rate"] for p in pts]
    assert rates[0] == pytest.approx(1 / 3, abs=1e-3) and rates[-1] == pytest.approx(2 / 3, abs=1e-3)
    assert rates == sorted(rates)


def test_frame_upload_and_snapshot(client):
    assert client.get("/api/cameras/cam01/snapshot.jpg").status_code == 404
    jpg = b"\xff\xd8fake-jpeg"
    assert client.post("/api/cameras/cam01/frame", content=jpg, headers={"Content-Type": "image/jpeg"}).status_code == 204
    r = client.get("/api/cameras/cam01/snapshot.jpg")
    assert r.content == jpg and r.headers["content-type"] == "image/jpeg"
    assert client.post("/api/cameras/cam01/frame", content=b"not-jpeg").status_code == 415


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Smart Parking" in r.text
    assert client.get("/static/app.js").status_code == 200
