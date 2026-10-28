"""Trạng thái chỉ cần trong bộ nhớ: kết nối WebSocket của dashboard và ảnh overlay mới nhất từ worker."""
import asyncio
import time
from dataclasses import dataclass

from fastapi import WebSocket


class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, message: dict) -> None:
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                self.clients.discard(ws)


@dataclass
class Frame:
    data: bytes
    seq: int
    received: float


class FrameStore:
    def __init__(self):
        self.frames: dict[str, Frame] = {}

    def put(self, camera_id: str, data: bytes) -> None:
        prev = self.frames.get(camera_id)
        self.frames[camera_id] = Frame(data, prev.seq + 1 if prev else 0, time.time())

    def get(self, camera_id: str) -> Frame | None:
        return self.frames.get(camera_id)

    async def mjpeg(self, camera_id: str, is_disconnected):
        last = -1
        while not await is_disconnected():
            frame = self.get(camera_id)
            if frame and frame.seq != last:
                last = frame.seq
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame.data + b"\r\n"
            await asyncio.sleep(0.1)
