"""Chạy worker (src.vision.worker) thành tiến trình riêng cho từng video được tải lên."""
import subprocess
import sys
import time
from pathlib import Path


class WorkerManager:
    def __init__(self, root: Path):
        self.root = root
        self.procs: dict[str, subprocess.Popen] = {}
        self.started: dict[str, float] = {}

    def start(self, key: str, args: list[str], log_path: Path) -> None:
        self.stop(key)
        log = open(log_path, "ab")
        self.procs[key] = subprocess.Popen([sys.executable, "-m", "src.vision.worker", *args],
                                           cwd=self.root, stdout=log, stderr=subprocess.STDOUT)
        self.started[key] = time.monotonic()

    def stop(self, key: str) -> None:
        proc = self.procs.pop(key, None)
        self.started.pop(key, None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def running(self, key: str) -> bool:
        proc = self.procs.get(key)
        return proc is not None and proc.poll() is None

    def running_keys(self) -> list[str]:
        """Các worker đang chạy, cũ nhất trước."""
        return sorted((k for k in self.procs if self.running(k)), key=lambda k: self.started.get(k, 0))

    def stop_all(self) -> None:
        for key in list(self.procs):
            self.stop(key)
