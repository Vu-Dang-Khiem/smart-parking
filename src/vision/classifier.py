"""Phân loại trạng thái ô đỗ bằng mô hình ONNX xuất từ notebooks/03_train_classifier."""
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class SlotClassifier:
    def __init__(self, model_path: str | Path, config_path: str | Path | None = None, threads: int = 0):
        model_path = Path(model_path)
        # classifier.json nằm cạnh file .onnx, ghi lại tiền xử lý lúc train
        config_path = Path(config_path) if config_path else model_path.with_suffix(".json")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.input_size = int(self.config["input_size"])
        self.runtime_pad = float(self.config["runtime_pad"])
        self.classes = self.config["classes"]
        self.mean = np.array(self.config["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(self.config["std"], dtype=np.float32).reshape(1, 3, 1, 1)

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.config.get("input_name", self.session.get_inputs()[0].name)

    def preprocess(self, crops_bgr: np.ndarray) -> np.ndarray:
        """(N, S, S, 3) BGR uint8 → (N, 3, S, S) float32 đã chuẩn hóa như lúc train."""
        if crops_bgr.shape[1:3] != (self.input_size, self.input_size):
            crops_bgr = np.stack([cv2.resize(c, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
                                  for c in crops_bgr])
        x = crops_bgr[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32) / 255.0  # BGR → RGB, NCHW
        return (x - self.mean) / self.std

    def predict_proba(self, crops_bgr: np.ndarray) -> np.ndarray:
        """Xác suất occupied của từng ô, shape (N,)."""
        if len(crops_bgr) == 0:
            return np.zeros(0, dtype=np.float32)
        logits = self.session.run(None, {self.input_name: self.preprocess(crops_bgr)})[0]
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return (exp[:, 1] / exp.sum(axis=1)).astype(np.float32)
