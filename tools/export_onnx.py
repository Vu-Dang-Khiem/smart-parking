"""Xuất trọng số .pt (notebook 03) sang ONNX + JSON tiền xử lý để chạy trong hệ thống / tools.

    python tools/export_onnx.py --arch resnet18 --weights models/checkpoints/resnet18.pt --out models/classifier_resnet18.onnx

Kiến trúc phải giống notebooks/03_train_classifier. Cần gói onnx.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]


class SmallCNN(nn.Module):
    # Giống hệt SmallCNN trong notebooks/03_train_classifier
    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, x):
        return self.head(self.features(x))


def build(arch: str) -> nn.Module:
    if arch == "small_cnn":
        return SmallCNN()
    if arch == "mobilenet_v3":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
        return m
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m
    raise ValueError(arch)


def export(arch: str, weights: Path, out: Path, input_size: int = 128) -> float:
    model = build(arch)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    torch.onnx.export(model, torch.randn(1, 3, input_size, input_size), out, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    config = json.loads((ROOT / "models" / "classifier.json").read_text(encoding="utf-8"))   # cùng tiền xử lý
    config.update({"model": arch, "weights": weights.name})
    for key in ("test_accuracy", "test_f1"):
        config.pop(key, None)
    out.with_suffix(".json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    import onnxruntime as ort   # kiểm tra ONNX cho cùng kết quả với PyTorch
    x = torch.randn(8, 3, input_size, input_size)
    with torch.no_grad():
        ref = model(x).numpy()
    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    return float(np.abs(ref - got).max())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arch", required=True, choices=["small_cnn", "mobilenet_v3", "resnet18"])
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    diff = export(args.arch, Path(args.weights), Path(args.out))
    print(f"Đã xuất {args.out} – chênh lệch PyTorch vs ONNX: {diff:.2e}")
    if diff > 1e-3:
        sys.exit("Chênh lệch quá lớn, kiểm tra lại kiến trúc")


if __name__ == "__main__":
    main()
