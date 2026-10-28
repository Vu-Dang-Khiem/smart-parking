"""Fine-tune MobileNetV3 (train trên PKLot) bằng vài khung hình có nhãn của bãi xe mới (CNR-EXT) – đo trước/sau.

Mô phỏng việc lắp hệ thống ở bãi mới: chỉ gán nhãn K khung hình cho mỗi camera rồi fine-tune.
Chia CNR-EXT theo NGÀY: các ngày fine-tune và các ngày test không trùng nhau.

    python tools/finetune_cnrpark.py --weights models/checkpoints/mobilenet_v3.pt --shots 5 20 50

Kết quả: data/cnrpark/finetune/results.json, finetune_curve.png và (nếu đã cài gói onnx)
models/classifier_cnr_k<K>.onnx + .json cho mức K lớn nhất.
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from eval_cnrpark import DATA, WEATHER, extract_once, load_labels, read_camera_boxes  # noqa: E402
from src.vision.slot_cropper import CameraConfig, Slot, SlotCropper  # noqa: E402

# Hình học và tiền xử lý giống notebooks/02_make_crops + 03_train_classifier
SAVE_SIZE, SAVE_PAD, INPUT_SIZE, RUNTIME_PAD = 160, 0.25, 128, 0.10
C0 = (SAVE_SIZE - INPUT_SIZE) // 2
MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD_T = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
OUT = DATA / "finetune"


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def normalize(x):
    return (x - MEAN_T) / STD_T


def eval_batch(x):
    """uint8 (N,3,160,160) → CenterCrop(128) = bbox + 10%."""
    return normalize(x[:, :, C0:C0 + INPUT_SIZE, C0:C0 + INPUT_SIZE].float() / 255)


def uniform(n, lo, hi):
    return torch.empty(n).uniform_(lo, hi)


def train_batch(x):
    """Augment giống notebook 03 (cắt lệch + xoay + lật, màu sắc, làm mờ, bóng đổ), chạy trên CPU."""
    n = x.shape[0]
    x = x.float() / 255
    area, ratio = uniform(n, 0.44, 1.0), torch.exp(uniform(n, math.log(0.85), math.log(1.18)))
    w, h = torch.sqrt(area * ratio).clamp(max=1.0), torch.sqrt(area / ratio).clamp(max=1.0)
    cx, cy = uniform(n, -1, 1) * (1 - w), uniform(n, -1, 1) * (1 - h)
    ang = uniform(n, -math.radians(10), math.radians(10))
    flip = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([w * cos * flip, -h * sin, cx], 1), torch.stack([w * sin * flip, h * cos, cy], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, (n, 3, INPUT_SIZE, INPUT_SIZE), align_corners=False), align_corners=False)
    x = x * uniform(n, 0.6, 1.4).view(n, 1, 1, 1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * uniform(n, 0.6, 1.4).view(n, 1, 1, 1) + mean
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    x = (x - gray) * uniform(n, 0.7, 1.3).view(n, 1, 1, 1) + gray
    blur = torch.rand(n) < 0.2
    if blur.any():
        sigma = random.uniform(0.5, 1.5)
        k = torch.exp(-(torch.arange(5, dtype=x.dtype) - 2) ** 2 / (2 * sigma ** 2))
        k = k / k.sum()
        x[blur] = F.conv2d(F.pad(x[blur], (2, 2, 2, 2), mode="reflect"), (k[:, None] * k[None, :]).expand(3, 1, 5, 5), groups=3)
    lin = torch.linspace(-0.5, 0.5, INPUT_SIZE)
    ys, xs = torch.meshgrid(lin, lin, indexing="ij")
    a = uniform(n, 0, 2 * math.pi).view(n, 1, 1)
    shadow = (torch.rand(n) < 0.3).view(n, 1, 1) & (xs * torch.cos(a) + ys * torch.sin(a) > uniform(n, -0.3, 0.3).view(n, 1, 1))
    factor = uniform(n, 0.4, 0.8).view(n, 1, 1)
    x = x * torch.where(shadow, factor, torch.ones_like(factor)).unsqueeze(1)
    return normalize(x.clamp(0, 1))


def build_model(weights: Path):
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    return m


@torch.no_grad()
def predict(model, x_uint8, batch=512):
    model.eval()
    return torch.cat([torch.softmax(model(eval_batch(x_uint8[i:i + batch])), 1)[:, 1]
                      for i in range(0, len(x_uint8), batch)]).numpy()


class CnrCrops:
    """Cắt ô (bbox nới 25%, 160×160 – giống crop train) từ ảnh toàn cảnh CNR-EXT cho các khung hình đã chọn."""

    def __init__(self, labels: pd.DataFrame):
        self.groups = labels.groupby(["cam", "weather", "date", "time"])
        self.full_root = extract_once(DATA / "CNR-EXT_FULL_IMAGE_1000x750.tar", "FULL_IMAGE_1000x750")
        boxes = read_camera_boxes(self.full_root)
        self.croppers, self.slot_ids = {}, {}
        for cam, b in boxes.items():
            slots = [Slot(str(r.slot_id), np.array([[r.x, r.y], [r.x + r.w, r.y], [r.x + r.w, r.y + r.h], [r.x, r.y + r.h]],
                                                   dtype=np.float32)) for r in b.itertuples()]
            self.croppers[cam] = SlotCropper(CameraConfig(f"cnr{cam}", (1000, 750), slots), pad=SAVE_PAD, size=SAVE_SIZE)
            self.slot_ids[cam] = [int(s.id) for s in slots]

    def frames(self, frame_keys):
        """Trả về (ảnh uint8 (N,3,160,160), nhãn, bảng thông tin) cho danh sách khung hình (cam, weather, date, time)."""
        groups = self.groups
        xs, ys, info = [], [], []
        for key in frame_keys:
            cam, weather, date, hhmm = key
            frame = cv2.imread(str(self.full_root / WEATHER[weather] / date / f"camera{cam}" / f"{date}_{hhmm}.jpg"))
            if frame is None:
                continue
            crops = self.croppers[cam].crop(frame)
            index = {sid: i for i, sid in enumerate(self.slot_ids[cam])}
            for r in groups.get_group(key).itertuples():
                if r.slot_id in index:
                    xs.append(crops[index[r.slot_id]][:, :, ::-1])   # BGR → RGB như lúc train
                    ys.append(r.occupancy)
                    info.append((cam, weather, date, hhmm, r.slot_id))
        x = torch.from_numpy(np.ascontiguousarray(np.stack(xs))).permute(0, 3, 1, 2)
        return x, np.array(ys), pd.DataFrame(info, columns=["cam", "weather", "date", "time", "slot_id"])


def pklot_demo_crops():
    """Ô của video demo PKLot (mỗi ảnh gốc một lần) để kiểm tra mô hình có quên bãi cũ không."""
    demo = ROOT / "data" / "pklot_demo"
    if not (demo / "info.json").exists():
        return None
    info = json.loads((demo / "info.json").read_text(encoding="utf-8"))
    gt = pd.read_csv(demo / "ground_truth.csv")
    cfg = CameraConfig.load(ROOT / info["config"])
    cropper = SlotCropper(cfg, pad=SAVE_PAD, size=SAVE_SIZE)
    cap = cv2.VideoCapture(str(next(demo.glob("video.*"))))
    xs, ys, idx = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % info["frames_per_photo"] == 0:
            g = gt[gt["frame"] == idx].set_index("slot_id")["status"]
            crops = cropper.crop(frame)
            for s, c in zip(cfg.slots, crops):
                if g.get(s.id) in ("vacant", "occupied"):
                    xs.append(c[:, :, ::-1]); ys.append(int(g[s.id] == "occupied"))
        idx += 1
    return torch.from_numpy(np.ascontiguousarray(np.stack(xs))).permute(0, 3, 1, 2), np.array(ys)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tp, fp, fn = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"n": int(len(y)), "accuracy": float((pred == y).mean()), "precision": float(prec), "recall": float(rec),
            "f1": float(2 * prec * rec / max(prec + rec, 1e-9))}


def predict_frames(model, data, keys, chunk=64):
    # Đọc và dự đoán theo từng nhóm khung hình để không phải giữ ~50 nghìn ảnh test trong RAM
    ps, ys, infos = [], [], []
    for i in range(0, len(keys), chunk):
        x, y, info = data.frames(keys[i:i + chunk])
        ps.append(predict(model, x)); ys.append(y); infos.append(info)
    return np.concatenate(ps), np.concatenate(ys), pd.concat(infos, ignore_index=True)


def evaluate(model, data, test_keys, pklot):
    p, y, info = predict_frames(model, data, test_keys)
    out = {"cnr_test": metrics(y, p),
           "by_camera": {f"camera{c}": metrics(y[m], p[m]) for c in sorted(info.cam.unique()) for m in [(info.cam == c).to_numpy()]},
           "by_weather": {WEATHER[w]: metrics(y[m], p[m]) for w in sorted(info.weather.unique()) for m in [(info.weather == w).to_numpy()]}}
    if pklot is not None:
        out["pklot_demo"] = metrics(pklot[1], predict(model, pklot[0]))
    return out


def finetune(base, x, y, steps, lr, batch):
    model = copy.deepcopy(base)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    yt = torch.from_numpy(y).long()
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(x), (min(batch, len(x)),))
        loss = F.cross_entropy(model(train_batch(x[idx])), yt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return model.eval()


def export_onnx(model, k):
    try:
        import onnx  # noqa: F401  – torch.onnx cần gói onnx
    except ImportError:
        print("Chưa cài gói onnx (pip install onnx) → bỏ qua bước xuất ONNX")
        return
    path = ROOT / "models" / f"classifier_cnr_k{k}.onnx"
    torch.onnx.export(model, torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE), path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    base_cfg = json.loads((ROOT / "models" / "classifier.json").read_text(encoding="utf-8"))
    base_cfg.update({"model": f"mobilenet_v3 fine-tune CNR-EXT ({k} khung hình/camera)"})
    path.with_suffix(".json").write_text(json.dumps(base_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Đã xuất", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", default=str(ROOT / "models" / "checkpoints" / "mobilenet_v3.pt"), help="state_dict từ notebook 03")
    p.add_argument("--shots", type=int, nargs="+", default=[5, 20, 50], help="số khung hình có nhãn mỗi camera")
    p.add_argument("--test-frac", type=float, default=0.35, help="tỷ lệ số ngày chỉ dùng để test")
    p.add_argument("--steps", type=int, default=300, help="số bước fine-tune (batch --batch ảnh)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--test-stride", type=int, default=1, help="chỉ test 1/N khung hình của các ngày test (chạy nhanh)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    torch.set_num_threads(os.cpu_count() or 4)
    seed_everything(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    labels = load_labels()
    days = sorted(labels["date"].unique())
    rng = random.Random(args.seed)
    test_days = set(rng.sample(days, max(1, round(len(days) * args.test_frac))))
    frames = labels[["cam", "weather", "date", "time"]].drop_duplicates()
    test_frames = frames[frames["date"].isin(test_days)].iloc[::args.test_stride]
    pool = frames[~frames["date"].isin(test_days)]
    print(f"{len(days)} ngày: {len(days) - len(test_days)} ngày để lấy dữ liệu fine-tune, {len(test_days)} ngày test "
          f"({len(test_frames)} khung hình)")

    data = CnrCrops(labels)
    test = [tuple(r) for r in test_frames.itertuples(index=False)]
    pklot = pklot_demo_crops()
    base = build_model(Path(args.weights))

    # Chọn khung hình fine-tune lồng nhau: tập K=5 nằm trong K=20, K=20 nằm trong K=50
    order = pool.sample(frac=1.0, random_state=args.seed).groupby("cam")
    results = {"setup": {"test_days": sorted(test_days), "steps": args.steps, "lr": args.lr, "batch": args.batch},
               "k0_no_finetune": evaluate(base, data, test, pklot)}
    print(f"K=0 (chưa fine-tune): CNR test acc {results['k0_no_finetune']['cnr_test']['accuracy']:.4f}"
          + (f" | PKLot demo acc {results['k0_no_finetune']['pklot_demo']['accuracy']:.4f}" if pklot else ""))
    best_model = None
    for k in sorted(args.shots):
        keys = [tuple(r) for _, g in order for r in g.head(k).itertuples(index=False)]
        x, y, _ = data.frames(keys)
        t0 = time.time()
        model = finetune(base, x, y, args.steps, args.lr, args.batch)
        res = evaluate(model, data, test, pklot)
        res["train_crops"] = int(len(y))
        results[f"k{k}"] = res
        best_model, best_k = model, k
        print(f"K={k:3d} khung hình/camera ({len(y):,} ô, {time.time() - t0:.0f}s): CNR test acc {res['cnr_test']['accuracy']:.4f} "
              f"recall {res['cnr_test']['recall']:.4f}" + (f" | PKLot demo acc {res['pklot_demo']['accuracy']:.4f}" if pklot else ""))

    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    plot(results, args.shots)
    if best_model is not None:
        export_onnx(best_model, best_k)
    print("Đã lưu", OUT / "results.json")


def plot(results, shots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [0] + sorted(shots)
    keys = ["k0_no_finetune"] + [f"k{k}" for k in sorted(shots)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(ks, [results[k]["cnr_test"]["accuracy"] for k in keys], "o-", lw=2, label="CNR-EXT (bãi mới, ngày test)")
    if "pklot_demo" in results[keys[0]]:
        axes[0].plot(ks, [results[k]["pklot_demo"]["accuracy"] for k in keys], "s--", label="PKLot demo (bãi cũ)")
    axes[0].set_xlabel("số khung hình có nhãn / camera"); axes[0].set_ylabel("accuracy")
    axes[0].set_title("Fine-tune bằng vài khung hình của bãi mới"); axes[0].grid(alpha=0.3); axes[0].legend()
    cams = list(results[keys[0]]["by_camera"])
    for c in cams:
        axes[1].plot(ks, [results[k]["by_camera"][c]["accuracy"] for k in keys], "o-", label=c)
    axes[1].set_xlabel("số khung hình có nhãn / camera"); axes[1].set_title("Theo camera")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(OUT / "finetune_curve.png", dpi=120)


if __name__ == "__main__":
    main()
