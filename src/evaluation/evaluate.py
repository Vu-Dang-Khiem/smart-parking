"""Đánh giá cả hệ thống trên video có nhãn thật: so log của worker (--log-csv) với ground truth.

    python -m src.evaluation.evaluate --log data/pklot_demo/worker_log.csv --gt data/pklot_demo/ground_truth.csv

- "Nhãn thô": xác suất ≥ 0.5 tại từng frame (chưa làm mượt).
- "Sau làm mượt": trạng thái của SlotSmoother. Ngay sau khi trạng thái thật đổi, làm mượt cần vài frame
  mới đổi theo (độ trễ có chủ đích), nên `--settle` frame đầu sau mỗi lần đổi được tính riêng là vùng chuyển tiếp.
- "Đổi trạng thái thừa": số lần trạng thái dự đoán đổi nhiều hơn số lần đổi thật – đo độ nhấp nháy.
"""
import argparse
import json

import numpy as np
import pandas as pd


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"n": int(len(y_true)), "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def count_changes(seq: pd.Series) -> int:
    seq = seq.dropna()
    return int((seq != seq.shift()).sum() - 1) if len(seq) else 0


def evaluate(log: pd.DataFrame, gt: pd.DataFrame, settle: int = 3) -> dict:
    df = log.merge(gt.rename(columns={"status": "truth"}), on=["frame", "slot_id"], how="inner")
    df = df[df["truth"].isin(["vacant", "occupied"])].sort_values(["slot_id", "frame"])
    df["y"] = (df["truth"] == "occupied").astype(int)
    df["raw"] = (df["prob"] >= 0.5).astype(int)
    # Số frame kể từ lần gần nhất trạng thái thật của ô thay đổi
    changed = df.groupby("slot_id")["y"].transform(lambda s: s.ne(s.shift()).cumsum())
    df["since_change"] = df.groupby(["slot_id", changed]).cumcount()
    first_seg = changed == 1

    smooth = df[df["state"].isin(["vacant", "occupied"])]
    settled = smooth[(smooth["since_change"] >= settle) | first_seg.loc[smooth.index]]
    true_changes = sum(count_changes(g["truth"]) for _, g in df.groupby("slot_id"))
    raw_changes = sum(count_changes(g["raw"]) for _, g in df.groupby("slot_id"))
    smooth_changes = sum(count_changes(g["state"].where(g["state"].isin(["vacant", "occupied"]))) for _, g in df.groupby("slot_id"))
    return {
        "frames": int(df["frame"].nunique()), "slots": int(df["slot_id"].nunique()),
        "raw": binary_metrics(df["y"].to_numpy(), df["raw"].to_numpy()),
        "smoothed_all": binary_metrics(smooth["y"].to_numpy(), (smooth["state"] == "occupied").astype(int).to_numpy()),
        "smoothed_settled": binary_metrics(settled["y"].to_numpy(), (settled["state"] == "occupied").astype(int).to_numpy()),
        "state_changes": {"true": true_changes, "raw": raw_changes, "smoothed": smooth_changes,
                          "extra_raw": raw_changes - true_changes, "extra_smoothed": smooth_changes - true_changes},
    }


def print_report(r: dict) -> None:
    print(f"{r['frames']} frame × {r['slots']} ô\n")
    print(f"{'':28s}{'n':>8s}{'accuracy':>10s}{'precision':>11s}{'recall':>9s}{'F1':>8s}")
    for key, name in [("raw", "Nhãn thô (chưa làm mượt)"), ("smoothed_all", "Sau làm mượt – mọi frame"),
                      ("smoothed_settled", "Sau làm mượt – đã ổn định")]:
        m = r[key]
        print(f"{name:28s}{m['n']:>8d}{m['accuracy']:>10.4f}{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}")
    c = r["state_changes"]
    print(f"\nSố lần đổi trạng thái: thật {c['true']} | nhãn thô {c['raw']} (thừa {c['extra_raw']}) | "
          f"sau làm mượt {c['smoothed']} (thừa {c['extra_smoothed']})")


def main(argv=None):
    p = argparse.ArgumentParser(description="Đánh giá hệ thống từ log worker và nhãn thật")
    p.add_argument("--log", required=True, help="file CSV từ worker --log-csv")
    p.add_argument("--gt", required=True, help="nhãn thật: frame, slot_id, status")
    p.add_argument("--settle", type=int, default=3, help="số frame chuyển tiếp sau mỗi lần trạng thái thật đổi")
    p.add_argument("--json", help="lưu kết quả ra file JSON")
    args = p.parse_args(argv)
    result = evaluate(pd.read_csv(args.log), pd.read_csv(args.gt), args.settle)
    print_report(result)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
