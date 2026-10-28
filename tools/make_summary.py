"""Gom mọi kết quả thành một bảng so sánh cho báo cáo → results/summary_all.csv.

    python tools/make_summary.py

Nguồn: results/classifier/results.csv (notebook 03), results/yolo*/results_yolo.csv (notebook 04/04b),
results/cnrpark/results_*.json (tools/eval_cnrpark.py), data/pklot_demo/log_*.csv (worker --log-csv).
Thiếu nguồn nào thì ô tương ứng để trống.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.evaluate import evaluate  # noqa: E402

RES = ROOT / "results"
DEMO = ROOT / "data" / "pklot_demo"

# (tên hiển thị, khóa trong results.csv/results_yolo.csv, file CNR-EXT, log video demo, ms/frame đo trên laptop)
METHODS = [
    ("Đoán lớp đa số", "majority", None, None, None),
    ("HOG + Linear SVM", "hog_svm", None, None, None),
    ("LBP + RBF SVM", "lbp_svm", None, None, None),
    ("CNN nhỏ (train từ đầu)", "small_cnn", "results_classifier_small_cnn.json", "log_classifier_small_cnn.csv", 191.8),
    ("MobileNetV3-Small (chính)", "mobilenet_v3", "results_classifier.json", "log_models.csv", 87.2),
    ("ResNet18", "resnet18", "results_classifier_resnet18.json", "log_classifier_resnet18.csv", 323.2),
    ("YOLOv8n-PKLot", "yolov8n_pklot", "results_yolov8n_pklot.json", "log_yolo_pklot.csv", 41.5),
    ("YOLOv8n-COCO 640", "yolov8n_coco_640", "results_yolov8n_coco_640.json", "log_yolov8n_coco_640.csv", 35.3),
    ("YOLOv8n-COCO 1280", "yolov8n_coco_1280", "results_yolov8n_coco_1280.json", "log_yolov8n_coco_1280.csv", 155.7),
    ("YOLO11n-PKLot", "yolo11n_pklot", "results_yolo11n_pklot.json", "log_yolo11n_pklot.csv", 37.1),
    ("YOLO11n-COCO 640", "yolo11n_coco_640", "results_yolo11n_coco_640.json", "log_yolo11n_coco_640.csv", 35.6),
    ("YOLO11n-COCO 1280", "yolo11n_coco_1280", "results_yolo11n_coco_1280.json", "log_yolo11n_coco_1280.csv", 177.7),
]


def kaggle_tables() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in [RES / "classifier" / "results.csv", *RES.glob("yolo*/results_yolo.csv")] if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["experiment", "model", "test_camera"])


def main():
    k = kaggle_tables()
    rows = []
    for name, key, cnr_file, demo_log, ms in METHODS:
        indomain = k[(k["experiment"] == "in-domain") & (k["model"] == key) & (k["test_camera"] == "all")]
        loco = k[(k["experiment"] == "LOCO") & (k["model"] == key)]
        cnr = RES / "cnrpark" / cnr_file if cnr_file else None
        log = DEMO / demo_log if demo_log else None
        if indomain.empty and not (cnr and cnr.exists()):
            continue  # phương pháp chưa có kết quả (ví dụ YOLO11n trước khi chạy notebook 04b)
        row = {"phương pháp": name,
               "PKLot test – acc": indomain["accuracy"].iloc[0] if len(indomain) else None,
               "PKLot test – F1": indomain["f1"].iloc[0] if len(indomain) else None,
               "PKLot bỏ một camera – acc TB": loco["accuracy"].mean() if len(loco) else None,
               "CNR-EXT – acc": None, "CNR-EXT – recall": None, "video demo – acc": None,
               "ms/frame (100 ô, CPU)": ms, "FPS": round(1000 / ms, 1) if ms else None}
        if cnr and cnr.exists():
            o = json.loads(cnr.read_text(encoding="utf-8"))["ours"]["overall"]
            row["CNR-EXT – acc"], row["CNR-EXT – recall"] = o["accuracy"], o["recall"]
        if log and log.exists():
            row["video demo – acc"] = evaluate(pd.read_csv(log), pd.read_csv(DEMO / "ground_truth.csv"))["raw"]["accuracy"]
        rows.append(row)
    table = pd.DataFrame(rows)
    out = RES / "summary_all.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))
    print(f"\nĐã lưu {out}")


if __name__ == "__main__":
    main()
