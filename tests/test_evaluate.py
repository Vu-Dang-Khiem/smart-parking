import pandas as pd
import pytest

from src.evaluation.evaluate import evaluate


def frames(slot, truths, probs, states):
    gt = pd.DataFrame({"frame": range(len(truths)), "slot_id": slot, "status": truths})
    log = pd.DataFrame({"frame": range(len(truths)), "slot_id": slot, "prob": probs, "state": states})
    return log, gt


def test_raw_vs_smoothed_and_transition_zone():
    truth = ["vacant"] * 4 + ["occupied"] * 6
    probs = [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]          # nhấp nháy ở frame 2
    states = [None, None, None, "vacant", "vacant", "vacant", "vacant", "occupied", "occupied", "occupied"]
    log, gt = frames("A", truth, probs, states)
    r = evaluate(log, gt, settle=3)
    assert r["raw"]["accuracy"] == pytest.approx(0.9)
    assert r["state_changes"] == {"true": 1, "raw": 3, "smoothed": 1, "extra_raw": 2, "extra_smoothed": 0}
    # 3 frame đầu sau khi xe vào (4, 5, 6) là vùng chuyển tiếp → bỏ khỏi "đã ổn định"
    assert r["smoothed_all"]["n"] == 7 and r["smoothed_all"]["accuracy"] == pytest.approx(4 / 7)
    assert r["smoothed_settled"]["n"] == 4 and r["smoothed_settled"]["accuracy"] == 1.0


def test_missing_ground_truth_ignored():
    log, gt = frames("A", ["occupied", None, "occupied"], [0.9, 0.1, 0.9], ["occupied"] * 3)
    r = evaluate(log, gt)
    assert r["raw"]["n"] == 2 and r["raw"]["accuracy"] == 1.0
