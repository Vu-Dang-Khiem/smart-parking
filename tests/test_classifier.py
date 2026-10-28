from pathlib import Path

import numpy as np

from src.vision.classifier import SlotClassifier

FIXTURE = Path(__file__).parent / "fixtures" / "red_green.onnx"  # đỏ → occupied, xanh → vacant


def solid(bgr, size=128):
    return np.full((size, size, 3), bgr, np.uint8)


def test_predicts_batch():
    clf = SlotClassifier(FIXTURE)
    probs = clf.predict_proba(np.stack([solid((0, 0, 255)), solid((0, 255, 0))]))
    assert probs.shape == (2,)
    assert probs[0] > 0.9 and probs[1] < 0.1


def test_preprocess_converts_bgr_to_normalized_rgb():
    clf = SlotClassifier(FIXTURE)
    x = clf.preprocess(solid((0, 0, 255))[None])  # đỏ thuần
    assert x.shape == (1, 3, 128, 128) and x.dtype == np.float32
    np.testing.assert_allclose(x[0, :, 0, 0], (np.array([1, 0, 0]) - clf.mean.ravel()) / clf.std.ravel(), rtol=1e-5)


def test_resizes_crops_of_other_size():
    clf = SlotClassifier(FIXTURE)
    assert clf.predict_proba(solid((0, 0, 255), size=96)[None])[0] > 0.9


def test_empty_batch():
    assert SlotClassifier(FIXTURE).predict_proba(np.zeros((0, 128, 128, 3), np.uint8)).shape == (0,)
