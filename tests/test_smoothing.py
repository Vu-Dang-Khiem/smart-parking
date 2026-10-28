import pytest

from src.vision.smoothing import OCCUPIED, VACANT, SlotSmoother


def run(raw: str, **kw):
    """Chạy một ô qua chuỗi nhãn thô ('V' = vacant, 'O' = occupied), trả về chuỗi trạng thái sau làm mượt."""
    sm = SlotSmoother(["A"], **kw)
    states = []
    for ch in raw:
        sm.update([0.9 if ch == "O" else 0.1])
        states.append({VACANT: "V", OCCUPIED: "O", None: "?"}[sm.state["A"]])
    return "".join(states)


def flips(seq: str) -> int:
    seq = seq.replace("?", "")
    return sum(a != b for a, b in zip(seq, seq[1:]))


def test_filters_flicker_and_switches_after_k_votes():
    # Người đi ngang ở frame 4, xe vào từ frame 9, bóng đổ ở frame 15 và 22 (ví dụ trên trang kiến trúc)
    raw = "VVVOVVVVOOOOOOVOOOOOOVOO"
    smooth = run(raw)
    assert flips(raw) == 7
    assert flips(smooth) == 1
    assert smooth.index("O") == 11  # trễ 3 frame sau khi xe vào


def test_state_unknown_until_k_frames():
    assert run("VVVVV") == "???VV"


def test_returns_changes_with_confidence():
    sm = SlotSmoother(["A", "B"], window=3, k=2)
    assert sm.update([0.9, 0.2]) == []
    changes = sm.update([0.8, 0.1])
    assert [(c.slot_id, c.status) for c in changes] == [("A", OCCUPIED), ("B", VACANT)]
    assert changes[0].confidence == pytest.approx(0.85)
    assert sm.counts() == {"vacant": 1, "occupied": 1, "unknown": 0}


def test_wrong_number_of_probs_raises():
    with pytest.raises(ValueError):
        SlotSmoother(["A", "B"]).update([0.5])
