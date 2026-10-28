"""Làm mượt trạng thái ô đỗ theo thời gian để tránh nhấp nháy.

Mỗi ô giữ `window` nhãn thô gần nhất. Trạng thái chỉ được đặt / đổi khi ít nhất `k` nhãn trong cửa sổ
cùng chỉ về trạng thái mới (mặc định 4/5). Người đi ngang, bóng mây hay xe đang lùi vào chỉ làm nhãn thô
sai vài frame nên không đủ phiếu để đổi trạng thái.
"""
from collections import deque
from dataclasses import dataclass

VACANT, OCCUPIED = "vacant", "occupied"


@dataclass
class Change:
    slot_id: str
    status: str
    confidence: float  # xác suất của trạng thái mới, trung bình trên các frame đã bỏ phiếu cho nó


class SlotSmoother:
    def __init__(self, slot_ids: list[str], window: int = 5, k: int = 4, threshold: float = 0.5):
        if not 0 < k <= window:
            raise ValueError("cần 0 < k <= window")
        self.slot_ids, self.window, self.k, self.threshold = list(slot_ids), window, k, threshold
        self.history = {sid: deque(maxlen=window) for sid in self.slot_ids}
        self.state: dict[str, str | None] = {sid: None for sid in self.slot_ids}  # None = chưa đủ dữ liệu

    def update(self, probs) -> list[Change]:
        """Nhận xác suất occupied của từng ô (cùng thứ tự slot_ids), trả về các ô vừa đổi trạng thái."""
        if len(probs) != len(self.slot_ids):
            raise ValueError(f"cần {len(self.slot_ids)} xác suất, nhận {len(probs)}")
        changes = []
        for sid, p in zip(self.slot_ids, probs):
            hist = self.history[sid]
            hist.append(float(p))
            votes = [q for q in hist if q >= self.threshold]
            if len(votes) >= self.k:
                new, conf = OCCUPIED, sum(votes) / len(votes)
            elif len(hist) - len(votes) >= self.k:
                new, conf = VACANT, sum(1 - q for q in hist if q < self.threshold) / (len(hist) - len(votes))
            else:
                continue
            if new != self.state[sid]:
                self.state[sid] = new
                changes.append(Change(sid, new, round(conf, 4)))
        return changes

    def counts(self) -> dict[str, int]:
        values = list(self.state.values())
        return {"vacant": values.count(VACANT), "occupied": values.count(OCCUPIED), "unknown": values.count(None)}
