// Các bước xử lý ảnh của hệ thống, viết lại bằng JavaScript để chạy trong trình duyệt.
// Bám sát bản Python: src/vision/slot_cropper.py, classifier.py, smoothing.py, yolo_detector.py, slot_proposer.py.
(function (root) {
  "use strict";

  // round() của Python/NumPy làm tròn nửa về số chẵn (0.5 → 0, 1.5 → 2)
  function roundHalfEven(x) {
    const r = Math.round(x);
    return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
  }

  // ---------- Cắt ô: hình chữ nhật bao polygon, nới `pad` mỗi phía (slot_cropper.crop_box) ----------
  function cropBox(polygon, pad) {
    const xs = polygon.map((p) => p[0]), ys = polygon.map((p) => p[1]);
    const x = Math.min(...xs), y = Math.min(...ys), w = Math.max(...xs) - x, h = Math.max(...ys) - y;
    return [roundHalfEven(x - pad * w), roundHalfEven(y - pad * h), roundHalfEven(x + w * (1 + pad)), roundHalfEven(y + h * (1 + pad))];
  }

  function scalePolygons(slots, frameSize, width, height) {
    const sx = width / frameSize[0], sy = height / frameSize[1];
    return slots.map((s) => s.polygon.map(([x, y]) => [x * sx, y * sy]));
  }

  // Hệ số nội suy theo một chiều giống cv2.resize(..., INTER_AREA):
  // thu nhỏ cả hai chiều → lấy trung bình theo diện tích; còn lại → nội suy tuyến tính kiểu "area" của OpenCV.
  function areaTable(src, dst) {
    const scale = src / dst, table = [];
    for (let d = 0; d < dst; d++) {
      const f1 = d * scale, f2 = f1 + scale, taps = [];
      const s1 = Math.ceil(f1), s2 = Math.floor(f2);
      if (s1 - f1 > 1e-3) taps.push([s1 - 1, (s1 - f1) / scale]);
      for (let s = s1; s < s2; s++) taps.push([s, 1 / scale]);
      if (f2 - s2 > 1e-3) taps.push([s2, (Math.min(f2, src) - s2) / scale]);
      table.push(taps);
    }
    return table;
  }

  function linearAreaTable(src, dst) {
    const scale = src / dst, inv = dst / src, table = [];
    for (let d = 0; d < dst; d++) {
      const s = Math.floor(d * scale);
      let f = d + 1 - (s + 1) * inv;
      f = f <= 0 ? 0 : f - Math.floor(f);
      table.push([[s, 1 - f], [s + 1, f]]);
    }
    return table;
  }

  // Cắt vùng box (có thể vượt ngoài khung → lặp điểm ảnh mép, như BORDER_REPLICATE) và resize về size×size.
  // rgba: Uint8ClampedArray của cả khung hình (width×height). Trả về Uint8 RGB (size*size*3), đã làm tròn như cv2.
  function cropResize(rgba, width, height, box, size) {
    const [x0, y0, x1, y1] = box, bw = x1 - x0, bh = y1 - y0;
    const area = bw >= size && bh >= size;
    const tx = area ? areaTable(bw, size) : linearAreaTable(bw, size);
    const ty = area ? areaTable(bh, size) : linearAreaTable(bh, size);
    const clampX = (s) => Math.min(width - 1, Math.max(0, x0 + Math.min(bw - 1, Math.max(0, s))));
    const clampY = (s) => Math.min(height - 1, Math.max(0, y0 + Math.min(bh - 1, Math.max(0, s))));
    // nội suy theo hàng trước (giống OpenCV), rồi theo cột
    const rows = new Map();
    const hrow = (sy) => {
      if (rows.has(sy)) return rows.get(sy);
      const row = new Float64Array(size * 3), base = clampY(sy) * width;
      for (let dx = 0; dx < size; dx++) {
        let r = 0, g = 0, b = 0;
        for (const [sx, w] of tx[dx]) {
          const i = (base + clampX(sx)) * 4;
          r += rgba[i] * w; g += rgba[i + 1] * w; b += rgba[i + 2] * w;
        }
        row[dx * 3] = r; row[dx * 3 + 1] = g; row[dx * 3 + 2] = b;
      }
      rows.set(sy, row);
      return row;
    };
    const out = new Uint8Array(size * size * 3);
    for (let dy = 0; dy < size; dy++) {
      const acc = new Float64Array(size * 3);
      for (const [sy, w] of ty[dy]) {
        if (w === 0) continue;
        const row = hrow(sy);
        for (let k = 0; k < acc.length; k++) acc[k] += row[k] * w;
      }
      for (let k = 0; k < acc.length; k++) out[dy * size * 3 + k] = Math.min(255, Math.max(0, roundHalfEven(acc[k])));
    }
    return out;
  }

  // ---------- Phân loại (classifier.preprocess + predict_proba) ----------
  function toTensor(crops, size, mean, std) {
    const n = crops.length, plane = size * size, data = new Float32Array(n * 3 * plane);
    crops.forEach((crop, i) => {
      for (let p = 0; p < plane; p++) {
        for (let c = 0; c < 3; c++) data[i * 3 * plane + c * plane + p] = (crop[p * 3 + c] / 255 - mean[c]) / std[c];
      }
    });
    return data;
  }

  function occupiedProb(logits, n) {
    const probs = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const a = logits[2 * i], b = logits[2 * i + 1], m = Math.max(a, b);
      const ea = Math.exp(a - m), eb = Math.exp(b - m);
      probs[i] = eb / (ea + eb);
    }
    return probs;
  }

  // ---------- Làm mượt theo thời gian (smoothing.SlotSmoother) ----------
  class SlotSmoother {
    constructor(slotIds, window = 5, k = 4, threshold = 0.5) {
      Object.assign(this, { slotIds, window, k, threshold });
      this.history = Object.fromEntries(slotIds.map((id) => [id, []]));
      this.state = Object.fromEntries(slotIds.map((id) => [id, null]));
    }

    update(probs) {
      const changes = [];
      this.slotIds.forEach((id, i) => {
        const hist = this.history[id];
        hist.push(probs[i]);
        if (hist.length > this.window) hist.shift();
        const votes = hist.filter((q) => q >= this.threshold);
        let next, conf;
        if (votes.length >= this.k) {
          next = "occupied"; conf = votes.reduce((a, b) => a + b, 0) / votes.length;
        } else if (hist.length - votes.length >= this.k) {
          next = "vacant"; conf = hist.filter((q) => q < this.threshold).reduce((a, q) => a + 1 - q, 0) / (hist.length - votes.length);
        } else return;
        if (next !== this.state[id]) { this.state[id] = next; changes.push({ slot_id: id, status: next, confidence: conf }); }
      });
      return changes;
    }

    counts() {
      const v = Object.values(this.state);
      return { vacant: v.filter((s) => s === "vacant").length, occupied: v.filter((s) => s === "occupied").length,
               unknown: v.filter((s) => s === null).length, total: v.length };
    }
  }

  // ---------- YOLO (yolo_detector.YoloOnnx) ----------
  const COCO_VEHICLES = [2, 3, 5, 7];   // car, motorcycle, bus, truck

  function letterboxParams(w, h, imgsz) {
    const r = Math.min(imgsz / h, imgsz / w), nw = Math.round(w * r), nh = Math.round(h * r);
    const px = (imgsz - nw) / 2, py = (imgsz - nh) / 2;
    return { r, nw, nh, px, py, left: Math.round(px - 0.1), top: Math.round(py - 0.1) };
  }

  function boxArea(b) { return Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]); }
  function inter(a, b) {
    return Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  }
  function iou(a, b) { const i = inter(a, b); return i / (boxArea(a) + boxArea(b) - i + 1e-9); }

  function nms(boxes, scores, thr) {
    const order = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
    const keep = [];
    while (order.length) {
      const i = order.shift();
      keep.push(i);
      for (let k = order.length - 1; k >= 0; k--) if (iou(boxes[i], boxes[order[k]]) >= thr) order.splice(k, 1);
    }
    return keep;
  }

  // out: Float32Array (4 + nc) × anchors (layout của ultralytics), trả về hộp xe theo toạ độ khung gốc
  function decodeYolo(out, dims, lb, frameW, frameH, conf = 0.25, iouThr = 0.5, classes = COCO_VEHICLES) {
    const [, rowsN, anchors] = dims, nc = rowsN - 4;
    const boxes = [], scores = [], cls = [];
    for (let a = 0; a < anchors; a++) {
      let best = 0, bc = 0;
      for (let c = 0; c < nc; c++) { const s = out[(4 + c) * anchors + a]; if (s > best) { best = s; bc = c; } }
      if (best < conf) continue;
      const cx = out[a], cy = out[anchors + a], w = out[2 * anchors + a], h = out[3 * anchors + a];
      boxes.push([(cx - w / 2 - lb.px) / lb.r, (cy - h / 2 - lb.py) / lb.r, (cx + w / 2 - lb.px) / lb.r, (cy + h / 2 - lb.py) / lb.r]);
      scores.push(best); cls.push(bc);
    }
    // NMS theo từng lớp: dịch hộp của mỗi lớp ra xa nhau rồi chạy NMS một lần
    const keep = nms(boxes.map((b, i) => b.map((v) => v + cls[i] * 10000)), scores, iouThr);
    return keep.filter((i) => classes.includes(cls[i])).map((i) => {
      const b = boxes[i];
      return [Math.min(Math.max(b[0], 0), frameW), Math.min(Math.max(b[1], 0), frameH),
              Math.min(Math.max(b[2], 0), frameW), Math.min(Math.max(b[3], 0), frameH)];
    });
  }

  // ---------- Tự đề xuất ô (slot_proposer.cluster_boxes) ----------
  function median(values) {
    const v = [...values].sort((a, b) => a - b), m = v.length >> 1;
    return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
  }

  function clusterBoxes(framesBoxes, nFrames, minSupport = 0.25, iouThr = 0.5) {
    const all = [], frameOf = [];
    framesBoxes.forEach((bs, f) => bs.forEach((b) => { all.push(b); frameOf.push(f); }));
    if (!all.length) return [];
    const n = all.length, M = Array.from({ length: n }, (_, i) => all.map((b) => iou(all[i], b)));
    const support = M.map((row) => new Set(row.flatMap((v, j) => (v >= iouThr ? [frameOf[j]] : []))).size);
    const assigned = new Array(n).fill(false), slots = [], scores = [];
    for (const i of support.map((s, i) => i).sort((a, b) => support[b] - support[a])) {
      if (assigned[i]) continue;
      const members = M[i].flatMap((v, j) => (v >= iouThr && !assigned[j] ? [j] : []));
      members.forEach((j) => { assigned[j] = true; });
      const hit = new Set(members.map((j) => frameOf[j])).size;
      if (hit >= Math.max(2, minSupport * nFrames)) {
        slots.push([0, 1, 2, 3].map((k) => median(members.map((j) => all[j][k]))));
        scores.push(hit);
      }
    }
    let kept = nms(slots, scores, 0.3).map((i) => slots[i]);
    const drop = new Set();     // bỏ ô nằm lọt ≥ 60% trong ô được ủng hộ nhiều hơn
    for (let i = 0; i < kept.length; i++) {
      if (drop.has(i)) continue;
      for (let j = i + 1; j < kept.length; j++) {
        if (inter(kept[i], kept[j]) / (Math.min(boxArea(kept[i]), boxArea(kept[j])) + 1e-9) >= 0.6) drop.add(j);
      }
    }
    kept = kept.filter((_, i) => !drop.has(i));
    // đánh số theo hàng (trên → dưới), trái → phải
    return kept.sort((a, b) => roundHalfEven(a[1] / 40) - roundHalfEven(b[1] / 40) || a[0] - b[0]);
  }

  const api = { roundHalfEven, cropBox, scalePolygons, cropResize, toTensor, occupiedProb, SlotSmoother,
                letterboxParams, decodeYolo, nms, iou, clusterBoxes, COCO_VEHICLES };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Vision = api;
})(typeof self !== "undefined" ? self : this);
