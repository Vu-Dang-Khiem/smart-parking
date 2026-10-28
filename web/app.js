// Dashboard chạy trong trình duyệt: phát ảnh/video, phân loại từng ô bằng MobileNetV3 (ONNX Runtime Web),
// làm mượt 4/5 khung rồi vẽ lên dashboard – cùng chuỗi xử lý với worker Python của hệ thống.
const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";
const STATUS_TEXT = { vacant: "Trống", occupied: "Có xe" };
const COLORS = { vacant: "#1f9a52", occupied: "#d23f3a", null: "#9aa3af" };
const TICK_MS = 500;            // 2 khung/giây như worker
const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";

const app = { cameras: [], current: null, playing: true, speed: 1, classifier: null, config: null, yolo: null, timer: null };
window.__smartParking = app;     // để kiểm thử tự động đọc được trạng thái

function svgEl(tag, attrs, parent) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}
// Canvas phải đọc lại điểm ảnh liên tục → tạo context với willReadFrequently ngay lần đầu
const ctx2d = (canvas) => canvas.getContext("2d", { willReadFrequently: true });
const fmtSec = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

// ---------- Mô hình ----------
async function loadClassifier() {
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? 0 : 1;   // GitHub Pages không bật được đa luồng
  app.config = await (await fetch("models/classifier.json")).json();
  app.classifier = await ort.InferenceSession.create("models/classifier.onnx", { executionProviders: ["wasm"] });
  $("model-status").className = "conn online";
  $("model-status").textContent = "Mô hình sẵn sàng";
}

// Khung hình (canvas) → xác suất có xe của từng ô, đúng tiền xử lý lúc train
async function classify(source, cam) {
  const { width, height } = source;
  const rgba = ctx2d(source).getImageData(0, 0, width, height).data;
  const size = app.config.input_size, pad = app.config.runtime_pad;
  const polys = Vision.scalePolygons(cam.slots, cam.frame_size, width, height);
  const crops = polys.map((p) => Vision.cropResize(rgba, width, height, Vision.cropBox(p, pad), size));
  const t0 = performance.now();
  const input = new ort.Tensor("float32", Vision.toTensor(crops, size, app.config.mean, app.config.std), [crops.length, 3, size, size]);
  const out = await app.classifier.run({ [app.config.input_name]: input });
  const probs = Vision.occupiedProb(out[app.config.output_name].data, crops.length);
  cam.inferMs = performance.now() - t0;
  return probs;
}

// ---------- Camera ----------
function makeCamera(def, kind) {
  const ids = def.slots.map((s) => s.id);
  return { ...def, kind, smoother: new Vision.SlotSmoother(ids), events: [], points: [], frameIdx: 0, tick: 0,
           cache: new Map(), canvas: document.createElement("canvas"), acc: { correct: 0, total: 0, last: null } };
}

function loadImage(src) {
  return new Promise((ok, fail) => { const img = new Image(); img.onload = () => ok(img); img.onerror = fail; img.src = src; });
}

async function demoFrame(cam) {
  const f = cam.frames[cam.frameIdx];
  f.image = f.image || await loadImage(f.src);
  cam.canvas.width = f.image.naturalWidth; cam.canvas.height = f.image.naturalHeight;
  ctx2d(cam.canvas).drawImage(f.image, 0, 0);
  return f;
}

function seekVideo(video, t) {
  return new Promise((ok) => { video.addEventListener("seeked", ok, { once: true }); video.currentTime = t; });
}

// Một bước 0,5 giây: ảnh demo giữ nguyên `hold` khung (chỉ cần chạy mô hình một lần), video thì lấy khung mới
async function step(cam) {
  let probs, label;
  if (cam.kind === "demo") {
    const f = await demoFrame(cam);
    if (!cam.cache.has(cam.frameIdx)) cam.cache.set(cam.frameIdx, await classify(cam.canvas, cam));
    probs = cam.cache.get(cam.frameIdx);
    label = f.label;
  } else {
    const v = $("user-video");
    cam.canvas.width = v.videoWidth; cam.canvas.height = v.videoHeight;
    ctx2d(cam.canvas).drawImage(v, 0, 0);
    probs = await classify(cam.canvas, cam);
    label = fmtSec(v.currentTime);
  }
  const changes = cam.smoother.update(probs);
  changes.forEach((c) => cam.events.unshift({ ...c, at: label }));
  cam.events.length = Math.min(cam.events.length, 100);
  if (cam !== app.current) return;
  renderFrame(cam, label);
  renderSummary(cam);
  renderEvents(cam, changes.length);
  changes.forEach((c) => updateSlot(cam, c));
  advance(cam, label);
}

function advance(cam, label) {
  const counts = cam.smoother.counts();
  if (cam.kind === "demo") {
    cam.tick++;
    if (cam.tick < cam.hold) return;
    cam.tick = 0;
    scorePhoto(cam);
    if (cam.newDay) { cam.points = []; cam.newDay = false; }   // biểu đồ ngày trước giữ tới khi có điểm mới
    cam.points.push({ label, rate: counts.occupied / counts.total });
    cam.frameIdx = (cam.frameIdx + 1) % cam.frames.length;
    if (cam.frameIdx === 0) {   // hết ngày → giữ kết quả cả vòng rồi phát lại từ đầu
      cam.loopAcc = { correct: cam.acc.correct, total: cam.acc.total };
      cam.newDay = true; cam.acc = { correct: 0, total: 0, last: cam.acc.last };
    }
    $("seek").value = cam.frameIdx;
  } else {
    const v = $("user-video");
    if (cam.points.length && v.currentTime < cam.lastTime) cam.points = [];   // video phát lại từ đầu
    cam.lastTime = v.currentTime;
    cam.points.push({ label, rate: counts.occupied / counts.total });
    $("seek").value = Math.floor(v.currentTime);
  }
  renderChart(cam);
}

// So trạng thái đã làm mượt ở cuối mỗi ảnh với nhãn thật của ảnh đó
function scorePhoto(cam) {
  const truth = cam.frames[cam.frameIdx].truth;
  let correct = 0, total = 0;
  for (const [id, y] of Object.entries(truth)) {
    const s = cam.smoother.state[id];
    if (s === null || s === undefined) continue;
    total++; correct += (s === "occupied") === (y === 1);
  }
  cam.acc.correct += correct; cam.acc.total += total; cam.acc.last = total ? correct / total : null;
  renderAccuracy(cam);
}

function schedule() {
  clearTimeout(app.timer);
  if (!app.playing || !app.current || !app.classifier) return;
  const cam = app.current, t0 = performance.now();
  step(cam).catch((e) => console.error(e)).finally(() => {
    const wait = Math.max(0, TICK_MS / app.speed - (performance.now() - t0));
    app.timer = setTimeout(schedule, wait);
  });
}

// ---------- Vẽ ----------
function renderFrame(cam, label) {
  const view = $("view"), ctx = view.getContext("2d"), src = cam.canvas;
  view.width = src.width; view.height = src.height;
  ctx.drawImage(src, 0, 0);
  const polys = Vision.scalePolygons(cam.slots, cam.frame_size, src.width, src.height);
  const lw = Math.max(2, src.width / 500);
  polys.forEach((p, i) => {
    const s = cam.smoother.state[cam.slots[i].id];
    ctx.beginPath(); p.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath();
    ctx.fillStyle = COLORS[s] + "40"; ctx.fill();
    ctx.strokeStyle = COLORS[s]; ctx.lineWidth = lw; ctx.stroke();
  });
  const c = cam.smoother.counts(), text = `Trong: ${c.vacant}/${c.total}   Co xe: ${c.occupied}`;
  ctx.font = `bold ${Math.round(src.width / 45)}px sans-serif`;
  ctx.fillStyle = "rgba(30,30,30,0.85)"; ctx.fillRect(0, 0, ctx.measureText(text).width + 24, src.width / 30);
  ctx.fillStyle = "#fff"; ctx.fillText(text, 12, src.width / 42);
  $("view-placeholder").hidden = true;
  const pos = cam.kind === "demo" ? `ảnh ${cam.frameIdx + 1}/${cam.frames.length} · ${label}` : `video ${label}`;
  $("frame-info").textContent = pos;
  $("speed-info").textContent = `MobileNetV3 trong trình duyệt: ${Math.round(cam.inferMs || 0)} ms cho ${cam.slots.length} ô ` +
    `(${(1000 / Math.max(cam.inferMs || 1, 1)).toFixed(1)} khung/giây). Bản Python trên CPU: 87 ms cho 100 ô.`;
}

function renderSummary(cam) {
  const c = cam.smoother.counts();
  $("n-vacant").textContent = c.vacant + c.occupied ? c.vacant : "–";
  $("n-occupied").textContent = c.vacant + c.occupied ? c.occupied : "–";
  const rate = c.vacant + c.occupied ? c.occupied / c.total : null;
  $("rate").textContent = rate === null ? "–" : `${Math.round(rate * 100)}%`;
  $("rate-bar").style.width = `${(rate ?? 0) * 100}%`;
  document.title = `(${c.vacant} trống) Smart Parking`;
}

function renderAccuracy(cam) {
  $("acc-card").hidden = cam.kind !== "demo";
  if (cam.kind !== "demo") return;
  const { correct, total, last } = cam.acc;
  $("acc").textContent = last === null ? "–" : `${(last * 100).toFixed(1)}%`;
  const loop = cam.loopAcc ? ` · cả ngày lượt trước ${(cam.loopAcc.correct / cam.loopAcc.total * 100).toFixed(2)}%` : "";
  $("acc-hint").textContent = total ? `ảnh vừa qua · trung bình ${(correct / total * 100).toFixed(2)}% trên ${total} lượt ô${loop}` : "chờ hết ảnh đầu tiên";
}

function renderMap(cam) {
  const map = $("map");
  map.replaceChildren();
  const [w, h] = cam.frame_size;
  const xs = cam.slots.flatMap((s) => s.polygon.map((p) => p[0])), ys = cam.slots.flatMap((s) => s.polygon.map((p) => p[1]));
  const pad = 0.04 * w;
  const x0 = Math.max(0, Math.min(...xs) - pad), y0 = Math.max(0, Math.min(...ys) - pad);
  const x1 = Math.min(w, Math.max(...xs) + pad), y1 = Math.min(h, Math.max(...ys) + pad);
  map.setAttribute("viewBox", xs.length ? `${x0} ${y0} ${x1 - x0} ${y1 - y0}` : `0 0 ${w} ${h}`);
  for (const slot of cam.slots) {
    const g = svgEl("g", { class: cam.smoother.state[slot.id] ?? "unknown", "data-slot": slot.id }, map);
    svgEl("polygon", { points: slot.polygon.map((p) => p.join(",")).join(" ") }, g);
    const px = slot.polygon.map((p) => p[0]), py = slot.polygon.map((p) => p[1]);
    const size = Math.min(Math.max(...px) - Math.min(...px), Math.max(...py) - Math.min(...py));
    svgEl("text", { x: px.reduce((a, b) => a + b) / px.length, y: py.reduce((a, b) => a + b) / py.length,
                    "font-size": Math.max(5, size / 3.2) }, g).textContent = slot.id;
    svgEl("title", {}, g).textContent = `${slot.id}: ${STATUS_TEXT[cam.smoother.state[slot.id]] ?? "Chưa rõ"}`;
  }
}

function updateSlot(cam, change) {
  const g = $("map").querySelector(`[data-slot="${CSS.escape(change.slot_id)}"]`);
  if (!g) return;
  g.setAttribute("class", `${change.status} flash`);
  g.querySelector("title").textContent = `${change.slot_id}: ${STATUS_TEXT[change.status]} (${Math.round(change.confidence * 100)}%)`;
  setTimeout(() => g.classList.remove("flash"), 1300);
}

function eventRow(e, isNew) {
  const tr = document.createElement("tr");
  if (isNew) tr.className = "new";
  tr.innerHTML = `<td></td><td></td><td><span class="pill ${e.status}">${STATUS_TEXT[e.status]}</span></td>` +
    `<td>${Math.round(e.confidence * 100)}%</td>`;
  tr.children[0].textContent = e.at;
  tr.children[1].textContent = e.slot_id;
  return tr;
}

function renderEvents(cam, fresh = 0) {
  $("events").replaceChildren(...cam.events.map((e, i) => eventRow(e, i < fresh)));
}

function renderChart(cam) {
  const svg = $("chart"), pts = cam.points;
  svg.replaceChildren();
  const W = svg.clientWidth || 600, H = 220, L = 40, R = 12, T = 10, B = 26;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const n = cam.kind === "demo" ? cam.frames.length : Math.max(pts.length, 2);
  const x = (i) => L + (i / Math.max(n - 1, 1)) * (W - L - R), y = (r) => T + (1 - r) * (H - T - B);
  for (const r of [0, 0.25, 0.5, 0.75, 1]) {
    svgEl("line", { class: "grid-line", x1: L, x2: W - R, y1: y(r), y2: y(r) }, svg);
    svgEl("text", { x: L - 6, y: y(r) + 4, "text-anchor": "end" }, svg).textContent = `${r * 100}%`;
  }
  if (!pts.length) {
    svgEl("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, svg).textContent = "Chưa có dữ liệu";
    return;
  }
  const labels = cam.kind === "demo" ? cam.frames.map((f) => f.label.split(" ")[0]) : pts.map((p) => p.label);
  for (const i of [0, Math.floor((labels.length - 1) / 2), labels.length - 1]) {
    if (labels[i] === undefined) continue;
    svgEl("text", { x: x(i), y: H - 6, "text-anchor": i === 0 ? "start" : i === labels.length - 1 ? "end" : "middle" }, svg)
      .textContent = labels[i];
  }
  const line = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.rate).toFixed(1)}`).join(" ");
  svgEl("path", { class: "area", d: `${line} L${x(pts.length - 1)},${y(0)} L${x(0)},${y(0)} Z` }, svg);
  svgEl("path", { class: "line", d: line }, svg);
  svgEl("circle", { class: "end", cx: x(pts.length - 1), cy: y(pts.at(-1).rate), r: 4 }, svg);
}

function selectCamera(id) {
  const cam = app.cameras.find((c) => c.id === id);
  if (!cam) return;
  app.current = cam;
  $("camera-select").value = id;
  const v = $("user-video");
  if (cam.kind === "video") { v.play(); $("seek").max = Math.floor(v.duration || 0); } else { v.pause(); $("seek").max = cam.frames.length - 1; }
  $("seek").value = cam.kind === "demo" ? cam.frameIdx : 0;
  $("chart-hint").textContent = cam.kind === "demo" ? "theo giờ chụp trong ngày" : "theo thời gian trong video";
  renderMap(cam); renderSummary(cam); renderEvents(cam); renderChart(cam); renderAccuracy(cam);
  history.replaceState(null, "", `?camera=${encodeURIComponent(id)}${location.hash}`);
  schedule();
}

function refreshCameraList() {
  $("camera-select").replaceChildren(...app.cameras.map((c) => new Option(`${c.name} (${c.slots.length} ô)`, c.id)));
  if (app.current) $("camera-select").value = app.current.id;
}

// ---------- Điều khiển phát ----------
$("camera-select").addEventListener("change", (e) => selectCamera(e.target.value));
$("play-btn").addEventListener("click", () => {
  app.playing = !app.playing;
  $("play-btn").textContent = app.playing ? "Tạm dừng" : "Phát";
  const v = $("user-video");
  if (app.current?.kind === "video") app.playing ? v.play() : v.pause();
  schedule();
});
$("speed").addEventListener("change", (e) => { app.speed = Number(e.target.value); $("user-video").playbackRate = app.speed; });
$("seek").addEventListener("input", (e) => {
  const cam = app.current;
  if (!cam) return;
  if (cam.kind === "demo") { cam.frameIdx = Number(e.target.value); cam.tick = 0; } else $("user-video").currentTime = Number(e.target.value);
});
window.addEventListener("resize", () => app.current && renderChart(app.current));

// ---------- Trang thử video của bạn ----------
const ed = { video: null, slots: [], selected: -1, scale: 1, drag: null, frame: document.createElement("canvas") };

function showView(name) {
  $("dashboard").hidden = name !== "dashboard";
  $("upload").hidden = name !== "upload";
  $("open-upload").hidden = name === "upload";
  if (name === "upload") { app.playing = false; clearTimeout(app.timer); $("user-video").pause(); }
  else { app.playing = true; $("play-btn").textContent = "Tạm dừng"; if (app.current) selectCamera(app.current.id); }
}
$("open-upload").addEventListener("click", () => { location.hash = "#upload"; });
$("back-btn").addEventListener("click", () => { location.hash = ""; });
window.addEventListener("hashchange", () => showView(location.hash === "#upload" ? "upload" : "dashboard"));

$("file").addEventListener("change", async () => {
  const file = $("file").files[0];
  if (!file) return;
  const v = $("user-video");
  v.src = URL.createObjectURL(file);
  try {
    await new Promise((ok, fail) => { v.onloadeddata = ok; v.onerror = () => fail(new Error("trình duyệt không đọc được video này (thử mp4 H.264)")); });
  } catch (e) { $("file-status").textContent = `Lỗi: ${e.message}`; return; }
  ed.video = { name: file.name, w: v.videoWidth, h: v.videoHeight, duration: v.duration };
  ed.slots = []; ed.selected = -1;
  $("file-status").textContent = `${file.name}: ${v.videoWidth}×${v.videoHeight}, ${v.duration.toFixed(1)} giây`;
  $("step2").hidden = $("step3").hidden = false;
  await loadEditorFrame();
});

async function grabFrame(t, canvas) {
  const v = $("user-video");
  await seekVideo(v, Math.min(Math.max(t, 0), Math.max(v.duration - 0.05, 0)));
  canvas.width = v.videoWidth; canvas.height = v.videoHeight;
  ctx2d(canvas).drawImage(v, 0, 0);
}

async function loadEditorFrame() { await grabFrame(Number($("position").value) * ed.video.duration, ed.frame); drawEditor(); }
$("position").addEventListener("change", () => ed.video && loadEditorFrame());

function drawEditor() {
  const canvas = $("canvas"), ctx = canvas.getContext("2d");
  ed.scale = Math.min(1, canvas.parentElement.clientWidth / ed.video.w);
  canvas.width = Math.round(ed.video.w * ed.scale); canvas.height = Math.round(ed.video.h * ed.scale);
  ctx.drawImage(ed.frame, 0, 0, canvas.width, canvas.height);
  ed.slots.forEach((r, i) => {
    const [x0, y0, x1, y1] = r.map((v) => v * ed.scale);
    ctx.fillStyle = i === ed.selected ? "rgba(29,95,196,0.35)" : "rgba(255,193,7,0.15)";
    ctx.strokeStyle = i === ed.selected ? "#1d5fc4" : "#ffc107";
    ctx.lineWidth = 2; ctx.fillRect(x0, y0, x1 - x0, y1 - y0); ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  });
  if (ed.drag?.moved) {
    const { x0, y0, x1, y1 } = ed.drag;
    ctx.setLineDash([6, 4]); ctx.strokeStyle = "#1d5fc4";
    ctx.strokeRect(x0 * ed.scale, y0 * ed.scale, (x1 - x0) * ed.scale, (y1 - y0) * ed.scale); ctx.setLineDash([]);
  }
  $("slot-count").textContent = `${ed.slots.length} ô`;
  $("delete-btn").disabled = ed.selected < 0;
  $("start-btn").disabled = ed.slots.length === 0;
}

function toVideo(e) {
  const rect = $("canvas").getBoundingClientRect();
  return [(e.clientX - rect.left) / ed.scale, (e.clientY - rect.top) / ed.scale];
}
$("canvas").addEventListener("mousedown", (e) => { const [x, y] = toVideo(e); ed.drag = { x0: x, y0: y, x1: x, y1: y, moved: false }; });
$("canvas").addEventListener("mousemove", (e) => {
  if (!ed.drag) return;
  [ed.drag.x1, ed.drag.y1] = toVideo(e);
  ed.drag.moved = Math.abs(ed.drag.x1 - ed.drag.x0) + Math.abs(ed.drag.y1 - ed.drag.y0) > 6;
  drawEditor();
});
$("canvas").addEventListener("mouseup", () => {
  const d = ed.drag;
  ed.drag = null;
  if (!d) return;
  if (d.moved) {
    const r = [Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.max(d.x0, d.x1), Math.max(d.y0, d.y1)];
    if (r[2] - r[0] > 8 && r[3] - r[1] > 8) { ed.slots.push(r); ed.selected = ed.slots.length - 1; }
  } else {
    const hits = ed.slots.map((r, i) => [r, i]).filter(([r]) => d.x0 >= r[0] && d.x0 <= r[2] && d.y0 >= r[1] && d.y0 <= r[3]);
    hits.sort((a, b) => (a[0][2] - a[0][0]) * (a[0][3] - a[0][1]) - (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]));
    ed.selected = hits.length ? hits[0][1] : -1;
  }
  drawEditor();
});
function deleteSelected() { if (ed.selected < 0) return; ed.slots.splice(ed.selected, 1); ed.selected = -1; drawEditor(); }
$("delete-btn").addEventListener("click", deleteSelected);
document.addEventListener("keydown", (e) => {
  if (!$("upload").hidden && ed.video && (e.key === "Delete" || e.key === "Backspace")) deleteSelected();
});
$("clear-btn").addEventListener("click", () => { ed.slots = []; ed.selected = -1; drawEditor(); });

// Tự tìm ô: YOLO11n-COCO quét các khung trải đều video, xe đứng yên ở nhiều khung → một ô (slot_proposer)
async function proposeSlots(nFrames = 12) {
  const status = $("propose-status"), bar = $("propose-bar");
  $("propose-meter").hidden = false;
  if (!app.yolo) {
    status.textContent = "Đang tải YOLO11n (11 MB, chỉ lần đầu)…";
    app.yolo = await ort.InferenceSession.create("models/yolo/yolo11n_coco_1280.onnx", { executionProviders: ["wasm"] });
  }
  const imgsz = 1280, frame = document.createElement("canvas"), lbCanvas = document.createElement("canvas");
  lbCanvas.width = lbCanvas.height = imgsz;
  const all = [], t0 = performance.now();
  for (let k = 0; k < nFrames; k++) {
    status.textContent = `Đang quét khung ${k + 1}/${nFrames} bằng YOLO…`;
    await grabFrame((k / Math.max(nFrames - 1, 1)) * ed.video.duration, frame);
    const lb = Vision.letterboxParams(frame.width, frame.height, imgsz), ctx = ctx2d(lbCanvas);
    ctx.fillStyle = "rgb(114,114,114)"; ctx.fillRect(0, 0, imgsz, imgsz);
    ctx.drawImage(frame, lb.left, lb.top, lb.nw, lb.nh);
    const px = ctx.getImageData(0, 0, imgsz, imgsz).data, plane = imgsz * imgsz, x = new Float32Array(3 * plane);
    for (let p = 0; p < plane; p++) { x[p] = px[p * 4] / 255; x[plane + p] = px[p * 4 + 1] / 255; x[2 * plane + p] = px[p * 4 + 2] / 255; }
    const out = await app.yolo.run({ [app.yolo.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, imgsz, imgsz]) });
    const o = out[app.yolo.outputNames[0]];
    all.push(Vision.decodeYolo(o.data, o.dims, lb, frame.width, frame.height));
    bar.style.width = `${((k + 1) / nFrames) * 100}%`;
  }
  ed.slots = Vision.clusterBoxes(all, nFrames);
  ed.selected = -1;
  status.textContent = `Tìm thấy ${ed.slots.length} ô sau ${((performance.now() - t0) / 1000).toFixed(0)} giây. Kiểm tra và sửa lại nếu cần.`;
  $("propose-meter").hidden = true;
  await loadEditorFrame();
}
$("propose-btn").addEventListener("click", async () => {
  $("propose-btn").disabled = true;
  try { await proposeSlots(); } catch (e) { console.error(e); $("propose-status").textContent = `Lỗi: ${e.message}`; }
  finally { $("propose-btn").disabled = false; }
});

$("start-btn").addEventListener("click", async () => {
  const slots = ed.slots.map((r, i) => ({ id: `S${String(i + 1).padStart(3, "0")}`,
                                          polygon: [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]] }));
  app.cameras = app.cameras.filter((c) => c.kind !== "video");
  const cam = makeCamera({ id: "video", name: `Video của bạn: ${ed.video.name}`, frame_size: [ed.video.w, ed.video.h], slots }, "video");
  app.cameras.push(cam);
  refreshCameraList();
  await seekVideo($("user-video"), 0);
  app.current = cam;
  location.hash = "";
});

// ---------- Khởi động ----------
(async function init() {
  const defs = await (await fetch("demo/cameras.json")).json();
  app.cameras = defs.map((d) => makeCamera(d, "demo"));
  refreshCameraList();
  const wanted = new URLSearchParams(location.search).get("camera");
  app.current = app.cameras.find((c) => c.id === wanted) ?? app.cameras[0];
  showView(location.hash === "#upload" ? "upload" : "dashboard");
  try {
    await loadClassifier();
  } catch (e) {
    console.error(e);
    $("model-status").textContent = "Không tải được mô hình";
    return;
  }
  if (location.hash !== "#upload") selectCamera(app.current.id);
})();
