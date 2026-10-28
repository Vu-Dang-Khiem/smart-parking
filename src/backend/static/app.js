// Dashboard: tải trạng thái qua REST khi mở trang, sau đó cập nhật realtime qua WebSocket.
const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";
const STATUS_TEXT = { vacant: "Trống", occupied: "Có xe" };
const state = { cameraId: null, camera: null };

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

function svgEl(tag, attrs, parent) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

function fmtTime(iso) {
  return new Date(iso).toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// ---------- Tổng quan ----------
function renderSummary(counts) {
  $("n-vacant").textContent = counts.vacant;
  $("n-occupied").textContent = counts.occupied;
  $("n-total").textContent = counts.total;
  const known = counts.vacant + counts.occupied;
  const rate = known ? counts.occupied / counts.total : null;
  $("rate").textContent = rate === null ? "–" : `${Math.round(rate * 100)}%`;
  $("rate-bar").style.width = `${(rate ?? 0) * 100}%`;
  document.title = `(${counts.vacant} trống) Smart Parking`;
}

// ---------- Sơ đồ ô đỗ: vẽ polygon theo toạ độ ảnh camera ----------
function renderMap(camera) {
  const map = $("map");
  map.replaceChildren();
  const [w, h] = camera.frame_size;
  // Khung nhìn ôm sát vùng có ô đỗ thay vì cả khung hình camera
  const xs = camera.slots.flatMap((s) => s.polygon.map((p) => p[0]));
  const ys = camera.slots.flatMap((s) => s.polygon.map((p) => p[1]));
  const pad = 0.04 * w;
  const x0 = Math.max(0, Math.min(...xs) - pad), y0 = Math.max(0, Math.min(...ys) - pad);
  const x1 = Math.min(w, Math.max(...xs) + pad), y1 = Math.min(h, Math.max(...ys) + pad);
  map.setAttribute("viewBox", xs.length ? `${x0} ${y0} ${x1 - x0} ${y1 - y0}` : `0 0 ${w} ${h}`);
  for (const slot of camera.slots) {
    const g = svgEl("g", { class: slot.status ?? "unknown", "data-slot": slot.id }, map);
    svgEl("polygon", { points: slot.polygon.map((p) => p.join(",")).join(" ") }, g);
    const px = slot.polygon.map((p) => p[0]), py = slot.polygon.map((p) => p[1]);
    const cx = px.reduce((a, b) => a + b) / px.length, cy = py.reduce((a, b) => a + b) / py.length;
    // Cỡ chữ theo kích thước ô để nhãn không tràn sang ô bên cạnh
    const size = Math.min(Math.max(...px) - Math.min(...px), Math.max(...py) - Math.min(...py));
    svgEl("text", { x: cx, y: cy, "font-size": Math.max(5, size / 3.2) }, g).textContent = slot.id;
    svgEl("title", {}, g).textContent = slotTitle(slot);
  }
}

function slotTitle(slot) {
  const status = STATUS_TEXT[slot.status] ?? "Chưa rõ";
  const conf = slot.confidence != null ? ` (${Math.round(slot.confidence * 100)}%)` : "";
  return `${slot.id}: ${status}${conf}${slot.updated_at ? " – " + fmtTime(slot.updated_at) : ""}`;
}

function updateSlot(change, ts) {
  const slot = state.camera.slots.find((s) => s.id === change.slot_id);
  if (!slot) return;
  Object.assign(slot, { status: change.status, confidence: change.confidence, updated_at: ts });
  const g = $("map").querySelector(`[data-slot="${CSS.escape(slot.id)}"]`);
  if (!g) return;
  g.setAttribute("class", `${slot.status} flash`);
  g.querySelector("title").textContent = slotTitle(slot);
  setTimeout(() => g.classList.remove("flash"), 1300);
}

// ---------- Bảng sự kiện ----------
function eventRow(e, isNew) {
  const tr = document.createElement("tr");
  if (isNew) tr.className = "new";
  tr.innerHTML = `<td>${fmtTime(e.ts)}</td><td></td><td><span class="pill ${e.status}">${STATUS_TEXT[e.status]}</span></td>` +
    `<td>${Math.round(e.confidence * 100)}%</td>`;
  tr.children[1].textContent = e.slot_id;
  return tr;
}

function renderEvents(events) {
  $("events").replaceChildren(...events.map((e) => eventRow(e, false)));
}

// ---------- Biểu đồ tỷ lệ lấp đầy (SVG, không cần thư viện ngoài) ----------
function renderChart(data) {
  const svg = $("chart");
  svg.replaceChildren();
  const W = svg.clientWidth || 600, H = 220, L = 40, R = 12, T = 10, B = 26;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const end = Date.now(), start = end - data.hours * 3600e3;
  const x = (t) => L + ((t - start) / (end - start)) * (W - L - R);
  const y = (r) => T + (1 - r) * (H - T - B);
  for (const r of [0, 0.25, 0.5, 0.75, 1]) {
    svgEl("line", { class: "grid-line", x1: L, x2: W - R, y1: y(r), y2: y(r) }, svg);
    svgEl("text", { x: L - 6, y: y(r) + 4, "text-anchor": "end" }, svg).textContent = `${r * 100}%`;
  }
  const ticks = data.hours <= 1 ? 4 : 6;
  for (let i = 0; i <= ticks; i++) {
    const t = start + (i / ticks) * (end - start);
    const label = new Date(t).toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
    svgEl("text", { x: x(t), y: H - 6, "text-anchor": i === 0 ? "start" : i === ticks ? "end" : "middle" }, svg).textContent = label;
  }
  const pts = data.points.map((p) => [x(Date.parse(p.t)), y(p.rate)]).filter(([px]) => px >= L);
  if (!pts.length) {
    svgEl("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, svg).textContent = "Chưa có dữ liệu trong khoảng này";
    return;
  }
  const line = pts.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join(" ");
  svgEl("path", { class: "area", d: `${line} L${pts.at(-1)[0]},${y(0)} L${pts[0][0]},${y(0)} Z` }, svg);
  svgEl("path", { class: "line", d: line }, svg);
  svgEl("circle", { class: "end", cx: pts.at(-1)[0], cy: pts.at(-1)[1], r: 4 }, svg);
}

async function loadChart() {
  if (!state.cameraId) return;
  renderChart(await api(`/api/stats/${encodeURIComponent(state.cameraId)}/occupancy?hours=${$("range").value}&bucket_minutes=${$("range").value === "1" ? 1 : 10}`));
}

// ---------- Hình ảnh camera: lấy ảnh mới nhất mỗi ~0.7 giây ----------
let videoTimer = null;
function startVideo() {
  clearInterval(videoTimer);
  const img = $("video");
  img.onload = () => { $("video-placeholder").hidden = true; $("frame-age").textContent = `cập nhật ${fmtTime(new Date().toISOString())}`; };
  img.onerror = () => { $("video-placeholder").hidden = false; };
  const tick = () => { img.src = `/api/cameras/${encodeURIComponent(state.cameraId)}/snapshot.jpg?t=${Date.now()}`; };
  tick();
  videoTimer = setInterval(tick, 700);
}

// ---------- Camera ----------
async function selectCamera(id) {
  state.cameraId = id;
  state.camera = await api(`/api/cameras/${encodeURIComponent(id)}`);
  renderSummary(state.camera.counts);
  renderMap(state.camera);
  renderEvents(await api(`/api/events/${encodeURIComponent(id)}?limit=50`));
  startVideo();
  loadChart();
  history.replaceState(null, "", `?camera=${encodeURIComponent(id)}`);
}

async function loadCameras() {
  const cameras = await api("/api/cameras");
  $("empty").hidden = cameras.length > 0;
  $("content").hidden = cameras.length === 0;
  const select = $("camera-select");
  select.replaceChildren(...cameras.map((c) => new Option(`${c.camera_id} (${c.counts.total} ô)`, c.camera_id)));
  if (!cameras.length) return;
  const wanted = new URLSearchParams(location.search).get("camera");
  const id = cameras.some((c) => c.camera_id === wanted) ? wanted : (state.cameraId ?? cameras[0].camera_id);
  select.value = id;
  await selectCamera(id);
}

// ---------- WebSocket ----------
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => {
    $("conn").className = "conn online";
    $("conn").textContent = "Realtime";
    loadCameras().catch(console.error); // đồng bộ lại những gì đã lỡ khi mất kết nối
  };
  ws.onclose = () => {
    $("conn").className = "conn offline";
    $("conn").textContent = "Mất kết nối";
    setTimeout(connect, 2000);
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type !== "slots") return;
    if (!state.camera) { loadCameras(); return; }
    if (msg.camera_id !== state.cameraId) return;
    for (const c of msg.changes) {
      updateSlot(c, msg.ts);
      $("events").prepend(eventRow({ ...c, ts: msg.ts }, true));
    }
    while ($("events").children.length > 100) $("events").lastChild.remove();
    renderSummary(msg.counts);
  };
}

$("camera-select").addEventListener("change", (e) => selectCamera(e.target.value));
$("range").addEventListener("change", loadChart);
window.addEventListener("resize", () => { clearTimeout(window._rz); window._rz = setTimeout(loadChart, 200); });
setInterval(loadChart, 60000);
connect();
