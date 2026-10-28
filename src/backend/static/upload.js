// Trang tải video: tải lên → tự tìm ô (YOLO) → sửa ô trên canvas → chạy worker theo dõi video.
const $ = (id) => document.getElementById(id);
const state = { video: null, slots: [], selected: -1, image: new Image(), scale: 1, drag: null };

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : null;
  if (!res.ok) throw new Error(body?.detail ? JSON.stringify(body.detail) : `${res.status}`);
  return body;
}

// ---------- 1. Tải lên (gửi thẳng file trong body, có thanh tiến trình) ----------
$("file").addEventListener("change", () => { $("upload-btn").disabled = !$("file").files.length; });
$("upload-btn").addEventListener("click", () => {
  const file = $("file").files[0];
  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/api/videos?filename=${encodeURIComponent(file.name)}`);
  $("upload-meter").hidden = false;
  $("upload-btn").disabled = true;
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    $("upload-bar").style.width = `${(e.loaded / e.total) * 100}%`;
    $("upload-status").textContent = `Đang tải ${(e.loaded / 1e6).toFixed(1)} / ${(e.total / 1e6).toFixed(1)} MB`;
  };
  xhr.onload = () => {
    $("upload-btn").disabled = false;
    const body = JSON.parse(xhr.responseText || "{}");
    if (xhr.status !== 200) { $("upload-status").textContent = `Lỗi: ${body.detail || xhr.status}`; return; }
    $("upload-status").textContent = `Đã tải ${body.name}: ${body.frame_size.join("×")}, ${body.frames} khung hình, ${body.fps} fps`;
    openVideo(body, []);
    loadVideos();
  };
  xhr.onerror = () => { $("upload-btn").disabled = false; $("upload-status").textContent = "Lỗi mạng khi tải lên"; };
  xhr.send(file);
});

// Bản public có giới hạn dung lượng, thời lượng và thời gian chạy → hiện cho người dùng biết trước
api("/api/videos/limits").then((l) => {
  const parts = [];
  if (l.max_upload_mb < 2048) parts.push(`≤ ${l.max_upload_mb} MB`);
  if (l.max_upload_seconds) parts.push(`≤ ${Math.round(l.max_upload_seconds / 60)} phút`);
  if (l.upload_minutes) parts.push(`theo dõi tối đa ${l.upload_minutes} phút`);
  if (parts.length) $("upload-hint").textContent += ` · ${parts.join(", ")}`;
}).catch(() => {});

// ---------- 2. Ô đỗ trên canvas ----------
function openVideo(video, slots) {
  state.video = video;
  state.slots = slots;
  state.selected = -1;
  $("step2").hidden = $("step3").hidden = false;
  $("propose-status").textContent = "";
  loadFrame();
  $("step2").scrollIntoView({ behavior: "smooth" });
}

function loadFrame() {
  state.image.onload = draw;
  state.image.src = `/api/videos/${state.video.video_id}/frame.jpg?position=${$("position").value}`;
}
$("position").addEventListener("change", () => state.video && loadFrame());

function draw() {
  const canvas = $("canvas"), ctx = canvas.getContext("2d");
  const [w, h] = state.video.frame_size;
  state.scale = Math.min(1, canvas.parentElement.clientWidth / w);
  canvas.width = Math.round(w * state.scale);
  canvas.height = Math.round(h * state.scale);
  ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  state.slots.forEach((r, i) => {
    const [x0, y0, x1, y1] = r.map((v) => v * state.scale);
    ctx.fillStyle = i === state.selected ? "rgba(29,95,196,0.35)" : "rgba(255,193,7,0.15)";
    ctx.strokeStyle = i === state.selected ? "#1d5fc4" : "#ffc107";
    ctx.lineWidth = 2;
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  });
  if (state.drag?.moved) {
    const { x0, y0, x1, y1 } = state.drag;
    ctx.setLineDash([6, 4]); ctx.strokeStyle = "#1d5fc4";
    ctx.strokeRect(x0 * state.scale, y0 * state.scale, (x1 - x0) * state.scale, (y1 - y0) * state.scale);
    ctx.setLineDash([]);
  }
  $("slot-count").textContent = `${state.slots.length} ô`;
  $("delete-btn").disabled = state.selected < 0;
  $("start-btn").disabled = state.slots.length === 0;
}

function toVideo(e) {
  const rect = $("canvas").getBoundingClientRect();
  return [(e.clientX - rect.left) / state.scale, (e.clientY - rect.top) / state.scale];
}

$("canvas").addEventListener("mousedown", (e) => {
  const [x, y] = toVideo(e);
  state.drag = { x0: x, y0: y, x1: x, y1: y, moved: false };
});
$("canvas").addEventListener("mousemove", (e) => {
  if (!state.drag) return;
  [state.drag.x1, state.drag.y1] = toVideo(e);
  state.drag.moved = Math.abs(state.drag.x1 - state.drag.x0) + Math.abs(state.drag.y1 - state.drag.y0) > 6;
  draw();
});
$("canvas").addEventListener("mouseup", () => {
  const d = state.drag;
  state.drag = null;
  if (!d) return;
  if (d.moved) {   // kéo chuột → thêm ô mới
    const r = [Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.max(d.x0, d.x1), Math.max(d.y0, d.y1)];
    if (r[2] - r[0] > 8 && r[3] - r[1] > 8) { state.slots.push(r); state.selected = state.slots.length - 1; }
  } else {         // bấm → chọn ô nhỏ nhất chứa điểm bấm
    const hits = state.slots.map((r, i) => [r, i]).filter(([r]) => d.x0 >= r[0] && d.x0 <= r[2] && d.y0 >= r[1] && d.y0 <= r[3]);
    hits.sort((a, b) => (a[0][2] - a[0][0]) * (a[0][3] - a[0][1]) - (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]));
    state.selected = hits.length ? hits[0][1] : -1;
  }
  draw();
});

function deleteSelected() {
  if (state.selected < 0) return;
  state.slots.splice(state.selected, 1);
  state.selected = -1;
  draw();
}
$("delete-btn").addEventListener("click", deleteSelected);
document.addEventListener("keydown", (e) => { if ((e.key === "Delete" || e.key === "Backspace") && state.video) deleteSelected(); });
$("clear-btn").addEventListener("click", () => { state.slots = []; state.selected = -1; draw(); });

$("propose-btn").addEventListener("click", async () => {
  $("propose-btn").disabled = true;
  $("propose-status").textContent = "Đang quét video bằng YOLO để tìm ô đỗ…";
  try {
    const res = await api(`/api/videos/${state.video.video_id}/propose`, { method: "POST" });
    state.slots = res.slots.map((s) => {
      const xs = s.polygon.map((p) => p[0]), ys = s.polygon.map((p) => p[1]);
      return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    });
    state.selected = -1;
    $("propose-status").textContent = `Tìm thấy ${state.slots.length} ô. Kiểm tra và sửa lại nếu cần.`;
    draw();
  } catch (err) {
    $("propose-status").textContent = `Lỗi: ${err.message}`;
  } finally {
    $("propose-btn").disabled = false;
  }
});

// ---------- 3. Bắt đầu theo dõi ----------
$("start-btn").addEventListener("click", async () => {
  $("start-btn").disabled = true;
  const slots = state.slots.map((r, i) => ({
    id: `S${String(i + 1).padStart(3, "0")}`,
    polygon: [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]],
  }));
  try {
    const res = await api(`/api/videos/${state.video.video_id}/start`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ slots }),
    });
    $("start-status").textContent = "Đã bắt đầu, đang mở dashboard…";
    location.href = `/?camera=${encodeURIComponent(res.camera_id)}`;
  } catch (err) {
    $("start-status").textContent = `Lỗi: ${err.message}`;
    $("start-btn").disabled = false;
  }
});

// ---------- Danh sách video ----------
async function loadVideos() {
  const videos = await api("/api/videos");
  $("videos").replaceChildren(...videos.map((v) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td></td><td>${v.frame_size.join("×")} · ${v.size_mb} MB</td>` +
      `<td><span class="pill ${v.running ? "vacant" : ""}">${v.running ? "Đang theo dõi" : "Dừng"}</span></td><td class="actions"></td>`;
    tr.children[0].textContent = v.name;
    const actions = tr.querySelector(".actions");
    const button = (label, onClick, ghost = true) => {
      const b = document.createElement("button");
      b.className = ghost ? "btn ghost small" : "btn small";
      b.textContent = label;
      b.addEventListener("click", onClick);
      actions.appendChild(b);
    };
    button("Sửa ô", async () => {
      const saved = await api(`/api/videos/${v.video_id}/slots`);
      openVideo(v, saved.slots.map((s) => {
        const xs = s.polygon.map((p) => p[0]), ys = s.polygon.map((p) => p[1]);
        return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
      }));
    });
    if (v.running) {
      button("Dashboard", () => { location.href = `/?camera=${encodeURIComponent(v.camera_id)}`; }, false);
      button("Dừng", async () => { await api(`/api/videos/${v.video_id}/stop`, { method: "POST" }); loadVideos(); });
    }
    return tr;
  }));
}
window.addEventListener("resize", () => state.video && draw());
loadVideos();
