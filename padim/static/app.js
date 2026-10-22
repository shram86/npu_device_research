const video = document.getElementById("video");
const details = document.getElementById("details");
let mode = "heatmap";
let resetting = false;
function setMode(value) {
  mode = value;
  video.src = `/video_feed?mode=${encodeURIComponent(mode)}`;
  document.querySelectorAll("[data-mode]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  });
  const toggle = document.getElementById("heatmap-toggle");
  toggle.textContent = mode === "original" ? "Heatmap OFF" : "Heatmap ON";
  toggle.setAttribute("aria-pressed", String(mode !== "original"));
}
document.querySelectorAll("[data-mode]").forEach(button => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});
document.getElementById("heatmap-toggle").addEventListener("click", () => {
  setMode(mode === "original" ? "heatmap" : "original");
});
function number(value, digits = 1) { return value === null ? "—" : Number(value).toFixed(digits); }
function render(data) {
  const status = document.getElementById("status");
  status.textContent = data.status;
  status.dataset.state = data.status;
  document.getElementById("connection").textContent = data.camera_connected ? "● Camera connected" : "Camera unavailable";
  document.getElementById("score").textContent = data.inference_ready ? number(data.score, 4) : "—";
  document.getElementById("threshold").textContent = number(data.threshold, 4);
  document.getElementById("camera-fps").textContent = number(data.camera_fps);
  document.getElementById("inference-fps").textContent = number(data.inference_fps);
  document.getElementById("latency").textContent = `${number(data.inference_ms)} ms`;
  const breakdown = document.getElementById("latency-breakdown");
  breakdown.replaceChildren();
  const stages = data.latency_ms || {};
  for (const [key, label] of [["preprocess_ms", "RGB / resize / tensor"], ["backend_ms", "Backend total"],
      ["input_ms", "Input encode + flush"], ["network_ms", "Feature extraction"],
      ["output_ms", "Output read / selected decode / resize"], ["postprocess_ms", "PaDiM CPU scoring"],
      ["feature_select_ms", "Selected features / FP16 packing"], ["score_input_ms", "Score input copy + flush"],
      ["score_network_ms", "PaDiM NPU scoring"], ["score_output_ms", "Score output read"],
      ["score_finish_ms", "Sqrt / smoothing / max"], ["bridge_ms", "Python / C bridge"], ["validation_ms", "Output checks"]]) {
    if (stages[key] == null) continue;
    const term = document.createElement("dt"); term.textContent = label;
    const value = document.createElement("dd");
    value.textContent = stages[key] == null ? "N/A" : `${number(stages[key], 2)} ms`;
    breakdown.append(term, value);
  }
  document.getElementById("persistence").textContent = `${data.persistence_hits} / ${data.persistence_frames}`;
  const temperatures = data.temperatures || {};
  for (const kind of ["cpu", "npu"]) {
    const value = temperatures[`${kind}_celsius`];
    document.getElementById(`${kind}-temperature`).textContent = value == null ? "N/A" : `${number(value)} °C`;
  }
  document.getElementById("calibration").hidden = data.threshold_calibrated;
  document.getElementById("backend").textContent = data.backend.toUpperCase();
  details.textContent = data.error || data.render_error || data.event_error ||
    (!data.inference_ready ? "Waiting for a fresh inference result…" :
      data.alarm ? "Alarm latched. Inspect the object, then reset." :
      data.status === "PENDING" ? "Anomaly above threshold; checking persistence…" : "Monitoring the PaDiM ROI.");
}
async function poll() {
  try {
    const response = await fetch("/api/status", {cache: "no-store"});
    if (!response.ok) throw new Error("Status unavailable");
    const data = await response.json();
    if (!resetting) render(data);
  } catch (error) {
    const status = document.getElementById("status");
    status.textContent = status.dataset.state = "UNKNOWN";
    document.getElementById("connection").textContent = "Server unavailable";
    document.getElementById("score").textContent = "—";
    details.textContent = "Connection lost. Retrying…";
  } finally { setTimeout(poll, 1000); }
}
document.getElementById("reset").addEventListener("click", async () => {
  const button = document.getElementById("reset");
  button.disabled = resetting = true;
  try {
    const response = await fetch("/api/reset_alarm", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: "{}"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Reset failed");
    render(data);
  } catch (error) { details.textContent = error.message; }
  finally { button.disabled = resetting = false; }
});
video.addEventListener("error", () => {
  setTimeout(() => { video.src = `/video_feed?mode=${encodeURIComponent(mode)}&t=${Date.now()}`; }, 2000);
});
poll();

const gallery = document.getElementById("events-gallery");
const eventMessage = document.getElementById("events-message");
const moreButton = document.getElementById("events-more");
const eventIds = new Set();
let oldestEvent = null;
let eventsBusy = false;
function eventCard(event) {
  const card = document.createElement("article");
  card.className = "event-card";
  card.dataset.id = event.id;
  const preview = document.createElement("a");
  preview.href = event.files["heatmap.jpg"];
  preview.target = "_blank"; preview.rel = "noopener";
  const image = document.createElement("img");
  image.src = event.files["heatmap.jpg"]; image.loading = "lazy";
  image.alt = `Anomaly event ${event.utc}`;
  preview.append(image); card.append(preview);
  const timestamp = document.createElement("p");
  const date = new Date(event.utc);
  timestamp.textContent = Number.isNaN(date.getTime()) ? event.utc : date.toLocaleString();
  const score = document.createElement("p");
  score.textContent = `Score ${number(event.score, 4)} · Threshold ${number(event.threshold, 4)} · ${event.backend.toUpperCase()}`;
  card.append(timestamp, score);
  const links = document.createElement("div"); links.className = "event-links";
  for (const [file, label] of [["original.jpg", "Original"], ["heatmap.jpg", "Heatmap"], ["anomaly_map.npy", "Map"], ["event.json", "Details"]]) {
    const link = document.createElement("a"); link.href = event.files[file]; link.textContent = label;
    if (file.endsWith(".jpg")) { link.target = "_blank"; link.rel = "noopener"; }
    else link.download = file;
    links.append(link);
  }
  card.append(links); return card;
}
async function loadEvents(older = false) {
  if (eventsBusy) return;
  eventsBusy = true; moreButton.disabled = true;
  try {
    const query = new URLSearchParams({limit: "12"});
    if (older && oldestEvent) query.set("before", oldestEvent);
    const response = await fetch(`/api/events?${query}`, {cache: "no-store"});
    if (!response.ok) throw new Error("Could not load saved events");
    const data = await response.json();
    const added = [];
    for (const event of data.events) {
      if (eventIds.has(event.id)) continue;
      eventIds.add(event.id); added.push(eventCard(event));
      if (!oldestEvent || event.id < oldestEvent) oldestEvent = event.id;
    }
    if (older) gallery.append(...added); else gallery.prepend(...added);
    document.getElementById("event-count").textContent = `${data.total} events`;
    eventMessage.textContent = data.total ? "" : "No saved anomaly events yet.";
    if (older || eventIds.size <= 12) moreButton.hidden = !data.has_more;
  } catch (error) { eventMessage.textContent = error.message; }
  finally { eventsBusy = false; moreButton.disabled = false; }
}
moreButton.addEventListener("click", () => loadEvents(true));
loadEvents();
setInterval(() => loadEvents(), 5000);
