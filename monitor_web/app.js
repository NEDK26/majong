const canvas = document.querySelector("#screenCanvas");
const ctx = canvas.getContext("2d");
const monitorSelect = document.querySelector("#monitorSelect");
const countSelect = document.querySelector("#countSelect");
const intervalSelect = document.querySelector("#intervalSelect");
const statusBadge = document.querySelector("#captureStatus");
const regionText = document.querySelector("#regionText");
const emptyPreview = document.querySelector("#emptyPreview");
const analysisEmpty = document.querySelector("#analysisEmpty");
const analysisContent = document.querySelector("#analysisContent");
const errorBox = document.querySelector("#errorBox");

let previewImage = null;
let region = null;
let dragStart = null;
let running = false;
let timer = null;

function setStatus(text, kind = "idle") {
  statusBadge.textContent = text;
  statusBadge.className = `status-badge ${kind}`;
}

function drawPreview() {
  if (!previewImage) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(previewImage, 0, 0, canvas.width, canvas.height);
  if (!region) return;
  ctx.save();
  ctx.fillStyle = "rgba(9, 31, 28, .48)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.clearRect(region.x, region.y, region.width, region.height);
  ctx.drawImage(previewImage, region.x, region.y, region.width, region.height, region.x, region.y, region.width, region.height);
  ctx.strokeStyle = "#f0bb4e";
  ctx.lineWidth = Math.max(2, canvas.width / 700);
  ctx.setLineDash([10, 6]);
  ctx.strokeRect(region.x, region.y, region.width, region.height);
  ctx.restore();
}

function updateRegionText() {
  regionText.textContent = region ? `x ${region.x} · y ${region.y} · ${region.width} × ${region.height}` : "尚未选择";
}

function defaultBottomRegion() {
  if (!canvas.width) return;
  region = { x: 0, y: Math.round(canvas.height * .64), width: canvas.width, height: Math.round(canvas.height * .36) };
  updateRegionText();
  drawPreview();
}

async function loadMonitors() {
  const response = await fetch("/api/monitors", { cache: "no-store" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "无法读取显示器列表");
  monitorSelect.innerHTML = data.monitors.map(item => `<option value="${item.id}">${item.name} · ${item.width}×${item.height}</option>`).join("");
}

async function refreshPreview(resetRegion = false) {
  setStatus(running ? "观察中" : "读取画面", running ? "running" : "idle");
  const image = new Image();
  image.src = `/api/frame?monitor=${monitorSelect.value}&t=${Date.now()}`;
  await image.decode();
  previewImage = image;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  emptyPreview.hidden = true;
  if (resetRegion || !region) defaultBottomRegion(); else drawPreview();
  if (!running) setStatus("可框选", "idle");
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(canvas.width, Math.round((event.clientX - rect.left) * canvas.width / rect.width))),
    y: Math.max(0, Math.min(canvas.height, Math.round((event.clientY - rect.top) * canvas.height / rect.height))),
  };
}

canvas.addEventListener("pointerdown", event => {
  dragStart = pointerPosition(event);
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", event => {
  if (!dragStart) return;
  const current = pointerPosition(event);
  region = { x: Math.min(dragStart.x, current.x), y: Math.min(dragStart.y, current.y), width: Math.abs(current.x - dragStart.x), height: Math.abs(current.y - dragStart.y) };
  drawPreview();
  updateRegionText();
});
canvas.addEventListener("pointerup", event => {
  if (!dragStart) return;
  canvas.releasePointerCapture(event.pointerId);
  dragStart = null;
  if (region.width < 30 || region.height < 30) defaultBottomRegion();
});

function tileMarkup(tile, extra = "") {
  const suit = tile.slice(-1);
  return `<span class="tile ${suit} ${extra}" title="${tile}">${tile}</span>`;
}

function shantenLabel(value) {
  if (value === -1) return "和牌";
  if (value === 0) return "听牌";
  return `${value} 向听`;
}

function effectiveText(items) {
  return items?.map(item => `${item.tile}×${item.remaining}`).join("、") || "—";
}

function renderAnalysis(data) {
  analysisEmpty.hidden = true;
  analysisContent.hidden = false;
  errorBox.hidden = true;
  document.querySelector("#capturedAt").textContent = data.capturedAt;
  document.querySelector("#shantenValue").textContent = shantenLabel(data.shanten);
  document.querySelector("#recommendTiles").innerHTML = data.recommendations.length ? data.recommendations.map(tile => tileMarkup(tile, "recommend")).join("") : `<span class="effective-text">${data.mode === "agari" ? "已经和牌" : "等待摸牌"}</span>`;
  document.querySelector("#tileRack").innerHTML = data.tiles.map(tile => tileMarkup(tile)).join("");
  document.querySelector("#confidenceText").textContent = `最低置信度 ${(data.minimumConfidence * 100).toFixed(0)}% · ${data.tiles.length} 张暗牌`;

  const drawBlock = document.querySelector("#drawBlock");
  if (data.mode === "draw") {
    drawBlock.hidden = false;
    document.querySelector("#drawTiles").innerHTML = data.effectiveDraws.map(item => `<span class="effective-pill">${item.tile} × ${item.remaining}</span>`).join("");
  } else drawBlock.hidden = true;

  const best = new Set(data.recommendations);
  document.querySelector("#candidateBody").innerHTML = data.candidates.map(item => `<tr class="${best.has(item.discard) ? "best" : ""}"><td>${tileMarkup(item.discard)}</td><td>${shantenLabel(item.shanten)}</td><td class="effective-text">${effectiveText(item.effectiveTiles)}</td><td>${item.ukeire}</td></tr>`).join("") || `<tr><td colspan="4" class="effective-text">当前没有舍牌候选。</td></tr>`;

  if (data.minimumConfidence < .62) {
    errorBox.hidden = false;
    errorBox.textContent = "部分牌识别置信度偏低，请核对框选区域或明确暗牌张数。";
  }
}

async function analyzeOnce() {
  if (!region) defaultBottomRegion();
  try {
    setStatus("识别中", "running");
    const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ monitor: Number(monitorSelect.value), region, expectedCount: countSelect.value || null }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "无法分析当前画面");
    renderAnalysis(data);
    setStatus(running ? "观察中" : "已更新", running ? "running" : "idle");
  } catch (error) {
    errorBox.hidden = false;
    errorBox.textContent = error.message;
    setStatus("未识别", "error");
  }
}

async function liveLoop() {
  if (!running) return;
  await analyzeOnce();
  if (running) timer = window.setTimeout(liveLoop, Number(intervalSelect.value));
}

document.querySelector("#startButton").addEventListener("click", () => {
  running = true;
  document.querySelector("#startButton").disabled = true;
  document.querySelector("#stopButton").disabled = false;
  liveLoop();
});
document.querySelector("#stopButton").addEventListener("click", () => {
  running = false;
  window.clearTimeout(timer);
  document.querySelector("#startButton").disabled = false;
  document.querySelector("#stopButton").disabled = true;
  setStatus("已停止", "idle");
});
document.querySelector("#onceButton").addEventListener("click", analyzeOnce);
document.querySelector("#exitButton").addEventListener("click", async () => {
  if (!window.confirm("退出牌理镜并停止本地程序？")) return;
  running = false;
  window.clearTimeout(timer);
  try { await fetch("/api/shutdown", { method: "POST" }); } catch (_) { /* 服务关闭时连接可能提前结束 */ }
  document.body.innerHTML = `<main class="closed-screen"><div class="empty-tile">✓</div><h1>牌理镜已退出</h1><p>现在可以关闭这个浏览器标签页。</p></main>`;
});
document.querySelector("#refreshButton").addEventListener("click", () => refreshPreview(false));
document.querySelector("#delayRefreshButton").addEventListener("click", () => {
  setStatus("3 秒后抓取", "idle");
  window.setTimeout(() => refreshPreview(false), 3000);
});
document.querySelector("#bottomRegionButton").addEventListener("click", defaultBottomRegion);
monitorSelect.addEventListener("change", () => refreshPreview(true));

(async function init() {
  try { await loadMonitors(); await refreshPreview(true); }
  catch (error) { emptyPreview.innerHTML = `<span>!</span><p>${error.message}</p>`; setStatus("读取失败", "error"); }
})();
