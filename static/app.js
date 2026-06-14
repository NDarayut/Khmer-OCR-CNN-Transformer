"use strict";

// ── Element refs ──────────────────────────────────────────────────────────
const dropZone       = document.getElementById("drop-zone");
const fileInput      = document.getElementById("file-input");
const previewCont    = document.getElementById("preview-container");
const previewImg     = document.getElementById("preview-img");
const changeFileBtn  = document.getElementById("change-file-btn");
const uploadSection  = document.getElementById("upload-section");
const detectorSel    = document.getElementById("detector-select");
const confRow        = document.getElementById("conf-row");
const confInput      = document.getElementById("conf-input");
const padRow         = document.getElementById("pad-row");
const padInput       = document.getElementById("pad-input");
const formatSel      = document.getElementById("format-select");
const runBtn         = document.getElementById("run-btn");

const loadingOverlay = document.getElementById("loading-overlay");

const resultsSection = document.getElementById("results-section");
const resultImg      = document.getElementById("result-img");
const resultText     = document.getElementById("result-text");
const downloadLink   = document.getElementById("download-link");
const newDocBtn      = document.getElementById("new-doc-btn");
const errorBanner    = document.getElementById("error-banner");
const errorMsg       = document.getElementById("error-msg");
const dismissErrBtn  = document.getElementById("dismiss-error-btn");

const docViewport    = document.getElementById("doc-viewport");
const boxOverlay     = document.getElementById("box-overlay");
const scanLine       = document.getElementById("scan-line");
const toggleBoxesBtn = document.getElementById("toggle-boxes-btn");

const SVG_NS = "http://www.w3.org/2000/svg";

// ── State ─────────────────────────────────────────────────────────────────
let uploadId = null;
let previewUrl = null;

// ── Helpers ───────────────────────────────────────────────────────────────
function show(el) { el.hidden = false; }
function hide(el) { el.hidden = true; }

function showError(msg) {
  errorMsg.textContent = msg;
  show(errorBanner);
}

function resetToUpload() {
  hide(resultsSection);
  hide(errorBanner);
  show(uploadSection);
  uploadId = null;
  previewUrl = null;
  previewImg.src = "";
  hide(previewCont);
  show(dropZone);
  runBtn.disabled = true;
  fileInput.value = "";
  clearResult();
}

// ── Result rendering & box overlay ─────────────────────────────────────────
function clearResult() {
  boxOverlay.replaceChildren();
  resultText.replaceChildren();
  scanLine.classList.remove("sweep");
  docViewport.classList.remove("boxes-hidden");
  hide(toggleBoxesBtn);
  toggleBoxesBtn.textContent = "Hide boxes";
}

function renderResult(data) {
  resultImg.onload = null;   // run once
  clearResult();

  const segments = data.segments || [];
  const [w, h] = data.image_size || [resultImg.naturalWidth, resultImg.naturalHeight];

  // No detected regions: show the plain message, skip the animation.
  if (!segments.length) {
    const p = document.createElement("div");
    p.className = "ocr-plain";
    p.textContent = data.text;
    resultText.appendChild(p);
    return;
  }

  boxOverlay.setAttribute("viewBox", `0 0 ${w} ${h}`);

  // Build a box + (for text) a matching line, both keyed by the same index.
  segments.forEach((seg, i) => {
    const [x1, y1, x2, y2] = seg.bbox;
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", x1);
    rect.setAttribute("y", y1);
    rect.setAttribute("width", Math.max(0, x2 - x1));
    rect.setAttribute("height", Math.max(0, y2 - y1));
    rect.setAttribute("rx", "2");
    rect.classList.add("seg-box", seg.type === "logo" ? "seg-logo" : "seg-text");
    rect.dataset.idx = i;
    // Approximate perimeter for the stroke-draw effect.
    const len = 2 * (Math.max(0, x2 - x1) + Math.max(0, y2 - y1));
    rect.style.setProperty("--len", len || 1);
    boxOverlay.appendChild(rect);

    if (seg.type === "text") {
      const line = document.createElement("div");
      line.className = "ocr-line";
      line.dataset.idx = i;
      line.textContent = seg.text || " ";
      resultText.appendChild(line);
    }
  });

  show(toggleBoxesBtn);
  wireHoverLink();
  animateReveal(segments.length);
}

function animateReveal(count) {
  // Cap total duration so large documents don't crawl.
  const total = Math.min(2200, 250 + count * 90);
  const step = count > 1 ? (total - 250) / (count - 1) : 0;

  scanLine.style.setProperty("--scan-dur", `${Math.max(0.9, total / 1000)}s`);
  scanLine.classList.remove("sweep");
  void scanLine.offsetWidth;   // restart animation
  scanLine.classList.add("sweep");

  const rects = boxOverlay.querySelectorAll(".seg-box");
  const lines = resultText.querySelectorAll(".ocr-line");
  rects.forEach((rect, i) => {
    const delay = i * step;
    setTimeout(() => rect.classList.add("drawn"), delay);
  });
  lines.forEach((line) => {
    const i = Number(line.dataset.idx);
    setTimeout(() => line.classList.add("shown"), i * step + 200);
  });
}

function wireHoverLink() {
  const setActive = (idx, on) => {
    const rect = boxOverlay.querySelector(`.seg-box[data-idx="${idx}"]`);
    const line = resultText.querySelector(`.ocr-line[data-idx="${idx}"]`);
    if (rect) rect.classList.toggle("seg-active", on);
    if (line) line.classList.toggle("line-active", on);
  };
  boxOverlay.querySelectorAll(".seg-box").forEach((rect) => {
    const idx = rect.dataset.idx;
    rect.addEventListener("mouseenter", () => setActive(idx, true));
    rect.addEventListener("mouseleave", () => setActive(idx, false));
  });
  resultText.querySelectorAll(".ocr-line").forEach((line) => {
    const idx = line.dataset.idx;
    line.addEventListener("mouseenter", () => setActive(idx, true));
    line.addEventListener("mouseleave", () => setActive(idx, false));
  });
}

// ── Detector toggle ───────────────────────────────────────────────────────
detectorSel.addEventListener("change", () => {
  const isYolo = detectorSel.value === "yolo";
  const needsPad = isYolo || detectorSel.value === "legacy";
  if (isYolo) show(confRow); else hide(confRow);
  if (needsPad) show(padRow); else hide(padRow);
});

// ── File selection ────────────────────────────────────────────────────────
function handleFile(file) {
  if (!file) return;

  // Local preview while uploading
  const reader = new FileReader();
  reader.onload = (e) => { previewImg.src = e.target.result; };
  reader.readAsDataURL(file);
  hide(dropZone);
  show(previewCont);
  runBtn.disabled = true;

  // Upload to server
  const fd = new FormData();
  fd.append("image", file);

  fetch("/upload", { method: "POST", body: fd })
    .then((r) => r.json())
    .then((data) => {
      if (!data.ok) throw new Error(data.error || "Upload failed.");
      uploadId = data.upload_id;
      previewUrl = data.preview_url;
      runBtn.disabled = false;
    })
    .catch((err) => {
      show(dropZone);
      hide(previewCont);
      alert("Upload error: " + err.message);
    });
}

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// Drag-and-drop
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// Change image
changeFileBtn.addEventListener("click", () => {
  show(dropZone);
  hide(previewCont);
  runBtn.disabled = true;
  uploadId = null;
  fileInput.value = "";
});

// ── Run OCR ───────────────────────────────────────────────────────────────
runBtn.addEventListener("click", () => {
  if (!uploadId) return;

  runBtn.disabled = true;
  hide(uploadSection);
  show(loadingOverlay);

  const body = {
    upload_id: uploadId,
    detector: detectorSel.value,
    output_format: formatSel.value,
    conf: detectorSel.value === "yolo" ? parseFloat(confInput.value) : null,
    pad: (detectorSel.value === "yolo" || detectorSel.value === "legacy") ? parseInt(padInput.value, 10) : null,
  };

  fetch("/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
    .then((r) => r.json())
    .then((data) => {
      hide(loadingOverlay);
      if (!data.ok) throw new Error(data.error || "Processing failed.");

      downloadLink.href = data.download_url;
      hide(errorBanner);
      show(resultsSection);

      // Render the document with its animated box overlay once the image
      // has loaded (so the SVG viewBox matches the natural dimensions).
      resultImg.onload = () => renderResult(data);
      resultImg.src = previewUrl;
      if (resultImg.complete) renderResult(data);
    })
    .catch((err) => {
      hide(loadingOverlay);
      show(uploadSection);
      runBtn.disabled = false;
      show(resultsSection);
      showError("Error: " + err.message);
    });
});

// ── Results actions ───────────────────────────────────────────────────────
newDocBtn.addEventListener("click", resetToUpload);
dismissErrBtn.addEventListener("click", () => hide(errorBanner));

toggleBoxesBtn.addEventListener("click", () => {
  const hidden = docViewport.classList.toggle("boxes-hidden");
  toggleBoxesBtn.textContent = hidden ? "Show boxes" : "Hide boxes";
});
