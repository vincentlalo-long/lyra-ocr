import { activeTab } from "./router.js";
import { escapeHtml, mustEl, showToast } from "./core.js";

// State
let artifacts = [];
let currentArtifact = null;
let dataset = null;
let currentFrameIndex = 0;
let selectedTrackId = null;
let selectedLabel = "lyric";
let isPlaying = false;
let playInterval = null;
let currentFilter = "all";
let frameImage = new Image();

// DOM Elements
const artifactSelect = mustEl("artifactSelect");
const trackList = mustEl("trackList");
const trackCount = mustEl("trackCount");
const frameCanvas = mustEl("frameCanvas");
const ctx = frameCanvas.getContext("2d");
const frameSlider = mustEl("frameSlider");
const frameNum = mustEl("frameNum");
const frameTotal = mustEl("frameTotal");
const ptsTime = mustEl("ptsTime");
const playBtn = mustEl("playBtn");
const prevBtn = mustEl("prevBtn");
const nextBtn = mustEl("nextBtn");
const fpsSelect = mustEl("fpsSelect");
const trackInfoCard = mustEl("trackInfoCard");
const labelOptions = mustEl("labelOptions");
const trackingErrorCheckbox = mustEl("trackingErrorCheckbox");
const correctedTextInput = mustEl("correctedTextInput");
const notesInput = mustEl("notesInput");
const saveLabelBtn = mustEl("saveLabelBtn");
const coveragePill = mustEl("coveragePill");
const refreshBtn = mustEl("refreshBtn");
const showAllBoxes = mustEl("showAllBoxes");

// Colors mapping
const COLORS = {
  lyric: "#10b981",
  non_lyric: "#ef4444"
};

// Load initial artifacts
export async function initReviewTab() {
  try {
    const res = await fetch("/api/artifacts");
    const data = await res.json();
    artifacts = data.artifacts || [];

    artifactSelect.innerHTML = "";
    artifacts.forEach(a => {
      const opt = document.createElement("option");
      opt.value = a.path;
      opt.textContent = `${a.name} (${a.coverage_percent}%)`;
      artifactSelect.appendChild(opt);
    });

    if (artifacts.length > 0) {
      loadArtifact(artifacts[0].path);
    }
  } catch (err) {
    console.error("Failed to load artifacts:", err);
  }
}

export async function loadArtifact(path) {
  currentArtifact = path;
  try {
    const res = await fetch(`/api/data?dir=${encodeURIComponent(path)}`);
    dataset = await res.json();
    updateUI();
  } catch (err) {
    console.error("Failed to load dataset:", err);
  }
}

// Deep-link target used by router.navigateToReview (Eval "Fix →" rows, Runs "Review").
export async function openReviewTarget(dir, trackId) {
  await loadArtifact(dir);
  if (trackId !== null && trackId !== undefined && dataset && dataset.tracks) {
    selectTrack(trackId, true);
  }
}

function updateUI() {
  if (!dataset) return;

  // Update Header & Coverage
  const cov = dataset.status?.coverage_percent || 0;
  coveragePill.textContent = `${cov}% Labeled (${dataset.status?.labeled_tracks || 0}/${dataset.status?.total_tracks || 0})`;
  coveragePill.className = `badge ${cov >= 95 ? "badge-lyric" : "badge-non_lyric"}`;

  // Frames setup
  const totalFrames = (dataset.frame_index && dataset.frame_index.length) || 0;
  frameTotal.textContent = totalFrames;
  frameSlider.max = Math.max(0, totalFrames - 1);
  currentFrameIndex = 0;

  // Render Track List
  renderTrackList();

  // Select first track if available
  if (dataset.tracks && dataset.tracks.length > 0) {
    selectTrack(dataset.tracks[0].track_id);
  }

  loadCurrentFrame();
}

function renderTrackList() {
  trackList.innerHTML = "";
  if (!dataset || !dataset.tracks) return;

  let filtered = dataset.tracks.filter(t => {
    const tid = String(t.track_id);
    const lbl = dataset.labels[tid]?.human_label;
    if (currentFilter === "unlabeled") return !lbl;
    if (currentFilter === "lyric") return (lbl === "lyric" || (!lbl && t.label === "lyric"));
    if (currentFilter === "non_lyric") return (lbl && lbl !== "lyric") || (!lbl && t.label !== "lyric");
    return true;
  });

  trackCount.textContent = filtered.length;

  filtered.forEach(t => {
    const tid = String(t.track_id);
    const card = document.createElement("div");
    card.className = `track-card ${String(selectedTrackId) === tid ? "active" : ""}`;
    card.id = `track-card-${tid}`;

    const humanLbl = dataset.labels[tid]?.human_label;
    const displayLbl = humanLbl || t.label || "non_lyric";
    const hasHuman = !!humanLbl;

    card.innerHTML = `
      <div class="track-card-header">
        <strong>Track #${t.track_id}</strong>
        <span class="badge badge-${displayLbl}">${displayLbl} ${hasHuman ? "✓" : ""}</span>
      </div>
      <div class="track-card-text">${escapeHtml(t.features?.dominant_text || "(empty text)")}</div>
      <div class="track-card-meta">
        <span>${(t.duration_ms / 1000).toFixed(1)}s (${t.occurrences?.length || t.num_occurrences || 0} f)</span>
        <span>Conf: ${(t.confidence || 0).toFixed(2)}</span>
      </div>
    `;

    card.addEventListener("click", () => selectTrack(t.track_id, true));
    trackList.appendChild(card);
  });
}

export function selectTrack(trackId, jumpToFrame = false) {
  selectedTrackId = String(trackId);
  const t = dataset.tracks.find(x => String(x.track_id) === selectedTrackId);
  if (!t) return;

  // Update Card Active State
  document.querySelectorAll(".track-card").forEach(c => c.classList.remove("active"));
  const activeCard = document.getElementById(`track-card-${selectedTrackId}`);
  if (activeCard) {
    activeCard.classList.add("active");
    activeCard.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  // If requested, seek frame to start of track
  if (jumpToFrame && t.occurrences && t.occurrences.length > 0) {
    const firstOccur = t.occurrences[0];
    const fIdx = dataset.frame_index.findIndex(f => f.pts_ms >= firstOccur.timestamp_ms);
    if (fIdx >= 0) {
      seekToFrame(fIdx);
    }
  }

  // Populate Right Inspector
  const humanRec = dataset.labels[selectedTrackId] || {};
  const currentHumanLabel = humanRec.human_label || (t.label || "non_lyric");
  setChosenLabel(currentHumanLabel);

  trackingErrorCheckbox.checked = !!humanRec.tracking_error;
  correctedTextInput.value = humanRec.corrected_text || "";
  notesInput.value = humanRec.notes || "";

  trackInfoCard.innerHTML = `
    <div class="info-row">
      <span class="info-label">Track ID:</span>
      <span class="info-val">#${t.track_id}</span>
    </div>
    <div class="info-row">
      <span class="info-label">Model Classification:</span>
      <span class="badge badge-${t.label || 'non_lyric'}">${t.label || 'non_lyric'} (${t.confidence?.toFixed(2) || '0.00'})</span>
    </div>
    <div class="info-row">
      <span class="info-label">Lifetime:</span>
      <span class="info-val">${(t.start_ms/1000).toFixed(2)}s → ${(t.end_ms/1000).toFixed(2)}s (${t.duration_ms} ms)</span>
    </div>
    <div class="info-row">
      <span class="info-label">Occurrences:</span>
      <span class="info-val">${t.occurrences?.length || t.num_occurrences || 0} frames</span>
    </div>
    <div class="section-title" style="margin-top:6px">Dominant Text</div>
    <div class="text-box-display">${escapeHtml(t.features?.dominant_text || "(empty)")}</div>
  `;

  drawFrame();
}

function setChosenLabel(label) {
  selectedLabel = label;
  document.querySelectorAll(".label-option").forEach(opt => {
    opt.classList.toggle("selected", opt.dataset.label === label);
  });
}

// Canvas & Frame Rendering
function loadCurrentFrame() {
  if (!dataset || !dataset.frame_index || dataset.frame_index.length === 0) return;

  const fInfo = dataset.frame_index[currentFrameIndex];
  if (!fInfo) return;

  frameNum.textContent = currentFrameIndex + 1;
  frameSlider.value = currentFrameIndex;

  // PTS Format MM:SS.mmm
  const pts = fInfo.pts_ms || 0;
  const min = Math.floor(pts / 60000);
  const sec = Math.floor((pts % 60000) / 1000);
  const ms = Math.floor(pts % 1000);
  ptsTime.textContent = `${String(min).padStart(2,'0')}:${String(sec).padStart(2,'0')}.${String(ms).padStart(3,'0')}`;

  frameImage.onload = () => {
    frameCanvas.width = frameImage.naturalWidth || 1280;
    frameCanvas.height = frameImage.naturalHeight || 720;
    drawFrame();
  };
  frameImage.src = `/api/frame?dir=${encodeURIComponent(currentArtifact)}&file=${encodeURIComponent(fInfo.filename)}`;
}

function drawFrame() {
  if (!frameImage.complete) return;

  ctx.clearRect(0, 0, frameCanvas.width, frameCanvas.height);
  ctx.drawImage(frameImage, 0, 0);

  if (!dataset || !dataset.tracks) return;

  const currentPts = dataset.frame_index[currentFrameIndex]?.pts_ms;
  if (currentPts === undefined) return;

  // Find occurrences in this frame
  dataset.tracks.forEach(t => {
    const tid = String(t.track_id);
    const isSelected = (tid === selectedTrackId);

    if (!showAllBoxes.checked && !isSelected) return;

    const occ = (t.occurrences || []).find(o => Math.abs(o.timestamp_ms - currentPts) < 300);
    if (!occ || !occ.bbox) return;

    const [x1, y1, x2, y2] = occ.bbox;
    const w = x2 - x1;
    const h = y2 - y1;

    const humanLbl = dataset.labels[tid]?.human_label;
    const color = COLORS[humanLbl || t.label] || "#3b82f6";

    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected ? 4 : 2;
    ctx.strokeRect(x1, y1, w, h);

    if (isSelected) {
      ctx.fillStyle = "rgba(59, 130, 246, 0.2)";
      ctx.fillRect(x1, y1, w, h);
    }

    // Label Tag
    ctx.fillStyle = color;
    const tagText = `#${t.track_id} ${t.features?.dominant_text?.substring(0, 16) || ''}`;
    ctx.font = "bold 13px monospace";
    const txtWidth = ctx.measureText(tagText).width;
    ctx.fillRect(x1, Math.max(0, y1 - 20), txtWidth + 8, 20);

    ctx.fillStyle = "#fff";
    ctx.fillText(tagText, x1 + 4, Math.max(14, y1 - 5));
  });
}

// Canvas click to select bounding box
frameCanvas.addEventListener("click", e => {
  if (!dataset || !dataset.tracks) return;

  const rect = frameCanvas.getBoundingClientRect();
  const scaleX = frameCanvas.width / rect.width;
  const scaleY = frameCanvas.height / rect.height;

  const clickX = (e.clientX - rect.left) * scaleX;
  const clickY = (e.clientY - rect.top) * scaleY;

  const currentPts = dataset.frame_index[currentFrameIndex]?.pts_ms;
  if (currentPts === undefined) return;

  for (let t of dataset.tracks) {
    const occ = (t.occurrences || []).find(o => Math.abs(o.timestamp_ms - currentPts) < 300);
    if (occ && occ.bbox) {
      const [x1, y1, x2, y2] = occ.bbox;
      if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
        selectTrack(t.track_id);
        break;
      }
    }
  }
});

// Save Label Action
async function saveCurrentLabel() {
  if (!selectedTrackId || !dataset) return;

  const payload = {
    artifact_dir: currentArtifact,
    video_id: dataset.manifest?.video_id || dataset.manifest?.video_path || "video",
    track_id: selectedTrackId,
    human_label: selectedLabel,
    tracking_error: trackingErrorCheckbox.checked,
    corrected_text: correctedTextInput.value.trim() || null,
    notes: notesInput.value.trim(),
    annotator: "web_reviewer"
  };

  try {
    const res = await fetch("/api/label", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();

    if (result.ok) {
      showToast(`Saved Track #${selectedTrackId} as ${selectedLabel.toUpperCase()}`);
      dataset.labels[selectedTrackId] = result.record;
      dataset.status = result.status;

      // Update UI
      renderTrackList();
      const cov = dataset.status?.coverage_percent || 0;
      coveragePill.textContent = `${cov}% Labeled (${dataset.status?.labeled_tracks || 0}/${dataset.status?.total_tracks || 0})`;

      // Advance to next track
      advanceNextTrack();
    }
  } catch (err) {
    console.error("Save label failed:", err);
  }
}

function advanceNextTrack() {
  if (!dataset || !dataset.tracks) return;
  const idx = dataset.tracks.findIndex(t => String(t.track_id) === selectedTrackId);
  if (idx >= 0 && idx + 1 < dataset.tracks.length) {
    selectTrack(dataset.tracks[idx + 1].track_id, true);
  }
}

function seekToFrame(idx) {
  if (!dataset || !dataset.frame_index) return;
  currentFrameIndex = Math.max(0, Math.min(dataset.frame_index.length - 1, idx));
  loadCurrentFrame();
}


function togglePlay() {
  isPlaying = !isPlaying;
  playBtn.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
  if (isPlaying) {
    const fps = parseFloat(fpsSelect.value) * 2.0; // 2.0 FPS baseline
    playInterval = setInterval(() => {
      if (currentFrameIndex + 1 < dataset.frame_index.length) {
        seekToFrame(currentFrameIndex + 1);
      } else {
        togglePlay();
      }
    }, 1000 / fps);
  } else {
    clearInterval(playInterval);
  }
}

// Event Listeners
artifactSelect.addEventListener("change", e => loadArtifact(e.target.value));
refreshBtn.addEventListener("click", () => loadArtifact(currentArtifact));
prevBtn.addEventListener("click", () => seekToFrame(currentFrameIndex - 1));
nextBtn.addEventListener("click", () => seekToFrame(currentFrameIndex + 1));
playBtn.addEventListener("click", togglePlay);
frameSlider.addEventListener("input", e => seekToFrame(parseInt(e.target.value)));
saveLabelBtn.addEventListener("click", saveCurrentLabel);
showAllBoxes.addEventListener("change", drawFrame);

document.querySelectorAll(".label-option").forEach(opt => {
  opt.addEventListener("click", () => setChosenLabel(opt.dataset.label));
});

document.querySelectorAll(".filter-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentFilter = btn.dataset.filter;
    renderTrackList();
  });
});

// Keyboard Shortcuts
window.addEventListener("keydown", e => {
  if (activeTab !== "review") return;
  // Ignore if typing in text input
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") {
    if (e.key === "Enter") saveCurrentLabel();
    return;
  }

  if (e.key === "1") setChosenLabel("lyric");
  else if (e.key === "2") setChosenLabel("non_lyric");
  else if (e.key === "e" || e.key === "E") trackingErrorCheckbox.checked = !trackingErrorCheckbox.checked;
  else if (e.key === "ArrowLeft" || e.key === "a" || e.key === "A") seekToFrame(currentFrameIndex - 1);
  else if (e.key === "ArrowRight" || e.key === "d" || e.key === "D") seekToFrame(currentFrameIndex + 1);
  else if (e.key === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "Enter") saveCurrentLabel();
});

