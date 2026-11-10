// =========================================================================
// LRC Studio: Video-to-LRC Generation, Cue Inspector, Synchronized Player
// =========================================================================

import { activeTab } from "./router.js";
import { clearHiddenArtifacts, getHiddenArtifacts, hideArtifact, showToast } from "./core.js";

let currentCues = [];
let currentMetadata = { title: "", artist: "", album: "", by: "Lyra OCR", offset_ms: 0 };
let currentLrcText = "";
let currentArtifactDir = "";
let currentVideoPath = "";
let currentOutLrc = "";
let playbackTimeMs = 0;
let isPlaying = false;
let playTimer = null;
let maxDurationMs = 180000;

let allRawArtifacts = [];
let showDatasetRuns = false;

export async function initLrcStudio() {
  await loadArtifactOptions();
  bindStudioEvents();
}

async function loadArtifactOptions() {
  const select = document.getElementById("lrcArtifactSelect");
  if (!select) return;

  try {
    const res = await fetch("/api/artifacts");
    if (!res.ok) return;
    const data = await res.json();
    allRawArtifacts = data.artifacts || [];
    renderArtifactDropdown();
  } catch (err) {
    console.warn("Failed to load artifacts for LRC studio", err);
  }
}

function renderArtifactDropdown() {
  const select = document.getElementById("lrcArtifactSelect");
  if (!select) return;

  const currentVal = select.value;
  const hidden = getHiddenArtifacts();
  const visible = allRawArtifacts.filter(a => !hidden.has(a.path));
  const outputs = visible.filter(a => a.kind === "output" || a.path.startsWith("outputs/") || a.path.startsWith("artifacts/"));
  const datasetRuns = visible.filter(a => a.kind === "dataset_run" || a.path.startsWith("datasets/"));

  select.innerHTML = '<option value="">-- Choose Existing Artifact Directory --</option>';

  if (outputs.length > 0) {
    const group = document.createElement("optgroup");
    group.label = "Outputs";
    for (const art of outputs) {
      const opt = document.createElement("option");
      opt.value = art.path;
      opt.textContent = `${art.name} (${art.title || art.video_id || "video"})`;
      group.appendChild(opt);
    }
    select.appendChild(group);
  }

  if (showDatasetRuns && datasetRuns.length > 0) {
    const group = document.createElement("optgroup");
    group.label = `Dataset Runs (${datasetRuns.length})`;
    for (const art of datasetRuns) {
      const opt = document.createElement("option");
      opt.value = art.path;
      opt.textContent = `${art.name} (${art.title || art.video_id || "video"})`;
      group.appendChild(opt);
    }
    select.appendChild(group);
  }

  if (currentVal && [...select.options].some(o => o.value === currentVal)) {
    select.value = currentVal;
  }

  const toggleBtn = document.getElementById("lrcToggleDatasetsBtn");
  if (toggleBtn) {
    toggleBtn.textContent = showDatasetRuns
      ? `Hide Dataset Runs (${datasetRuns.length})`
      : `Show Dataset Runs (${datasetRuns.length})`;
    toggleBtn.style.display = datasetRuns.length ? "" : "none";
  }

  const restoreBtn = document.getElementById("lrcRestoreHiddenBtn");
  if (restoreBtn) {
    restoreBtn.style.display = hidden.size > 0 ? "" : "none";
    restoreBtn.textContent = `↺ Restore ${hidden.size} Hidden`;
  }
}

export function resetStudioView() {
  currentCues = [];
  currentMetadata = { title: "", artist: "", album: "", by: "Lyra OCR", offset_ms: 0 };
  currentLrcText = "";
  currentArtifactDir = "";
  currentVideoPath = "";
  currentOutLrc = "";
  playbackTimeMs = 0;
  if (isPlaying) togglePlayback();

  const artSelect = document.getElementById("lrcArtifactSelect");
  if (artSelect) artSelect.value = "";
  const vidIn = document.getElementById("lrcVideoInput");
  if (vidIn) vidIn.value = "";
  const titleIn = document.getElementById("lrcTitleInput");
  if (titleIn) titleIn.value = "";
  const artistIn = document.getElementById("lrcArtistInput");
  if (artistIn) artistIn.value = "";
  const offsetIn = document.getElementById("lrcOffsetMs");
  if (offsetIn) offsetIn.value = "0";
  const revIn = document.getElementById("lrcIncludeReview");
  if (revIn) revIn.checked = false;
  const forceIn = document.getElementById("lrcForceRerun");
  if (forceIn) forceIn.checked = false;

  const banner = document.getElementById("lrcStatusBanner");
  if (banner) banner.style.display = "none";
  const mainCont = document.getElementById("studioMainContainer");
  if (mainCont) mainCont.style.display = "none";

  renderCuesList();
  renderKaraokeView();
  syncCuesToRawLrc();
  showToast("Studio view cleared");
}

function bindStudioEvents() {
  const genBtn = document.getElementById("lrcGenerateBtn");
  if (genBtn && !genBtn.dataset.bound) {
    genBtn.dataset.bound = "true";
    genBtn.addEventListener("click", handleGenerateLrc);
  }

  const artSelect = document.getElementById("lrcArtifactSelect");
  if (artSelect && !artSelect.dataset.bound) {
    artSelect.dataset.bound = "true";
    artSelect.addEventListener("change", () => {
      if (artSelect.value) {
        document.getElementById("lrcVideoInput").value = "";
      }
    });
  }

  const hideBtn = document.getElementById("lrcHideArtifactBtn");
  if (hideBtn && !hideBtn.dataset.bound) {
    hideBtn.dataset.bound = "true";
    hideBtn.addEventListener("click", () => {
      const val = artSelect ? artSelect.value : "";
      if (!val) { showToast("Select an artifact to hide from web view"); return; }
      hideArtifact(val);
      showToast(`Hidden from view: ${basename(val)}`);
      renderArtifactDropdown();
    });
  }

  const restoreBtn = document.getElementById("lrcRestoreHiddenBtn");
  if (restoreBtn && !restoreBtn.dataset.bound) {
    restoreBtn.dataset.bound = "true";
    restoreBtn.addEventListener("click", () => {
      clearHiddenArtifacts();
      showToast("All hidden artifacts restored");
      renderArtifactDropdown();
    });
  }

  const toggleDsBtn = document.getElementById("lrcToggleDatasetsBtn");
  if (toggleDsBtn && !toggleDsBtn.dataset.bound) {
    toggleDsBtn.dataset.bound = "true";
    toggleDsBtn.addEventListener("click", () => {
      showDatasetRuns = !showDatasetRuns;
      renderArtifactDropdown();
    });
  }

  const resetBtn = document.getElementById("lrcResetStudioBtn");
  if (resetBtn && !resetBtn.dataset.bound) {
    resetBtn.dataset.bound = "true";
    resetBtn.addEventListener("click", resetStudioView);
  }

  const copyBtn = document.getElementById("lrcCopyBtn");
  if (copyBtn && !copyBtn.dataset.bound) {
    copyBtn.dataset.bound = "true";
    copyBtn.addEventListener("click", copyLrcToClipboard);
  }

  const dlBtn = document.getElementById("lrcDownloadBtn");
  if (dlBtn && !dlBtn.dataset.bound) {
    dlBtn.dataset.bound = "true";
    dlBtn.addEventListener("click", downloadLrcFile);
  }

  const saveBtn = document.getElementById("lrcSaveBtn");
  if (saveBtn && !saveBtn.dataset.bound) {
    saveBtn.dataset.bound = "true";
    saveBtn.addEventListener("click", handleSaveLrcChanges);
  }

  const addCueBtn = document.getElementById("lrcAddCueBtn");
  if (addCueBtn && !addCueBtn.dataset.bound) {
    addCueBtn.dataset.bound = "true";
    addCueBtn.addEventListener("click", handleAddCue);
  }

  const playBtn = document.getElementById("studioPlayBtn");
  if (playBtn && !playBtn.dataset.bound) {
    playBtn.dataset.bound = "true";
    playBtn.addEventListener("click", togglePlayback);
  }

  const scrubber = document.getElementById("studioScrubber");
  if (scrubber && !scrubber.dataset.bound) {
    scrubber.dataset.bound = "true";
    scrubber.addEventListener("input", (e) => {
      seekTo(parseInt(e.target.value, 10));
    });
  }

  // Switch between Cues Editor and Raw LRC view
  const tabCues = document.getElementById("studioTabCues");
  const tabRaw = document.getElementById("studioTabRaw");
  if (tabCues && tabRaw) {
    tabCues.addEventListener("click", () => {
      tabCues.classList.add("active");
      tabRaw.classList.remove("active");
      document.getElementById("studioCuesView").style.display = "";
      document.getElementById("studioRawView").style.display = "none";
    });
    tabRaw.addEventListener("click", () => {
      tabRaw.classList.add("active");
      tabCues.classList.remove("active");
      document.getElementById("studioCuesView").style.display = "none";
      document.getElementById("studioRawView").style.display = "";
      syncCuesToRawLrc();
    });
  }
}

async function handleGenerateLrc() {
  const genBtn = document.getElementById("lrcGenerateBtn");
  const statusEl = document.getElementById("lrcStatusBanner");
  const artDir = document.getElementById("lrcArtifactSelect").value.trim();
  const videoPath = document.getElementById("lrcVideoInput").value.trim();

  if (!artDir && !videoPath) {
    showStatus("Please specify a video file path or choose an existing artifact directory.", "error");
    return;
  }

  const payload = {
    artifact_dir: artDir || undefined,
    video_path: videoPath || undefined,
    tracker: document.getElementById("lrcTrackerSelect").value,
    correct_method: document.getElementById("lrcCorrectSelect").value,
    include_review: document.getElementById("lrcIncludeReview").checked,
    refine: document.getElementById("lrcRefineTiming").checked,
    offset_ms: parseInt(document.getElementById("lrcOffsetMs").value || "0", 10),
    title: document.getElementById("lrcTitleInput").value.trim(),
    artist: document.getElementById("lrcArtistInput").value.trim(),
    force: document.getElementById("lrcForceRerun").checked,
  };

  genBtn.disabled = true;
  const isUrl = videoPath.startsWith("http://") || videoPath.startsWith("https://");
  genBtn.innerHTML = isUrl
    ? '<span class="spinner"></span> Downloading & Processing Video...'
    : '<span class="spinner"></span> Processing OCR & Generating LRC...';
  showStatus(
    isUrl
      ? "Downloading YouTube video & extracting synchronized lyrics..."
      : "Processing video and extracting synchronized lyrics...",
    "info"
  );

  try {
    const res = await fetch("/api/lrc/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: jsonStringify(payload),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Generation failed");
    }

    currentCues = data.cues || [];
    currentMetadata = data.metadata || payload;
    currentLrcText = data.lrc_text || "";
    currentOutLrc = data.out_lrc || "";
    currentArtifactDir = artDir;
    currentVideoPath = videoPath;

    // Estimate duration from cues
    if (currentCues.length > 0) {
      maxDurationMs = Math.max(...currentCues.map(c => c.end_ms || c.start_ms)) + 5000;
    }
    const scrubber = document.getElementById("studioScrubber");
    if (scrubber) {
      scrubber.max = maxDurationMs;
    }

    renderCuesList();
    renderKaraokeView();
    syncCuesToRawLrc();
    updateStatsSummary(data);

    showStatus(`Generated ${data.n_cues} lyric cues successfully! Saved to: ${data.out_lrc}`, "success");
    document.getElementById("studioMainContainer").style.display = "grid";
  } catch (err) {
    showStatus(`Error: ${err.message}`, "error");
  } finally {
    genBtn.disabled = false;
    genBtn.innerHTML = "▶ Generate LRC";
  }
}

function updateStatsSummary(data) {
  const container = document.getElementById("studioStatsBar");
  if (!container) return;
  container.innerHTML = `
    <span class="stat-badge"><strong>${data.n_cues || 0}</strong> Accepted Cues</span>
    <span class="stat-badge"><strong>${data.n_skipped || 0}</strong> Filtered Distractors</span>
    <span class="stat-badge ${data.review_required > 0 ? "warning" : ""}"><strong>${data.review_required || 0}</strong> Review Flags</span>
    <span class="stat-badge file-path" title="${data.out_lrc}">📄 ${basename(data.out_lrc)}</span>
  `;
}

function renderCuesList() {
  const container = document.getElementById("studioCuesList");
  if (!container) return;
  container.innerHTML = "";

  if (currentCues.length === 0) {
    container.innerHTML = '<div class="empty-state">No lyric cues found. Try toggling "Include Review Cues" or checking video text region.</div>';
    return;
  }

  currentCues.forEach((cue, idx) => {
    const row = document.createElement("div");
    row.className = "cue-row";
    row.dataset.idx = idx;

    const timeTag = formatTimestamp(cue.start_ms);
    const endTag = formatTimestamp(cue.end_ms);

    row.innerHTML = `
      <div class="cue-time" title="Click to seek player" data-time="${cue.start_ms}">
        <span class="cue-tag">${timeTag}</span>
      </div>
      <div class="cue-input-box">
        <input type="text" class="cue-text-input" value="${escapeHtml(cue.text)}" data-idx="${idx}">
      </div>
      <div class="cue-meta">
        <span class="chip-conf ${cue.confidence >= 0.9 ? 'high' : 'med'}">${Math.round((cue.confidence || 0) * 100)}%</span>
        ${cue.review_required ? '<span class="chip-rev">REV</span>' : ''}
        <button class="mini-btn danger cue-del-btn" title="Remove line" data-idx="${idx}">✕</button>
      </div>
    `;

    row.querySelector(".cue-time").addEventListener("click", () => seekTo(cue.start_ms));
    row.querySelector(".cue-text-input").addEventListener("input", (e) => {
      currentCues[idx].text = e.target.value;
      renderKaraokeView();
    });
    row.querySelector(".cue-del-btn").addEventListener("click", () => {
      currentCues.splice(idx, 1);
      renderCuesList();
      renderKaraokeView();
      syncCuesToRawLrc();
    });

    container.appendChild(row);
  });
}

function renderKaraokeView() {
  const karaokeBox = document.getElementById("studioKaraokeStage");
  if (!karaokeBox) return;

  const currentIdx = currentCues.findIndex((c, i) => {
    const nextStart = currentCues[i + 1] ? currentCues[i + 1].start_ms : Infinity;
    return playbackTimeMs >= c.start_ms && playbackTimeMs < nextStart;
  });

  const prevText = currentIdx > 0 ? currentCues[currentIdx - 1].text : "";
  const currText = currentIdx >= 0 ? currentCues[currentIdx].text : "(Instrumental / Outro)";
  const nextText = currentIdx >= 0 && currentIdx < currentCues.length - 1 ? currentCues[currentIdx + 1].text : "";

  karaokeBox.innerHTML = `
    <div class="karaoke-line prev">${escapeHtml(prevText)}</div>
    <div class="karaoke-line active">✦ ${escapeHtml(currText)}</div>
    <div class="karaoke-line next">${escapeHtml(nextText)}</div>
  `;

  // Highlight corresponding row in list
  document.querySelectorAll(".cue-row").forEach((row, i) => {
    row.classList.toggle("playing", i === currentIdx);
  });
}

function syncCuesToRawLrc() {
  const textarea = document.getElementById("studioRawLrcText");
  if (!textarea) return;

  const lines = [
    `[ti:${currentMetadata.title || ""}]`,
    `[ar:${currentMetadata.artist || ""}]`,
    `[by:${currentMetadata.by || "Lyra OCR"}]`,
    `[offset:${currentMetadata.offset_ms || 0}]`,
  ];

  currentCues.forEach(c => {
    const ts = formatTimestamp(c.start_ms);
    lines.push(`${ts}${c.text}`);
  });

  currentLrcText = lines.join("\n") + "\n";
  textarea.value = currentLrcText;
}

function handleAddCue() {
  const newCue = {
    start_ms: playbackTimeMs,
    end_ms: playbackTimeMs + 2500,
    text: "New Lyric Line",
    confidence: 1.0,
    review_required: false,
    source_track_ids: [],
    source_segment_ids: [],
  };

  currentCues.push(newCue);
  currentCues.sort((a, b) => a.start_ms - b.start_ms);
  renderCuesList();
  renderKaraokeView();
  syncCuesToRawLrc();
}

async function handleSaveLrcChanges() {
  if (!currentOutLrc) {
    showStatus("No active output file to save to.", "error");
    return;
  }

  const rawTextarea = document.getElementById("studioRawLrcText");
  const updatedLrcText = rawTextarea ? rawTextarea.value : currentLrcText;

  const payload = {
    out_lrc: currentOutLrc,
    lrc_text: updatedLrcText,
    cues: currentCues,
    metadata: currentMetadata,
    video_path: currentVideoPath,
  };

  try {
    const res = await fetch("/api/lrc/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: jsonStringify(payload),
    });
    if (!res.ok) throw new Error("Failed to save changes to disk");
    showStatus("LRC and cues successfully saved to disk!", "success");
  } catch (err) {
    showStatus(`Save error: ${err.message}`, "error");
  }
}

function copyLrcToClipboard() {
  syncCuesToRawLrc();
  navigator.clipboard.writeText(currentLrcText).then(() => {
    showStatus("LRC contents copied to clipboard!", "success");
  }).catch(() => {
    showStatus("Failed to copy to clipboard", "error");
  });
}

function downloadLrcFile() {
  syncCuesToRawLrc();
  const blob = new Blob([currentLrcText], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = basename(currentOutLrc) || "lyrics.lrc";
  a.click();
  URL.revokeObjectURL(a.href);
}

function togglePlayback() {
  const btn = document.getElementById("studioPlayBtn");
  if (isPlaying) {
    clearInterval(playTimer);
    isPlaying = false;
    if (btn) btn.innerHTML = "▶ Play";
  } else {
    isPlaying = true;
    if (btn) btn.innerHTML = "⏸ Pause";
    const startReal = performance.now();
    const startPlayback = playbackTimeMs;
    playTimer = setInterval(() => {
      const elapsed = performance.now() - startReal;
      playbackTimeMs = Math.min(maxDurationMs, Math.round(startPlayback + elapsed));
      updatePlaybackUI();
      if (playbackTimeMs >= maxDurationMs) {
        togglePlayback();
      }
    }, 50);
  }
}

function seekTo(ms) {
  playbackTimeMs = Math.max(0, Math.min(maxDurationMs, ms));
  updatePlaybackUI();
}

function updatePlaybackUI() {
  const scrubber = document.getElementById("studioScrubber");
  if (scrubber) scrubber.value = playbackTimeMs;
  const timeLabel = document.getElementById("studioCurrentTime");
  if (timeLabel) timeLabel.textContent = formatTimestamp(playbackTimeMs).replace(/[\[\]]/g, "");
  renderKaraokeView();
}

function formatTimestamp(ms) {
  const totalSecs = Math.max(0, ms) / 1000.0;
  const mins = Math.floor(totalSecs / 60);
  const secs = totalSecs % 60;
  const mm = String(mins).padStart(2, "0");
  const ss = secs.toFixed(2).padStart(5, "0");
  return `[${mm}:${ss}]`;
}

function showStatus(msg, type = "info") {
  const banner = document.getElementById("lrcStatusBanner");
  if (!banner) return;
  banner.textContent = msg;
  banner.className = `status-banner ${type}`;
  banner.style.display = "block";
  if (type === "success") {
    setTimeout(() => { banner.style.display = "none"; }, 5000);
  }
}

function jsonStringify(obj) {
  return JSON.stringify(obj);
}

function escapeHtml(str) {
  return (str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function basename(pathStr) {
  if (!pathStr) return "";
  return pathStr.split(/[\\/]/).pop();
}
