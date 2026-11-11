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
let allLayers = [];
let selectedLayerId = "all";
let playbackTimeMs = 0;

let isPlaying = false;
let playTimer = null;
let maxDurationMs = 180000;
let studioVideo = null;

let allRawArtifacts = [];
let showDatasetRuns = false;

export async function initLrcStudio() {
  await loadArtifactOptions();
  bindStudioEvents();
  attachVideoEvents();
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
  allLayers = [];
  selectedLayerId = "all";
  playbackTimeMs = 0;
  if (isPlaying) togglePlayback();

  const layerSel = document.getElementById("studioLayerSelect");
  if (layerSel) {
    layerSel.innerHTML = "";
    layerSel.style.display = "none";
  }


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
  if (forceIn) forceIn.checked = true;
  const loadIn = document.getElementById("lrcLoadPath");
  if (loadIn) loadIn.value = "";
  const refIn = document.getElementById("lrcReferencePath");
  if (refIn) refIn.value = "";
  const refCheck = document.getElementById("lrcUseReference");
  if (refCheck) refCheck.checked = false;
  setVideoSource("");

  const banner = document.getElementById("lrcStatusBanner");
  if (banner) banner.style.display = "none";
  const statsBar = document.getElementById("studioStatsBar");
  if (statsBar) statsBar.innerHTML = "";
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

  const loadBtn = document.getElementById("lrcLoadBtn");
  if (loadBtn && !loadBtn.dataset.bound) {
    loadBtn.dataset.bound = "true";
    loadBtn.addEventListener("click", handleLoadLrc);
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

  const layerSel = document.getElementById("studioLayerSelect");
  if (layerSel && !layerSel.dataset.bound) {
    layerSel.dataset.bound = "true";
    layerSel.addEventListener("change", (e) => {
      selectedLayerId = e.target.value;
      renderCuesList();
      syncCuesToRawLrc();
    });
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

function attachVideoEvents() {
  studioVideo = document.getElementById("studioVideo");
  if (!studioVideo || studioVideo.dataset.bound) return;
  studioVideo.dataset.bound = "true";

  studioVideo.addEventListener("timeupdate", () => {
    if (studioVideo.src && !studioVideo.paused) {
      playbackTimeMs = Math.round(studioVideo.currentTime * 1000);
      updatePlaybackUI();
    }
  });
  studioVideo.addEventListener("seeked", () => {
    playbackTimeMs = Math.round(studioVideo.currentTime * 1000);
    updatePlaybackUI();
  });
  studioVideo.addEventListener("loadedmetadata", () => {
    if (isFinite(studioVideo.duration) && studioVideo.duration > 0) {
      maxDurationMs = Math.round(studioVideo.duration * 1000);
      const scrubber = document.getElementById("studioScrubber");
      if (scrubber) scrubber.max = maxDurationMs;
    }
  });
  studioVideo.addEventListener("play", () => {
    isPlaying = true;
    clearInterval(playTimer);
    const btn = document.getElementById("studioPlayBtn");
    if (btn) btn.innerHTML = "⏸ Pause";
  });
  studioVideo.addEventListener("pause", () => {
    isPlaying = false;
    const btn = document.getElementById("studioPlayBtn");
    if (btn) btn.innerHTML = "▶ Play";
  });
  studioVideo.addEventListener("ended", () => {
    isPlaying = false;
    const btn = document.getElementById("studioPlayBtn");
    if (btn) btn.innerHTML = "▶ Play";
  });
}

function setVideoSource(path) {
  if (!studioVideo) studioVideo = document.getElementById("studioVideo");
  const wrapper = document.getElementById("studioVideoWrapper");
  if (!studioVideo) return;

  if (!path) {
    studioVideo.pause();
    studioVideo.removeAttribute("src");
    studioVideo.load();
    if (wrapper) wrapper.style.display = "none";
    return;
  }

  const isRemote = path.startsWith("http://") || path.startsWith("https://");
  studioVideo.src = isRemote ? path : `/api/media?path=${encodeURIComponent(path)}`;
  if (wrapper) wrapper.style.display = "";
}

function parseLrcText(text) {
  const cues = [];
  const timestampRe = /\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]/g;
  for (const raw of (text || "").split(/\r?\n/)) {
    timestampRe.lastIndex = 0;
    const starts = [];
    let match;
    while ((match = timestampRe.exec(raw)) !== null) {
      const minutes = parseInt(match[1], 10);
      const seconds = parseInt(match[2], 10);
      const fraction = match[3] ? parseInt(match[3].padEnd(3, "0"), 10) : 0;
      starts.push((minutes * 60 + seconds) * 1000 + fraction);
    }
    const lyricText = raw.replace(/\[[^\]]*\]/g, "").trim();
    if (!lyricText || starts.length === 0) continue;
    for (const start of starts) {
      cues.push({ start_ms: start, end_ms: start + 3000, text: lyricText, confidence: 1.0, review_required: false });
    }
  }
  cues.sort((a, b) => a.start_ms - b.start_ms);
  cues.forEach((cue, idx) => {
    cue.end_ms = cues[idx + 1] ? cues[idx + 1].start_ms : cue.start_ms + 4000;
  });
  return cues;
}

async function handleLoadLrc() {
  const input = document.getElementById("lrcLoadPath");
  const lrcPath = (input ? input.value : "").trim();
  if (!lrcPath) {
    showStatus("Enter a .lrc file path to load for side-by-side testing.", "error");
    return;
  }

  try {
    const res = await fetch(`/api/lrc?path=${encodeURIComponent(lrcPath)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to load LRC");

    currentCues = Array.isArray(data.cues) && data.cues.length ? data.cues : parseLrcText(data.lrc_text || "");
    currentMetadata = data.metadata || { title: "", artist: "", album: "", by: "Lyra OCR", offset_ms: 0 };
    currentLrcText = data.lrc_text || "";
    currentOutLrc = data.path || lrcPath;
    currentArtifactDir = "";
    currentVideoPath = data.video_path || "";
    playbackTimeMs = 0;
    setVideoSource(currentVideoPath);
    maxDurationMs = currentCues.length
      ? Math.max(...currentCues.map(c => c.end_ms || c.start_ms)) + 5000
      : 180000;
    const scrubber = document.getElementById("studioScrubber");
    if (scrubber) scrubber.max = maxDurationMs;

    renderCuesList();
    renderKaraokeView();
    syncCuesToRawLrc();
    allLayers = data.layers || [];
    renderLayerSelector(allLayers, data.primary_lang);
    document.getElementById("studioMainContainer").style.display = "grid";
    showStatus(
      `Loaded ${currentCues.length} cues${currentVideoPath ? " with video" : " (no video path found)"}.`,
      "success"
    );
  } catch (err) {
    showStatus(`Load error: ${err.message}`, "error");
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
    use_reference: document.getElementById("lrcUseReference").checked,
    no_caption_fallback: document.getElementById("lrcNoCaptionFallback") ? document.getElementById("lrcNoCaptionFallback").checked : false,
    reference_path: document.getElementById("lrcReferencePath").value.trim() || undefined,
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
    currentVideoPath = data.video_path || videoPath;
    setVideoSource(currentVideoPath);

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
    allLayers = data.layers || [];
    renderLayerSelector(allLayers, data.primary_lang);
    updateStatsSummary(data);

    showStatus(`Generated ${data.n_cues} lyric cues successfully! Saved to: ${data.out_lrc}`, "success");

    document.getElementById("studioMainContainer").style.display = "grid";
    loadArtifactOptions();
  } catch (err) {
    showStatus(`Error: ${err.message}`, "error");
  } finally {
    genBtn.disabled = false;
    genBtn.innerHTML = "▶ Generate LRC";
  }
}

function renderLayerSelector(layers, primaryLang) {
  const select = document.getElementById("studioLayerSelect");
  if (!select) return;
  allLayers = Array.isArray(layers) ? layers : [];
  if (allLayers.length <= 1) {
    select.innerHTML = "";
    select.style.display = "none";
    selectedLayerId = "all";
    return;
  }

  select.innerHTML = "";
  const allOpt = document.createElement("option");
  allOpt.value = "all";
  allOpt.textContent = `All Layers (${currentCues.length} cues)`;
  select.appendChild(allOpt);

  allLayers.forEach((layer) => {
    const opt = document.createElement("option");
    opt.value = String(layer.layer_id);
    const isPrimary = primaryLang && layer.lang === primaryLang ? " ★" : "";
    opt.textContent = `Layer ${layer.layer_id} [${layer.lang}] (${layer.n_cues} cues)${isPrimary}`;
    select.appendChild(opt);
  });

  select.style.display = "inline-block";
  selectedLayerId = "all";
}

function updateStatsSummary(data) {
  const container = document.getElementById("studioStatsBar");
  if (!container) return;
  const hasRef = (data.n_reference_matches || 0) > 0 || (data.n_reference_gap_fills || 0) > 0;
  const isCaption = Boolean(data.reference_source && data.reference_source.includes("caption"));
  const layerBadges = (data.layers && data.layers.length > 1)
    ? data.layers.map(l => `<span class="stat-badge">🌐 Layer ${l.layer_id} [${l.lang}] (${l.n_cues})</span>`).join(" ")
    : "";
  container.innerHTML = `
    <span class="stat-badge"><strong>${data.n_cues || 0}</strong> Accepted Cues</span>
    <span class="stat-badge"><strong>${data.n_skipped || 0}</strong> Filtered Distractors</span>
    <span class="stat-badge ${data.review_required > 0 ? "warning" : ""}"><strong>${data.review_required || 0}</strong> Review Flags</span>
    ${layerBadges}
    ${data.cached ? `<span class="stat-badge warning">♻️ Loaded from cached LRC</span>` : ""}
    ${hasRef ? `<span class="stat-badge"><strong>${data.n_reference_matches || 0}</strong> Ref Matches, <strong>${data.n_reference_gap_fills || 0}</strong> Gap Fills (${data.reference_source || "ref"})</span>` : ""}
    ${isCaption ? `<span class="stat-badge warning">⚠️ ASR caption fallback used</span>` : ""}
    <span class="stat-badge file-path" title="${data.out_lrc}">📄 ${basename(data.out_lrc)}</span>
  `;
}

function renderCuesList() {
  const container = document.getElementById("studioCuesList");
  if (!container) return;
  container.innerHTML = "";

  const filteredCues = (selectedLayerId === "all" || !selectedLayerId)
    ? currentCues
    : currentCues.filter(c => String(c.layer_id) === String(selectedLayerId));

  if (filteredCues.length === 0) {
    container.innerHTML = '<div class="empty-state">No lyric cues found for this layer.</div>';
    return;
  }

  filteredCues.forEach((cue) => {
    const actualIdx = currentCues.indexOf(cue);
    const row = document.createElement("div");
    row.className = "cue-row";
    row.dataset.idx = actualIdx;

    const timeTag = formatTimestamp(cue.start_ms);
    const endTag = formatTimestamp(cue.end_ms);
    const layerBadge = (cue.layer_id !== undefined && cue.layer_id !== null && allLayers.length > 1)
      ? `<span class="cue-tag" style="background:#334155;color:#94a3b8;font-size:10px;margin-right:4px;">L${cue.layer_id}:${cue.lang || 'und'}</span>`
      : '';

    row.innerHTML = `
      <div class="cue-time" title="Click to seek player" data-time="${cue.start_ms}">
        ${layerBadge}<span class="cue-tag">${timeTag}</span>
      </div>
      <div class="cue-input-box">
        <input type="text" class="cue-text-input" value="${escapeHtml(cue.text)}" data-idx="${actualIdx}">
      </div>
      <div class="cue-meta">
        <span class="chip-conf ${cue.confidence >= 0.9 ? 'high' : 'med'}">${Math.round((cue.confidence || 0) * 100)}%</span>
        ${cue.review_required ? '<span class="chip-rev">REV</span>' : ''}
        <button class="mini-btn danger cue-del-btn" title="Remove line" data-idx="${actualIdx}">✕</button>
      </div>
    `;

    row.querySelector(".cue-time").addEventListener("click", () => seekTo(cue.start_ms));
    row.querySelector(".cue-text-input").addEventListener("input", (e) => {
      currentCues[actualIdx].text = e.target.value;
      renderKaraokeView();
    });
    row.querySelector(".cue-del-btn").addEventListener("click", () => {
      currentCues.splice(actualIdx, 1);
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

  const targetCues = (selectedLayerId === "all" || !selectedLayerId)
    ? currentCues
    : currentCues.filter(c => String(c.layer_id) === String(selectedLayerId));

  const lines = [
    `[ti:${currentMetadata.title || ""}]`,
    `[ar:${currentMetadata.artist || ""}]`,
    `[by:${currentMetadata.by || "Lyra OCR"}]`,
    `[offset:${currentMetadata.offset_ms || 0}]`,
  ];

  targetCues.forEach(c => {
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

  let filename = "lyrics.lrc";
  if (selectedLayerId !== "all") {
    const matched = allLayers.find(l => String(l.layer_id) === String(selectedLayerId));
    if (matched && matched.path) {
      filename = basename(matched.path);
    } else if (currentOutLrc) {
      filename = basename(currentOutLrc);
    }
  } else if (currentOutLrc) {
    filename = basename(currentOutLrc);
  }

  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}


function togglePlayback() {
  const btn = document.getElementById("studioPlayBtn");
  if (studioVideo && studioVideo.src) {
    if (studioVideo.paused) {
      studioVideo.play().catch(() => {});
    } else {
      studioVideo.pause();
    }
    return;
  }
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
  if (studioVideo && studioVideo.src) {
    try {
      studioVideo.currentTime = playbackTimeMs / 1000;
    } catch (_) { /* metadata not ready yet */ }
  }
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
