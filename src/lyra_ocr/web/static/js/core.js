function showToast(msg) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2000);
}

function escapeHtml(str) {
  return (str || "").replace(/[&<>'"]/g, tag => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", "\"": "&quot;"
  }[tag] || tag));
}

// Playback loop

// =========================================================================
// Tabs & deep-link routing
// =========================================================================
let activeTab = "review";
function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("active", v.id === `view-${tab}`));
  document.getElementById("reviewHeaderControls").style.display = tab === "review" ? "" : "none";
  if (tab !== "review") { stopJobsPolling(); stopRunPolling(); }
  if (tab === "datasets") initDatasetsTab();
  if (tab === "runs") initRunsTab();
  if (tab === "eval") initEvalTab();
}
document.querySelectorAll(".tab-btn").forEach(b => b.addEventListener("click", () => switchTab(b.dataset.tab)));

function parseHash() {
  const h = location.hash.replace(/^#/, "");
  if (!h) return null;
  const p = new URLSearchParams(h);
  return { tab: p.get("tab"), dir: p.get("dir"), track: p.get("track"), dataset: p.get("dataset") };
}

function deepLinkToReview(dir, trackId) {
  location.hash = `tab=review&dir=${encodeURIComponent(dir)}&track=${encodeURIComponent(trackId)}`;
  switchTab("review");
  loadArtifact(dir).then(() => {
    if (trackId !== null && trackId !== undefined && dataset && dataset.tracks) {
      selectTrack(trackId, true);
    }
  });
}

async function jsonFetch(url, opts) {
  const res = await fetch(url, opts);
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) {
    const msg = (body && (body.error || body.message)) || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return body;
}

