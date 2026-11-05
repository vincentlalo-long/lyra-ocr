// =========================================================================
// Tabs & deep-link routing
//
// One-way dependency: features import from here, never the reverse. Features
// register their activate/deactivate hooks from main.js at boot.
// =========================================================================

export let activeTab = "review";

const tabs = new Map();
let reviewDeeplink = null;

export function registerTab(name, { activate, deactivate } = {}) {
  tabs.set(name, { activate, deactivate });
}

export function setReviewDeeplinkHandler(fn) {
  reviewDeeplink = fn;
}

export function switchTab(tab) {
  const previous = tabs.get(activeTab);
  if (tab !== activeTab && previous && previous.deactivate) previous.deactivate();
  activeTab = tab;

  document.querySelectorAll(".tab-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".view").forEach(v =>
    v.classList.toggle("active", v.id === `view-${tab}`));

  const header = document.getElementById("reviewHeaderControls");
  if (header) header.style.display = tab === "review" ? "" : "none";

  const next = tabs.get(tab);
  if (next && next.activate) next.activate();
}

export function bindTabButtons() {
  document.querySelectorAll(".tab-btn").forEach(b =>
    b.addEventListener("click", () => switchTab(b.dataset.tab)));
}

export function parseHash() {
  const h = location.hash.replace(/^#/, "");
  if (!h) return null;
  const p = new URLSearchParams(h);
  return { tab: p.get("tab"), dir: p.get("dir"), track: p.get("track"), dataset: p.get("dataset") };
}

export function navigateToReview(dir, trackId) {
  const track = trackId === null || trackId === undefined ? "" : trackId;
  location.hash = `tab=review&dir=${encodeURIComponent(dir)}&track=${encodeURIComponent(track)}`;
  switchTab("review");
  if (reviewDeeplink) return reviewDeeplink(dir, trackId);
  return Promise.resolve();
}
