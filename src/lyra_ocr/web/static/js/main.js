// =========================================================================
// Deep-link entry: #tab=review&dir=...&track=...
// =========================================================================
function applyInitialHashRoute() {
  const hl = parseHash();
  if (!hl) return;
  if (hl.tab && hl.tab !== "review") {
    switchTab(hl.tab);
    return;
  }
  if (hl.dir) {
    switchTab("review");
    loadArtifact(hl.dir).then(() => {
      if (hl.track && dataset && dataset.tracks) selectTrack(hl.track, true);
    });
  }
}

// Start
init().then(applyInitialHashRoute);
