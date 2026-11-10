// =========================================================================
// Entry point: wire the tab registry, bind chrome, then apply the deep link.
// =========================================================================

import { bindTabButtons, navigateToReview, parseHash, registerTab, setReviewDeeplinkHandler, switchTab } from "./router.js";
import { initDatasetsTab, stopJobsPolling } from "./datasets.js";
import { initRunsTab, stopRunPolling } from "./runs.js";
import { initEvalTab } from "./eval.js";
import { initReviewTab, openReviewTarget } from "./review.js";
import { initLrcStudio } from "./lrc_studio.js";

registerTab("lrc", { activate: initLrcStudio });
registerTab("review", { activate: initReviewTab });
registerTab("datasets", { activate: initDatasetsTab, deactivate: stopJobsPolling });
registerTab("runs", { activate: initRunsTab, deactivate: stopRunPolling });
registerTab("eval", { activate: initEvalTab });

setReviewDeeplinkHandler(openReviewTarget);
bindTabButtons();

async function applyInitialHashRoute() {
  const hl = parseHash();
  if (!hl) {
    switchTab("lrc");
    return;
  }
  if (hl.tab && hl.tab !== "review") {
    switchTab(hl.tab);
    return;
  }
  if (hl.dir) {
    await navigateToReview(hl.dir, hl.track || null);
    return;
  }
  if (hl.tab === "review") {
    switchTab("review");
    return;
  }
  switchTab("lrc");
}

initLrcStudio().then(applyInitialHashRoute);
