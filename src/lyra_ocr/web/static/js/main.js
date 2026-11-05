// =========================================================================
// Entry point: wire the tab registry, bind chrome, then apply the deep link.
// =========================================================================

import { bindTabButtons, navigateToReview, parseHash, registerTab, setReviewDeeplinkHandler, switchTab } from "./router.js";
import { initDatasetsTab, stopJobsPolling } from "./datasets.js";
import { initRunsTab, stopRunPolling } from "./runs.js";
import { initEvalTab } from "./eval.js";
import { initReviewTab, openReviewTarget } from "./review.js";

registerTab("datasets", { activate: initDatasetsTab, deactivate: stopJobsPolling });
registerTab("runs", { activate: initRunsTab, deactivate: stopRunPolling });
registerTab("eval", { activate: initEvalTab });

setReviewDeeplinkHandler(openReviewTarget);
bindTabButtons();

async function applyInitialHashRoute() {
  const hl = parseHash();
  if (!hl) return;
  if (hl.tab && hl.tab !== "review") {
    switchTab(hl.tab);
    return;
  }
  if (hl.dir) await navigateToReview(hl.dir, hl.track || null);
}

initReviewTab().then(applyInitialHashRoute);
