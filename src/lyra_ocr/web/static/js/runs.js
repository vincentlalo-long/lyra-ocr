// =========================================================================
// Runs tab
// =========================================================================
import { activeTab, navigateToReview } from "./router.js";
import { escapeHtml, jsonFetch, mustEl, showToast } from "./core.js";
import { evalPick } from "./eval.js";

// Run selection state lives here (not in the Datasets tab).
let currentRunDs = null;
let currentRunId = null;
let runPollTimer = null;
let currentRunDataset = null;

const runDsSelect = mustEl("runDsSelect");
const runSelect = mustEl("runSelect");
const runVideoTable = mustEl("runVideoTable");

// Called by the Datasets tab before switching over, then expanded by initRunsTab.
export function setSelectedRunDataset(name) {
  currentRunDs = name;
  currentRunId = null;
}

export async function initRunsTab() {
  try {
    const data = await jsonFetch("/api/datasets");
    const list = data.datasets || [];
    const prev = currentRunDs || runDsSelect.value;
    runDsSelect.innerHTML = '<option value="">— select dataset —</option>';
    list.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.name;
      opt.textContent = `${d.name} (${d.video_count} videos · ${d.runs_count || 0} runs)`;
      runDsSelect.appendChild(opt);
    });
    if (prev && list.some(d => d.name === prev)) currentRunDs = prev;
    else currentRunDs = null;
    if (currentRunDs) {
      runDsSelect.value = currentRunDs;
      await loadRunsForDataset(currentRunDs);
    } else {
      renderNoRun();
    }
  } catch (err) {
    document.getElementById("runDetailError").textContent = `Failed to load runs: ${err.message}`;
  }
}

runDsSelect.addEventListener("change", () => {
  currentRunDs = runDsSelect.value || null;
  currentRunId = null;
  if (currentRunDs) loadRunsForDataset(currentRunDs);
  else renderNoRun();
});

runSelect.addEventListener("change", () => {
  currentRunId = runSelect.value || null;
  loadRunDetail();
});

async function loadRunsForDataset(name) {
  currentRunDs = name;
  try {
    currentRunDataset = await jsonFetch(`/api/datasets/${encodeURIComponent(name)}`);
    const runs = currentRunDataset.runs || [];
    const keep = currentRunId && runs.some(r => r.run_id === currentRunId);
    if (!keep) currentRunId = runs.length ? runs[0].run_id : null;
    runSelect.innerHTML = "";
    if (!runs.length) {
      runSelect.innerHTML = '<option value="">— no runs yet —</option>';
      currentRunId = null;
    } else {
      runs.forEach(r => {
        const opt = document.createElement("option");
        opt.value = r.run_id;
        opt.textContent = `${r.tracker} @ ${r.fps}fps · ${r.video_count || 0} videos`;
        runSelect.appendChild(opt);
      });
      runSelect.value = currentRunId;
    }
    await loadRunDetail();
  } catch (err) {
    document.getElementById("runDetailError").textContent = err.message;
  }
}

async function loadRunDetail() {
  const errEl = document.getElementById("runDetailError");
  errEl.textContent = "";
  if (!currentRunDs || !currentRunId) { renderNoRun(); return; }
  let run;
  try {
    run = await jsonFetch(`/api/datasets/${encodeURIComponent(currentRunDs)}/runs/${encodeURIComponent(currentRunId)}`);
  } catch (err) {
    errEl.textContent = err.message;
    return;
  }
  document.getElementById("runTitle").textContent = `${run.tracker} @ ${run.fps}fps — ${run.run_id}`;
  const opts = Object.entries(run.options || {}).map(([k, v]) => `${k}=${v}`).join(", ");
  document.getElementById("runMeta").innerHTML =
    `Tracker <strong>${escapeHtml(run.tracker)}</strong> · FPS <strong>${run.fps}</strong>` +
    (opts ? `<br>${escapeHtml(opts)}` : "") +
    `<br>Output: <code>${escapeHtml(run.artifact_root || "")}</code>`;

  renderRunStats(run);
  renderRunVideoTable(run);

  const active = (run.job_stats?.queued || 0) + (run.job_stats?.running || 0);
  if (active > 0) scheduleRunPolling();
  else stopRunPolling();
}

function renderNoRun() {
  document.getElementById("runTitle").textContent = "No run selected";
  document.getElementById("runMeta").textContent = "";
  document.getElementById("runStats").innerHTML = "";
  document.getElementById("runDetailError").textContent = "";
  runVideoTable.innerHTML = '';
}

function renderRunStats(run) {
  const videos = currentRunDataset?.videos || [];
  const done = (run.videos || []).length;
  const stats = run.job_stats || {};
  const gtCount = videos.filter(v => (v.ground_truth_segments || 0) > 0).length;
  const chips = [
    ["Videos", videos.length],
    ["Completed", done],
    ["Ground Truth", gtCount],
    ["Queued", stats.queued || 0],
    ["Running", stats.running || 0],
    ["Failed", stats.failed || 0],
  ];
  document.getElementById("runStats").innerHTML = chips
    .map(([l, v]) => `<div class="stat-chip"><strong>${v}</strong>${l}</div>`)
    .join("");
}

function renderRunVideoTable(run) {
  const videos = currentRunDataset?.videos || [];
  const entries = {};
  (run.videos || []).forEach(v => { entries[v.video_id] = v; });
  const jobs = {};
  (run.jobs || []).forEach(j => {
    const prev = jobs[j.video_id];
    if (!prev || (j.updated_at || 0) >= (prev.updated_at || 0)) jobs[j.video_id] = j;
  });
  const vids = new Set([...videos.map(v => v.video_id), ...Object.keys(entries), ...Object.keys(jobs)]);

  let html = `<tr><th>Video</th><th>Run state</th><th>Artifact</th><th>GT</th><th>Actions</th></tr>`;
  if (!vids.size) {
    html += `<tr><td colspan="5" style="color:var(--text-muted);font-style:italic;padding:16px">No videos in this dataset yet.</td></tr>`;
  }
  [...vids].forEach(vid => {
    const v = videos.find(x => x.video_id === vid) || {};
    const e = entries[vid];
    const j = jobs[vid];
    const title = e?.title || v.title || vid;
    const gt = v.ground_truth_segments || 0;
    let state = "not run", stateCls = "queued", pct = 0, stage = "";
    if (j && ["queued", "running", "failed", "cancelled"].includes(j.state)) {
      state = j.state; stateCls = j.state; pct = Math.round(j.percent || 0); stage = j.stage || "";
    } else if (e) {
      state = "done"; stateCls = "done"; pct = 100;
    }
    const errTip = j?.error ? ` title="${escapeHtml(j.error)}"` : "";
    let actions = "";
    if (e?.artifact_dir) {
      const dir = `datasets/${run.dataset}/runs/${run.run_id}/${e.artifact_dir}`;
      actions += `<button class="mini-btn" data-review="${escapeHtml(dir)}">Review</button> `;
    }
    if (j && ["failed", "cancelled"].includes(j.state))
      actions += `<button class="mini-btn" data-run-retry="${j.job_id}"${errTip}>Retry</button> `;
    if (j && ["queued", "running"].includes(j.state))
      actions += `<button class="mini-btn" data-run-cancel="${j.job_id}">Cancel</button>`;

    html += `<tr>
      <td class="wrap"><strong>${escapeHtml(String(title).slice(0, 60))}</strong><br>
          <span style="color:var(--text-muted);font-size:11px">${escapeHtml(vid)}</span></td>
      <td><span class="state-chip state-${stateCls}"${errTip}>${escapeHtml(state)}</span>
          ${j && ["queued", "running"].includes(j.state)
            ? `<br><span class="pbar"><div style="width:${pct}%"></div></span> ${pct}% <span style="font-size:11px;color:var(--text-muted)">${escapeHtml(stage)}</span>`
            : (state === "failed" && j.failed_stage ? `<br><span style="font-size:11px;color:var(--channel)">at ${escapeHtml(j.failed_stage)}</span>` : "")}</td>
      <td>${e?.artifact_dir ? escapeHtml(e.artifact_dir) : (v.downloaded ? "—" : "not downloaded")}</td>
      <td>${gt}</td>
      <td>${actions}</td>
    </tr>`;
  });
  runVideoTable.innerHTML = html;

  runVideoTable.querySelectorAll("[data-review]").forEach(b =>
    b.addEventListener("click", () => navigateToReview(b.dataset.review, null)));
  runVideoTable.querySelectorAll("[data-run-retry]").forEach(b =>
    b.addEventListener("click", () => runJobAction(b.dataset.runRetry, "retry", { force: true })));
  runVideoTable.querySelectorAll("[data-run-cancel]").forEach(b =>
    b.addEventListener("click", () => runJobAction(b.dataset.runCancel, "cancel")));
}

async function runJobAction(jobId, action, payload) {
  try {
    await jsonFetch(`/api/jobs/${jobId}/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {})
    });
    showToast(action === "cancel" ? "Job cancelled" : "Job re-queued");
    loadRunDetail();
  } catch (err) {
    showToast(`${action} failed: ${err.message}`);
  }
}

document.getElementById("runCreateBtn").addEventListener("click", async () => {
  if (!currentRunDs) { showToast("Select a dataset first"); return; }
  const tracker = document.getElementById("runTracker").value;
  const fps = parseFloat(document.getElementById("runFps").value) || 1.0;
  const options = {
    device: document.getElementById("runDevice").value,
    rec_batch_size: parseInt(document.getElementById("runRecBatch").value, 10) || 8,
    dedup: document.getElementById("runDedup").checked,
  };
  try {
    const res = await jsonFetch(`/api/datasets/${encodeURIComponent(currentRunDs)}/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tracker, fps, options })
    });
    currentRunId = res.run.run_id;
    showToast(`Run created: ${currentRunId}`);
    await loadRunsForDataset(currentRunDs);
  } catch (err) {
    showToast(`Create run failed: ${err.message}`);
  }
});

async function enqueueRun(videoIds, force) {
  if (!currentRunDs || !currentRunId) { showToast("Select a run first"); return; }
  if (!videoIds.length) { showToast("No videos to queue"); return; }
  try {
    const res = await jsonFetch(`/api/datasets/${encodeURIComponent(currentRunDs)}/runs/${encodeURIComponent(currentRunId)}/items`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_ids: videoIds, options: { force: !!force } })
    });
    showToast(`Queued ${(res.enqueued || []).length} video(s)`);
    await loadRunDetail();
  } catch (err) {
    showToast(`Queue failed: ${err.message}`);
  }
}

document.getElementById("runQueueAllBtn").addEventListener("click", () => {
  const vids = (currentRunDataset?.videos || []).map(v => v.video_id);
  enqueueRun(vids, document.getElementById("runForce").checked);
});

document.getElementById("runQueueMissingBtn").addEventListener("click", async () => {
  if (!currentRunDs || !currentRunId) { showToast("Select a run first"); return; }
  try {
    const run = await jsonFetch(`/api/datasets/${encodeURIComponent(currentRunDs)}/runs/${encodeURIComponent(currentRunId)}`);
    const done = new Set((run.videos || []).map(v => v.video_id));
    const vids = (currentRunDataset?.videos || []).map(v => v.video_id).filter(v => !done.has(v));
    enqueueRun(vids, document.getElementById("runForce").checked);
  } catch (err) { showToast(`Failed to list missing videos: ${err.message}`); }
});

document.getElementById("runPromoteBtn").addEventListener("click", async () => {
  if (!currentRunDs || !currentRunId) { showToast("Select a run first"); return; }
  if (!confirm("Promote labels from this run into the dataset's ground truth? Existing ground truth for these videos will be overwritten.")) return;
  try {
    const res = await jsonFetch(`/api/datasets/${encodeURIComponent(currentRunDs)}/ground_truth/promote`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: currentRunId })
    });
    showToast(`Promoted ${res.total ?? Object.keys(res.promoted || {}).length} segment(s) from ${Object.keys(res.promoted || {}).length} video(s)`);
    await loadRunsForDataset(currentRunDs);
  } catch (err) {
    showToast(`Promote failed: ${err.message}`);
  }
});

document.getElementById("runEvalBtn").addEventListener("click", () => {
  if (!currentRunDs || !currentRunId) { showToast("Select a run first"); return; }
  evalPick(currentRunDs, currentRunId);
});

async function cancelAllRunJobs() {
  if (!confirm("Are you sure you want to cancel all queued and running jobs?")) return;
  try {
    const res = await jsonFetch("/api/jobs/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: currentRunDs || undefined })
    });
    showToast(`Cancelled ${res.cancelled || 0} job(s)`);
    if (currentRunDs) await loadRunDetail();
  } catch (err) {
    showToast(`Cancel failed: ${err.message}`);
  }
}

async function deleteCurrentRun() {
  if (!currentRunDs || !currentRunId) { showToast("Select a run first"); return; }
  if (!confirm(`Are you sure you want to delete run '${currentRunId}' and all its files? This action cannot be undone.`)) return;
  try {
    const res = await jsonFetch(`/api/datasets/${encodeURIComponent(currentRunDs)}/runs/${encodeURIComponent(currentRunId)}/delete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    showToast(`Deleted run: ${res.deleted || currentRunId}`);
    currentRunId = null;
    await loadRunsForDataset(currentRunDs);
  } catch (err) {
    showToast(`Delete failed: ${err.message}`);
  }
}

document.getElementById("runDeleteBtn")?.addEventListener("click", deleteCurrentRun);
document.getElementById("runTableDeleteBtn")?.addEventListener("click", deleteCurrentRun);
document.getElementById("runCancelAllBtn")?.addEventListener("click", cancelAllRunJobs);
document.getElementById("runTableCancelAllBtn")?.addEventListener("click", cancelAllRunJobs);

document.getElementById("runRefreshBtn").addEventListener("click", () => {
  if (currentRunDs) loadRunsForDataset(currentRunDs);
});

function scheduleRunPolling() {
  stopRunPolling();
  runPollTimer = setInterval(() => {
    if (activeTab !== "runs") { stopRunPolling(); return; }
    loadRunDetail();
  }, 2000);
}
export function stopRunPolling() {
  if (runPollTimer) { clearInterval(runPollTimer); runPollTimer = null; }
}

