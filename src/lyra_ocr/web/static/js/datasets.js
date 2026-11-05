// =========================================================================
// Datasets tab
// =========================================================================
import { activeTab, switchTab } from "./router.js";
import { escapeHtml, jsonFetch, mustEl, showToast } from "./core.js";
import { setSelectedRunDataset } from "./runs.js";

let currentDs = null;
let ytFetchedItems = [];
let dsPollTimer = null;

const dsSelect = mustEl("dsSelect");
const ytResults = mustEl("ytResults");
const ytTable = mustEl("ytTable");
const dsVideoTable = mustEl("dsVideoTable");

export async function initDatasetsTab() {
  await loadDatasetsList();
}

async function loadDatasetsList() {
  try {
    const data = await jsonFetch("/api/datasets");
    const list = data.datasets || [];
    dsSelect.innerHTML = '<option value="">— select dataset —</option>';
    list.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.name;
      opt.textContent = `${d.name} (${d.video_count} videos · ${d.ground_truth_count || 0} GT · ${d.runs_count || 0} runs)`;
      dsSelect.appendChild(opt);
    });
    if (currentDs) {
      dsSelect.value = currentDs;
      loadDatasetDetail(currentDs);
    }
  } catch (err) {
    document.getElementById("dsDetailError").textContent = `Failed to list datasets: ${err.message}`;
  }
}

dsSelect.addEventListener("change", () => {
  currentDs = dsSelect.value || null;
  if (currentDs) loadDatasetDetail(currentDs);
});

document.getElementById("dsCreateBtn").addEventListener("click", async () => {
  const name = document.getElementById("dsNewName").value.trim();
  const description = document.getElementById("dsNewDesc").value.trim();
  if (!name) { showToast("Dataset name is required"); return; }
  try {
    await jsonFetch("/api/datasets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, description })
    });
    showToast(`Dataset '${name}' created`);
    document.getElementById("dsNewName").value = "";
    document.getElementById("dsNewDesc").value = "";
    currentDs = name;
    await loadDatasetsList();
  } catch (err) {
    showToast(`Create failed: ${err.message}`);
  }
});

async function loadDatasetDetail(name) {
  let data;
  try {
    data = await jsonFetch(`/api/datasets/${encodeURIComponent(name)}`);
  } catch (err) {
    document.getElementById("dsTitle").textContent = name;
    document.getElementById("dsDetailError").textContent = err.message;
    return;
  }
  document.getElementById("dsDetailError").textContent = "";
  const videos = data.videos || [];
  const gtCount = videos.filter(v => (v.ground_truth_segments || 0) > 0).length;
  const runs = data.runs || [];
  document.getElementById("dsTitle").textContent =
    `${name} — ${videos.length} videos · ${gtCount} with GT · ${runs.length} runs`;

  renderDsStats(data);
  renderDsVideoTable(data);

  const active = (data.job_stats?.queued || 0) + (data.job_stats?.running || 0);
  if (active > 0) scheduleJobsPolling(name);
  else stopJobsPolling();
}

function renderDsStats(data) {
  const videos = data.videos || [];
  const stats = data.job_stats || {};
  const gtVideos = videos.filter(v => (v.ground_truth_segments || 0) > 0).length;
  const downloaded = videos.filter(v => v.downloaded).length;
  const chips = [
    ["Videos", videos.length],
    ["Downloaded", downloaded],
    ["Ground Truth", gtVideos],
    ["Runs", (data.runs || []).length],
    ["Queued", stats.queued || 0],
    ["Failed", stats.failed || 0],
  ];
  document.getElementById("dsStats").innerHTML = chips
    .map(([l, v]) => `<div class="stat-chip"><strong>${v}</strong>${l}</div>`)
    .join("");
}

function renderDsVideoTable(data) {
  const jobs = (data.jobs || []).filter(j => j.kind !== "run");
  const jobByVid = {};
  jobs.forEach(j => {
    const prev = jobByVid[j.video_id];
    if (!prev || (j.updated_at || 0) >= (prev.updated_at || 0)) jobByVid[j.video_id] = j;
  });
  const entryByVid = {};
  data.videos.forEach(v => { entryByVid[v.video_id] = v; });

  const rows = [];
  const seen = new Set();
  jobs.forEach(j => { seen.add(j.video_id); rows.push({ vid: j.video_id, job: j, entry: entryByVid[j.video_id] }); });
  data.videos.forEach(v => { if (!seen.has(v.video_id)) rows.push({ vid: v.video_id, job: null, entry: v }); });

  let html = `<tr><th>Video</th><th>Download</th><th>Ground Truth</th><th>Split</th><th>Actions</th></tr>`;
  if (!rows.length) {
    html += `<tr><td colspan="5" style="color:var(--text-muted);font-style:italic;padding:16px">No videos yet — fetch a channel or add local paths, then download.</td></tr>`;
  }
  rows.forEach(r => {
    const j = r.job, e = r.entry;
    const title = (j?.source?.title) || e?.title || r.vid;
    const pct = j ? Math.round(j.percent || 0) : 100;
    const errTip = j?.error ? ` title="${escapeHtml(j.error)}"` : "";
    const stage = j && j.state === "running" ? escapeHtml(j.stage || "") : "";
    const downloaded = !!e?.downloaded;
    const state = j && ["queued", "running", "failed", "cancelled"].includes(j.state)
      ? j.state
      : (downloaded ? "done" : "queued");
    const stateText = j && ["queued", "running", "failed", "cancelled"].includes(j.state)
      ? j.state
      : (downloaded ? "downloaded" : "not downloaded");
    const gt = e?.ground_truth_segments || 0;

    let actions = "";
    if (downloaded) actions += `<button class="mini-btn" data-open-run="${escapeHtml(r.vid)}">Runs →</button> `;
    if (j && ["failed", "cancelled"].includes(j.state))
      actions += `<button class="mini-btn" data-retry="${j.job_id}"${errTip}>Retry</button> `;
    if (j && ["queued", "running"].includes(j.state))
      actions += `<button class="mini-btn" data-cancel="${j.job_id}">Cancel</button>`;

    const splitSel = e
      ? `<select data-split="${escapeHtml(r.vid)}" class="mini-btn">
           <option value="train"${e.split === "train" ? " selected" : ""}>train</option>
           <option value="test"${e.split !== "train" ? " selected" : ""}>test</option>
         </select>`
      : `<span style="color:var(--text-muted)">—</span>`;

    html += `<tr>
      <td class="wrap"><strong>${escapeHtml(String(title).slice(0, 60))}</strong><br>
          <span style="color:var(--text-muted);font-size:11px">${escapeHtml(r.vid)}</span></td>
      <td><span class="state-chip state-${state}"${errTip}>${escapeHtml(stateText)}</span>
          ${j && ["queued", "running"].includes(j.state)
            ? `<br><span class="pbar"><div style="width:${pct}%"></div></span> ${pct}% <span style="font-size:11px;color:var(--text-muted)">${stage}</span>`
            : (state === "failed" && j.failed_stage ? `<br><span style="font-size:11px;color:var(--channel)">at ${escapeHtml(j.failed_stage)}</span>` : "")}</td>
      <td>${gt}</td>
      <td>${splitSel}</td>
      <td>${actions}</td>
    </tr>`;
  });
  dsVideoTable.innerHTML = html;

  dsVideoTable.querySelectorAll("[data-open-run]").forEach(b =>
    b.addEventListener("click", () => {
      setSelectedRunDataset(data.name);
      switchTab("runs");
    }));
  dsVideoTable.querySelectorAll("[data-retry]").forEach(b =>
    b.addEventListener("click", () => jobAction(b.dataset.retry, "retry", {})));
  dsVideoTable.querySelectorAll("[data-cancel]").forEach(b =>
    b.addEventListener("click", () => jobAction(b.dataset.cancel, "cancel")));
  dsVideoTable.querySelectorAll("[data-split]").forEach(sel =>
    sel.addEventListener("change", async () => {
      try {
        await jsonFetch(`/api/datasets/${encodeURIComponent(data.name)}/split`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ video_id: sel.dataset.split, split: sel.value })
        });
        showToast(`Split set to ${sel.value}`);
      } catch (err) { showToast(`Split failed: ${err.message}`); }
    }));
}

async function jobAction(jobId, action, payload) {
  try {
    await jsonFetch(`/api/jobs/${jobId}/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {})
    });
    showToast(action === "cancel" ? "Job cancelled" : "Job re-queued");
    if (currentDs) loadDatasetDetail(currentDs);
  } catch (err) {
    showToast(`${action} failed: ${err.message}`);
  }
}

function scheduleJobsPolling(name) {
  stopJobsPolling();
  dsPollTimer = setInterval(() => {
    if (activeTab !== "datasets" || currentDs !== name) { stopJobsPolling(); return; }
    loadDatasetDetail(name);
  }, 2000);
}
export function stopJobsPolling() {
  if (dsPollTimer) { clearInterval(dsPollTimer); dsPollTimer = null; }
}

// --- YouTube fetch & selection ---
document.getElementById("ytFetchBtn").addEventListener("click", async () => {
  const url = document.getElementById("ytUrl").value.trim();
  const limit = document.getElementById("ytLimit").value || "200";
  const status = document.getElementById("ytStatus");
  if (!url) { showToast("Paste a channel/playlist/watch URL first"); return; }
  const btn = document.getElementById("ytFetchBtn");
  btn.disabled = true;
  status.innerHTML = `<span class="spinner-inline"></span> Listing videos (can take 10–60s)…`;
  try {
    const data = await jsonFetch(`/api/youtube/list?url=${encodeURIComponent(url)}&limit=${encodeURIComponent(limit)}`);
    ytFetchedItems = data.items || [];
    renderYtTable();
    status.textContent = `${ytFetchedItems.length} videos found`;
  } catch (err) {
    status.textContent = "";
    ytResults.style.display = "none";
    showToast(`YouTube listing failed: ${err.message}`);
  } finally {
    btn.disabled = false;
  }
});

function renderYtTable() {
  if (!ytFetchedItems.length) { ytResults.style.display = "none"; return; }
  ytResults.style.display = "block";
  let html = `<tr>
    <th><input type="checkbox" id="ytSelectAll" checked></th>
    <th>Title</th><th>Duration</th><th>ID</th></tr>`;
  ytFetchedItems.forEach((v, i) => {
    const dur = v.duration ? `${Math.floor(v.duration / 60)}:${String(Math.floor(v.duration % 60)).padStart(2, "0")}` : "?";
    html += `<tr>
      <td><input type="checkbox" class="yt-check" data-idx="${i}" checked></td>
      <td class="wrap">${escapeHtml(v.title || "")}</td>
      <td>${dur}</td>
      <td style="color:var(--text-muted);font-size:11px">${escapeHtml(v.id)}</td></tr>`;
  });
  ytTable.innerHTML = html;
  document.getElementById("ytSelectAll").addEventListener("change", e => {
    ytTable.querySelectorAll(".yt-check").forEach(c => c.checked = e.target.checked);
  });
}

// --- Download / ingest ---
document.getElementById("downloadBtn").addEventListener("click", async () => {
  if (!currentDs) { showToast("Select or create a dataset first"); return; }
  const videos = [];
  ytTable.querySelectorAll(".yt-check:checked").forEach(c => {
    const v = ytFetchedItems[parseInt(c.dataset.idx)];
    if (v) videos.push({ kind: "youtube", url: v.url || `https://www.youtube.com/watch?v=${v.id}`,
                         yt_id: v.id, title: v.title || "", channel: v.channel || "", duration: v.duration });
  });
  document.getElementById("localPaths").value.split("\n").map(s => s.trim()).filter(Boolean)
    .forEach(p => videos.push({ kind: "local", path: p }));

  if (!videos.length) { showToast("Nothing selected — tick YouTube videos or add local paths"); return; }

  const options = { max_height: parseInt(document.getElementById("optMaxHeight").value) || null };
  try {
    const res = await jsonFetch(`/api/datasets/${encodeURIComponent(currentDs)}/download`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ videos, options })
    });
    const n = (res.enqueued || []).length;
    const rej = res.rejected || [];
    showToast(n ? `Queued ${n} download(s)${rej.length ? ` · ${rej.length} rejected` : ""}` : `No new jobs (${rej.length} rejected)`);
    if (rej.length) console.warn("Rejected sources:", rej);
    if (n) {
      document.getElementById("localPaths").value = "";
      ytTable.querySelectorAll(".yt-check").forEach(c => c.checked = false);
      loadDatasetDetail(currentDs);
    } else if (rej.length) {
      showToast(`Rejected: ${rej[0]}${rej.length > 1 ? ` (+${rej.length - 1} more)` : ""}`);
    }
  } catch (err) {
    showToast(`Download queue failed: ${err.message}`);
  }
});

document.getElementById("dsRefreshBtn").addEventListener("click", () => { loadDatasetsList(); if (currentDs) loadDatasetDetail(currentDs); });

document.getElementById("dsRunsBtn").addEventListener("click", () => {
  if (!currentDs) { showToast("No dataset selected"); return; }
  setSelectedRunDataset(currentDs);
  switchTab("runs");
});

