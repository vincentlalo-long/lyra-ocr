// =========================================================================
// Eval tab
// =========================================================================
let currentEvalDs = null;
let currentEvalRun = "all";

async function initEvalTab() {
  const sel = document.getElementById("evalDatasetSelect");
  try {
    const data = await jsonFetch("/api/datasets");
    const list = data.datasets || [];
    const prev = currentEvalDs || sel.value;
    sel.innerHTML = '<option value="">Select a dataset…</option>';
    list.forEach(d => {
      const opt = document.createElement("option");
      opt.value = d.name;
      opt.textContent = `${d.name} (${d.video_count} videos · ${d.runs_count || 0} runs)`;
      sel.appendChild(opt);
    });
    if (prev && list.some(d => d.name === prev)) currentEvalDs = prev;
    else currentEvalDs = list.length ? list[0].name : null;
    if (currentEvalDs) {
      sel.value = currentEvalDs;
      await populateEvalRuns(currentEvalDs);
    } else {
      document.getElementById("evalRunSelect").innerHTML = '<option value="all">All runs</option>';
    }
  } catch (err) { /* keep static options */ }
}

async function populateEvalRuns(name) {
  const sel = document.getElementById("evalRunSelect");
  sel.innerHTML = '<option value="all">All runs</option>';
  if (!name) return;
  try {
    const data = await jsonFetch(`/api/datasets/${encodeURIComponent(name)}`);
    (data.runs || []).forEach(r => {
      const opt = document.createElement("option");
      opt.value = r.run_id;
      opt.textContent = `${r.tracker} @ ${r.fps}fps (${r.video_count || 0} videos)`;
      sel.appendChild(opt);
    });
    const keep = currentEvalRun !== "all" && [...sel.options].some(o => o.value === currentEvalRun);
    currentEvalRun = keep ? currentEvalRun : "all";
    sel.value = currentEvalRun;
  } catch (err) { /* leave runs empty */ }
}

document.getElementById("evalDatasetSelect").addEventListener("change", async e => {
  currentEvalDs = e.target.value || null;
  currentEvalRun = "all";
  await populateEvalRuns(currentEvalDs);
});
document.getElementById("evalRunSelect").addEventListener("change", e => {
  currentEvalRun = e.target.value || "all";
});

async function evalPick(dsName, runId) {
  currentEvalDs = dsName;
  currentEvalRun = runId || "all";
  switchTab("eval");
  await populateEvalRuns(dsName);
  const dsSel = document.getElementById("evalDatasetSelect");
  if ([...dsSel.options].some(o => o.value === dsName)) dsSel.value = dsName;
  currentEvalRun = runId || "all";
  document.getElementById("evalRunSelect").value = currentEvalRun;
  runEvaluation();
}

document.getElementById("evalRunBtn").addEventListener("click", runEvaluation);

async function runEvaluation() {
  const ds = document.getElementById("evalDatasetSelect").value;
  const run = document.getElementById("evalRunSelect").value || "all";
  const status = document.getElementById("evalStatus");
  const content = document.getElementById("evalContent");
  if (!ds) { showToast("Pick a dataset"); return; }
  status.innerHTML = `<span class="spinner-inline"></span> Evaluating…`;
  content.innerHTML = "";
  try {
    const data = await jsonFetch(`/api/eval?dataset=${encodeURIComponent(ds)}&run=${encodeURIComponent(run)}`);
    if (data.mode === "all") {
      status.textContent = `${ds} · ${(data.reports || []).length} run(s)`;
      renderEvalAll(data.reports || [], ds);
      return;
    }
    const errData = await jsonFetch(
      `/api/eval/errors?dataset=${encodeURIComponent(ds)}&run=${encodeURIComponent(run)}`
    ).catch(() => ({ errors: [] }));
    status.textContent = `${ds} · run ${run}`;
    renderEvalReport(data.report, errData.errors || [], ds, run);
  } catch (err) {
    status.textContent = "";
    content.innerHTML = `<div class="gate-banner gate-fail">Evaluation failed<ul><li>${escapeHtml(err.message)}</li></ul></div>`;
  }
}

function renderEvalAll(reports, dsName) {
  const content = document.getElementById("evalContent");
  let html = `<div style="font-size:12px;color:var(--text-muted)">Each run is evaluated against the same dataset ground truth. Click a row to inspect its errors.</div>`;
  if (!reports.length) {
    html += `<div class="gate-banner gate-fail">No runs to evaluate. Create a run in the Runs tab first.</div>`;
    content.innerHTML = html;
    return;
  }
  html += `<div style="overflow-x:auto"><table class="data-table"><tr>
    <th>Run</th><th>Gate</th><th>Coverage</th><th>Lyric Precision</th><th>Lyric Recall</th>
    <th>Lyric F1</th><th>Leakage</th><th>False Drops</th><th>Tracks</th></tr>`;
  reports.forEach(item => {
    const r = item.report || {};
    const b = r.binary_lyric || {};
    const gate = r.gate_passed
      ? `<span class="state-chip state-done">pass</span>`
      : `<span class="state-chip state-failed" title="${escapeHtml((r.gate_reasons || []).join('; '))}">fail</span>`;
    html += `<tr data-eval-run="${escapeHtml(item.run_id)}" style="cursor:pointer">
      <td><code>${escapeHtml(item.run_id)}</code></td>
      <td>${gate}</td>
      <td>${((r.coverage_rate || 0) * 100).toFixed(1)}%</td>
      <td>${(b.precision ?? 0).toFixed(3)}</td>
      <td>${(b.recall ?? 0).toFixed(3)}</td>
      <td>${(b.f1 ?? 0).toFixed(3)}</td>
      <td>${(((r.leakage || {}).count_rate || 0) * 100).toFixed(1)}%</td>
      <td>${(((r.false_drop || {}).rate || 0) * 100).toFixed(1)}%</td>
      <td>${r.evaluated_tracks || 0}/${r.total_tracks || 0}</td>
    </tr>`;
  });
  html += `</table></div>`;
  content.innerHTML = html;
  content.querySelectorAll("[data-eval-run]").forEach(tr =>
    tr.addEventListener("click", () => {
      currentEvalRun = tr.dataset.evalRun;
      document.getElementById("evalRunSelect").value = currentEvalRun;
      runEvaluation();
    }));
}

function renderEvalReport(rep, errors, dsName, runId) {
  const content = document.getElementById("evalContent");
  const ci = rep.binary_lyric_ci || {};
  const ciStr = k => ci[k] ? ` [${ci[k][0].toFixed(2)}, ${ci[k][1].toFixed(2)}]` : "";

  let html = "";

  const gateCls = rep.gate_passed ? "gate-pass" : "gate-fail";
  const reasons = (rep.gate_reasons || []).map(r => `<li>• ${escapeHtml(r)}</li>`).join("");
  html += `<div class="gate-banner ${gateCls}">
    ${rep.gate_passed ? "✅ GATE PASSED — rule engine meets Phase 1 targets" : "❌ GATE FAILED"}
    ${reasons ? `<ul>${reasons}</ul>` : ""}
  </div>`;

  const cards = [
    ["Coverage", `${(rep.coverage_rate * 100).toFixed(1)}%`, `${rep.evaluated_tracks}/${rep.total_tracks} tracks labeled`],
    ["Lyric Precision", rep.binary_lyric.precision.toFixed(3), `95% CI${ciStr("precision")}`],
    ["Lyric Recall", rep.binary_lyric.recall.toFixed(3), `95% CI${ciStr("recall")}`],
    ["Lyric F1", rep.binary_lyric.f1.toFixed(3), `support: ${rep.binary_lyric.support_lyric} lyric / ${rep.binary_lyric.support_non_lyric} non-lyric`],
    ["Distractor Leakage", `${(rep.leakage.count_rate * 100).toFixed(1)}%`, `${rep.leakage.leaked_tracks_count} non-lyric tracks classified lyric`],
    ["False Drops", `${(rep.false_drop.rate * 100).toFixed(1)}%`, `${rep.false_drop.count}/${rep.false_drop.total_golden_lyrics} golden lyrics missed`],
    ["Tracking Miss", `${(rep.tracking_miss_rate * 100).toFixed(1)}%`, `${rep.tracking_miss_tracks} tracks · ${rep.tracking_error_tracks} flagged fragmented`],
  ];
  html += `<div class="metrics-grid">` + cards.map(([l, v, s]) =>
    `<div class="metric-card"><div class="m-label">${l}</div><div class="m-val">${v}</div><div class="m-sub">${escapeHtml(s)}</div></div>`).join("") + `</div>`;

  const REAL_ERRORS = ["leakage", "false_drop", "tracking_miss", "missing_track", "misclassification"];
  const shown = errors.filter(e => REAL_ERRORS.includes(e.error_type));
  const unlabeledCount = errors.filter(e => e.error_type === "unlabeled").length;
  let errHtml = `<div class="section-title" style="margin-top:8px">Errors (${shown.length})</div>`;
  if (unlabeledCount) errHtml += `<div style="font-size:12px;color:var(--text-muted);margin-bottom:6px">${unlabeledCount} unlabeled track(s) — label them in Review to raise coverage.</div>`;
  if (!shown.length) {
    errHtml += `<div style="color:var(--lyric);padding:10px">No misclassified tracks 🎉</div>`;
  } else {
    errHtml += `<div style="overflow-x:auto"><table class="data-table"><tr>
      <th>Video</th><th>Track</th><th>Text</th><th>Model</th><th>Human</th><th>Type</th><th>Conf</th><th></th></tr>`;
    shown.slice(0, 200).forEach(e => {
      const artDir = dsName && runId
        ? `datasets/${dsName}/runs/${runId}/videos/${e.video_id}`
        : "";
      const fix = artDir
        ? `<button class="link-btn" data-err-dir="${escapeHtml(artDir)}" data-err-track="${escapeHtml(String(e.track_id))}">Fix →</button>`
        : "";
      errHtml += `<tr>
        <td style="font-size:11px;color:var(--text-muted)">${escapeHtml(String(e.video_id).slice(-12))}</td>
        <td>#${escapeHtml(String(e.track_id))}</td>
        <td class="wrap">${escapeHtml((e.text || "").slice(0, 60))}</td>
        <td>${escapeHtml(e.model_label || "")}</td>
        <td>${escapeHtml(e.human_label || "")}</td>
        <td><span class="err-reason" style="color:var(--channel)">${escapeHtml(e.error_type)}</span></td>
        <td>${escapeHtml(String(e.model_confidence ?? ""))}</td>
        <td>${fix}</td></tr>`;
    });
    errHtml += `</table></div>`;
    if (shown.length > 200) errHtml += `<div style="font-size:12px;color:var(--text-muted)">… ${shown.length - 200} more</div>`;
  }
  html += errHtml;

  content.innerHTML = html;
  content.querySelectorAll("[data-err-dir]").forEach(b =>
    b.addEventListener("click", () => deepLinkToReview(b.dataset.errDir, b.dataset.errTrack)));
}

