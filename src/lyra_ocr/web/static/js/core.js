// =========================================================================
// Leaf utilities shared by every tab. This module must not import features.
// =========================================================================

export function mustEl(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing #${id}`);
  return el;
}

export function showToast(msg) {
  const toast = mustEl("toast");
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2000);
}

export function escapeHtml(str) {
  return (str || "").replace(/[&<>'"]/g, tag => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", "\"": "&quot;"
  }[tag] || tag));
}

export async function jsonFetch(url, opts) {
  const res = await fetch(url, opts);
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) {
    const msg = (body && (body.error || body.message)) || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return body;
}

export function getHiddenArtifacts() {
  try {
    return new Set(JSON.parse(localStorage.getItem("lyra_hidden_artifacts") || "[]"));
  } catch (e) {
    return new Set();
  }
}

export function hideArtifact(path) {
  const hidden = getHiddenArtifacts();
  hidden.add(path);
  localStorage.setItem("lyra_hidden_artifacts", JSON.stringify([...hidden]));
}

export function clearHiddenArtifacts() {
  localStorage.removeItem("lyra_hidden_artifacts");
}
