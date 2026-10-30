"""Local web UI and background job manager for MP4 downloads."""

from __future__ import annotations

import json
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .downloader import download_video

HOST = "127.0.0.1"
PORT = 8765

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _update_job(job_id: str, **values: object) -> None:
    with _jobs_lock:
        _jobs[job_id].update(values)


def _run_download(job_id: str, url: str, output_dir: str) -> None:
    try:
        result = download_video(
            url,
            output_dir,
            progress=lambda event: _update_job(job_id, **event),
        )
        _update_job(job_id, status="completed", **result)
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc))


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lyra OCR - Video Downloader</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { max-width: 760px; margin: 3rem auto; padding: 0 1rem; background: #111827; color: #e5e7eb; }
    h1 { margin-bottom: .35rem; } .muted { color: #9ca3af; }
    form, #job { background: #1f2937; padding: 1.25rem; border-radius: 12px; margin-top: 1.25rem; }
    label { display: block; margin-top: .9rem; margin-bottom: .35rem; }
    input { box-sizing: border-box; width: 100%; padding: .7rem; border: 1px solid #4b5563; border-radius: 7px; background: #111827; color: inherit; }
    button { margin-top: 1rem; padding: .7rem 1rem; border: 0; border-radius: 7px; background: #22c55e; color: #052e16; font-weight: 700; cursor: pointer; }
    progress { width: 100%; height: 1.2rem; margin-top: 1rem; } code { overflow-wrap: anywhere; }
    .error { color: #fca5a5; } .ok { color: #86efac; }
  </style>
</head>
<body>
  <h1>Lyra OCR</h1>
  <div class="muted">Prototype: download a YouTube video as MP4 for the future OCR pipeline.</div>
  <form id="download-form">
    <label for="url">YouTube URL</label>
    <input id="url" name="url" type="url" placeholder="https://www.youtube.com/watch?v=..." required>
    <label for="output_dir">Output directory</label>
    <input id="output_dir" name="output_dir" value="./downloads" required>
    <button type="submit">Download MP4</button>
  </form>
  <section id="job" hidden>
    <strong id="status">Starting...</strong>
    <progress id="progress" max="100" value="0"></progress>
    <p id="message" class="muted"></p>
    <p id="result"></p>
  </section>
  <script>
    const form = document.querySelector("#download-form");
    const jobBox = document.querySelector("#job");
    const status = document.querySelector("#status");
    const progress = document.querySelector("#progress");
    const message = document.querySelector("#message");
    const result = document.querySelector("#result");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      result.textContent = ""; jobBox.hidden = false; status.textContent = "Starting...";
      const body = new URLSearchParams(new FormData(form));
      const response = await fetch("/api/download", {method: "POST", body});
      const data = await response.json();
      if (!response.ok) { status.className = "error"; status.textContent = data.error; return; }
      poll(data.job_id);
    });
    async function poll(jobId) {
      const response = await fetch("/api/jobs/" + jobId);
      const job = await response.json();
      status.className = job.status === "failed" ? "error" : job.status === "completed" ? "ok" : "";
      status.textContent = job.status + (job.stage ? " - " + job.stage : "");
      progress.value = job.percent || 0;
      message.textContent = job.message || "";
      if (job.status === "completed") {
        result.innerHTML = "MP4: <code>" + escapeHtml(job.video_path) + "</code>";
      } else if (job.status === "failed") {
        result.textContent = job.error || "Download failed";
      } else {
        setTimeout(() => poll(jobId), 1000);
      }
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[char]));
    }
  </script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {format % args}")

    def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            with _jobs_lock:
                job = _jobs.get(job_id)
            if not job:
                self._json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
            else:
                self._json(job)
            return
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/download":
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        values = parse_qs(self.rfile.read(length).decode())
        url = values.get("url", [""])[0].strip()
        output_dir = values.get("output_dir", ["./downloads"])[0].strip()
        if not url:
            self._json({"error": "YouTube URL is required"}, HTTPStatus.BAD_REQUEST)
            return
        if not output_dir:
            self._json({"error": "Output directory is required"}, HTTPStatus.BAD_REQUEST)
            return

        job_id = uuid.uuid4().hex
        with _jobs_lock:
            _jobs[job_id] = {"job_id": job_id, "status": "queued", "percent": 0}
        thread = threading.Thread(
            target=_run_download,
            args=(job_id, url, output_dir),
            daemon=True,
        )
        thread.start()
        self._json({"job_id": job_id}, HTTPStatus.ACCEPTED)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Lyra OCR web UI: http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()

