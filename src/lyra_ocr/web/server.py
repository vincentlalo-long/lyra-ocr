"""HTTP routing and server bootstrap for the local review/dataset web UI."""

from __future__ import annotations

import threading
import urllib.parse
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from lyra_ocr.dataset import registry
from lyra_ocr.dataset.youtube import list_channel_or_playlist
from lyra_ocr.web.base import ReviewHandlerBase
from lyra_ocr.web.dataset_api import DatasetAPIMixin
from lyra_ocr.web.eval_api import EvalAPIMixin
from lyra_ocr.web.jobs import JobQueue
from lyra_ocr.web.review_api import ReviewAPIMixin


class LyraReviewHandler(ReviewAPIMixin, DatasetAPIMixin, EvalAPIMixin, ReviewHandlerBase):
    """Composed request handler: routing here, API handlers grouped by domain."""

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. API: List available artifacts
        if path == "/api/artifacts":
            self._handle_get_artifacts()
            return

        # 2. API: Load full artifact dataset
        if path == "/api/data":
            art_dir = query.get("dir", [None])[0]
            if not art_dir:
                self._send_error("Missing query parameter: dir", 400)
                return
            self._handle_get_data(art_dir)
            return

        # 3. API: Serve extracted frame image
        if path == "/api/frame":
            art_dir = query.get("dir", [None])[0]
            filename = query.get("file", [None])[0]
            if not art_dir or not filename:
                self._send_error("Missing query parameters: dir and file", 400)
                return
            self._handle_get_frame(art_dir, filename)
            return

        # 4. API: Labeling coverage & distribution status
        if path == "/api/status":
            art_dir = query.get("dir", [None])[0]
            if not art_dir:
                self._send_error("Missing query parameter: dir", 400)
                return
            self._handle_get_status(art_dir)
            return

        # 5. API: Live evaluation against ground truth
        if path == "/api/eval":
            self._handle_get_eval(
                query.get("golden", [None])[0],
                query.get("dir", [None])[0],
                query.get("dataset", [None])[0],
                query.get("run", [None])[0],
            )
            return

        # 5b. API: Evaluation error rows (for dashboard click-through)
        if path == "/api/eval/errors":
            self._handle_get_eval_errors(
                query.get("golden", [None])[0],
                query.get("dir", [None])[0],
                query.get("dataset", [None])[0],
                query.get("run", [None])[0],
            )
            return

        # 6. API: Dataset registry
        if path == "/api/datasets":
            self._send_json({"datasets": registry.list_datasets(self._workspace_root())})
            return

        if path.startswith("/api/datasets/"):
            self._handle_dataset_get(path[len("/api/datasets/"):].strip("/"))
            return

        # 7. API: Job queue
        if path == "/api/jobs":
            dataset = query.get("dataset", [None])[0]
            queue = self._job_queue()
            if queue is None:
                self._send_error("Job queue unavailable", HTTPStatus.SERVICE_UNAVAILABLE)
                return
            jobs = [j.to_dict() for j in queue.list(dataset)]
            self._send_json({"jobs": jobs, "stats": queue.stats(dataset)})
            return

        if path.startswith("/api/jobs/"):
            job_id = path[len("/api/jobs/"):].strip("/")
            queue = self._job_queue()
            rec = queue.get(job_id) if queue else None
            if rec is None:
                self._send_error("Job not found", 404)
                return
            self._send_json(rec.to_dict())
            return

        # 8. API: YouTube channel/playlist listing
        if path == "/api/youtube/list":
            url = query.get("url", [None])[0]
            if not url:
                self._send_error("Missing query parameter: url", 400)
                return
            try:
                items = list_channel_or_playlist(url, limit=int(query.get("limit", ["1000"])[0]))
                self._send_json({"items": items})
            except Exception as exc:  # noqa: BLE001
                self._send_error(f"YouTube listing failed: {exc}", 502)
            return

        # 9. Static files (index.html, JS, CSS)
        self._handle_static_file(path)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/label":
            self._handle_post_label()
            return

        if path == "/api/datasets":
            self._handle_create_dataset()
            return

        if path.startswith("/api/datasets/"):
            self._handle_dataset_post(path[len("/api/datasets/"):].strip("/"))
            return

        if path.startswith("/api/jobs/"):
            self._handle_job_post(path[len("/api/jobs/"):].strip("/"))
            return

        self._send_error(f"Unknown endpoint: {path}", 404)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    workspace_root: str | Path | None = None,
    job_queue: Optional[JobQueue] = None,
) -> ThreadingHTTPServer:
    """Create a configured ThreadingHTTPServer instance with dataset job queue."""
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, LyraReviewHandler)
    httpd.workspace_root = Path(workspace_root or Path.cwd()).resolve()  # type: ignore[attr-defined]
    queue = job_queue if job_queue is not None else JobQueue(httpd.workspace_root)
    queue.start()
    httpd.job_queue = queue  # type: ignore[attr-defined]
    return httpd


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = False,
) -> None:
    """Run the review web server synchronously."""
    httpd = create_server(host, port)
    url = f"http://{host}:{port}"
    print(f"\n=======================================================")
    print(f"🚀 Lyra OCR Interactive Review UI running at:")
    print(f"   {url}")
    print(f"   Workspace root: {httpd.workspace_root}")
    print(f"   Press Ctrl+C to terminate.")
    print(f"=======================================================\n")

    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[WebUI] Server shutting down...")
    finally:
        httpd.job_queue.shutdown()  # type: ignore[attr-defined]
        httpd.server_close()
