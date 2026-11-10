"""HTTP routing and server bootstrap for the local review/dataset web UI."""

from __future__ import annotations

import threading
import urllib.parse
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

from lyra_ocr.dataset import registry
from lyra_ocr.dataset.youtube import list_channel_or_playlist
from lyra_ocr.web.base import ReviewHandlerBase
from lyra_ocr.web.dataset_api import DatasetAPIMixin
from lyra_ocr.web.eval_api import EvalAPIMixin
from lyra_ocr.web.jobs import JobQueue
from lyra_ocr.web.lrc_api import LrcAPIMixin
from lyra_ocr.web.review_api import ReviewAPIMixin

#: POST bodies larger than this are rejected before being read into memory.
MAX_BODY_BYTES = 8 * 1024 * 1024

Query = dict[str, list[str]]


class _BadRequest(Exception):
    """Raised by a route helper to emit a JSON error response."""

    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = int(status)


def _query_param(query: Query, name: str) -> str:
    value = query.get(name, [None])[0]
    if not value:
        raise _BadRequest(f"Missing query parameter: {name}")
    return value


def _optional(query: Query, name: str) -> Optional[str]:
    return query.get(name, [None])[0]


def _get_datasets(handler: Any, query: Query) -> None:
    handler._send_json({"datasets": registry.list_datasets(handler._workspace_root())})


def _list_jobs(handler: Any, query: Query) -> None:
    dataset = _optional(query, "dataset")
    queue = handler._job_queue()
    if queue is None:
        raise _BadRequest("Job queue unavailable", HTTPStatus.SERVICE_UNAVAILABLE)
    jobs = [j.to_dict() for j in queue.list(dataset)]
    handler._send_json({"jobs": jobs, "stats": queue.stats(dataset)})


def _get_job(handler: Any, query: Query, job_id: str) -> None:
    queue = handler._job_queue()
    rec = queue.get(job_id) if queue else None
    if rec is None:
        raise _BadRequest("Job not found", HTTPStatus.NOT_FOUND)
    handler._send_json(rec.to_dict())


def _youtube_list(handler: Any, query: Query) -> None:
    url = _query_param(query, "url")
    try:
        items = list_channel_or_playlist(url, limit=int(_optional(query, "limit") or 1000))
    except Exception as exc:  # noqa: BLE001
        raise _BadRequest(f"YouTube listing failed: {exc}", HTTPStatus.BAD_GATEWAY) from exc
    handler._send_json({"items": items})


GET_ROUTES: dict[str, Callable[[Any, Query], None]] = {
    "/api/artifacts": lambda handler, query: handler._handle_get_artifacts(query),
    "/api/data": lambda handler, query: handler._handle_get_data(_query_param(query, "dir")),
    "/api/frame": lambda handler, query: handler._handle_get_frame(
        _query_param(query, "dir"), _query_param(query, "file")
    ),
    "/api/status": lambda handler, query: handler._handle_get_status(_query_param(query, "dir")),
    "/api/eval": lambda handler, query: handler._handle_get_eval(
        _optional(query, "golden"),
        _optional(query, "dir"),
        _optional(query, "dataset"),
        _optional(query, "run"),
    ),
    "/api/eval/errors": lambda handler, query: handler._handle_get_eval_errors(
        _optional(query, "golden"),
        _optional(query, "dir"),
        _optional(query, "dataset"),
        _optional(query, "run"),
    ),
    "/api/datasets": _get_datasets,
    "/api/jobs": _list_jobs,
    "/api/youtube/list": _youtube_list,
    "/api/lrc": lambda handler, query: handler._handle_get_lrc(_query_param(query, "path")),
}

POST_ROUTES: dict[str, Callable[[Any], None]] = {
    "/api/label": lambda handler: handler._handle_post_label(),
    "/api/datasets": lambda handler: handler._handle_create_dataset(),
    "/api/lrc/generate": lambda handler: handler._handle_post_lrc_generate(),
    "/api/lrc/save": lambda handler: handler._handle_post_lrc_save(),
}


class LyraReviewHandler(ReviewAPIMixin, DatasetAPIMixin, EvalAPIMixin, LrcAPIMixin, ReviewHandlerBase):
    """Composed request handler: routing here, API handlers grouped by domain."""

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        try:
            route = GET_ROUTES.get(path)
            if route is not None:
                route(self, query)
                return

            # Parameterised API routes
            if path.startswith("/api/datasets/"):
                self._handle_dataset_get(path[len("/api/datasets/"):].strip("/"))
                return

            if path.startswith("/api/jobs/"):
                _get_job(self, query, path[len("/api/jobs/"):].strip("/"))
                return

            # Static files (index.html, JS, CSS)
            self._handle_static_file(path)
        except _BadRequest as exc:
            self._send_error(str(exc), exc.status)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > MAX_BODY_BYTES:
            self._send_error(
                f"Request body too large (max {MAX_BODY_BYTES} bytes)",
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return

        try:
            route = POST_ROUTES.get(path)
            if route is not None:
                route(self)
                return

            if path.startswith("/api/datasets/"):
                self._handle_dataset_post(path[len("/api/datasets/"):].strip("/"))
                return

            if path.startswith("/api/jobs/"):
                self._handle_job_post(path[len("/api/jobs/"):].strip("/"))
                return

            self._send_error(f"Unknown endpoint: {path}", 404)
        except _BadRequest as exc:
            self._send_error(str(exc), exc.status)

    def do_DELETE(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path.startswith("/api/datasets/"):
                rest = path[len("/api/datasets/"):].strip("/")
                parts = [p for p in rest.split("/") if p]
                if len(parts) == 3 and parts[1] == "runs":
                    self._handle_delete_run(parts[0], parts[2])
                    return
                if len(parts) == 2 and parts[1] == "runs":
                    self._handle_delete_run(parts[0], "all")
                    return
            self._send_error(f"Unknown endpoint: {path}", 404)
        except _BadRequest as exc:
            self._send_error(str(exc), exc.status)


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
    workspace_root: str | Path | None = None,
) -> None:
    """Run the review web server synchronously."""
    httpd = create_server(host, port, workspace_root=workspace_root)
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
