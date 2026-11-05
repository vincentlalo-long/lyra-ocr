"""Shared HTTP plumbing for the local review/dataset web server."""

from __future__ import annotations

import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _safe_resolve(path_str: str, base_dir: Path | None = None) -> Optional[Path]:
    """Safely resolve path without allowing directory traversal outside permitted roots."""
    p = Path(path_str)
    if not p.is_absolute():
        p = (base_dir or Path.cwd()) / p
    try:
        resolved = p.resolve()
    except Exception:
        return None
    base = (base_dir or Path.cwd()).resolve()
    if resolved == base or base in resolved.parents:
        return resolved
    return None


class ReviewHandlerBase(BaseHTTPRequestHandler):
    """JSON helpers, request parsing, and static file serving."""

    server_version = "LyraReviewServer/1.0"

    def _send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message, "status": status}, status=status)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _workspace_root(self) -> Path:
        return getattr(self.server, "workspace_root", Path.cwd())

    def _job_queue(self):
        return getattr(self.server, "job_queue", None)

    def _read_json_body(self) -> Optional[dict[str, Any]]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def _handle_static_file(self, req_path: str) -> None:
        """Serve static files from STATIC_DIR."""
        if req_path in ("/", "/index.html"):
            req_path = "/index.html"

        rel_path = req_path.lstrip("/")
        target_file = (STATIC_DIR / rel_path).resolve()

        # Prevent traversal outside STATIC_DIR
        if STATIC_DIR not in target_file.parents and target_file != STATIC_DIR:
            self._send_error("Forbidden", 403)
            return

        if not target_file.exists() or not target_file.is_file():
            self._send_error("File not found", 404)
            return

        mime_type, _ = mimetypes.guess_type(str(target_file))
        mime_type = mime_type or "application/octet-stream"

        with open(target_file, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", f"{mime_type}; charset=utf-8" if "text" in mime_type or "javascript" in mime_type else mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        """Custom concise log output."""
        sys.stderr.write(f"[WebUI] {self.address_string()} - {format % args}\n")
