"""Lightweight HTTP server for interactive track review, frame inspection, and labeling."""

from __future__ import annotations

import json
import mimetypes
import re
import sys
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from lyra_ocr.dataset import registry
from lyra_ocr.dataset.models import VideoSource
from lyra_ocr.dataset.youtube import list_channel_or_playlist
from lyra_ocr.eval.loader import load_artifacts_evaluation_items, load_golden_dataset
from lyra_ocr.eval.metrics import evaluate_predictions
from lyra_ocr.eval.report import build_prediction_rows
from lyra_ocr.labels import (
    VALID_DECISIONS,
    VALID_LABELS,
    LabelRecord,
    append_label_record,
    get_label_status,
    load_labels_jsonl,
    normalize_label,
)
from lyra_ocr.web.jobs import JobQueue

STATIC_DIR = Path(__file__).resolve().parent / "static"

_YOUTUBE_URL_RE = re.compile(r"^https?://(www\.|m\.)?(youtube\.com|youtu\.be)/")


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


class LyraReviewHandler(BaseHTTPRequestHandler):
    """HTTP request handler providing REST API and serving static review interface."""

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
            golden_dir = query.get("golden", [None])[0]
            art_dir = query.get("dir", [None])[0]
            self._handle_get_eval(golden_dir, art_dir)
            return

        # 5b. API: Evaluation error rows (for dashboard click-through)
        if path == "/api/eval/errors":
            golden_dir = query.get("golden", [None])[0]
            art_dir = query.get("dir", [None])[0]
            self._handle_get_eval_errors(golden_dir, art_dir)
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

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # API Handlers
    # -------------------------------------------------------------------------

    def _handle_get_artifacts(self) -> None:
        """Scan workspace for directories containing tracks.json and manifest.json."""
        root = Path.cwd()
        candidate_dirs: list[Path] = []

        search_roots = [
            root / "outputs",
            root / "golden",
            root,
        ]

        for s_root in search_roots:
            if not s_root.exists():
                continue
            for item in s_root.glob("**/tracks.json"):
                art_dir = item.parent
                if (art_dir / "manifest.json").exists():
                    candidate_dirs.append(art_dir)

        # Deduplicate
        seen = set()
        artifacts_list = []
        for d in candidate_dirs:
            resolved = d.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            rel_path = str(d.relative_to(root)) if d.is_relative_to(root) else str(d)
            try:
                status = get_label_status(d)
                manifest_path = d / "manifest.json"
                v_title = ""
                v_channel = ""
                if manifest_path.exists():
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                        meta = m_data.get("metadata", {})
                        v_title = meta.get("title", "")
                        v_channel = meta.get("channel", "")

                artifacts_list.append({
                    "name": d.name,
                    "path": rel_path,
                    "video_id": status.get("video_id", d.name),
                    "title": v_title,
                    "channel": v_channel,
                    "total_tracks": status.get("total_tracks", 0),
                    "labeled_tracks": status.get("labeled_tracks", 0),
                    "coverage_percent": status.get("coverage_percent", 0.0),
                })
            except Exception:
                continue

        artifacts_list.sort(key=lambda x: x["name"])
        self._send_json({"artifacts": artifacts_list})

    def _handle_get_data(self, art_dir_str: str) -> None:
        """Load manifest, tracks, frame_index, and latest labels for an artifact directory."""
        art_path = _safe_resolve(art_dir_str)
        if not art_path or not art_path.exists():
            self._send_error(f"Artifact directory not found: {art_dir_str}", 404)
            return

        tracks_file = art_path / "tracks.json"
        manifest_file = art_path / "manifest.json"
        frames_file = art_path / "frame_index.json"
        labels_file = art_path / "labels.jsonl"

        if not tracks_file.exists():
            self._send_error("tracks.json missing in artifacts directory", 400)
            return

        with open(tracks_file, "r", encoding="utf-8") as f:
            tracks_data = json.load(f)

        manifest_data = {}
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

        frames_data = []
        if frames_file.exists():
            with open(frames_file, "r", encoding="utf-8") as f:
                raw_f = json.load(f)
                if isinstance(raw_f, list):
                    frames_data = raw_f
                elif isinstance(raw_f, dict):
                    frames_data = raw_f.get("frames", [])

        # Normalize frame fields for frontend
        normalized_frames = []
        for idx, fr in enumerate(frames_data):
            img_p = fr.get("image_path") or fr.get("filename") or f"frame_{idx:06d}.jpg"
            normalized_frames.append({
                "frame_id": fr.get("frame_id", idx),
                "pts_ms": fr.get("timestamp_ms") or fr.get("pts_ms", 0),
                "filename": Path(img_p).name,
                "image_path": img_p,
            })

        # Load existing labels
        labels_map = {}
        if labels_file.exists():
            raw_labels = load_labels_jsonl(labels_file)
            for (vid, tid), row in raw_labels.items():
                labels_map[tid] = row

        status = get_label_status(art_path)

        tracks_payload = tracks_data.get("tracks", [])
        for tr in tracks_payload:
            tr["label"] = normalize_label(tr.get("label"))

        self._send_json({
            "artifact_dir": str(art_dir_str),
            "manifest": manifest_data,
            "tracks": tracks_payload,
            "frame_index": normalized_frames,
            "labels": labels_map,
            "status": status,
            "valid_labels": VALID_LABELS,
        })

    def _handle_get_frame(self, art_dir_str: str, filename: str) -> None:
        """Serve extracted JPEG frame image."""
        art_path = _safe_resolve(art_dir_str)
        if not art_path:
            self._send_error("Invalid artifact directory", 400)
            return

        # Sanitize filename (no path components)
        clean_name = Path(filename).name
        frame_path = art_path / "frames" / clean_name
        if not frame_path.exists():
            # Check directly in art_path
            frame_path = art_path / clean_name

        if not frame_path.exists() or not frame_path.is_file():
            self._send_error(f"Frame image not found: {filename}", 404)
            return

        with open(frame_path, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(content)

    def _handle_get_status(self, art_dir_str: str) -> None:
        art_path = _safe_resolve(art_dir_str)
        if not art_path or not art_path.exists():
            self._send_error(f"Artifact directory not found: {art_dir_str}", 404)
            return
        status = get_label_status(art_path)
        self._send_json(status)

    def _handle_get_eval(self, golden_dir_str: str | None, art_dir_str: str | None) -> None:
        try:
            if golden_dir_str:
                g_path = _safe_resolve(golden_dir_str, base_dir=self._workspace_root())
                if not g_path or not g_path.exists():
                    self._send_error(f"Golden dataset directory not found: {golden_dir_str}", 404)
                    return
                items = load_golden_dataset(g_path)
            elif art_dir_str:
                art_path = _safe_resolve(art_dir_str, base_dir=self._workspace_root())
                if not art_path or not art_path.exists():
                    self._send_error(f"Artifact directory not found: {art_dir_str}", 404)
                    return
                items = load_artifacts_evaluation_items(art_path)
            else:
                self._send_error("Must specify either golden or dir query parameter", 400)
                return

            rep = evaluate_predictions(items)
            self._send_json(rep.to_dict())
        except Exception as e:
            self._send_error(f"Evaluation failed: {e}", 500)

    def _handle_post_label(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            self._send_error("Empty request body", 400)
            return

        body_raw = self.rfile.read(content_len).decode("utf-8")
        try:
            payload = json.loads(body_raw)
        except json.JSONDecodeError:
            self._send_error("Invalid JSON body", 400)
            return

        art_dir_str = payload.get("artifact_dir")
        if not art_dir_str:
            self._send_error("Missing required field: artifact_dir", 400)
            return

        art_path = _safe_resolve(art_dir_str)
        if not art_path or not art_path.exists():
            self._send_error(f"Artifact directory not found: {art_dir_str}", 404)
            return

        video_id = payload.get("video_id")
        track_id = payload.get("track_id")
        human_label_raw = payload.get("human_label")
        if not isinstance(human_label_raw, str) or not human_label_raw.strip():
            self._send_error("human_label is required and must be a string", 400)
            return
        human_label = human_label_raw.strip().lower()

        if not video_id or track_id is None:
            self._send_error("video_id and track_id are required", 400)
            return

        if human_label not in VALID_LABELS:
            self._send_error(f"Invalid human_label '{human_label}'. Allowed: {VALID_LABELS}", 400)
            return

        human_decision = str(payload.get("human_decision", "") or "").strip()
        if human_decision and human_decision not in VALID_DECISIONS:
            self._send_error(
                f"Invalid human_decision '{human_decision}'. Allowed: {VALID_DECISIONS}", 400
            )
            return

        # Get track info from tracks.json
        tracks_file = art_path / "tracks.json"
        model_label = None
        model_confidence = None
        if tracks_file.exists():
            with open(tracks_file, "r", encoding="utf-8") as f:
                td = json.load(f)
                for tr in td.get("tracks", []):
                    if str(tr.get("track_id")) == str(track_id):
                        model_label = normalize_label(tr.get("label"))
                        model_confidence = tr.get("confidence")
                        break

        labels_path = art_path / "labels.jsonl"
        record = LabelRecord(
            video_id=str(video_id),
            track_id=str(track_id),
            human_label=human_label,
            tracking_error=bool(payload.get("tracking_error", False)),
            human_decision=human_decision,
            corrected_text=payload.get("corrected_text") or None,
            notes=str(payload.get("notes", "")),
            annotator=str(payload.get("annotator", "web_reviewer")),
            model_label=model_label,
            model_confidence=model_confidence,
        )

        append_label_record(labels_path, record)
        updated_status = get_label_status(art_path)

        self._send_json({
            "ok": True,
            "record": record.to_dict(),
            "status": updated_status,
        })

    def _handle_dataset_get(self, rest: str) -> None:
        parts = rest.split("/")
        name = parts[0]
        sub = parts[1] if len(parts) > 1 else ""
        try:
            if sub == "eval":
                self._handle_get_eval(golden_dir_str=f"datasets/{name}", art_dir_str=None)
                return
            if sub:
                self._send_error(f"Unknown dataset sub-resource: {sub}", 404)
                return
            data = registry.get_dataset(self._workspace_root(), name)
        except FileNotFoundError as exc:
            self._send_error(str(exc), 404)
            return
        except ValueError as exc:
            self._send_error(str(exc), 400)
            return
        queue = self._job_queue()
        data["jobs"] = [j.to_dict() for j in queue.list(name)] if queue else []
        data["job_stats"] = queue.stats(name) if queue else {}
        self._send_json(data)

    def _handle_create_dataset(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            self._send_error("Invalid JSON body", 400)
            return
        name = str(payload.get("name", "")).strip()
        description = str(payload.get("description", ""))
        try:
            path = registry.create_dataset(self._workspace_root(), name, description)
        except (ValueError, FileExistsError) as exc:
            self._send_error(str(exc), 400)
            return
        self._send_json({"ok": True, "name": name, "path": str(path)}, status=201)

    def _handle_dataset_post(self, rest: str) -> None:
        parts = rest.split("/")
        if len(parts) != 2:
            self._send_error(f"Unknown dataset endpoint: {rest}", 404)
            return
        name, action = parts[0], parts[1]
        payload = self._read_json_body()
        if payload is None:
            self._send_error("Invalid JSON body", 400)
            return

        if action == "items":
            self._handle_add_items(name, payload)
        elif action == "split":
            try:
                registry.set_split(self._workspace_root(), name, str(payload.get("video_id", "")), str(payload.get("split", "")))
            except (KeyError, ValueError, FileNotFoundError) as exc:
                self._send_error(str(exc), 400)
                return
            self._send_json({"ok": True})
        elif action == "freeze":
            try:
                result = registry.freeze_dataset(self._workspace_root(), name)
            except (FileNotFoundError, OSError) as exc:
                self._send_error(str(exc), 400)
                return
            self._send_json({"ok": True, **result})
        else:
            self._send_error(f"Unknown dataset action: {action}", 404)

    def _handle_add_items(self, name: str, payload: dict[str, Any]) -> None:
        queue = self._job_queue()
        if queue is None:
            self._send_error("Job queue unavailable", HTTPStatus.SERVICE_UNAVAILABLE)
            return
        try:
            registry.get_dataset(self._workspace_root(), name)
        except (FileNotFoundError, ValueError) as exc:
            self._send_error(str(exc), 404)
            return

        raw_videos = payload.get("videos")
        if raw_videos is None:
            raw_videos = []
            for u in payload.get("youtube_urls", []):
                raw_videos.append({"kind": "youtube", "url": u})
            for p in payload.get("local_paths", []):
                raw_videos.append({"kind": "local", "path": p})

        options = dict(payload.get("options") or {})
        sources: list[VideoSource] = []
        rejected: list[str] = []
        for item in raw_videos:
            src = VideoSource.from_dict(item if isinstance(item, dict) else {"kind": "youtube", "url": str(item)})
            if src.kind == "youtube":
                if not _YOUTUBE_URL_RE.match(src.url or ""):
                    rejected.append(f"invalid youtube url: {src.url!r}")
                    continue
                if not src.yt_id:
                    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", src.url)
                    src.yt_id = m.group(1) if m else ""
                if not src.yt_id:
                    rejected.append(f"cannot derive video id: {src.url!r}")
                    continue
            elif src.kind == "local":
                resolved = _safe_resolve(src.path, base_dir=self._workspace_root())
                if not resolved or not resolved.is_file():
                    rejected.append(f"local file not accessible: {src.path!r}")
                    continue
                src.path = str(resolved)
            else:
                rejected.append(f"unknown source kind: {src.kind!r}")
                continue
            sources.append(src)

        try:
            created = queue.enqueue(name, sources, options)
        except FileNotFoundError as exc:
            self._send_error(str(exc), 404)
            return
        self._send_json(
            {"ok": True, "enqueued": [j.to_dict() for j in created], "rejected": rejected},
            status=HTTPStatus.ACCEPTED,
        )

    def _handle_job_post(self, rest: str) -> None:
        queue = self._job_queue()
        if queue is None:
            self._send_error("Job queue unavailable", HTTPStatus.SERVICE_UNAVAILABLE)
            return
        parts = rest.split("/")
        if len(parts) != 2:
            self._send_error(f"Unknown job endpoint: {rest}", 404)
            return
        job_id, action = parts
        payload = self._read_json_body() or {}
        try:
            if action == "retry":
                rec = queue.retry(job_id, force=bool(payload.get("force", False)))
            elif action == "cancel":
                rec = queue.cancel(job_id)
            else:
                self._send_error(f"Unknown job action: {action}", 404)
                return
        except KeyError:
            self._send_error("Job not found", 404)
            return
        except ValueError as exc:
            self._send_error(str(exc), 400)
            return
        self._send_json({"ok": True, "job": rec.to_dict()})

    def _handle_get_eval_errors(self, golden_dir_str: Optional[str], art_dir_str: Optional[str]) -> None:
        try:
            if golden_dir_str:
                g_path = _safe_resolve(golden_dir_str, base_dir=self._workspace_root())
                if not g_path or not g_path.exists():
                    self._send_error(f"Golden dataset directory not found: {golden_dir_str}", 404)
                    return
                items = load_golden_dataset(g_path)
            elif art_dir_str:
                art_path = _safe_resolve(art_dir_str, base_dir=self._workspace_root())
                if not art_path or not art_path.exists():
                    self._send_error(f"Artifact directory not found: {art_dir_str}", 404)
                    return
                items = load_artifacts_evaluation_items(art_path)
            else:
                self._send_error("Must specify either golden or dir query parameter", 400)
                return

            rep = evaluate_predictions(items)
            rows = build_prediction_rows(items)
            error_rows = [
                r for r in rows
                if r["error_type"] in ("leakage", "false_drop", "misclassification", "tracking_miss", "missing_track")
            ]
            self._send_json({"report": rep.to_dict(), "errors": error_rows})
        except Exception as exc:  # noqa: BLE001
            self._send_error(f"Evaluation failed: {exc}", 500)

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
