"""Dataset, run, job, and media-ingestion endpoints."""

from __future__ import annotations

import re
from http import HTTPStatus
from typing import Any

from lyra_ocr.dataset import ground_truth, registry
from lyra_ocr.dataset.models import VideoSource
from lyra_ocr.web.base import _safe_resolve

_YOUTUBE_URL_RE = re.compile(r"^https?://(www\.|m\.)?(youtube\.com|youtu\.be)/")


class DatasetAPIMixin:
    """Datasets tab, runs, job control, and YouTube listing."""

    def _handle_dataset_get(self, rest: str) -> None:
        parts = [p for p in rest.split("/") if p]
        if not parts:
            self._send_error("Missing dataset name", 400)
            return
        name = parts[0]
        sub = parts[1] if len(parts) > 1 else ""
        try:
            if sub == "runs":
                if len(parts) >= 3:
                    run = registry.get_run(self._workspace_root(), name, parts[2])
                    run["artifact_root"] = f"datasets/{name}/runs/{parts[2]}"
                    queue = self._job_queue()
                    jobs = [j.to_dict() for j in queue.list(name)] if queue else []
                    jobs = [j for j in jobs if j.get("run_id") == parts[2]]
                    run["jobs"] = jobs
                    stats: dict[str, int] = {}
                    for j in jobs:
                        stats[j["state"]] = stats.get(j["state"], 0) + 1
                    run["job_stats"] = stats
                    self._send_json(run)
                    return
                self._send_json({"runs": registry.list_runs(self._workspace_root(), name)})
                return
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
        parts = [p for p in rest.split("/") if p]
        if len(parts) < 2:
            self._send_error(f"Unknown dataset endpoint: {rest}", 404)
            return
        name, action = parts[0], parts[1]
        payload = self._read_json_body()
        if payload is None:
            self._send_error("Invalid JSON body", 400)
            return

        if action == "items" and len(parts) == 2:
            self._handle_add_items(name, payload)
        elif action == "download" and len(parts) == 2:
            self._handle_download(name, payload)
        elif action == "runs" and len(parts) == 2:
            self._handle_create_run(name, payload)
        elif action == "runs" and len(parts) == 4 and parts[3] == "items":
            self._handle_run_items(name, parts[2], payload)
        elif action == "ground_truth" and len(parts) == 3 and parts[2] == "promote":
            self._handle_promote_ground_truth(name, payload)
        elif action == "split" and len(parts) == 2:
            try:
                registry.set_split(self._workspace_root(), name, str(payload.get("video_id", "")), str(payload.get("split", "")))
            except (KeyError, ValueError, FileNotFoundError) as exc:
                self._send_error(str(exc), 400)
                return
            self._send_json({"ok": True})
        elif action == "freeze" and len(parts) == 2:
            try:
                result = registry.freeze_dataset(self._workspace_root(), name)
            except (FileNotFoundError, OSError) as exc:
                self._send_error(str(exc), 400)
                return
            self._send_json({"ok": True, **result})
        else:
            self._send_error(f"Unknown dataset action: {rest}", 404)

    def _parse_sources(self, payload: dict[str, Any]) -> tuple[list[VideoSource], list[str]]:
        raw_videos = payload.get("videos")
        if raw_videos is None:
            raw_videos = []
            for u in payload.get("youtube_urls", []):
                raw_videos.append({"kind": "youtube", "url": u})
            for p in payload.get("local_paths", []):
                raw_videos.append({"kind": "local", "path": p})

        sources: list[VideoSource] = []
        rejected: list[str] = []
        for item in raw_videos:
            src = VideoSource.from_dict(
                item if isinstance(item, dict) else {"kind": "youtube", "url": str(item)}
            )
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
        return sources, rejected

    def _handle_add_items(self, name: str, payload: dict[str, Any]) -> None:
        """Legacy endpoint: download + run in one job."""
        queue = self._job_queue()
        if queue is None:
            self._send_error("Job queue unavailable", HTTPStatus.SERVICE_UNAVAILABLE)
            return
        try:
            registry.get_dataset(self._workspace_root(), name)
        except (FileNotFoundError, ValueError) as exc:
            self._send_error(str(exc), 404)
            return
        sources, rejected = self._parse_sources(payload)
        try:
            created = queue.enqueue(name, sources, dict(payload.get("options") or {}))
        except FileNotFoundError as exc:
            self._send_error(str(exc), 404)
            return
        self._send_json(
            {"ok": True, "enqueued": [j.to_dict() for j in created], "rejected": rejected},
            status=HTTPStatus.ACCEPTED,
        )

    def _handle_download(self, name: str, payload: dict[str, Any]) -> None:
        queue = self._job_queue()
        if queue is None:
            self._send_error("Job queue unavailable", HTTPStatus.SERVICE_UNAVAILABLE)
            return
        try:
            registry.get_dataset(self._workspace_root(), name)
        except (FileNotFoundError, ValueError) as exc:
            self._send_error(str(exc), 404)
            return
        sources, rejected = self._parse_sources(payload)
        try:
            created = queue.enqueue_ingest(name, sources, dict(payload.get("options") or {}))
        except FileNotFoundError as exc:
            self._send_error(str(exc), 404)
            return
        self._send_json(
            {"ok": True, "enqueued": [j.to_dict() for j in created], "rejected": rejected},
            status=HTTPStatus.ACCEPTED,
        )

    def _handle_create_run(self, name: str, payload: dict[str, Any]) -> None:
        try:
            run = registry.create_run(
                self._workspace_root(),
                name,
                tracker=str(payload.get("tracker", "greedy")),
                fps=float(payload.get("fps", 1.0)),
                options=dict(payload.get("options") or {}),
                run_id=payload.get("run_id"),
            )
        except (FileNotFoundError, ValueError, FileExistsError) as exc:
            self._send_error(str(exc), 400)
            return
        self._send_json({"ok": True, "run": run}, status=201)

    def _handle_run_items(self, name: str, run_id: str, payload: dict[str, Any]) -> None:
        queue = self._job_queue()
        if queue is None:
            self._send_error("Job queue unavailable", HTTPStatus.SERVICE_UNAVAILABLE)
            return
        video_ids = payload.get("video_ids")
        if video_ids in (None, "all", []):
            selected = None
        elif isinstance(video_ids, list):
            selected = [str(v) for v in video_ids]
        else:
            self._send_error("video_ids must be a list or 'all'", 400)
            return
        try:
            created = queue.enqueue_run(
                name,
                run_id,
                selected,
                options=dict(payload.get("options") or {}),
            )
        except (FileNotFoundError, ValueError) as exc:
            self._send_error(str(exc), 404)
            return
        self._send_json(
            {"ok": True, "enqueued": [j.to_dict() for j in created]},
            status=HTTPStatus.ACCEPTED,
        )

    def _handle_promote_ground_truth(self, name: str, payload: dict[str, Any]) -> None:
        run_id = str(payload.get("run_id", "")).strip()
        if not run_id:
            self._send_error("Missing required field: run_id", 400)
            return
        video_ids = payload.get("video_ids")
        selected = set(str(v) for v in video_ids) if isinstance(video_ids, list) else None
        try:
            ds_dir = registry.dataset_dir(self._workspace_root(), name)
            run = registry.get_run(self._workspace_root(), name, run_id)
            result = ground_truth.promote_run(ds_dir, run, selected)
        except (FileNotFoundError, ValueError, OSError) as exc:
            self._send_error(str(exc), 400)
            return
        self._send_json({"ok": True, **result})

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
