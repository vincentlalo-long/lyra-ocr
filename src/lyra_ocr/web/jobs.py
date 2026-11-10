"""Background job queue for dataset building: download -> Phase-1 pipeline.

A single worker thread serializes OCR (CPU-bound, onnxruntime already saturates
cores) so models load once and are reused across all queued videos. Job state is
persisted per-dataset to ``jobs.json`` (atomic replace) and resumed on restart.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from lyra_ocr.dataset import registry
from lyra_ocr.dataset.models import JobRecord, VideoSource
from lyra_ocr.pipeline import PipelineCancelled, PipelineEngines
from lyra_ocr.web.runners import (
    PipelineRunner,
    _download_with_retry,
    _make_default_runner,
    _make_ingest_runner,
    _make_run_runner,
)

class JobQueue:
    def __init__(
        self,
        workspace_root: str | Path,
        pipeline_runner: Optional[PipelineRunner] = None,
        engines: Optional[PipelineEngines] = None,
        ingest_runner: Optional[PipelineRunner] = None,
        run_runner: Optional[PipelineRunner] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self._engines = engines or PipelineEngines()
        self._runner = pipeline_runner or _make_default_runner(self.workspace_root, self._engines)
        self._ingest_runner = ingest_runner or _make_ingest_runner(self.workspace_root)
        self._run_runner = run_runner or _make_run_runner(self.workspace_root, self._engines)

        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._order: list[str] = []
        self._cancel_events: dict[str, threading.Event] = {}
        self._pending_delete: set[str] = set()

        self._stop = threading.Event()
        self._wake = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_persist: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._load_all()
        self._worker = threading.Thread(target=self._worker_loop, name="lyra-job-queue", daemon=True)
        self._worker.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None

    # ------------------------------------------------------------------ #
    # Persistence (per-dataset jobs.json)
    # ------------------------------------------------------------------ #

    def _jobs_path(self, dataset: str) -> Path:
        return registry.dataset_dir(self.workspace_root, dataset) / "jobs.json"

    def _load_all(self) -> None:
        root = registry.datasets_root(self.workspace_root)
        if not root.exists():
            return
        with self._lock:
            for ds_dir in sorted(root.iterdir()):
                jp = ds_dir / "jobs.json"
                if not jp.is_file():
                    continue
                try:
                    with open(jp, "r", encoding="utf-8") as f:
                        rows = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                for row in rows:
                    try:
                        rec = JobRecord.from_dict(row)
                    except TypeError:
                        continue
                    if rec.state == "running":
                        rec.state = "queued"  # interrupted by restart
                    self._jobs[rec.job_id] = rec
                    self._order.append(rec.job_id)
                    self._cancel_events[rec.job_id] = threading.Event()

    def _persist_dataset(self, dataset: str, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_persist.get(dataset, 0.0) < 1.0:
            return
        self._last_persist[dataset] = now
        rows = [j.to_dict() for jid in self._order if (j := self._jobs[jid]) and j.dataset == dataset]
        jp = self._jobs_path(dataset)
        try:
            jp.parent.mkdir(parents=True, exist_ok=True)
            tmp = jp.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False)
            os.replace(tmp, jp)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def enqueue(self, dataset: str, sources: list[VideoSource], options: Optional[dict[str, Any]] = None) -> list[JobRecord]:
        registry.get_dataset(self.workspace_root, dataset)  # raises if dataset missing
        created: list[JobRecord] = []
        existing_ids = {j.video_id for j in self._jobs.values() if j.dataset == dataset and j.state not in ("failed", "cancelled")}
        now = int(time.time())
        with self._lock:
            for src in sources:
                vid = registry.derive_video_id(src)
                if vid in existing_ids:
                    continue
                rec = JobRecord(
                    job_id=uuid.uuid4().hex,
                    dataset=dataset,
                    video_id=vid,
                    source=src.to_dict(),
                    state="queued",
                    created_at=now,
                    updated_at=now,
                    kind="pipeline",
                    options=dict(options or {}),
                )
                self._jobs[rec.job_id] = rec
                self._order.append(rec.job_id)
                self._cancel_events[rec.job_id] = threading.Event()
                existing_ids.add(vid)
                created.append(rec)
            if created:
                self._persist_dataset(dataset, force=True)
        self._wake.set()
        return created

    def enqueue_ingest(
        self,
        dataset: str,
        sources: list[VideoSource],
        options: Optional[dict[str, Any]] = None,
    ) -> list[JobRecord]:
        """Queue download/copy jobs only; no OCR or tracking is performed."""
        registry.get_dataset(self.workspace_root, dataset)
        created: list[JobRecord] = []
        existing_ids = {
            j.video_id
            for j in self._jobs.values()
            if j.dataset == dataset and j.kind == "ingest" and j.state not in ("failed", "cancelled")
        }
        now = int(time.time())
        with self._lock:
            for src in sources:
                vid = registry.derive_video_id(src)
                if vid in existing_ids:
                    continue
                rec = JobRecord(
                    job_id=uuid.uuid4().hex,
                    dataset=dataset,
                    video_id=vid,
                    source=src.to_dict(),
                    state="queued",
                    created_at=now,
                    updated_at=now,
                    kind="ingest",
                    options=dict(options or {}),
                )
                self._jobs[rec.job_id] = rec
                self._order.append(rec.job_id)
                self._cancel_events[rec.job_id] = threading.Event()
                existing_ids.add(vid)
                created.append(rec)
            if created:
                self._persist_dataset(dataset, force=True)
        self._wake.set()
        return created

    def enqueue_run(
        self,
        dataset: str,
        run_id: str,
        video_ids: Optional[list[str]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> list[JobRecord]:
        """Queue a pipeline run for selected/all dataset videos."""
        run = registry.get_run(self.workspace_root, dataset, run_id)
        data = registry.get_dataset(self.workspace_root, dataset)
        selected = set(video_ids) if video_ids else None
        existing = {
            (j.run_id, j.video_id)
            for j in self._jobs.values()
            if j.dataset == dataset and j.kind == "run" and j.state not in ("failed", "cancelled")
        }
        created: list[JobRecord] = []
        now = int(time.time())
        with self._lock:
            for video in data.get("videos", []):
                vid = str(video.get("video_id", ""))
                if not vid or (selected is not None and vid not in selected):
                    continue
                if (run_id, vid) in existing:
                    continue
                source = dict(video.get("source") or {})
                if not source:
                    video_path = str(video.get("video_path", ""))
                    source = {"kind": "local", "path": video_path}
                rec = JobRecord(
                    job_id=uuid.uuid4().hex,
                    dataset=dataset,
                    video_id=vid,
                    source=source,
                    state="queued",
                    created_at=now,
                    updated_at=now,
                    kind="run",
                    run_id=run_id,
                    options={**dict(run.get("options") or {}), **dict(options or {})},
                )
                self._jobs[rec.job_id] = rec
                self._order.append(rec.job_id)
                self._cancel_events[rec.job_id] = threading.Event()
                existing.add((run_id, vid))
                created.append(rec)
            if created:
                self._persist_dataset(dataset, force=True)
        self._wake.set()
        return created

    def retry(self, job_id: str, force: bool = False) -> JobRecord:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                raise KeyError(job_id)
            if rec.state not in ("failed", "cancelled", "done"):
                raise ValueError(f"job is {rec.state}; only failed/cancelled/done can be retried")
            rec.state = "queued"
            rec.stage = ""
            rec.percent = 0.0
            rec.error = ""
            rec.failed_stage = ""
            rec.attempts += 1
            rec.updated_at = int(time.time())
            if force:
                rec.options["force"] = True
            self._cancel_events[job_id] = threading.Event()
            self._persist_dataset(rec.dataset, force=True)
        self._wake.set()
        return rec

    def cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                raise KeyError(job_id)
            if rec.state == "queued":
                rec.state = "cancelled"
                rec.updated_at = int(time.time())
                self._persist_dataset(rec.dataset, force=True)
            elif rec.state == "running":
                self._cancel_events.get(job_id, threading.Event()).set()
            else:
                raise ValueError(f"job is {rec.state}; cannot cancel")
        return rec

    def cancel_all(self, dataset: Optional[str] = None) -> list[JobRecord]:
        """Cancel all queued and running jobs, optionally restricted to a dataset."""
        cancelled: list[JobRecord] = []
        datasets_to_persist: set[str] = set()
        with self._lock:
            for jid in self._order:
                rec = self._jobs.get(jid)
                if rec is None:
                    continue
                if dataset and rec.dataset != dataset:
                    continue
                if rec.state == "queued":
                    rec.state = "cancelled"
                    rec.updated_at = int(time.time())
                    cancelled.append(rec)
                    datasets_to_persist.add(rec.dataset)
                elif rec.state == "running":
                    ev = self._cancel_events.get(rec.job_id)
                    if ev is not None:
                        ev.set()
                    cancelled.append(rec)
            for ds in datasets_to_persist:
                self._persist_dataset(ds, force=True)
        return cancelled

    def delete_run_jobs(self, dataset: str, run_id: str) -> int:
        """Cancel and purge all jobs associated with a specific run."""
        removed = 0
        with self._lock:
            to_remove = [
                jid for jid, rec in self._jobs.items()
                if rec.dataset == dataset and rec.run_id == run_id
            ]
            for jid in to_remove:
                rec = self._jobs.get(jid)
                ev = self._cancel_events.setdefault(jid, threading.Event())
                ev.set()
                if rec and rec.state == "running":
                    rec.state = "cancelled"
                    rec.updated_at = int(time.time())
                    self._pending_delete.add(jid)
                else:
                    self._jobs.pop(jid, None)
                    if jid in self._order:
                        self._order.remove(jid)
                    self._cancel_events.pop(jid, None)
                removed += 1
            if removed:
                self._persist_dataset(dataset, force=True)
        return removed

    def delete_all_run_jobs(self, dataset: str) -> int:
        """Cancel and purge all pipeline run jobs for a dataset."""
        removed = 0
        with self._lock:
            to_remove = [
                jid for jid, rec in self._jobs.items()
                if rec.dataset == dataset and rec.kind == "run"
            ]
            for jid in to_remove:
                rec = self._jobs.get(jid)
                ev = self._cancel_events.setdefault(jid, threading.Event())
                ev.set()
                if rec and rec.state == "running":
                    rec.state = "cancelled"
                    rec.updated_at = int(time.time())
                    self._pending_delete.add(jid)
                else:
                    self._jobs.pop(jid, None)
                    if jid in self._order:
                        self._order.remove(jid)
                    self._cancel_events.pop(jid, None)
                removed += 1
            if removed:
                self._persist_dataset(dataset, force=True)
        return removed

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def list(self, dataset: Optional[str] = None) -> list[JobRecord]:
        recs = [self._jobs[jid] for jid in self._order if jid in self._jobs]
        if dataset:
            recs = [r for r in recs if r.dataset == dataset]
        return recs

    def stats(self, dataset: Optional[str] = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in self.list(dataset):
            counts[rec.state] = counts.get(rec.state, 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    # Worker
    # ------------------------------------------------------------------ #

    def _next_queued(self) -> Optional[JobRecord]:
        with self._lock:
            for jid in self._order:
                rec = self._jobs.get(jid)
                if rec and rec.state == "queued":
                    return rec
        return None

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            rec = self._next_queued()
            if rec is None:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            self._run_item(rec)

    def _run_item(self, rec: JobRecord) -> None:
        src = VideoSource.from_dict(rec.source)
        cancel_event = self._cancel_events.setdefault(rec.job_id, threading.Event())
        with self._lock:
            rec.state = "running"
            rec.stage = "starting"
            rec.updated_at = int(time.time())
            self._persist_dataset(rec.dataset, force=True)

        def progress(stage: str, percent: float, message: str = "") -> None:
            with self._lock:
                rec.stage = stage
                rec.percent = round(max(0.0, min(100.0, percent)), 1)
                rec.updated_at = int(time.time())
            self._persist_dataset(rec.dataset)

        try:
            if cancel_event.is_set():
                raise PipelineCancelled()
            if rec.kind == "ingest":
                result = self._ingest_runner(rec, src, cancel_event, progress)
            elif rec.kind == "run":
                result = self._run_runner(rec, src, cancel_event, progress)
            else:
                result = self._runner(rec, src, cancel_event, progress)
            with self._lock:
                if rec.job_id not in self._pending_delete and not cancel_event.is_set():
                    rec.state = "done"
                    rec.stage = "done"
                    rec.percent = 100.0
                    rec.error = ""
                    rec.updated_at = int(time.time())
                    if rec.artifact_dir:
                        rec.options["force"] = False
            if rec.job_id not in self._pending_delete and not cancel_event.is_set():
                if rec.kind == "run":
                    self._sync_run(rec, result or {})
                elif rec.kind == "pipeline":
                    self._sync_manifest(rec, result or {})
        except PipelineCancelled:
            with self._lock:
                rec.state = "cancelled"
                rec.updated_at = int(time.time())
        except Exception as exc:  # noqa: BLE001 - job boundary
            with self._lock:
                rec.state = "failed"
                rec.error = str(exc)[-2000:]
                rec.failed_stage = rec.stage or "unknown"
                rec.updated_at = int(time.time())
        finally:
            with self._lock:
                if rec.job_id in self._pending_delete:
                    self._pending_delete.remove(rec.job_id)
                    self._jobs.pop(rec.job_id, None)
                    if rec.job_id in self._order:
                        self._order.remove(rec.job_id)
                    self._cancel_events.pop(rec.job_id, None)
            self._persist_dataset(rec.dataset, force=True)

    def _sync_manifest(self, rec: JobRecord, result: dict[str, Any]) -> None:
        if not rec.artifact_dir:
            return
        try:
            registry.sync_entry(
                self.workspace_root,
                rec.dataset,
                video_id=result.get("video_id", rec.video_id),
                artifact_dir=rec.artifact_dir,
                title=result.get("title", rec.source.get("title", "")),
                channel=result.get("channel", rec.source.get("channel", "")),
            )
        except (FileNotFoundError, OSError):
            pass

    def _sync_run(self, rec: JobRecord, result: dict[str, Any]) -> None:
        if not rec.run_id or not rec.artifact_dir:
            return
        try:
            registry.sync_run_entry(
                self.workspace_root,
                rec.dataset,
                rec.run_id,
                video_id=result.get("video_id", rec.video_id),
                artifact_dir=rec.artifact_dir,
                title=result.get("title", rec.source.get("title", "")),
                channel=result.get("channel", rec.source.get("channel", "")),
            )
        except (FileNotFoundError, OSError):
            pass
