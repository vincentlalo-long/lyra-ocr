"""Background job queue for dataset building: download → Phase-1 pipeline.

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

PipelineRunner = Callable[[JobRecord, VideoSource, threading.Event, Callable[[str, float, str], None]], dict[str, Any]]


def _make_default_runner(workspace_root: Path, engines: PipelineEngines) -> PipelineRunner:
    """Build the production runner: download (YouTube) then run Phase-1 pipeline.

    The runner sets ``rec.video_path`` and ``rec.artifact_dir`` as it progresses
    so cancellation/restart can resume and the manifest sync knows the target.
    """
    from lyra_ocr.downloader import download_video
    from lyra_ocr.pipeline import run_phase1_pipeline

    def runner(
        rec: JobRecord,
        src: VideoSource,
        cancel_event: threading.Event,
        progress: Callable[[str, float, str], None],
    ) -> dict[str, Any]:
        ds_dir = registry.dataset_dir(workspace_root, rec.dataset)
        out_dir = ds_dir / "artifacts" / rec.video_id
        rec.artifact_dir = f"artifacts/{rec.video_id}"

        title = src.title or ""
        channel = src.channel or ""
        fps = float(rec.options.get("fps", 1.0))
        tracker = str(rec.options.get("tracker", "greedy"))
        force = bool(rec.options.get("force", False))

        if src.kind == "youtube":
            videos_dir = ds_dir / "videos"
            existing = sorted(videos_dir.glob(f"*[{src.yt_id}].mp4")) if src.yt_id else []
            if existing:
                video_path = existing[-1]
            else:
                video_path = _download_with_retry(
                    src.url, videos_dir, progress, max_height=int(rec.options.get("max_height", 720))
                )
            rec.video_path = str(video_path)
            src.title = src.title or title
            src.channel = src.channel or channel
        else:
            video_path = Path(src.path)
            rec.video_path = str(video_path)

        def pipeline_progress(stage: str, percent: float, message: str = "") -> None:
            progress(stage, 40.0 + 0.6 * percent, message)

        result = run_phase1_pipeline(
            video_path,
            out_dir,
            title=src.title or title,
            channel=src.channel or channel,
            fps=fps,
            tracker=tracker,
            video_id=rec.video_id,
            source_url=src.url,
            engines=engines,
            progress=pipeline_progress,
            cancel_event=cancel_event,
            force=force,
        )
        return {
            "video_id": result["video_id"],
            "title": src.title or title,
            "channel": src.channel or channel,
        }

    return runner


def _download_with_retry(
    url: str,
    videos_dir: Path,
    progress: Callable[[str, float, str], None],
    max_height: int,
    attempts: int = 3,
) -> Path:
    from lyra_ocr.downloader import download_video

    backoffs = [0, 5, 15]
    last_err: Optional[Exception] = None
    for i in range(attempts):
        if i:
            time.sleep(backoffs[min(i, len(backoffs) - 1)])
        try:
            progress("download", 0.0, f"downloading (attempt {i + 1}/{attempts})")

            def dl_hook(evt: dict[str, Any]) -> None:
                if evt.get("stage") == "downloading":
                    progress("download", 40.0 * (evt.get("percent", 0.0) / 100.0), "downloading")
                else:
                    progress("download", 20.0, str(evt.get("message", evt.get("stage", ""))))

            info = download_video(url, str(videos_dir), progress=dl_hook, max_height=max_height)
            return Path(info["video_path"])
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"download failed after {attempts} attempts: {last_err}")


class JobQueue:
    def __init__(
        self,
        workspace_root: str | Path,
        pipeline_runner: Optional[PipelineRunner] = None,
        engines: Optional[PipelineEngines] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self._engines = engines or PipelineEngines()
        self._runner = pipeline_runner or _make_default_runner(self.workspace_root, self._engines)

        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._order: list[str] = []
        self._cancel_events: dict[str, threading.Event] = {}

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
        cancel_event = self._cancel_events.get(rec.job_id, threading.Event())
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
            result = self._runner(rec, src, cancel_event, progress)
            with self._lock:
                rec.state = "done"
                rec.stage = "done"
                rec.percent = 100.0
                rec.error = ""
                rec.updated_at = int(time.time())
                if rec.artifact_dir:
                    rec.options["force"] = False
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
