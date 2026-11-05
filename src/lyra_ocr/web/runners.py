"""Pipeline runners used by the background job queue."""

from __future__ import annotations

import shutil
import threading
import time
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
    from lyra_ocr.video.downloader import download_video
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

def _make_ingest_runner(workspace_root: Path) -> PipelineRunner:
    """Download or copy media into a dataset without running the pipeline."""
    from lyra_ocr.video.downloader import download_video

    def runner(
        rec: JobRecord,
        src: VideoSource,
        cancel_event: threading.Event,
        progress: Callable[[str, float, str], None],
    ) -> dict[str, Any]:
        ds_dir = registry.dataset_dir(workspace_root, rec.dataset)
        videos_dir = ds_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        if src.kind == "youtube":
            existing = sorted(videos_dir.glob(f"*[{src.yt_id}].mp4")) if src.yt_id else []
            if existing:
                video_path = existing[-1]
            else:
                video_path = _download_with_retry(
                    src.url,
                    videos_dir,
                    progress,
                    max_height=int(rec.options.get("max_height", 720)),
                )
        elif src.kind == "local":
            source_path = Path(src.path)
            if not source_path.is_file():
                raise FileNotFoundError(f"local video not found: {source_path}")
            target = videos_dir / f"{rec.video_id}{source_path.suffix.lower() or '.mp4'}"
            if source_path.resolve() != target.resolve():
                progress("copy", 20.0, "copying local video into dataset")
                shutil.copy2(source_path, target)
            video_path = target
        else:
            raise ValueError(f"unsupported source kind: {src.kind}")

        if cancel_event.is_set():
            raise PipelineCancelled("cancelled during ingest")

        rec.video_path = str(video_path)
        try:
            rel_path = str(video_path.resolve().relative_to(ds_dir.resolve()))
        except ValueError:
            rel_path = str(video_path)
        registry.upsert_video(
            workspace_root,
            rec.dataset,
            video_id=rec.video_id,
            video_path=rel_path,
            source=src.to_dict(),
            title=src.title,
            channel=src.channel,
            duration=src.duration,
        )
        progress("done", 100.0, "media ready")
        return {
            "video_id": rec.video_id,
            "video_path": str(video_path),
            "title": src.title,
            "channel": src.channel,
        }

    return runner

def _make_run_runner(workspace_root: Path, engines: PipelineEngines) -> PipelineRunner:
    """Run the Phase-1 pipeline for an already downloaded dataset video."""

    def runner(
        rec: JobRecord,
        src: VideoSource,
        cancel_event: threading.Event,
        progress: Callable[[str, float, str], None],
    ) -> dict[str, Any]:
        from lyra_ocr.pipeline import run_phase1_pipeline

        ds_dir = registry.dataset_dir(workspace_root, rec.dataset)
        video = registry.get_video(workspace_root, rec.dataset, rec.video_id)
        video_path = Path(str(video.get("video_path", "")))
        if not video_path.is_absolute():
            video_path = ds_dir / video_path
        if not video_path.is_file():
            raise FileNotFoundError(
                f"video {rec.video_id} has not been downloaded yet ({video_path})"
            )

        run = registry.get_run(workspace_root, rec.dataset, rec.run_id)
        out_dir = registry.run_dir(workspace_root, rec.dataset, rec.run_id) / "videos" / rec.video_id
        rec.artifact_dir = f"videos/{rec.video_id}"
        tracker = str(run.get("tracker", "greedy"))
        fps = float(rec.options.get("fps", run.get("fps", 1.0)))
        force = bool(rec.options.get("force", False))

        def pipeline_progress(stage: str, percent: float, message: str = "") -> None:
            progress(stage, 30.0 + 0.7 * percent, message)

        result = run_phase1_pipeline(
            video_path,
            out_dir,
            title=str(video.get("title", "") or ""),
            channel=str(video.get("channel", "") or ""),
            fps=fps,
            tracker=tracker,
            video_id=rec.video_id,
            source_url=str((video.get("source") or {}).get("url", "") or ""),
            engines=engines,
            progress=pipeline_progress,
            cancel_event=cancel_event,
            force=force,
        )
        return {
            "video_id": result["video_id"],
            "title": str(video.get("title", "") or ""),
            "channel": str(video.get("channel", "") or ""),
        }

    return runner

def _download_with_retry(
    url: str,
    videos_dir: Path,
    progress: Callable[[str, float, str], None],
    max_height: int,
    attempts: int = 3,
) -> Path:
    from lyra_ocr.video.downloader import download_video

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
