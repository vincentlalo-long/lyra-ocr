"""Reusable Phase-1 pipeline: probe → sample frames → OCR → track → classify → export.

Extracted from cli.run_smoke so the web dataset builder and CLI share one
implementation. All progress reporting goes through callbacks (no prints).

Frame source, device, recognition batch size, and dedup are opt-in accelerators
with kill switches; pass the conservative values (disk/cpu/1/dedup off) to get
the original behaviour.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import cv2

from lyra_ocr.classify import classify_all_tracks
from lyra_ocr.ocr.detect import TextDetector
from lyra_ocr.ocr.engine import OCREngine, TextLine
from lyra_ocr.ocr.runtime import RuntimeConfig, resolve_runtime
from lyra_ocr.video.dedup import clone_text_line, frame_signature, is_duplicate
from lyra_ocr.video.frames import (
    FrameStreamError,
    iter_sampled_frames,
    sample_frames,
    write_frame_index,
)
from lyra_ocr.video.probe import probe_video
from lyra_ocr.provenance import get_tool_versions
from lyra_ocr.review import (
    export_review_csv,
    export_review_tracks_csv,
    export_tracks_json,
    render_phase1_overlay,
)
from lyra_ocr.track import merge_fragmented_tracks, track_ocr_results

ProgressCb = Callable[[str, float, str], None]  # (stage, percent 0-100, message)
FrameCb = Callable[[int, int, list, float], None]  # (frame_id, ts_ms, text_lines, elapsed_ms)

PIPELINE_VERSION = "2"
FRAME_SOURCES = ("pipe", "disk")


class PipelineCancelled(Exception):
    """Raised when a running pipeline observes its cancel event."""


def compute_video_id(video_path: str | Path, override: Optional[str] = None) -> str:
    """Stable artifact id: explicit override (YouTube id) or legacy stem+path-hash."""
    if override:
        return override
    resolved = Path(video_path).resolve()
    v_stem = resolved.stem
    v_hash = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
    return f"{v_stem}_{v_hash}"


def _runtime_key(runtime: RuntimeConfig) -> tuple:
    return (
        runtime.device,
        runtime.rec_batch_size,
        runtime.intra_op_threads,
        runtime.inter_op_threads,
        runtime.use_coreml,
    )


@dataclass
class PipelineEngines:
    """Lazy singleton holder so a worker thread loads OCR models once."""

    detector: Optional[TextDetector] = None
    ocr_engine: Optional[OCREngine] = None
    runtime: Optional[RuntimeConfig] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _key: Optional[tuple] = field(default=None, repr=False, compare=False)
    _owned: bool = field(default=False, repr=False, compare=False)

    def get_or_init(
        self,
        runtime: Optional[RuntimeConfig] = None,
    ) -> tuple[TextDetector, OCREngine]:
        """Return OCR engines, rebuilding them when the runtime config changes."""
        with self._lock:
            if self.detector is not None and self.ocr_engine is not None:
                # Injected engines (tests / custom callers) are never replaced.
                if not self._owned:
                    return self.detector, self.ocr_engine
                if runtime is None or self._key == _runtime_key(runtime):
                    return self.detector, self.ocr_engine

            if runtime is None:
                runtime = self.runtime or resolve_runtime()

            self.detector = TextDetector(runtime=runtime)
            self.ocr_engine = OCREngine(runtime=runtime)
            self.runtime = runtime
            self._key = _runtime_key(runtime)
            self._owned = True
            return self.detector, self.ocr_engine


def _engine_providers(*engines: Any) -> list[str]:
    """Best-effort execution providers from a live detector/OCREngine."""
    for engine in engines:
        info = getattr(engine, "runtime_info", None)
        if not callable(info):
            continue
        try:
            data = info()
        except Exception:  # noqa: BLE001 - provenance must never fail a run
            continue
        providers = data.get("providers") if isinstance(data, dict) else None
        if providers:
            return [str(p) for p in providers]
    return []


def _config_hash(
    fps: float,
    start_sec: float,
    max_frames: Optional[int],
    tracker: str,
    title: str,
    channel: str,
    *,
    pipeline_version: str = PIPELINE_VERSION,
    device: str = "cpu",
    rec_batch_size: int = 1,
    dedup: bool = False,
    frame_source: str = "disk",
) -> str:
    cfg_dict = {
        "sample_fps": fps,
        "start_sec": start_sec,
        "sample_frames": max_frames,
        "tracker": tracker,
        "title": title or "",
        "channel": channel or "",
        "pipeline_version": pipeline_version,
        "device": device,
        "rec_batch_size": rec_batch_size,
        "dedup": bool(dedup),
        "frame_source": frame_source,
    }
    return hashlib.sha256(json.dumps(cfg_dict, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _iter_disk_frames(
    video_path: Path,
    out_dir: Path,
    fps: float,
    max_frames: Optional[int],
    start_sec: float,
) -> Iterator[dict[str, Any]]:
    """Legacy path: ffmpeg writes JPEGs, the consumer reads them back."""
    entries = sample_frames(
        video_path=video_path,
        output_dir=out_dir,
        fps=fps,
        max_frames=max_frames,
        start_sec=start_sec,
    )
    for entry in entries:
        image_bgr = cv2.imread(str(out_dir / entry["image_path"]))
        if image_bgr is None:
            continue
        yield {**entry, "image_bgr": image_bgr, "jpeg_bytes": None, "pts_source": "showinfo"}


def _iter_pipe_frames(
    video_path: Path,
    out_dir: Path,
    fps: float,
    max_frames: Optional[int],
    start_sec: float,
    *,
    jpeg_quality: int,
    queue_size: int,
    cancel_event: Optional[threading.Event],
) -> Iterator[dict[str, Any]]:
    """Streaming path: one ffmpeg process, frames consumed while they decode."""
    for packet in iter_sampled_frames(
        video_path=video_path,
        output_dir=out_dir,
        fps=fps,
        max_frames=max_frames,
        start_sec=start_sec,
        jpeg_quality=jpeg_quality,
        queue_size=queue_size,
        save_frames=False,
        cancel_event=cancel_event,
    ):
        yield {
            "frame_id": packet.frame_id,
            "timestamp_ms": packet.timestamp_ms,
            "pts_time": packet.pts_time,
            "image_path": packet.image_path,
            "image_bgr": packet.image_bgr,
            "jpeg_bytes": packet.jpeg_bytes,
            "pts_source": packet.pts_source,
        }


def run_phase1_pipeline(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    title: str = "",
    channel: str = "",
    fps: float = 1.0,
    start_sec: float = 0.0,
    max_frames: Optional[int] = None,
    tracker: str = "greedy",
    video_id: Optional[str] = None,
    source_url: str = "",
    engines: Optional[PipelineEngines] = None,
    progress: Optional[ProgressCb] = None,
    on_frame: Optional[FrameCb] = None,
    cancel_event: Optional[threading.Event] = None,
    force: bool = False,
    device: str = "auto",
    rec_batch_size: int = 8,
    dedup: bool = False,
    frame_source: str = "pipe",
    jpeg_quality: int = 2,
    queue_size: int = 8,
) -> dict[str, Any]:
    """Run the full Phase-1 pipeline for one video, writing standard artifacts.

    Returns {video_id, config_hash, tracks, frames_processed, out_dir, skipped}.
    """
    video_path = Path(video_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if frame_source not in FRAME_SOURCES:
        raise ValueError(f"frame_source must be one of {FRAME_SOURCES} (got {frame_source!r})")

    engine_holder = engines if engines is not None else PipelineEngines()
    # Always resolve from this run's options: a shared holder may still carry the
    # runtime of an earlier run (e.g. CPU after a previous job), and the config
    # hash must describe what this run actually requested.
    runtime = resolve_runtime(device, rec_batch_size)

    vid = compute_video_id(video_path, video_id)
    cfg_hash = _config_hash(
        fps,
        start_sec,
        max_frames,
        tracker,
        title,
        channel,
        device=runtime.device,
        rec_batch_size=runtime.rec_batch_size,
        dedup=dedup,
        frame_source=frame_source,
    )

    def report(stage: str, percent: float, message: str = "") -> None:
        if progress:
            progress(stage, percent, message)

    def check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineCancelled(f"cancelled before/at stage")

    manifest_path = out_dir / "manifest.json"
    tracks_path = out_dir / "tracks.json"
    if not force and manifest_path.exists() and tracks_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("config_hash") == cfg_hash:
                report("skipped", 100.0, "artifacts already complete for this config")
                return {
                    "video_id": vid,
                    "config_hash": cfg_hash,
                    "tracks": [],
                    "frames_processed": 0,
                    "out_dir": str(out_dir),
                    "skipped": True,
                }
        except (json.JSONDecodeError, OSError):
            pass

    # 1. Probe
    check_cancel()
    report("probe", 2.0, f"probing {video_path.name}")
    meta = probe_video(video_path)

    # 2. Frame sampling (streaming by default, with automatic disk fallback)
    check_cancel()
    frame_source_used = frame_source
    if frame_source == "pipe":
        report("sampling", 5.0, f"streaming frames at {fps} fps")
        stream_iter: Iterator[dict[str, Any]] = _iter_pipe_frames(
            video_path,
            out_dir,
            fps,
            max_frames,
            start_sec,
            jpeg_quality=jpeg_quality,
            queue_size=queue_size,
            cancel_event=cancel_event,
        )
        try:
            first_frame = next(stream_iter, None)
        except FrameStreamError as exc:
            report("sampling", 6.0, f"pipe extraction failed; falling back to disk mode: {exc}")
            frame_source_used = "disk"
            stream_iter = _iter_disk_frames(video_path, out_dir, fps, max_frames, start_sec)
            first_frame = None
        frame_iter: Iterator[dict[str, Any]] = (
            itertools.chain([first_frame], stream_iter) if first_frame is not None else stream_iter
        )
    else:
        report("sampling", 5.0, f"sampling frames at {fps} fps")
        stream_iter = _iter_disk_frames(video_path, out_dir, fps, max_frames, start_sec)
        frame_iter = stream_iter

    # 3. Detection + recognition
    check_cancel()
    detector, ocr_engine = engine_holder.get_or_init(runtime)
    frames_dir = out_dir / "frames"

    processed_frames: list[dict[str, Any]] = []
    total_time_ms = 0.0
    best_frame_path: Optional[Path] = None
    best_frame_ts = 0
    max_box_count = -1
    total_ocr_calls = 0

    prev_signature = None
    prev_lines: list[TextLine] = []
    prev_frame_id: Optional[int] = None
    prev_image_path: Optional[str] = None

    try:
        for i, frame in enumerate(frame_iter):
            check_cancel()
            fid = int(frame["frame_id"])
            t_ms = int(frame["timestamp_ms"])
            rel_fpath = str(frame["image_path"])
            img_bgr = frame["image_bgr"]
            signature = None
            reused = False

            if dedup:
                signature = frame_signature(img_bgr)
                reused = prev_signature is not None and is_duplicate(prev_signature, signature)

            if reused:
                text_lines = [clone_text_line(line) for line in prev_lines]
                rel_fpath = prev_image_path or rel_fpath
                detect_ms = 0.0
                recognize_ms = 0.0
                elapsed_ms = 0.0
            else:
                t0 = time.perf_counter()
                detections = detector.detect(img_bgr)
                t1 = time.perf_counter()
                text_lines = ocr_engine.process_detections(img_bgr, detections)
                t2 = time.perf_counter()
                detect_ms = round((t1 - t0) * 1000.0, 2)
                recognize_ms = round((t2 - t1) * 1000.0, 2)
                elapsed_ms = round(detect_ms + recognize_ms, 2)
                total_time_ms += elapsed_ms
                total_ocr_calls += 1

                if frame_source_used == "pipe":
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    abs_path = out_dir / rel_fpath
                    jpeg = frame.get("jpeg_bytes")
                    if jpeg:
                        abs_path.write_bytes(jpeg)
                    else:
                        cv2.imwrite(
                            str(abs_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                        )

                if dedup:
                    prev_signature = signature
                    prev_lines = text_lines
                    prev_frame_id = fid
                    prev_image_path = rel_fpath

            if on_frame:
                on_frame(fid, t_ms, text_lines, elapsed_ms)

            entry: dict[str, Any] = {
                "frame_id": fid,
                "timestamp_ms": t_ms,
                "pts_time": frame.get("pts_time", round(t_ms / 1000.0, 4)),
                "image_path": rel_fpath,
                "elapsed_ms": elapsed_ms,
                "detect_ms": detect_ms,
                "recognize_ms": recognize_ms,
                "boxes": [line.to_dict() for line in text_lines],
            }
            if reused:
                entry["reused"] = True
                entry["dedup_of_frame"] = prev_frame_id
            if frame.get("pts_source") == "estimated":
                entry["pts_source"] = "estimated"
            processed_frames.append(entry)

            if len(text_lines) > max_box_count:
                max_box_count = len(text_lines)
                best_frame_path = out_dir / rel_fpath
                best_frame_ts = t_ms

            expected = max_frames if max_frames and max_frames > 0 else None
            if expected:
                frac = min(1.0, (i + 1) / expected)
            else:
                frac = (i + 1) / (i + 1 + 25.0)
            report("ocr", 8.0 + 72.0 * frac, f"frame {i + 1}")
    finally:
        close_stream = getattr(stream_iter, "close", None)
        if close_stream is not None:
            close_stream()

    if frame_source_used == "pipe":
        write_frame_index(
            out_dir,
            [
                {
                    key: entry[key]
                    for key in (
                        "frame_id",
                        "timestamp_ms",
                        "pts_time",
                        "image_path",
                        "reused",
                        "dedup_of_frame",
                        "pts_source",
                    )
                    if key in entry
                }
                for entry in processed_frames
            ],
        )

    avg_time_ms = round(total_time_ms / max(1, len(processed_frames)), 2)

    # 4. manifest.json (schema must stay identical to the Phase-0 contract)
    check_cancel()
    report("exporting", 82.0, "writing manifest.json")
    providers = _engine_providers(detector, ocr_engine) or list(runtime.providers)
    manifest_data = {
        "video_id": vid,
        "video_path": str(video_path),
        "metadata": {
            "title": title or "",
            "channel": channel or "",
        },
        "tracker": {
            "name": tracker,
            "iou_threshold": 0.30,
            "max_gap_ms": 2000,
        },
        "config_hash": cfg_hash,
        "probe": meta,
        "config": {
            "sample_fps": fps,
            "start_sec": start_sec,
            "sample_frames": max_frames,
        },
        "runtime": {
            "requested": runtime.request,
            "device": runtime.device,
            "providers": providers,
            "rec_batch_size": runtime.rec_batch_size,
            "intra_op_threads": runtime.intra_op_threads,
            "inter_op_threads": runtime.inter_op_threads,
            "dedup": bool(dedup),
            "frame_source": frame_source_used,
            "pipeline_version": PIPELINE_VERSION,
        },
        "tool_versions": get_tool_versions(),
        "models": {
            "det": "PP-OCRv6_det_small",
            "rec": ocr_engine.DEFAULT_MODEL,
        },
        "timestamp_created": int(time.time()),
    }
    if source_url:
        manifest_data["source_url"] = source_url
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    # 5. ocr_raw.json
    ocr_raw_data = {
        "video_path": str(video_path),
        "manifest_path": "manifest.json",
        "total_frames_processed": len(processed_frames),
        "avg_frame_time_ms": avg_time_ms,
        "total_ocr_calls": total_ocr_calls,
        "frames": processed_frames,
    }
    with open(out_dir / "ocr_raw.json", "w", encoding="utf-8") as f:
        json.dump(ocr_raw_data, f, indent=2, ensure_ascii=False)

    # 6. Tracking
    check_cancel()
    report("tracking", 85.0, f"tracking with strategy '{tracker}'")
    v_dur_ms = int(meta.get("duration", 0) * 1000)
    tracks = track_ocr_results(
        frames_ocr_data=processed_frames,
        video_duration_ms=v_dur_ms,
        frame_width=meta["width"],
        frame_height=meta["height"],
        iou_threshold=0.30,
        max_gap_ms=2000,
        tracker_name=tracker,
    )
    tracks = merge_fragmented_tracks(
        tracks,
        video_duration_ms=v_dur_ms,
        frame_width=meta["width"],
        frame_height=meta["height"],
    )

    # 7. Classification
    check_cancel()
    report("classifying", 90.0, "rule-based classification")
    classify_all_tracks(
        tracks=tracks,
        metadata={
            "title": title or "",
            "channel": channel or "",
            "duration": meta["duration"],
        },
        video_duration_ms=v_dur_ms,
        frame_width=meta["width"],
        frame_height=meta["height"],
    )

    # 8. Exports
    check_cancel()
    report("exporting", 93.0, "writing tracks.json + review sheets")
    export_tracks_json(tracks, str(tracks_path))

    labels_jsonl_path = out_dir / "labels.jsonl"
    labels_path_str = str(labels_jsonl_path) if labels_jsonl_path.exists() else None
    export_review_csv(
        tracks,
        str(out_dir / "review.csv"),
        labels_jsonl_path=labels_path_str,
        video_id=vid,
    )
    export_review_tracks_csv(
        tracks,
        str(out_dir / "review_tracks.csv"),
        video_id=vid,
        labels_jsonl_path=labels_path_str,
    )

    preview_path = out_dir / "roi_preview.jpg"
    target_img = best_frame_path or (
        out_dir / processed_frames[0]["image_path"] if processed_frames else None
    )
    if target_img and target_img.exists():
        best_ts = best_frame_ts if best_frame_path else (
            processed_frames[0]["timestamp_ms"] if processed_frames else 0
        )
        render_phase1_overlay(
            image_path=str(target_img),
            tracks=tracks,
            out_path=str(preview_path),
            target_timestamp_ms=best_ts,
        )

    report("done", 100.0, f"{len(tracks)} track(s) from {len(processed_frames)} frame(s)")
    return {
        "video_id": vid,
        "config_hash": cfg_hash,
        "tracks": tracks,
        "frames_processed": len(processed_frames),
        "out_dir": str(out_dir),
        "skipped": False,
    }
