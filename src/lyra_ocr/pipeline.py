"""Reusable Phase-1 pipeline: probe → sample frames → OCR → track → classify → export.

Extracted from cli.run_smoke so the web dataset builder and CLI share one
implementation. All progress reporting goes through callbacks (no prints).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2

from lyra_ocr.classify import classify_all_tracks
from lyra_ocr.ocr.detect import TextDetector
from lyra_ocr.video.frames import sample_frames
from lyra_ocr.ocr import OCREngine, TextLine
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


@dataclass
class PipelineEngines:
    """Lazy singleton holder so a worker thread loads OCR models once."""

    detector: Optional[TextDetector] = None
    ocr_engine: Optional[OCREngine] = None

    def get_or_init(self) -> tuple[TextDetector, OCREngine]:
        if self.detector is None:
            self.detector = TextDetector()
        if self.ocr_engine is None:
            self.ocr_engine = OCREngine()
        return self.detector, self.ocr_engine


def _config_hash(
    fps: float,
    start_sec: float,
    max_frames: Optional[int],
    tracker: str,
    title: str,
    channel: str,
) -> str:
    cfg_dict = {
        "sample_fps": fps,
        "start_sec": start_sec,
        "sample_frames": max_frames,
        "tracker": tracker,
        "title": title or "",
        "channel": channel or "",
    }
    return hashlib.sha256(json.dumps(cfg_dict, sort_keys=True).encode("utf-8")).hexdigest()[:12]


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
) -> dict[str, Any]:
    """Run the full Phase-1 pipeline for one video, writing standard artifacts.

    Returns {video_id, config_hash, tracks, frames_processed, out_dir, skipped}.
    """
    video_path = Path(video_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vid = compute_video_id(video_path, video_id)
    cfg_hash = _config_hash(fps, start_sec, max_frames, tracker, title, channel)

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

    # 2. Frame sampling
    check_cancel()
    report("sampling", 5.0, f"sampling frames at {fps} fps")
    frame_entries = sample_frames(
        video_path=video_path,
        output_dir=out_dir,
        fps=fps,
        max_frames=max_frames,
        start_sec=start_sec,
    )
    report("sampling", 8.0, f"extracted {len(frame_entries)} frames")

    # 3. Detection + recognition
    check_cancel()
    engine_holder = engines if engines is not None else PipelineEngines()
    detector, ocr_engine = engine_holder.get_or_init()

    processed_frames: list[dict[str, Any]] = []
    total_time_ms = 0.0
    best_frame_path: Optional[Path] = None
    best_frame_ts = 0
    max_box_count = -1
    n_frames = max(1, len(frame_entries))

    for i, frame in enumerate(frame_entries):
        check_cancel()
        rel_fpath = frame["image_path"]
        abs_fpath = out_dir / rel_fpath
        fid = frame["frame_id"]
        t_ms = frame["timestamp_ms"]

        img_bgr = cv2.imread(str(abs_fpath))
        if img_bgr is None:
            continue

        t0 = time.perf_counter()
        detections = detector.detect(img_bgr)
        text_lines = ocr_engine.process_detections(img_bgr, detections)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        total_time_ms += elapsed_ms

        if on_frame:
            on_frame(fid, t_ms, text_lines, elapsed_ms)

        processed_frames.append(
            {
                "frame_id": fid,
                "timestamp_ms": t_ms,
                "pts_time": frame.get("pts_time", round(t_ms / 1000.0, 4)),
                "image_path": rel_fpath,
                "elapsed_ms": elapsed_ms,
                "boxes": [line.to_dict() for line in text_lines],
            }
        )

        if len(text_lines) > max_box_count:
            max_box_count = len(text_lines)
            best_frame_path = abs_fpath
            best_frame_ts = t_ms

        report("ocr", 8.0 + 72.0 * (i + 1) / n_frames, f"frame {i + 1}/{n_frames}")

    avg_time_ms = round(total_time_ms / max(1, len(processed_frames)), 2)

    # 4. manifest.json (schema must stay identical to the Phase-0 contract)
    check_cancel()
    report("exporting", 82.0, "writing manifest.json")
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
        out_dir / frame_entries[0]["image_path"] if frame_entries else None
    )
    if target_img and target_img.exists():
        best_ts = best_frame_ts if best_frame_path else (
            frame_entries[0]["timestamp_ms"] if frame_entries else 0
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
