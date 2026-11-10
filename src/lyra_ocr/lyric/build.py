from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from lyra_ocr.lyric.export import write_cues_json, write_lrc
from lyra_ocr.lyric.grouping import dedupe_and_group
from lyra_ocr.lyric.models import LrcMetadata
from lyra_ocr.lyric.select import load_artifacts, select_cues

logger = logging.getLogger(__name__)


def build_lrc_from_artifacts(
    artifacts_dir: str | Path,
    out_lrc: str | Path,
    *,
    video_path: str | Path | None = None,
    include_review: bool = False,
    correct_method: str = "dictionary",
    refine: bool = False,
    offset_ms: int = 0,
    title: str | None = None,
    artist: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    art_path = Path(artifacts_dir)
    out_path = Path(out_lrc)

    cues_json_path = out_path.with_suffix(".cues.json")
    if not force and out_path.exists() and cues_json_path.exists():
        try:
            cached_cues = json.loads(cues_json_path.read_text(encoding="utf-8"))
            return {
                "video_id": cached_cues.get("video_id", out_path.stem),
                "out_lrc": str(out_path),
                "out_cues_json": str(cues_json_path),
                "n_cues": len(cached_cues.get("cues", [])),
                "n_skipped": len(cached_cues.get("skipped", [])),
                "review_required": sum(1 for c in cached_cues.get("cues", []) if c.get("review_required")),
                "skipped": cached_cues.get("skipped", []),
            }
        except Exception:
            pass

    manifest, tracks, human_labels = load_artifacts(art_path)

    if not tracks:
        raise ValueError(
            f"No tracks found in artifacts directory '{art_path}'. "
            "Please run the Lyra OCR pipeline first or provide valid artifacts."
        )

    video_id = manifest.get("video_id") or art_path.name
    eff_video_path = str(video_path) if video_path else (manifest.get("video_path") or "")
    duration_ms = int(manifest.get("probe", {}).get("duration", 0) * 1000)
    tracker_name = manifest.get("tracker", {}).get("name", "bytetrack")

    m_title = title if title is not None else manifest.get("metadata", {}).get("title", "")
    m_artist = artist if artist is not None else manifest.get("metadata", {}).get("channel", "")

    metadata = LrcMetadata(
        title=m_title or "",
        artist=m_artist or "",
        offset_ms=offset_ms,
    )

    frame_width = manifest.get("probe", {}).get("width", 1280)
    frame_height = manifest.get("probe", {}).get("height", 720)
    cues, skipped = select_cues(
        tracks,
        human_labels,
        include_review=include_review,
        video_id=video_id,
        correct_method=correct_method,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    if refine:
        try:
            from lyra_ocr.lyric.timing import refine_timings
            cues = refine_timings(cues, eff_video_path, manifest, str(art_path), enabled=True)
        except Exception as e:
            logger.warning("Refine timings failed, falling back to coarse: %s", e)

    cues = dedupe_and_group(cues)

    write_lrc(out_path, cues, metadata)

    cues_json_path = out_path.with_suffix(".cues.json")
    write_cues_json(
        cues_json_path,
        cues,
        metadata,
        video_id=video_id,
        video_path=eff_video_path,
        duration_ms=duration_ms,
        tracker=tracker_name,
        skipped=skipped,
    )

    return {
        "video_id": video_id,
        "out_lrc": str(out_path),
        "out_cues_json": str(cues_json_path),
        "n_cues": len(cues),
        "n_skipped": len(skipped),
        "review_required": sum(1 for c in cues if c.review_required),
        "skipped": skipped,
    }


def build_lrc_from_video(
    video_path: str | Path,
    out_lrc: str | Path,
    *,
    artifacts_dir: str | Path | None = None,
    tracker: str = "bytetrack",
    fps: float = 2.0,
    start_sec: float = 0.0,
    max_frames: int | None = None,
    title: str = "",
    channel: str = "",
    device: str = "auto",
    rec_batch_size: int = 8,
    dedup: bool = False,
    frame_source: str = "pipe",
    force: bool = False,
    **build_kwargs: Any,
) -> dict[str, Any]:
    v_path = Path(video_path)
    if artifacts_dir is None:
        art_dir = Path("./lyra_output") / v_path.stem
    else:
        art_dir = Path(artifacts_dir)

    tracks_json_path = art_dir / "tracks.json"
    manifest_json_path = art_dir / "manifest.json"

    needs_pipeline = force or not (tracks_json_path.exists() and manifest_json_path.exists())

    if not needs_pipeline:
        try:
            manifest_data = json.loads(manifest_json_path.read_text(encoding="utf-8"))
            # 1. Tracker check
            manifest_tracker = manifest_data.get("tracker", {}).get("name")
            if manifest_tracker is not None and manifest_tracker != tracker:
                needs_pipeline = True

            # 2. Config check
            manifest_cfg = manifest_data.get("config", {})
            if "sample_fps" in manifest_cfg and abs(float(manifest_cfg["sample_fps"]) - float(fps)) > 1e-4:
                needs_pipeline = True
            if "start_sec" in manifest_cfg and abs(float(manifest_cfg.get("start_sec", 0.0)) - float(start_sec)) > 1e-4:
                needs_pipeline = True
            if "sample_frames" in manifest_cfg and manifest_cfg.get("sample_frames") != max_frames:
                needs_pipeline = True
            if "dedup" in manifest_cfg and bool(manifest_cfg.get("dedup")) != dedup:
                needs_pipeline = True

            # 3. Config hash check if available
            manifest_hash = manifest_data.get("config_hash")
            if manifest_hash and not needs_pipeline:
                try:
                    from lyra_ocr.pipeline import _config_hash, resolve_runtime
                    runtime = resolve_runtime(device, rec_batch_size)
                    expected_hash = _config_hash(
                        fps=fps,
                        start_sec=start_sec,
                        max_frames=max_frames,
                        tracker=tracker,
                        title=title,
                        channel=channel,
                        device=runtime.device,
                        rec_batch_size=runtime.rec_batch_size,
                        dedup=dedup,
                        frame_source=frame_source,
                    )
                    if manifest_hash != expected_hash:
                        needs_pipeline = True
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Manifest config check failed: %s", e)
            needs_pipeline = True

    if needs_pipeline:
        if not v_path.exists():
            raise FileNotFoundError(f"Video file not found: {v_path}")

        from lyra_ocr.pipeline import run_phase1_pipeline

        run_phase1_pipeline(
            video_path=str(v_path),
            out_dir=str(art_dir),
            fps=fps,
            start_sec=start_sec,
            max_frames=max_frames,
            device=device,
            rec_batch_size=rec_batch_size,
            tracker=tracker,
            title=title,
            channel=channel,
            dedup=dedup,
            frame_source=frame_source,
            force=force,
        )

    return build_lrc_from_artifacts(
        artifacts_dir=art_dir,
        out_lrc=out_lrc,
        title=title or None,
        artist=channel or None,
        force=force,
        **build_kwargs,
    )
