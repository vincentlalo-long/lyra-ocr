"""CLI dispatcher for lyra-ocr."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont

from .detect import TextDetector
from .frames import sample_frames
from .ocr import OCREngine, TextLine
from .probe import probe_video
from .track import track_ocr_results
from .classify import classify_all_tracks
from .review import export_tracks_json, export_review_csv, render_phase1_overlay


def get_tool_versions() -> dict[str, str]:
    """Inspect installed versions of dependencies and tools."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
    }
    try:
        from importlib.metadata import version
        versions["rapidocr"] = version("rapidocr")
    except Exception:
        versions["rapidocr"] = "unknown"

    try:
        import onnxruntime
        versions["onnxruntime"] = onnxruntime.__version__
    except Exception:
        versions["onnxruntime"] = "unknown"

    try:
        import cv2
        versions["opencv"] = cv2.__version__
    except Exception:
        versions["opencv"] = "unknown"

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        res = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True, check=False)
        first_line = res.stdout.splitlines()[0] if res.stdout else ""
        versions["ffmpeg"] = first_line.split("Copyright")[0].strip()
    else:
        versions["ffmpeg"] = "missing"

    return versions


def _find_unicode_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Find a system font capable of rendering Latin/Vietnamese/CJK text."""
    candidate_fonts = [
        "/usr/share/fonts/google-noto-sans-mono-cjk-vf-fonts/NotoSansMonoCJK-VF.ttc",
        "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    ]
    for font_path in candidate_fonts:
        if os.path.isfile(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_roi_preview(
    image_path: str | Path,
    boxes: list[TextLine],
    output_path: str | Path,
) -> None:
    """Render bounding boxes and text labels onto an image frame for visual audit."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _find_unicode_font(size=max(16, int(img.height * 0.025)))

    for item in boxes:
        poly = [(p[0], p[1]) for p in item.polygon]
        draw.polygon(poly, outline="#22c55e", width=2)

        xmin, ymin, xmax, ymax = item.bbox
        display_text = item.text_norm or item.text_raw
        label = f"{display_text} ({item.confidence:.2f})"

        text_bbox = draw.textbbox((xmin, ymin), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        lbl_y0 = max(0, ymin - th - 4)
        lbl_y1 = lbl_y0 + th + 4
        lbl_x1 = min(img.width, xmin + tw + 6)

        draw.rectangle([xmin, lbl_y0, lbl_x1, lbl_y1], fill="#111827")
        draw.text((xmin + 3, lbl_y0 + 2), label, fill="#f9fafb", font=font)

    img.save(output_path, "JPEG", quality=90)


def run_smoke(args: argparse.Namespace) -> int:
    """Execute smoke test verifying probe, frames (PTS), det-only, and ocr rec."""
    video_path = Path(args.video).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== [Lyra OCR Smoke Test — Phase 0] ===")
    print(f"Video : {video_path}")
    print(f"Output: {out_dir}")

    # 1. Probe video
    print("\n1. Probing video metadata...")
    meta = probe_video(video_path)
    print(
        f"   Resolution: {meta['width']}x{meta['height']} | "
        f"FPS: {meta['fps']} | "
        f"Duration: {meta['duration']}s | "
        f"Audio: {'Yes' if meta['has_audio'] else 'No'}"
    )

    # 2. Extract frames with accurate PTS
    sample_count = args.sample_frames
    start_sec = args.start_sec
    sample_fps = args.fps
    print(f"\n2. Sampling {sample_count} frames (from t={start_sec}s at {sample_fps} fps)...")
    frame_entries = sample_frames(
        video_path=video_path,
        output_dir=out_dir,
        fps=sample_fps,
        max_frames=sample_count,
        start_sec=start_sec,
    )
    print(f"   Extracted {len(frame_entries)} frames into {out_dir / 'frames'}")

    # 3. Detect and Recognize text using separated modules
    print("\n3. Initializing TextDetector (DBNet det-only) & OCREngine (PP-OCRv6 REC)...")
    detector = TextDetector()
    ocr_engine = OCREngine()

    print("   Running detection + recognition per frame...")
    processed_frames: list[dict[str, Any]] = []
    total_time_ms = 0.0
    best_frame_path: Path | None = None
    best_frame_lines: list[TextLine] = []
    max_box_count = -1

    for frame in frame_entries:
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

        print(f"   - Frame {fid} (PTS={t_ms}ms): {len(text_lines)} boxes in {elapsed_ms:.1f}ms")
        for line in text_lines:
            print(f"       [{line.script}] \"{line.text_norm}\" (conf: {line.confidence:.2f}, model: {line.model_id})")

        frame_data = {
            "frame_id": fid,
            "timestamp_ms": t_ms,
            "pts_time": frame.get("pts_time", round(t_ms / 1000.0, 4)),
            "image_path": rel_fpath,
            "elapsed_ms": elapsed_ms,
            "boxes": [line.to_dict() for line in text_lines],
        }
        processed_frames.append(frame_data)

        if len(text_lines) > max_box_count:
            max_box_count = len(text_lines)
            best_frame_path = abs_fpath
            best_frame_lines = text_lines

    avg_time_ms = round(total_time_ms / max(1, len(processed_frames)), 2)
    print(f"\n   Average OCR time per frame: {avg_time_ms}ms (~{1000.0/max(1.0, avg_time_ms):.1f} FPS)")

    # 4. Save manifest.json
    manifest_data = {
        "video_path": str(video_path),
        "probe": meta,
        "config": {
            "sample_fps": sample_fps,
            "start_sec": start_sec,
            "sample_frames": sample_count,
        },
        "tool_versions": get_tool_versions(),
        "models": {
            "det": "PP-OCRv6_det_small",
            "rec": ocr_engine.DEFAULT_MODEL,
        },
        "timestamp_created": int(time.time()),
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    print(f"\n4. Saved manifest data to: {manifest_path}")

    # 5. Save ocr_raw.json
    ocr_raw_data = {
        "video_path": str(video_path),
        "manifest_path": "manifest.json",
        "total_frames_processed": len(processed_frames),
        "avg_frame_time_ms": avg_time_ms,
        "frames": processed_frames,
    }
    raw_json_path = out_dir / "ocr_raw.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(ocr_raw_data, f, indent=2, ensure_ascii=False)
    print(f"5. Saved OCR raw data to: {raw_json_path}")

    # 6. Temporal Tracking (Phase 1)
    print("\n6. Running Temporal Tracking across frames...")
    v_dur_ms = int(meta.get("duration", 0) * 1000)
    tracks = track_ocr_results(
        frames_ocr_data=processed_frames,
        video_duration_ms=v_dur_ms,
        frame_width=meta["width"],
        frame_height=meta["height"],
        iou_threshold=0.30,
        max_gap_ms=2000,
    )
    print(f"   Clustered {len(tracks)} TextTrack stream(s).")

    # 7. Rule-based Classification (Phase 1)
    print("\n7. Classifying TextTracks (Rule Engine)...")
    track_meta = {
        "title": getattr(args, "title", "") or "",
        "channel": getattr(args, "channel", "") or "",
        "duration": meta["duration"],
    }
    classify_all_tracks(
        tracks=tracks,
        metadata=track_meta,
        video_duration_ms=v_dur_ms,
        frame_width=meta["width"],
        frame_height=meta["height"],
    )

    for tr in tracks:
        lbl = tr.classification.label if tr.classification else "unknown"
        dec = tr.classification.decision if tr.classification else "review"
        rsn = tr.classification.reason if tr.classification else ""
        txt = tr.features.get("dominant_text", "")
        print(f"   - Track #{tr.track_id} [{lbl.upper()}] ({dec}): \"{txt}\" — {rsn}")

    # 8. Export Phase 1 Artifacts: tracks.json & review.csv
    tracks_json_path = out_dir / "tracks.json"
    export_tracks_json(tracks, str(tracks_json_path))
    print(f"\n8. Saved tracks data to: {tracks_json_path}")

    review_csv_path = out_dir / "review.csv"
    labels_jsonl_path = out_dir / "labels.jsonl"
    export_review_csv(
        tracks,
        str(review_csv_path),
        labels_jsonl_path=str(labels_jsonl_path) if labels_jsonl_path.exists() else None,
    )
    print(f"9. Saved human review sheet to: {review_csv_path}")

    # 10. Render roi_preview.jpg (Color-coded by classification category)
    preview_path = out_dir / "roi_preview.jpg"
    target_img = best_frame_path or (out_dir / frame_entries[0]["image_path"] if frame_entries else None)
    if target_img and target_img.exists():
        best_ts = frame_entries[0]["timestamp_ms"] if frame_entries else 0
        render_phase1_overlay(
            image_path=str(target_img),
            tracks=tracks,
            out_path=str(preview_path),
            target_timestamp_ms=best_ts,
        )
        print(f"10. Generated color-coded visual audit preview: {preview_path}")

    print("\n=== Lyra OCR Smoke & Identification Completed Successfully! ===")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lyra-ocr", description="Lyra OCR CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    for cmd_name, cmd_help in [
        ("smoke", "Run OCR smoke test on a local video"),
        ("identify", "Run Phase 1 lyric identification and tracking on a video"),
    ]:
        p = subparsers.add_parser(cmd_name, help=cmd_help)
        p.add_argument("video", help="Path to input video file")
        p.add_argument(
            "--out",
            "-o",
            default="./smoke_output",
            help="Output directory (default: ./smoke_output)",
        )
        p.add_argument(
            "--sample-frames",
            "-n",
            type=int,
            default=5,
            help="Number of frames to sample (default: 5)",
        )
        p.add_argument(
            "--start-sec",
            "-s",
            type=float,
            default=0.0,
            help="Seconds offset to start sampling frames (default: 0.0)",
        )
        p.add_argument(
            "--fps",
            type=float,
            default=2.0,
            help="Sample FPS (default: 2.0)",
        )
        p.add_argument(
            "--title",
            default="",
            help="Song title metadata (used for intro title card discrimination)",
        )
        p.add_argument(
            "--channel",
            default="",
            help="Channel/Artist metadata (used for persistent watermark discrimination)",
        )

    args = parser.parse_args(argv)
    if args.command in ("smoke", "identify"):
        return run_smoke(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
