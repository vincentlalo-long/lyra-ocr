"""Frame extraction and PTS indexing using ffmpeg showinfo filter."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def sample_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 2.0,
    max_frames: int | None = None,
    start_sec: float = 0.0,
) -> list[dict[str, Any]]:
    """Extract frames at a given sampling rate, recording accurate Presentation Time Stamps (PTS).

    Args:
        video_path: Path to the video file.
        output_dir: Directory where 'frames/' and 'frame_index.json' are saved.
        fps: Target frame sampling rate (default: 2.0 fps).
        max_frames: Optional upper limit on the number of frames to extract.
        start_sec: Optional start time in seconds (default: 0.0).

    Returns:
        List of frame index entries with frame_id, timestamp_ms, pts_time, and relative image_path.
    """
    video = Path(video_path).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg is not found on PATH")

    out_root = Path(output_dir).resolve()
    frames_dir = out_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cmd = [ffmpeg_bin, "-y"]
    if start_sec > 0:
        cmd.extend(["-ss", f"{start_sec:.3f}"])

    cmd.extend([
        "-i",
        str(video),
        "-vf",
        f"fps={fps},showinfo",
        "-q:v",
        "2",
    ])

    if max_frames and max_frames > 0:
        cmd.extend(["-frames:v", str(max_frames)])

    frame_pattern = str(frames_dir / "frame_%06d.jpg")
    cmd.append(frame_pattern)

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr.strip()}")

    # Collect extracted image files in sorted order
    extracted_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not extracted_files:
        raise RuntimeError("ffmpeg completed but no frames were extracted")

    # Parse PTS from ffmpeg showinfo filter output: n: <n> pts: <pts> pts_time:<pts_time>
    pts_matches = re.findall(r"n:\s*(\d+)\s+pts:\s*(\d+)\s+pts_time:([\d.]+)", result.stderr)
    pts_by_idx: dict[int, float] = {}
    for n_str, _pts_str, pts_time_str in pts_matches:
        try:
            pts_by_idx[int(n_str)] = float(pts_time_str)
        except ValueError:
            pass

    index_entries: list[dict[str, Any]] = []
    step_sec = 1.0 / max(0.1, fps)

    for idx, fpath in enumerate(extracted_files):
        # Prefer exact PTS from showinfo filter; fallback to calculated step
        if idx in pts_by_idx:
            rel_pts = pts_by_idx[idx]
        else:
            rel_pts = idx * step_sec

        t_ms = int(round((start_sec + rel_pts) * 1000.0))
        rel_path = f"frames/{fpath.name}"

        index_entries.append(
            {
                "frame_id": idx,
                "timestamp_ms": t_ms,
                "pts_time": round(start_sec + rel_pts, 4),
                "image_path": rel_path,
            }
        )

    # Save frame_index.json
    index_file = out_root / "frame_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_entries, f, indent=2, ensure_ascii=False)

    return index_entries
