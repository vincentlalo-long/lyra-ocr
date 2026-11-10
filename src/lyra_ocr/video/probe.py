"""Video metadata inspection using ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def probe_video(video_path: str | Path) -> dict[str, Any]:
    """Inspect video file metadata using ffprobe.

    Returns:
        dict containing duration (seconds), fps, width, height, rotation, has_audio.
    """
    path = Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        raise RuntimeError("ffprobe is not found on PATH")

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed with error: {result.stderr.strip()}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse ffprobe JSON output: {exc}") from exc

    streams = data.get("streams", [])
    format_info = data.get("format", {})

    video_stream: dict[str, Any] | None = None
    has_audio = False

    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = stream
        elif codec_type == "audio":
            has_audio = True

    if not video_stream:
        raise RuntimeError(f"No video stream found in {path}")

    # Calculate FPS
    fps = 0.0
    r_frame_rate = video_stream.get("r_frame_rate", "")
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/", 1)
        try:
            den_val = float(den)
            if den_val > 0:
                fps = round(float(num) / den_val, 3)
        except ValueError:
            pass
    if fps <= 0:
        avg_frame_rate = video_stream.get("avg_frame_rate", "")
        if "/" in avg_frame_rate:
            num, den = avg_frame_rate.split("/", 1)
            try:
                den_val = float(den)
                if den_val > 0:
                    fps = round(float(num) / den_val, 3)
            except ValueError:
                pass

    # Duration: prefer format.duration, fallback to video stream duration
    duration = 0.0
    raw_duration = format_info.get("duration") or video_stream.get("duration")
    if raw_duration:
        try:
            duration = round(float(raw_duration), 3)
        except ValueError:
            pass

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    # Rotation: check tags or side_data_list
    rotation = 0
    tags = video_stream.get("tags", {})
    if "rotate" in tags:
        try:
            rotation = int(float(tags["rotate"])) % 360
        except ValueError:
            pass
    if rotation == 0:
        for side_data in video_stream.get("side_data_list", []):
            if "rotation" in side_data:
                try:
                    rotation = int(float(side_data["rotation"])) % 360
                    break
                except ValueError:
                    pass

    # When video is rotated 90 or 270 degrees, ffmpeg auto-rotates decoded frames,
    # swapping display width and height. Swap dimensions to match decoded frames.
    if rotation in (90, 270):
        width, height = height, width

    return {
        "video_path": str(path),
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "rotation": rotation,
        "has_audio": has_audio,
    }
