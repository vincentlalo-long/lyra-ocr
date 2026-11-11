"""YouTube video downloader adapted from Lyra's yt-dlp pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import yt_dlp


ProgressCallback = Callable[[dict[str, Any]], None]


def _format_bytes(value: float | int | None) -> str:
    if not value:
        return ""
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return ""


def download_video(
    url: str,
    output_dir: str,
    progress: ProgressCallback | None = None,
    max_height: int | None = None,
) -> dict[str, Any]:
    """Download one YouTube video as a playable MP4.

    max_height caps video resolution (e.g. 720) to save disk/OCR time.
    """
    url = url.strip()
    if not url:
        raise ValueError("YouTube URL is required")

    destination = Path(os.path.abspath(os.path.expanduser(output_dir)))
    destination.mkdir(parents=True, exist_ok=True)

    def emit(event: dict[str, Any]) -> None:
        if progress:
            progress(event)

    def progress_hook(data: dict[str, Any]) -> None:
        status = data.get("status")
        if status == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
            downloaded = data.get("downloaded_bytes") or 0
            percent = round(downloaded / total * 100, 1) if total else 0
            emit(
                {
                    "stage": "downloading",
                    "percent": percent,
                    "speed": _format_bytes(data.get("speed")) + "/s"
                    if data.get("speed")
                    else "",
                    "eta": data.get("eta"),
                }
            )
        elif status == "finished":
            emit({"stage": "processing", "message": "Merging video and audio into MP4"})

    if max_height:
        fmt = (
            f"bestvideo[height<={max_height}]+bestaudio"
            f"/best[height<={max_height}]/best"
        )
    else:
        fmt = "bestvideo+bestaudio/best"


    options: dict[str, Any] = {
        # Prefer separate MP4/M4A streams, then fall back to a ready MP4.
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": str(destination / "%(title)s [%(id)s].%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "progress_hooks": [progress_hook],
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "web"],
            }
        },
    }

    emit({"stage": "resolving", "message": "Reading YouTube video information"})
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            raise RuntimeError("YouTube returned no video information")
        filename = Path(ydl.prepare_filename(info))
        if filename.suffix.lower() != ".mp4":
            filename = filename.with_suffix(".mp4")

    if not filename.is_file():
        # yt-dlp's prepare_filename can report a pre-merge fragment name; the
        # merged file is the one whose name ends with "[<video id>].mp4".
        vid = str(info.get("id", ""))
        candidates = sorted(
            p for p in destination.glob("*.mp4") if p.name.endswith(f"[{vid}].mp4")
        ) if vid else []
        if candidates:
            filename = candidates[-1]
        else:
            raise RuntimeError("Download finished but the MP4 file was not found")

    result = {
        "video_path": str(filename),
        "title": info.get("title", filename.stem),
        "id": info.get("id", ""),
        "duration": info.get("duration"),
        "webpage_url": info.get("webpage_url", url),
        "height": info.get("height"),
        "format_id": info.get("format_id"),
    }

    emit({"stage": "complete", **result})
    return result

