"""YouTube channel/playlist listing via yt-dlp flat extraction (no stream URLs)."""

from __future__ import annotations

import re
import time
from typing import Any

import yt_dlp


_CACHE_TTL_SEC = 600.0
_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

_WATCH_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")


def _normalize_url(url: str) -> str:
    return url.strip()


def list_channel_or_playlist(url: str, limit: int = 1000) -> list[dict[str, Any]]:
    """Return [{id, title, duration, url, channel}] for a channel/playlist URL.

    Uses ``extract_flat: in_playlist`` so one request enumerates entries without
    resolving stream URLs (fast even for hundreds of videos). Single watch URLs
    fall back to a one-item extraction.
    """
    url = _normalize_url(url)
    if not url:
        raise ValueError("YouTube URL is required")

    watch_match = _WATCH_RE.search(url)
    if watch_match and "/channel/" not in url and "/playlist" not in url and not url.rstrip("/").endswith(("videos", "streams", "shorts", "live")):
        info = video_from_url(url)
        return [info] if info else []

    cached = _cache.get(url)
    if cached and time.time() - cached[0] < _CACHE_TTL_SEC:
        return cached[1]

    options = {
        "extract_flat": "in_playlist",
        "playlist_items": f"1-{limit}",
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": False,
    }
    items: list[dict[str, Any]] = []
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
        if not info:
            return []
        channel = info.get("channel") or info.get("uploader") or info.get("title", "")
        entries = info.get("entries") or []
        for e in entries:
            if not e:
                continue
            vid = str(e.get("id") or "").strip()
            if vid.startswith(("http://", "https://")):
                m = _WATCH_RE.search(vid)
                if m:
                    vid = m.group(1)
                else:
                    continue
            elif not vid:
                entry_url = str(e.get("url") or e.get("webpage_url") or "")
                m = _WATCH_RE.search(entry_url)
                if m:
                    vid = m.group(1)
                else:
                    continue

            final_url = str(e.get("webpage_url") or e.get("url") or "").strip()
            if not final_url or not final_url.startswith(("http://", "https://")):
                final_url = f"https://www.youtube.com/watch?v={vid}"

            items.append(
                {
                    "id": vid,
                    "title": e.get("title", ""),
                    "duration": e.get("duration"),
                    "url": final_url,
                    "channel": e.get("channel") or channel,
                }
            )

    _cache[url] = (time.time(), items)
    return items


def video_from_url(url: str) -> dict[str, Any] | None:
    """Resolve a single watch URL to {id, title, duration, url, channel}."""
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        return None
    return {
        "id": info.get("id", ""),
        "title": info.get("title", ""),
        "duration": info.get("duration"),
        "url": info.get("webpage_url", url),
        "channel": info.get("channel") or info.get("uploader", ""),
    }
