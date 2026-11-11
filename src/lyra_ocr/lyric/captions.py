"""YouTube caption fallback provider for reference lyric text repair.

Captions are strictly a secondary fallback when trusted references (LRCLIB,
manual) are missing. Auto-translated captions for non-English originals are
never used, and confidence is marked lower than curated lyrics.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from difflib import SequenceMatcher
from typing import Any

from lyra_ocr.lyric.reference import ReferenceBundle, ReferenceLine

logger = logging.getLogger(__name__)

DEFAULT_CAPTION_TIMEOUT = 6.0
_YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_URL_ID_RE = re.compile(
    r"(?:youtu\.be/|v=|/watch\?v=|/shorts/|/embed/)([A-Za-z0-9_-]{11})"
)
_YT_BRACKET_ID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]")

_MUSIC_TAG_RE = re.compile(r"\[(?:music|applause|cheering|laughter|instrumental)\]", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_VTT_TIMESTAMP_RE = re.compile(
    r"(?:(\d{1,2}):)?(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(?:(\d{1,2}):)?(\d{2}):(\d{2})[.,](\d{3})"
)


def resolve_caption_video_ref(
    source_url: str = "",
    video_id: str = "",
    manifest: dict[str, Any] | None = None,
) -> str:
    """Resolve a YouTube URL or 11-char ID, never a local media path."""
    manifest = manifest if isinstance(manifest, dict) else {}
    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}

    url_candidates = [
        source_url,
        manifest.get("source_url"),
        manifest.get("webpage_url"),
        manifest.get("original_url"),
    ]
    for candidate in url_candidates:
        value = str(candidate or "").strip()
        if value and ("youtube.com" in value or "youtu.be" in value):
            return value

    text_candidates = [
        video_id,
        manifest.get("video_id"),
        manifest.get("video_path"),
        metadata.get("title"),
    ]
    for candidate in text_candidates:
        value = str(candidate or "").strip()
        if not value:
            continue
        if _YT_ID_RE.match(value):
            return value
        match = _YT_URL_ID_RE.search(value) or _YT_BRACKET_ID_RE.search(value)
        if match:
            return match.group(1)
    return ""


def _clean_caption_text(text: str) -> str:
    cleaned = _HTML_TAG_RE.sub("", text or "")
    cleaned = _MUSIC_TAG_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _vtt_time_to_ms(hours: str | None, mins: str, secs: str, ms: str) -> int:
    h = int(hours or "0")
    m = int(mins)
    s = int(secs)
    millis = int(ms)
    return h * 3600000 + m * 60000 + s * 1000 + millis


def parse_caption_vtt(text: str) -> list[ReferenceLine]:
    """Parse WebVTT subtitle text into cleaned ReferenceLine entries."""
    lines: list[ReferenceLine] = []
    blocks = re.split(r"\n\s*\n", text or "")

    for block in blocks:
        block_lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not block_lines:
            continue

        ts_idx = -1
        ts_match = None
        for idx, line in enumerate(block_lines):
            m = _VTT_TIMESTAMP_RE.search(line)
            if m:
                ts_idx = idx
                ts_match = m
                break

        if ts_match is None or ts_idx == -1:
            continue

        start_ms = _vtt_time_to_ms(
            ts_match.group(1), ts_match.group(2), ts_match.group(3), ts_match.group(4)
        )
        end_ms = _vtt_time_to_ms(
            ts_match.group(5), ts_match.group(6), ts_match.group(7), ts_match.group(8)
        )

        content_lines = block_lines[ts_idx + 1 :]
        combined = " ".join(content_lines)
        clean = _clean_caption_text(combined)
        if not clean:
            continue

        # Skip adjacent duplicate text
        if lines and lines[-1].text == clean:
            continue

        lines.append(
            ReferenceLine(
                text=clean,
                start_ms=start_ms,
                end_ms=end_ms,
                source="youtube_captions",
                confidence=0.8,
            )
        )

    return lines


def parse_caption_json3(data: dict[str, Any]) -> list[ReferenceLine]:
    """Parse YouTube JSON3 events into deduplicated ReferenceLine entries."""
    raw_lines: list[ReferenceLine] = []
    events = data.get("events") or []

    for event in events:
        if not isinstance(event, dict):
            continue
        segs = event.get("segs") or []
        combined = "".join(s.get("utf8", "") for s in segs if isinstance(s, dict))
        clean = _clean_caption_text(combined)
        if not clean:
            continue

        start_ms = int(event.get("tStartMs", 0))
        duration_ms = int(event.get("dDurationMs", 0))
        end_ms = start_ms + duration_ms if duration_ms > 0 else None

        raw_lines.append(
            ReferenceLine(
                text=clean,
                start_ms=start_ms,
                end_ms=end_ms,
                source="youtube_captions",
                confidence=0.8,
            )
        )

    # Deduplicate rolling ASR chunks. YouTube auto-captions often repeat the
    # same rolling window with small word substitutions, so exact prefix/suffix
    # checks alone leave duplicate lyric fragments behind.
    deduped: list[ReferenceLine] = []
    for line in raw_lines:
        if not deduped:
            deduped.append(line)
            continue

        prev = deduped[-1]
        p_text = prev.text.strip().lower()
        c_text = line.text.strip().lower()
        gap_ms = (
            line.start_ms - prev.start_ms
            if line.start_ms is not None and prev.start_ms is not None
            else None
        )
        close_in_time = gap_ms is not None and 0 <= gap_ms < 4000

        if p_text == c_text:
            continue
        if close_in_time and p_text.endswith(c_text):
            continue
        if close_in_time and c_text.startswith(p_text):
            deduped[-1] = line
            continue
        if (
            close_in_time
            and len(p_text.split()) >= 3
            and len(c_text.split()) >= 3
            and SequenceMatcher(None, p_text, c_text).ratio() >= 0.85
        ):
            if len(c_text) > len(p_text):
                deduped[-1] = line
            continue

        deduped.append(line)

    return deduped


def fetch_youtube_caption_bundle(
    video_url_or_id: str,
    *,
    timeout: float = DEFAULT_CAPTION_TIMEOUT,
) -> ReferenceBundle | None:
    """Extract and parse official or automatic YouTube captions."""
    if not (video_url_or_id or "").strip():
        return None

    try:
        import yt_dlp
    except ImportError:
        logger.warning("yt-dlp is not installed; cannot fetch YouTube captions")
        return None

    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        ydl = yt_dlp.YoutubeDL(ydl_opts)
        info = ydl.extract_info(video_url_or_id, download=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to extract YouTube info for captions %s: %s", video_url_or_id, exc)
        return None

    if not isinstance(info, dict):
        return None

    orig_lang = (info.get("language") or "").strip().lower()
    auto_caps = info.get("automatic_captions") or {}
    manual_subs = info.get("subtitles") or {}

    # Disallow auto-translated 'en' captions for non-English original
    if orig_lang and orig_lang not in ("en", "en-us", "en-gb", "und", ""):
        # For non-English songs, we only allow captions in the original language
        caps_pool = {k: v for k, v in {**auto_caps, **manual_subs}.items() if k.lower().startswith(orig_lang[:2])}
    else:
        caps_pool = {**auto_caps, **manual_subs}

    if not caps_pool:
        return None

    # Priority of language selection
    chosen_entries = None
    chosen_lang = ""
    candidate_langs = []
    if orig_lang:
        candidate_langs.append(orig_lang)
    candidate_langs.extend(["en-orig", "en"])

    for lang_code in candidate_langs:
        for k, v in caps_pool.items():
            if k.lower() == lang_code:
                chosen_entries = v
                chosen_lang = k
                break
        if chosen_entries:
            break

    if not chosen_entries:
        # Fallback to any remaining language entry in pool
        for k, v in caps_pool.items():
            chosen_entries = v
            chosen_lang = k
            break

    if not chosen_entries or not isinstance(chosen_entries, list):
        return None

    # Pick format: prefer json3, then vtt
    target_item = None
    for item in chosen_entries:
        if isinstance(item, dict) and item.get("ext") == "json3":
            target_item = item
            break
    if not target_item:
        for item in chosen_entries:
            if isinstance(item, dict) and item.get("ext") == "vtt":
                target_item = item
                break
    if not target_item:
        target_item = chosen_entries[0] if isinstance(chosen_entries[0], dict) else None

    if not target_item or not target_item.get("url"):
        return None

    cap_url = target_item["url"]
    ext = target_item.get("ext", "")

    try:
        req = urllib.request.Request(cap_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_bytes = resp.read()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch caption data from %s: %s", cap_url, exc)
        return None

    lines: list[ReferenceLine] = []
    if ext == "json3" or cap_url.endswith(".json3"):
        try:
            data = json.loads(content_bytes.decode("utf-8"))
            lines = parse_caption_json3(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    else:
        text = content_bytes.decode("utf-8", errors="replace")
        lines = parse_caption_vtt(text)

    if not lines:
        return None

    return ReferenceBundle(
        lines=lines,
        source="youtube_captions",
        provider="youtube",
        language=chosen_lang,
    )
