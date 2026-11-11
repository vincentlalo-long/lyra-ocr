from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lyra_ocr.lyric.models import LyricCue, LrcMetadata


def format_lrc_timestamp(ms: int) -> str:
    cs = max(0, int(round(ms / 10.0)))
    minutes = cs // 6000
    rem = cs % 6000
    seconds = rem // 100
    centi = rem % 100
    return f"[{minutes:02d}:{seconds:02d}.{centi:02d}]"


def _sanitize_tag_value(val: str) -> str:
    return val.replace("[", " ").replace("]", " ")


def write_lrc(
    out_path: str | Path,
    cues: list[LyricCue],
    metadata: LrcMetadata,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    header_lines: list[str] = []
    if metadata.title.strip():
        header_lines.append(f"[ti:{_sanitize_tag_value(metadata.title)}]")
    if metadata.artist.strip():
        header_lines.append(f"[ar:{_sanitize_tag_value(metadata.artist)}]")
    if metadata.album.strip():
        header_lines.append(f"[al:{_sanitize_tag_value(metadata.album)}]")
    if metadata.by.strip():
        header_lines.append(f"[by:{_sanitize_tag_value(metadata.by)}]")
    header_lines.append(f"[offset:{metadata.offset_ms}]")

    # Sort cues by (start_ms, y_center)
    sorted_cues = sorted(cues, key=lambda c: (c.start_ms, c.y_center))

    cue_lines: list[str] = []
    seen: set[tuple[int, str]] = set()
    prev_start_ms = 0

    for cue in sorted_cues:
        cleaned_text = cue.text.replace("\r", " ").replace("\n", " ").strip()
        if not cleaned_text:
            continue

        # Monotonic timestamp: ensure start_ms does not decrease
        current_start_ms = max(cue.start_ms, prev_start_ms)
        prev_start_ms = current_start_ms

        cs = max(0, int(round(current_start_ms / 10.0)))
        key = (cs, cleaned_text)
        if key in seen:
            continue
        seen.add(key)

        cue_lines.append(f"{format_lrc_timestamp(current_start_ms)}{cleaned_text}")

    content = "\n".join(header_lines + cue_lines) + "\n"
    out.write_text(content, encoding="utf-8")


def write_cues_json(
    out_path: str | Path,
    cues: list[LyricCue],
    metadata: LrcMetadata,
    *,
    video_id: str,
    video_path: str,
    duration_ms: int,
    tracker: str,
    skipped: list[dict[str, Any]],
    options: dict[str, Any] | None = None,
    has_labels: bool = False,
    reference_source: str = "",
    n_reference_matches: int = 0,
    n_reference_gap_fills: int = 0,
    reference_warnings: list[str] | None = None,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "video_id": video_id,
        "video_path": video_path,
        "duration_ms": duration_ms,
        "tracker": tracker,
        "options": options or {},
        "has_labels": has_labels,
        "reference_source": reference_source,
        "n_reference_matches": n_reference_matches,
        "n_reference_gap_fills": n_reference_gap_fills,
        "reference_warnings": reference_warnings or [],
        "metadata": {
            "title": metadata.title,
            "artist": metadata.artist,
            "album": metadata.album,
            "by": metadata.by,
            "offset_ms": metadata.offset_ms,
        },
        "n_cues": len(cues),
        "n_skipped": len(skipped),
        "review_required": sum(1 for c in cues if c.review_required),
        "cues": [c.to_dict() for c in cues],
        "skipped": skipped,
    }

    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
