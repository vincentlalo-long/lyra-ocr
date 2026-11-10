from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


def load_synced_lrc(path: str | Path) -> list[tuple[int, str]]:
    p = Path(path)
    if not p.is_file():
        return []

    content = p.read_text(encoding="utf-8")
    results: list[tuple[int, str]] = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        tag_matches = list(re.finditer(r"\[(\d{1,3}):(\d{2})(?:\.(\d{1,3}))?\]", line))
        if not tag_matches:
            continue

        text = re.sub(r"\[.*?\]", "", line).strip()
        if not text:
            continue

        for m in tag_matches:
            mins = int(m.group(1))
            secs = int(m.group(2))
            cs_str = m.group(3) or ""
            if len(cs_str) == 2:
                ms = int(cs_str) * 10
            elif len(cs_str) == 3:
                ms = int(cs_str)
            elif len(cs_str) == 1:
                ms = int(cs_str) * 100
            else:
                ms = 0

            total_ms = (mins * 60 + secs) * 1000 + ms
            results.append((total_ms, text))

    results.sort(key=lambda x: x[0])
    return results


def _clean_plain_lyrics(raw_lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        # Filter section headers like [Verse 1], [Chorus], (Chorus), [Bridge], etc.
        if re.match(r"^\[\s*(verse|chorus|bridge|intro|outro|hook|pre-chorus|drop|breakdown|solo|instrumental).*?\]$", line, re.IGNORECASE):
            continue
        if re.match(r"^\(\s*(verse|chorus|bridge|intro|outro|hook|pre-chorus|drop|breakdown|solo|instrumental).*?\)$", line, re.IGNORECASE):
            continue
        cleaned.append(line)
    return cleaned


def load_plain_lyrics(dataset_dir: str | Path, video_id: str) -> list[str]:
    d = Path(dataset_dir)

    # 1. Check lyrics_cache/<video_id>.json
    cache_json = d / "lyrics_cache" / f"{video_id}.json"
    if cache_json.is_file():
        try:
            data = json.loads(cache_json.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return _clean_plain_lyrics([str(line) for line in data])
            if isinstance(data, dict):
                lines = data.get("lines") or data.get("plain_lyrics") or []
                if isinstance(lines, list):
                    return _clean_plain_lyrics([str(l) for l in lines])
                if isinstance(lines, str):
                    return _clean_plain_lyrics(lines.splitlines())
        except Exception:
            pass

    # 2. Check lyrics/<video_id>.txt
    txt_path = d / "lyrics" / f"{video_id}.txt"
    if txt_path.is_file():
        try:
            content = txt_path.read_text(encoding="utf-8")
            return _clean_plain_lyrics(content.splitlines())
        except Exception:
            pass

    return []
