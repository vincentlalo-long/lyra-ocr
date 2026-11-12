from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from lyra_ocr.lang import vote_layer_lang
from lyra_ocr.lyric.grouping import consolidate_language_layers
from lyra_ocr.lyric.models import LyricCue, LrcMetadata



def format_lrc_timestamp(ms: int) -> str:
    cs = max(0, int(round(ms / 10.0)))
    minutes = cs // 6000
    rem = cs % 6000
    seconds = rem // 100
    centi = rem % 100
    return f"[{minutes:02d}:{seconds:02d}.{centi:02d}]"


def format_enhanced_tag(ms: int) -> str:
    return format_lrc_timestamp(ms).replace("[", "<").replace("]", ">")


def _sanitize_tag_value(val: str) -> str:
    return val.replace("[", " ").replace("]", " ")


def write_lrc(
    out_path: str | Path,
    cues: list[LyricCue],
    metadata: LrcMetadata,
    *,
    enhanced: bool = False,
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

        if enhanced and cue.words:
            tagged_words = []
            for w in cue.words:
                w_tag = format_enhanced_tag(w.get("start_ms", current_start_ms))
                w_text = str(w.get("word", "")).strip()
                if w_text:
                    tagged_words.append(f"{w_tag}{w_text}")
            line_content = " ".join(tagged_words) if tagged_words else cleaned_text
        else:
            line_content = cleaned_text

        cue_lines.append(f"{format_lrc_timestamp(current_start_ms)}{line_content}")

    content = "\n".join(header_lines + cue_lines) + "\n"
    out.write_text(content, encoding="utf-8")


def _layer_priority_key(lang: str, n_cues: int) -> tuple[int, int, str]:
    """Sort key for choosing primary layer. Lower tuple is higher priority:
    1. 'ja-Latn' (priority 0)
    2. Non-English original language with the most cues (priority 1, -n_cues)
    3. 'en' (priority 2, -n_cues)
    4. Other / 'und' (priority 3, -n_cues)
    """
    if lang == "ja-Latn":
        return (0, -n_cues, lang)
    if lang not in ("en", "und"):
        return (1, -n_cues, lang)
    if lang == "en":
        return (2, -n_cues, lang)
    return (3, -n_cues, lang)


def write_lrc_layers(
    out_path: str | Path,
    cues: list[LyricCue],
    metadata: LrcMetadata,
    *,
    enhanced: bool = False,
) -> dict[str, Any]:
    """Write name.lrc (primary) + name.<lang>.lrc per lyric layer.

    Returns:
    {
        "primary": {"lang": ..., "path": ...},
        "layers": [{"layer_id": ..., "lang": ..., "path": ..., "n_cues": ...}]
    }
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not cues:
        write_lrc(out, [], metadata, enhanced=enhanced)
        return {
            "primary": {"lang": "und", "path": str(out)},
            "layers": [
                {
                    "layer_id": 0,
                    "lang": "und",
                    "path": str(out),
                    "n_cues": 0,
                }
            ],
        }

    cues = consolidate_language_layers(cues)

    # Group cues by layer_id
    by_layer: dict[Any, list[LyricCue]] = defaultdict(list)
    for c in cues:
        by_layer[c.layer_id].append(c)

    # Filter out trivial/ghost layers when major lyric layers exist
    max_cues_in_layer = max(len(c) for c in by_layer.values()) if by_layer else 0
    if max_cues_in_layer >= 8:
        significant_layers = {
            lid: lcues
            for lid, lcues in by_layer.items()
            if len(lcues) >= 4 or len(lcues) >= 0.15 * max_cues_in_layer
        }
        if significant_layers:
            by_layer = significant_layers

    # If only 1 layer, write single name.lrc without .<lang>.lrc
    if len(by_layer) <= 1:
        single_layer_id = next(iter(by_layer)) if by_layer else 0
        single_cues = by_layer[single_layer_id] if by_layer else []
        single_lang = vote_layer_lang([c.lang for c in single_cues if c.lang]) or "und"
        write_lrc(out, single_cues, metadata, enhanced=enhanced)
        return {
            "primary": {"lang": single_lang, "path": str(out)},
            "layers": [
                {
                    "layer_id": single_layer_id if single_layer_id is not None else 0,
                    "lang": single_lang,
                    "path": str(out),
                    "n_cues": len(single_cues),
                }
            ],
        }

    # Consolidate by language so each language produces exactly 1 .<lang>.lrc file (no -2, -3 suffixes)
    by_lang_cues: dict[str, list[LyricCue]] = defaultdict(list)
    lang_to_layer_id: dict[str, Any] = {}
    for lid in sorted(by_layer.keys(), key=lambda x: (0, x) if isinstance(x, int) else (1, str(x))):
        lcues = by_layer[lid]
        llang = vote_layer_lang([c.lang for c in lcues if c.lang]) or "und"
        if llang not in lang_to_layer_id:
            lang_to_layer_id[llang] = lid
        by_lang_cues[llang].extend(lcues)

    layer_items: list[dict[str, Any]] = []
    for llang, lcues in by_lang_cues.items():
        lcues.sort(key=lambda c: (c.start_ms, c.end_ms))
        layer_file = out.parent / f"{out.stem}.{llang}.lrc"
        write_lrc(layer_file, lcues, metadata, enhanced=enhanced)
        layer_items.append({
            "layer_id": lang_to_layer_id[llang],
            "lang": llang,
            "path": str(layer_file),
            "n_cues": len(lcues),
            "cues": lcues,
        })

    # 2. Select primary layer according to priority
    sorted_by_priority = sorted(
        layer_items,
        key=lambda it: _layer_priority_key(it["lang"], it["n_cues"]),
    )
    primary_layer = sorted_by_priority[0]

    # 3. Write primary name.lrc
    write_lrc(out, primary_layer["cues"], metadata, enhanced=enhanced)

    return {
        "primary": {"lang": primary_layer["lang"], "path": str(out)},
        "layers": [
            {
                "layer_id": it["layer_id"],
                "lang": it["lang"],
                "path": it["path"],
                "n_cues": it["n_cues"],
            }
            for it in layer_items
        ],
    }


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
    layers: list[dict[str, Any]] | None = None,
    primary_lang: str = "",
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
        "primary_lang": primary_lang,
        "layers": layers or [],
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

    def _json_default(obj: Any) -> Any:
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    out.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")

