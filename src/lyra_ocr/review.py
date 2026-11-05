"""Review artifact generators for Phase 1.

Exports structured `tracks.json`, tabular `review.csv`, and annotated
visual audit overlays (`roi_preview.jpg`).
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

from lyra_ocr.labels import (
    LabelRecord,
    append_label_record,
    load_labels_jsonl,
)
from lyra_ocr.track import TextTrack


FONT_CANDIDATES = [
    "/usr/share/fonts/google-noto-sans-mono-cjk-vf-fonts/NotoSansMonoCJK-VF.ttc",
    "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
]


def _find_unicode_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Find a system font that can render Latin/Vietnamese/CJK overlay text."""
    for font_path in FONT_CANDIDATES:
        if os.path.isfile(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


LABEL_COLORS = {
    "lyric": (34, 197, 94),      # Green
    "non_lyric": (239, 68, 68),  # Red
}


def export_tracks_json(tracks: list[TextTrack], out_path: str) -> None:
    """Export all TextTrack instances with features and segments to tracks.json."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    payload = {
        "total_tracks": len(tracks),
        "tracks": [t.to_dict() for t in tracks],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_human_labels(labels_path: str, video_id: Optional[str] = None) -> dict[str, dict[str, Any]]:
    """Load human labels from labels.jsonl, indexing latest entry by track_id."""
    records = load_labels_jsonl(labels_path)
    res: dict[str, dict[str, Any]] = {}
    for (vid, tid), rec in records.items():
        if video_id is None or vid == video_id or not vid:
            res[tid] = rec
    return res


def append_human_label(
    labels_path: str,
    track_id: str,
    human_label: str,
    human_decision: str = "",
    annotator: str = "human",
    note: str = "",
    video_id: str = "unknown_video",
    tracking_error: bool = False,
) -> None:
    """Append a validated human annotation record to labels.jsonl."""
    rec = LabelRecord(
        video_id=video_id,
        track_id=str(track_id),
        human_label=human_label,
        human_decision=human_decision,
        annotator=annotator,
        notes=note,
        tracking_error=tracking_error,
    )
    append_label_record(labels_path, rec)


def export_review_tracks_csv(
    tracks: list[TextTrack],
    out_path: str,
    video_id: str = "unknown_video",
    labels_jsonl_path: Optional[str] = None,
) -> None:
    """Export 1 row per track to review_tracks.csv for track-level Phase 1 ground truth."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    human_annotations = load_human_labels(labels_jsonl_path, video_id=video_id) if labels_jsonl_path else {}

    fieldnames = [
        "video_id",
        "track_id",
        "start_ms",
        "end_ms",
        "duration_ms",
        "dominant_text",
        "machine_label",
        "machine_decision",
        "confidence",
        "machine_reason",
        "qa_flags",
        "suggested_text",
        "human_label",
        "human_decision",
        "corrected_text",
        "tracking_error",
        "notes",
        "annotator",
        "reviewed_at",
    ]

    rows = []
    for tr in tracks:
        tid_str = str(tr.track_id)
        label = tr.classification.label if tr.classification else "non_lyric"
        confidence = tr.classification.confidence if tr.classification else 0.50
        decision = tr.classification.decision if tr.classification else "review"
        reason = tr.classification.reason if tr.classification else ""
        flags = ";".join(tr.classification.qa_flags) if tr.classification else ""
        h_annot = human_annotations.get(tid_str, {})

        dur_ms = max(0, tr.end_ms - tr.start_ms)
        rows.append(
            {
                "video_id": video_id,
                "track_id": tid_str,
                "start_ms": tr.start_ms,
                "end_ms": tr.end_ms,
                "duration_ms": dur_ms,
                "dominant_text": tr.features.get("dominant_text", ""),
                "machine_label": label,
                "machine_decision": decision,
                "confidence": f"{confidence:.2f}",
                "machine_reason": reason,
                "qa_flags": flags,
                "suggested_text": tr.features.get("dominant_text", ""),
                "human_label": h_annot.get("human_label", ""),
                "human_decision": h_annot.get("human_decision", ""),
                "corrected_text": h_annot.get("corrected_text") or "",
                "tracking_error": "true" if h_annot.get("tracking_error") else "false",
                "notes": h_annot.get("notes", ""),
                "annotator": h_annot.get("annotator", ""),
                "reviewed_at": h_annot.get("reviewed_at", ""),
            }
        )

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_review_csv(
    tracks: list[TextTrack],
    out_path: str,
    labels_jsonl_path: Optional[str] = None,
    video_id: Optional[str] = None,
) -> None:
    """Export human-in-the-loop audit rows to review.csv matching formal data contract."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    human_annotations = (
        load_human_labels(labels_jsonl_path, video_id=video_id)
        if labels_jsonl_path
        else {}
    )

    fieldnames = [
        "track_id",
        "start_ms",
        "end_ms",
        "machine_label",
        "machine_decision",
        "confidence",
        "suggested_text",
        "qa_flags",
        "human_label",
        "human_decision",
        "corrected_text",
        "review_note",
    ]

    rows = []
    for tr in tracks:
        label = tr.classification.label if tr.classification else "non_lyric"
        confidence = tr.classification.confidence if tr.classification else 0.50
        decision = tr.classification.decision if tr.classification else "review"
        reason = tr.classification.reason if tr.classification else ""
        flags = ";".join(tr.classification.qa_flags) if tr.classification else ""

        # If a track has distinct segments (e.g. lyrics advancing over time), export each segment
        if tr.segments:
            for seg in tr.segments:
                seg_id = (
                    f"{tr.track_id}"
                    if len(tr.segments) == 1
                    else f"{tr.track_id}.{seg.segment_id}"
                )
                h_annot = human_annotations.get(seg_id) or human_annotations.get(str(tr.track_id), {})
                rows.append(
                    {
                        "track_id": seg_id,
                        "start_ms": seg.start_ms,
                        "end_ms": seg.end_ms,
                        "machine_label": label,
                        "machine_decision": decision,
                        "confidence": f"{confidence:.2f}",
                        "suggested_text": seg.text,
                        "qa_flags": flags,
                        "human_label": h_annot.get("human_label", ""),
                        "human_decision": h_annot.get("human_decision", ""),
                        "corrected_text": h_annot.get("corrected_text") or "",
                        "review_note": reason,
                    }
                )
        else:
            # Fallback for track without parsed segments
            suggested = (
                tr.occurrences[0].text_norm if tr.occurrences else ""
            )
            tr_id_str = f"{tr.track_id}"
            h_annot = human_annotations.get(tr_id_str, {})
            rows.append(
                {
                    "track_id": tr_id_str,
                    "start_ms": tr.start_ms,
                    "end_ms": tr.end_ms,
                    "machine_label": label,
                    "machine_decision": decision,
                    "confidence": f"{confidence:.2f}",
                    "suggested_text": suggested,
                    "qa_flags": flags,
                    "human_label": h_annot.get("human_label", ""),
                    "human_decision": h_annot.get("human_decision", ""),
                    "corrected_text": h_annot.get("corrected_text") or "",
                    "review_note": reason,
                }
            )

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_phase1_overlay(
    image_path: str,
    tracks: list[TextTrack],
    out_path: str,
    target_timestamp_ms: Optional[int] = None,
) -> None:
    """Render an annotated image frame highlighting detected tracks with category color-coding."""
    if not os.path.exists(image_path):
        return

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _find_unicode_font(size=16)

    for tr in tracks:
        label = tr.classification.label if tr.classification else "non_lyric"
        color = LABEL_COLORS.get(label, (255, 255, 255))

        # Check if track is active at this timestamp (if timestamp provided)
        if target_timestamp_ms is not None:
            if not (tr.start_ms <= target_timestamp_ms <= tr.end_ms):
                continue

        # Find occurrence matching target timestamp, or use track bbox
        matching_occ = None
        if target_timestamp_ms is not None:
            for occ in tr.occurrences:
                if occ.timestamp_ms == target_timestamp_ms:
                    matching_occ = occ
                    break

        if matching_occ and matching_occ.polygon:
            poly = [tuple(p) for p in matching_occ.polygon]
            draw.polygon(poly, outline=color, width=3)
            box_text = matching_occ.text_norm
            top_left = poly[0]
        else:
            bbox = tr.bbox
            draw.rectangle(bbox, outline=color, width=3)
            box_text = tr.features.get("dominant_text", "")
            top_left = (bbox[0], bbox[1])

        # Draw tag header with a Unicode-capable font (JP/VI glyphs must render).
        tag = f"[{label.upper()}] {box_text}"
        tag_x = max(0, top_left[0])
        text_bbox = draw.textbbox((0, 0), tag, font=font)
        tag_w = text_bbox[2] - text_bbox[0]
        tag_h = text_bbox[3] - text_bbox[1]
        tag_y = max(0, top_left[1] - tag_h - 6)

        draw.rectangle([tag_x, tag_y, tag_x + tag_w + 6, tag_y + tag_h + 4], fill=(20, 20, 20))
        draw.text((tag_x + 3, tag_y + 1), tag, fill=color, font=font)

    img.save(out_path, quality=92)
