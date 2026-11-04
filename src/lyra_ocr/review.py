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

from lyra_ocr.track import TextTrack


LABEL_COLORS = {
    "lyric": (34, 197, 94),          # Green
    "channel": (239, 68, 68),        # Red
    "song_title": (234, 179, 8),     # Yellow
    "supplementary": (6, 182, 212),  # Cyan
    "scene_text": (217, 70, 239),    # Magenta
    "unknown": (249, 115, 22),       # Orange
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


def load_human_labels(labels_path: str) -> dict[str, dict[str, Any]]:
    """Load human labels from labels.jsonl, indexing latest entry by track_id."""
    if not os.path.exists(labels_path):
        return {}
    labels: dict[str, dict[str, Any]] = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                tid = str(record.get("track_id", ""))
                if tid:
                    labels[tid] = record
            except json.JSONDecodeError:
                continue
    return labels


def append_human_label(
    labels_path: str,
    track_id: str,
    human_label: str,
    human_decision: str,
    annotator: str = "human",
    note: str = "",
) -> None:
    """Append a human annotation record to labels.jsonl."""
    import time
    os.makedirs(os.path.dirname(os.path.abspath(labels_path)), exist_ok=True)
    record = {
        "track_id": str(track_id),
        "human_label": human_label,
        "human_decision": human_decision,
        "annotator": annotator,
        "note": note,
        "timestamp": int(time.time()),
    }
    with open(labels_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_review_csv(
    tracks: list[TextTrack],
    out_path: str,
    labels_jsonl_path: Optional[str] = None,
) -> None:
    """Export human-in-the-loop audit rows to review.csv matching formal data contract."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    human_annotations = load_human_labels(labels_jsonl_path) if labels_jsonl_path else {}

    fieldnames = [
        "track_id",
        "start_ms",
        "end_ms",
        "machine_label",
        "machine_decision",
        "confidence",
        "suggested_text",
        "human_label",
        "human_decision",
        "review_note",
    ]

    rows = []
    for tr in tracks:
        label = tr.classification.label if tr.classification else "unknown"
        confidence = tr.classification.confidence if tr.classification else 0.50
        decision = tr.classification.decision if tr.classification else "review"
        reason = tr.classification.reason if tr.classification else ""

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
                        "human_label": h_annot.get("human_label", ""),
                        "human_decision": h_annot.get("human_decision", ""),
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
                    "human_label": h_annot.get("human_label", ""),
                    "human_decision": h_annot.get("human_decision", ""),
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

    for tr in tracks:
        label = tr.classification.label if tr.classification else "unknown"
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

        # Draw tag header
        tag = f"[{label.upper()}] {box_text}"
        tag_x = max(0, top_left[0])
        tag_y = max(0, top_left[1] - 18)

        # Draw background rectangle for readability
        draw.rectangle([tag_x, tag_y, tag_x + len(tag) * 8 + 6, tag_y + 16], fill=(20, 20, 20))
        draw.text((tag_x + 3, tag_y + 1), tag, fill=color)

    img.save(out_path, quality=92)
