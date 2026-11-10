from __future__ import annotations

from lyra_ocr.lyric.models import LyricCue
from lyra_ocr.track import string_similarity


def _bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    if not box_a or not box_b or len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h
    if inter <= 0:
        return 0.0

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe_and_group(cues: list[LyricCue]) -> list[LyricCue]:
    if not cues:
        return []

    # 1. Clamp start_ms >= 0
    for cue in cues:
        cue.start_ms = max(0, cue.start_ms)

    # 2. Sort by (start_ms, y_center, track_id)
    sorted_cues = sorted(
        cues,
        key=lambda c: (
            c.start_ms,
            c.y_center,
            c.source_track_ids[0] if c.source_track_ids else 0,
        ),
    )

    merged: list[LyricCue] = []

    for cue in sorted_cues:
        matched_idx = -1
        for i, existing in enumerate(merged):
            # Check dedupe criteria
            sim = string_similarity(existing.text, cue.text)
            if sim >= 0.90 and abs(existing.start_ms - cue.start_ms) <= 500:
                y_diff = abs(existing.y_center - cue.y_center)
                iou = _bbox_iou(existing.bbox, cue.bbox)
                if y_diff <= 0.08 or iou >= 0.3:
                    matched_idx = i
                    break

        if matched_idx >= 0:
            target = merged[matched_idx]
            # Decide whether to upgrade target with cue's text/confidence
            if cue.confidence > target.confidence or (
                len(cue.text) >= len(target.text) + 3 and cue.confidence >= target.confidence - 0.05
            ):
                target.text = cue.text
                target.raw_text = cue.raw_text
                target.confidence = cue.confidence
                target.bbox = cue.bbox
                target.x_center = cue.x_center
                target.y_center = cue.y_center
                target.correction = cue.correction

            target.source_track_ids = sorted(list(set(target.source_track_ids + cue.source_track_ids)))
            target.source_segment_ids = sorted(list(set(target.source_segment_ids + cue.source_segment_ids)))
            target.review_required = target.review_required or cue.review_required
            target.end_ms = max(target.end_ms, cue.end_ms)
        else:
            merged.append(cue)

    # Re-sort final list
    return sorted(
        merged,
        key=lambda c: (
            c.start_ms,
            c.y_center,
            c.source_track_ids[0] if c.source_track_ids else 0,
        ),
    )
