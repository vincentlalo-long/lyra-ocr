from __future__ import annotations

from typing import Any

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


def _track_sort_key(tid: Any) -> tuple[int, int | str]:
    s = str(tid)
    return (0, int(s)) if s.isdigit() else (1, s)


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
            _track_sort_key(c.source_track_ids[0]) if c.source_track_ids else (0, -1),
        ),
    )

    merged: list[LyricCue] = []

    for cue in sorted_cues:
        matched_idx = -1
        for i, existing in enumerate(merged):
            # Cues from different layers must NEVER be merged
            if (
                existing.layer_id is not None
                and cue.layer_id is not None
                and existing.layer_id != cue.layer_id
            ):
                continue

            # Check dedupe criteria. A line sampled at 2 fps can appear as two
            # tracks starting ~1s apart; allow a wider window only for
            # near-identical text so genuine short consecutive lines survive.
            sim = string_similarity(existing.text, cue.text)
            start_delta = abs(existing.start_ms - cue.start_ms)
            time_close = start_delta <= 500 or (
                sim >= 0.92
                and start_delta <= 1500
                and min(existing.end_ms, cue.end_ms)
                - max(existing.start_ms, cue.start_ms)
                >= -500
            )
            if sim >= 0.90 and time_close:
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

            all_tids = list(set(target.source_track_ids + cue.source_track_ids))
            target.source_track_ids = sorted(all_tids, key=_track_sort_key)
            all_sids = list(set(target.source_segment_ids + cue.source_segment_ids))
            target.source_segment_ids = sorted(all_sids, key=_track_sort_key)
            target.review_required = target.review_required or cue.review_required
            target.end_ms = max(target.end_ms, cue.end_ms)
            if target.layer_id is None and cue.layer_id is not None:
                target.layer_id = cue.layer_id
            if (not target.lang or target.lang == "und") and cue.lang and cue.lang != "und":
                target.lang = cue.lang
        else:
            merged.append(cue)

    # 3. Cap end_ms at the next cue's start_ms within the same layer
    from collections import defaultdict

    by_layer: dict[Any, list[LyricCue]] = defaultdict(list)
    for c in merged:
        by_layer[c.layer_id].append(c)

    for layer_id, lcues in by_layer.items():
        lcues_sorted = sorted(lcues, key=lambda c: (c.start_ms, c.end_ms))
        for idx in range(len(lcues_sorted) - 1):
            curr = lcues_sorted[idx]
            next_cue = lcues_sorted[idx + 1]
            if next_cue.start_ms > curr.start_ms and curr.end_ms > next_cue.start_ms:
                curr.end_ms = next_cue.start_ms

    # Re-sort final list
    return sorted(
        merged,
        key=lambda c: (
            c.start_ms,
            c.y_center,
            _track_sort_key(c.source_track_ids[0]) if c.source_track_ids else (0, -1),
        ),
    )
