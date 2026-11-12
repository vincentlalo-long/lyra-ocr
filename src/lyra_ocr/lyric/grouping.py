from __future__ import annotations

import re
from collections import Counter, defaultdict
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


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _texts_match(a: str, b: str) -> bool:
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    # OCR often drops the tail of a line on one of two duplicate bands.
    if len(shorter) >= 8 and shorter in longer:
        return True
    return string_similarity(a, b) >= 0.90


def _text_in(texts: list[str], needle: str) -> bool:
    return any(_texts_match(needle, t) for t in texts)


def _layer_lang(cues: list[LyricCue]) -> str:
    counts = Counter(c.lang for c in cues if c.lang and c.lang != "und")
    return counts.most_common(1)[0][0] if counts else "und"


def merge_duplicate_layers(
    cues: list[LyricCue],
    *,
    min_containment: float = 0.70,
    max_y_gap: float = 0.12,
) -> list[LyricCue]:
    """Collapse spatial bands that render the same lyric stream (e.g. a two-line
    display split into two tracks) into a single layer before export.

    A layer is treated as a duplicate when it sits within ``max_y_gap`` of
    another layer of the same language and at least ``min_containment`` of its
    distinct lines also occur in that layer. The most complete layer is kept;
    cues from the duplicate band that only repeat lines already present are
    dropped, and genuinely new lines are moved onto the kept layer.
    """
    if not cues:
        return []

    by_layer: dict[Any, list[LyricCue]] = defaultdict(list)
    for cue in cues:
        if cue.layer_id is not None:
            by_layer[cue.layer_id].append(cue)

    layer_ids = list(by_layer.keys())
    if len(layer_ids) < 2:
        return cues

    profiles: dict[Any, dict[str, Any]] = {}
    for lid, lcues in by_layer.items():
        texts: list[str] = []
        for cue in lcues:
            norm = _normalize_text(cue.text)
            if norm and norm not in texts:
                texts.append(norm)
        profiles[lid] = {
            "texts": texts,
            "lang": _layer_lang(lcues),
            "y": sum(c.y_center for c in lcues) / len(lcues),
        }

    parent = {lid: lid for lid in layer_ids}

    def find(lid: Any) -> Any:
        while parent[lid] != lid:
            parent[lid] = parent[parent[lid]]
            lid = parent[lid]
        return lid

    def union(a: Any, b: Any) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(layer_ids):
        for b in layer_ids[i + 1 :]:
            pa, pb = profiles[a], profiles[b]
            if not pa["texts"] or not pb["texts"]:
                continue
            if pa["lang"] != pb["lang"] and "und" not in (pa["lang"], pb["lang"]):
                continue
            if abs(pa["y"] - pb["y"]) > max_y_gap:
                continue
            smaller, larger = (pa, pb) if len(pa["texts"]) <= len(pb["texts"]) else (pb, pa)
            matched = sum(1 for t in smaller["texts"] if _text_in(larger["texts"], t))
            if matched / len(smaller["texts"]) >= min_containment:
                union(a, b)

    components: dict[Any, list[Any]] = defaultdict(list)
    for lid in layer_ids:
        components[find(lid)].append(lid)

    dropped: set[int] = set()
    for members in components.values():
        if len(members) < 2:
            continue
        keeper = max(
            members,
            key=lambda lid: (
                len(profiles[lid]["texts"]),
                len(by_layer[lid]),
                -profiles[lid]["y"],
            ),
        )
        keeper_texts = profiles[keeper]["texts"]
        keeper_lang = profiles[keeper]["lang"]
        for lid in members:
            if lid == keeper:
                continue
            for cue in by_layer[lid]:
                norm = _normalize_text(cue.text)
                if norm and _text_in(keeper_texts, norm):
                    dropped.add(id(cue))
                    continue
                cue.layer_id = keeper
                if not cue.lang or cue.lang == "und":
                    cue.lang = keeper_lang

    if not dropped:
        return cues
    return [c for c in cues if id(c) not in dropped]


def consolidate_language_layers(
    cues: list[LyricCue],
    *,
    overlap_threshold: float = 0.25,
) -> list[LyricCue]:
    """Consolidate layers so that each distinct language has at most ONE layer.

    - First applies merge_duplicate_layers to collapse duplicate bands of the same text stream.
    - Then for remaining layers of the same language:
      - If they are non-overlapping/complementary (e.g. ad-libs, alternating lines),
        merge the secondary layer into the primary layer.
      - If they overlap significantly, keep the dominant layer and filter noisy/duplicate cues.
    """
    if not cues:
        return []

    cues = merge_duplicate_layers(cues)

    by_layer: dict[Any, list[LyricCue]] = defaultdict(list)
    for c in cues:
        if c.layer_id is not None:
            by_layer[c.layer_id].append(c)

    if len(by_layer) < 2:
        return cues

    # Group layers by voted language
    layers_by_lang: dict[str, list[Any]] = defaultdict(list)
    for lid, lcues in by_layer.items():
        lang = _layer_lang(lcues)
        layers_by_lang[lang].append(lid)

    dropped_cues: set[int] = set()

    for lang, lids in layers_by_lang.items():
        if len(lids) < 2 or lang == "und":
            continue

        # Sort layers by cue count descending
        sorted_lids = sorted(lids, key=lambda lid: len(by_layer[lid]), reverse=True)
        dominant_lid = sorted_lids[0]
        dom_cues = by_layer[dominant_lid]

        for sec_lid in sorted_lids[1:]:
            sec_cues = by_layer[sec_lid]
            if not sec_cues:
                continue

            # Compute temporal overlap
            n_overlapping = 0
            for sc in sec_cues:
                for dc in dom_cues:
                    if max(sc.start_ms, dc.start_ms) < min(sc.end_ms, dc.end_ms) - 200:
                        n_overlapping += 1
                        break
            overlap_ratio = n_overlapping / len(sec_cues)
            dom_texts = [_normalize_text(c.text) for c in dom_cues]

            if overlap_ratio <= overlap_threshold:
                # Complementary non-overlapping layer: merge completely into dominant layer
                for sc in sec_cues:
                    sc.layer_id = dominant_lid
                    sc.lang = lang
                dom_cues.extend(sec_cues)
            else:
                # Overlapping layer: only keep non-duplicate cues, or drop if trivial/noisy
                for sc in sec_cues:
                    norm = _normalize_text(sc.text)
                    if norm and _text_in(dom_texts, norm):
                        dropped_cues.add(id(sc))
                    elif len(sec_cues) < 0.35 * len(dom_cues) or len(sec_cues) < 10:
                        dropped_cues.add(id(sc))
                    else:
                        sc.layer_id = dominant_lid
                        sc.lang = lang
                        dom_cues.append(sc)
                        dom_texts.append(norm)

    if dropped_cues:
        cues = [c for c in cues if id(c) not in dropped_cues]

    cues.sort(key=lambda c: (c.start_ms, c.end_ms))
    return cues


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
