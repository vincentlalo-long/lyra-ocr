"""Within-layer cue deduplication and simultaneous multi-line grouping."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Any

from lyra_ocr.config import DEFAULT_GROUPING_THRESHOLDS, GroupingThresholds
from lyra_ocr.lang import get_language_profile
from lyra_ocr.lyric.layer_consolidation import (
    _cue_sort_key,
    _detect_parallel_bilingual_langs,
    _layer_lang,
    _layer_spanish_rate,
    _normalize_text,
    _text_in,
    _texts_match,
    _track_sort_key,
    consolidate_language_layers,
    merge_duplicate_layers,
    share_solitary_cross_layer_cues,
)
from lyra_ocr.lyric.models import LyricCue
from lyra_ocr.track import string_similarity

__all__ = [
    "consolidate_language_layers",
    "dedupe_and_group",
    "merge_duplicate_layers",
    "resolve_simultaneous_multiline_cues",
    "share_solitary_cross_layer_cues",
]


def _bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    if not box_a or not box_b or len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0

    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _merge_cue_words(
    words_a: list[dict[str, Any]] | None,
    words_b: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge word timing dictionaries, deduplicating identical spans and sorting chronologically."""
    if not words_a and not words_b:
        return []
    if not words_a:
        merged = [dict(w) for w in (words_b or [])]
    elif not words_b:
        merged = [dict(w) for w in (words_a or [])]
    else:
        merged = [dict(w) for w in words_a]
        existing_spans = {(w.get("start_ms"), w.get("end_ms"), w.get("word")) for w in merged}
        for cw in words_b:
            span_key = (cw.get("start_ms"), cw.get("end_ms"), cw.get("word"))
            if span_key not in existing_spans:
                merged.append(dict(cw))
                existing_spans.add(span_key)
    merged.sort(key=lambda w: (w.get("start_ms", 0), w.get("end_ms", 0)))
    return merged


def _is_cjk_heavy(text: str) -> bool:
    no_space = re.sub(r"\s+", "", text)
    return bool(no_space and re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", no_space))


def _collapse_same_y_band(band: list[LyricCue], cfg: GroupingThresholds) -> LyricCue:
    """Collapse overlapping flicker cues at the same vertical band into one cue."""
    if len(band) == 1:
        return band[0]
    best = max(
        band,
        key=lambda c: (
            len(c.text.strip()) >= cfg.flicker_min_meaningful_chars,
            c.confidence * len(c.text.strip()),
            c.confidence,
            len(c.text.strip()),
        ),
    )
    tids = sorted({tid for c in band for tid in c.source_track_ids}, key=_track_sort_key)
    sids = sorted({sid for c in band for sid in c.source_segment_ids}, key=_track_sort_key)
    merged_words: list[dict[str, Any]] = []
    for bc in band:
        if bc.words:
            merged_words = _merge_cue_words(merged_words, bc.words)
    lang = best.lang if best.lang and best.lang != "und" else next(
        (c.lang for c in band if c.lang and c.lang != "und"), "und"
    )
    layer_id = best.layer_id if best.layer_id is not None else next(
        (c.layer_id for c in band if c.layer_id is not None), None
    )
    return replace(
        best,
        start_ms=min(c.start_ms for c in band),
        end_ms=max(c.end_ms for c in band),
        source_track_ids=tids,
        source_segment_ids=sids,
        qa_flags=list(best.qa_flags),
        words=merged_words,
        lang=lang,
        layer_id=layer_id,
    )


def _try_merge_short_wrap(
    grp: list[LyricCue],
    t_start: int,
    t_end: int,
    max_dur_ms: int,
    max_words: int,
    cfg: GroupingThresholds,
) -> LyricCue | None:
    """Merge a short 2-line wrapped sentence in adjacent vertical bands if within bounds."""
    if len(grp) != 2 or max(0, t_end - t_start) > max_dur_ms:
        return None
    l1, l2 = grp[0], grp[1]
    y_diff = abs(l2.y_center - l1.y_center)
    if not (cfg.wrap_y_diff_min <= y_diff <= cfg.wrap_y_diff_max):
        return None

    combined = f"{l1.text.strip()} {l2.text.strip()}"
    lang = l1.lang if l1.lang and l1.lang != "und" else l2.lang
    lang_profile = get_language_profile(lang or "und")
    is_unspaced = lang_profile.is_cjk_unspaced or _is_cjk_heavy(combined)
    if is_unspaced:
        if len(re.sub(r"\s+", "", combined)) > cfg.short_wrap_max_cjk_chars:
            return None
    elif len(combined.split()) > max_words:
        return None

    sep = "" if (lang_profile.is_cjk_unspaced or (lang == "und" and _is_cjk_heavy(combined))) else " "
    raw_1 = (l1.raw_text or l1.text).strip()
    raw_2 = (l2.raw_text or l2.text).strip()
    tids = sorted(set(l1.source_track_ids + l2.source_track_ids), key=_track_sort_key)
    sids = sorted(set(l1.source_segment_ids + l2.source_segment_ids), key=_track_sort_key)
    qa = list(dict.fromkeys(list(l1.qa_flags) + list(l2.qa_flags) + ["multiline_merged"]))
    return replace(
        l1,
        start_ms=t_start,
        end_ms=t_end,
        text=f"{l1.text.strip()}{sep}{l2.text.strip()}",
        raw_text=f"{raw_1}{sep}{raw_2}",
        source_track_ids=tids,
        source_segment_ids=sids,
        confidence=round((l1.confidence + l2.confidence) / 2.0, 4),
        review_required=l1.review_required or l2.review_required,
        x_center=round((l1.x_center + l2.x_center) / 2.0, 4),
        y_center=round((l1.y_center + l2.y_center) / 2.0, 4),
        qa_flags=qa,
        lang=lang,
        words=_merge_cue_words(l1.words, l2.words),
    )


def resolve_simultaneous_multiline_cues(
    cues: list[LyricCue],
    *,
    asr_transcript: Any = None,
    short_wrap_max_dur_ms: int | None = None,
    short_wrap_max_words: int | None = None,
    min_stagger_gap_ms: int | None = None,
    thresholds: GroupingThresholds | None = None,
) -> list[LyricCue]:
    """Resolve simultaneous multi-line cues inside each layer via wrap-merging or de-colliding."""
    if not cues:
        return []

    cfg = thresholds or DEFAULT_GROUPING_THRESHOLDS
    eff_max_dur = cfg.short_wrap_max_dur_ms if short_wrap_max_dur_ms is None else short_wrap_max_dur_ms
    eff_max_words = cfg.short_wrap_max_words if short_wrap_max_words is None else short_wrap_max_words
    eff_stagger_gap = cfg.min_stagger_gap_ms if min_stagger_gap_ms is None else min_stagger_gap_ms

    by_layer: dict[Any, list[LyricCue]] = defaultdict(list)
    for cue in cues:
        by_layer[cue.layer_id].append(cue)

    resolved: list[LyricCue] = []
    for lcues in by_layer.values():
        ordered = sorted(lcues, key=lambda c: (c.start_ms, c.y_center, c.end_ms))
        groups: list[list[LyricCue]] = []
        for cue in ordered:
            placed = False
            c_dur = max(1, cue.end_ms - cue.start_ms)
            for grp in groups:
                g_start = min(c.start_ms for c in grp)
                g_end = max(c.end_ms for c in grp)
                ov = min(cue.end_ms, g_end) - max(cue.start_ms, g_start)
                union = max(1, g_end - g_start) + c_dur - ov if ov > 0 else 0
                iou = ov / union if union > 0 else 0.0
                same_y_contained = any(
                    abs(cue.y_center - gc.y_center) < cfg.same_y_band_tolerance
                    and (min(cue.end_ms, gc.end_ms) - max(cue.start_ms, gc.start_ms))
                    / max(1, min(c_dur, max(1, gc.end_ms - gc.start_ms)))
                    >= cfg.same_y_min_containment
                    for gc in grp
                )
                if (
                    abs(cue.start_ms - g_start) <= cfg.simultaneous_start_delta_ms
                    and iou >= cfg.simultaneous_min_iou
                ) or same_y_contained:
                    grp.append(cue)
                    placed = True
                    break
            if not placed:
                groups.append([cue])

        for grp in groups:
            if len(grp) < 2:
                resolved.extend(grp)
                continue

            grp.sort(key=lambda c: (c.y_center, c.start_ms))
            y_bands: list[list[LyricCue]] = []
            for cue in grp:
                if y_bands and abs(cue.y_center - y_bands[-1][0].y_center) < cfg.same_y_band_tolerance:
                    y_bands[-1].append(cue)
                else:
                    y_bands.append([cue])

            grp = [_collapse_same_y_band(band, cfg) for band in y_bands]
            if len(grp) < 2:
                resolved.extend(grp)
                continue

            t_start = min(c.start_ms for c in grp)
            t_end = max(c.end_ms for c in grp)
            merged_wrap = _try_merge_short_wrap(grp, t_start, t_end, eff_max_dur, eff_max_words, cfg)
            if merged_wrap is not None:
                resolved.append(merged_wrap)
                continue

            asr_ok = False
            if asr_transcript is not None:
                try:
                    from lyra_ocr.audio.asr import align_multiline_cues_with_asr

                    asr_ok = bool(align_multiline_cues_with_asr(grp, asr_transcript))
                except ImportError:
                    asr_ok = False

            if not asr_ok:
                weights = [max(1, len(re.sub(r"\s+", "", c.text))) for c in grp]
                total_w = sum(weights)
                delta_eff = max(max(0, t_end - t_start), len(grp) * eff_stagger_gap)
                cum_w = 0
                for cue, w in zip(grp, weights):
                    cue.start_ms = t_start + int((delta_eff * cum_w) // total_w)
                    cum_w += w
                    cue.end_ms = t_start + int((delta_eff * cum_w) // total_w)
                    if "multiline_staggered" not in cue.qa_flags:
                        cue.qa_flags = list(cue.qa_flags) + ["multiline_staggered"]

            resolved.extend(grp)

    resolved.sort(key=_cue_sort_key)
    return resolved


def _shared_affix_len(a: str, b: str, *, from_end: bool = False) -> int:
    limit = min(len(a), len(b))
    count = 0
    for i in range(limit):
        ca = a[-1 - i] if from_end else a[i]
        cb = b[-1 - i] if from_end else b[i]
        if ca != cb:
            break
        count += 1
    return count


def _fade_text_matches(a: str, b: str, sim: float, cfg: GroupingThresholds) -> bool:
    """Return True when two same-Y cues are strong or affix/substring fade-transition variants."""
    if sim < cfg.fade_min_similarity:
        return False
    if sim >= cfg.fade_strong_similarity:
        return True
    a_compact = re.sub(r"\s+", "", (a or "").lower())
    b_compact = re.sub(r"\s+", "", (b or "").lower())
    if not a_compact or not b_compact:
        return False
    shorter, longer = (
        (a_compact, b_compact) if len(a_compact) <= len(b_compact) else (b_compact, a_compact)
    )
    if len(shorter) <= cfg.fade_short_text_max_chars or shorter in longer:
        return True
    prefix_len = _shared_affix_len(a_compact, b_compact, from_end=False)
    suffix_len = _shared_affix_len(a_compact, b_compact, from_end=True)
    best_affix = max(prefix_len, suffix_len)
    return best_affix >= cfg.fade_affix_min_chars and (
        best_affix >= len(shorter) - 1 or (best_affix / len(shorter)) >= 0.75
    )


def _should_merge_duplicate_cue(
    existing: LyricCue,
    cue: LyricCue,
    asr_transcript: Any,
    cfg: GroupingThresholds,
    *,
    has_intervening_same_layer_cue: bool = False,
) -> bool:
    """Return True if ``cue`` is a temporal/fade duplicate of ``existing`` in the same layer."""
    if existing.layer_id is not None and cue.layer_id is not None and existing.layer_id != cue.layer_id:
        return False

    y_diff = abs(existing.y_center - cue.y_center)
    iou = _bbox_iou(existing.bbox, cue.bbox)
    if y_diff > cfg.dedupe_max_y_diff and iou < cfg.dedupe_min_bbox_iou:
        return False

    sim = string_similarity(existing.text, cue.text)
    text_match = sim >= cfg.dedupe_text_similarity
    if not text_match:
        e_clean = existing.text.strip().lower()
        c_clean = cue.text.strip().lower()
        if (len(e_clean) >= cfg.dedupe_affix_min_chars or len(c_clean) >= cfg.dedupe_affix_min_chars) and sim >= cfg.dedupe_affix_similarity:
            text_match = (
                e_clean.startswith(c_clean)
                or e_clean.endswith(c_clean)
                or c_clean.startswith(e_clean)
                or c_clean.endswith(e_clean)
            )

    start_delta = abs(existing.start_ms - cue.start_ms)
    gap_ms = cue.start_ms - existing.end_ms
    min_dur = min(max(0, existing.end_ms - existing.start_ms), max(0, cue.end_ms - cue.start_ms))
    fade_transition_match = (
        y_diff < cfg.same_y_band_tolerance
        and gap_ms <= cfg.fade_max_gap_ms
        and min_dur <= cfg.fade_max_dur_ms
        and _fade_text_matches(existing.text, cue.text, sim, cfg)
    )
    if fade_transition_match and asr_transcript is not None and gap_ms > 0:
        from lyra_ocr.audio.asr import asr_has_new_prefix_onset

        if asr_has_new_prefix_onset(
            cue.text,
            cue.start_ms,
            asr_transcript,
            first_start_ms=existing.start_ms,
        ):
            fade_transition_match = False
    if not text_match and not fade_transition_match:
        return False
    if has_intervening_same_layer_cue and (
        gap_ms > cfg.merge_overlap_gap_ms or start_delta > cfg.merge_max_start_delta_ms
    ):
        return False
    if fade_transition_match:
        return True
    same_track_or_band = (
        bool(set(existing.source_track_ids) & set(cue.source_track_ids))
        or y_diff < cfg.same_y_band_tolerance
    )
    max_gap = (
        cfg.same_track_max_gap_ms
        if (not has_intervening_same_layer_cue and same_track_or_band)
        else cfg.merge_split_max_gap_ms
    )
    in_merge_window = (
        start_delta <= cfg.merge_max_start_delta_ms
        or gap_ms <= cfg.merge_overlap_gap_ms
        or gap_ms <= max_gap
    )
    if not in_merge_window:
        return False
    if gap_ms <= 0:
        return True

    from lyra_ocr.audio.asr import asr_has_new_prefix_onset

    return not asr_has_new_prefix_onset(
        cue.text,
        cue.start_ms,
        asr_transcript,
        first_start_ms=existing.start_ms,
    )


def _merge_cue_into_target(target: LyricCue, cue: LyricCue, cfg: GroupingThresholds) -> None:
    """Merge ``cue`` metadata, text, timing, and words into ``target`` in-place."""
    target_dur = max(0, target.end_ms - target.start_ms)
    cue_dur = max(0, cue.end_ms - cue.start_ms)
    prefer_longer_stable = (
        target_dur <= cfg.fade_max_dur_ms < cue_dur
        and len(cue.text.strip()) >= len(target.text.strip())
    )
    avoid_shorter_fade = (
        cue_dur <= cfg.fade_max_dur_ms < target_dur
        and len(cue.text.strip()) < len(target.text.strip())
    )
    if prefer_longer_stable or (
        not avoid_shorter_fade
        and (
            cue.confidence > target.confidence
            or (
                len(cue.text) >= len(target.text) + cfg.upgrade_min_extra_chars
                and cue.confidence >= target.confidence - cfg.upgrade_max_conf_drop
            )
        )
    ):
        target.text = cue.text
        target.raw_text = cue.raw_text
        target.confidence = cue.confidence
        target.bbox = cue.bbox
        target.x_center = cue.x_center
        target.y_center = cue.y_center
        target.correction = cue.correction

    target.source_track_ids = sorted(
        set(target.source_track_ids + cue.source_track_ids), key=_track_sort_key
    )
    target.source_segment_ids = sorted(
        set(target.source_segment_ids + cue.source_segment_ids), key=_track_sort_key
    )
    target.review_required = target.review_required or cue.review_required
    target.end_ms = max(target.end_ms, cue.end_ms)
    if target.layer_id is None and cue.layer_id is not None:
        target.layer_id = cue.layer_id
    if (not target.lang or target.lang == "und") and cue.lang and cue.lang != "und":
        target.lang = cue.lang
    if cue.words:
        target.words = _merge_cue_words(target.words, cue.words)


def dedupe_and_group(
    cues: list[LyricCue],
    *,
    asr_transcript: Any = None,
    thresholds: GroupingThresholds | None = None,
) -> list[LyricCue]:
    """Deduplicate repeated OCR cues within each layer, resolve multiline cues, and clamp overlaps."""
    if not cues:
        return []

    cfg = thresholds or DEFAULT_GROUPING_THRESHOLDS
    for cue in cues:
        cue.start_ms = max(0, cue.start_ms)

    sorted_cues = sorted(cues, key=_cue_sort_key)
    merged: list[LyricCue] = []
    for cue in sorted_cues:
        matched_idx = -1
        for i, existing in enumerate(merged):
            has_intervening = any(
                j != i
                and other.layer_id == existing.layer_id
                and existing.start_ms < other.start_ms < cue.start_ms
                and string_similarity(other.text, cue.text) < cfg.dedupe_text_similarity
                and string_similarity(other.text, existing.text) < cfg.dedupe_text_similarity
                for j, other in enumerate(merged)
            )
            if _should_merge_duplicate_cue(
                existing,
                cue,
                asr_transcript,
                cfg,
                has_intervening_same_layer_cue=has_intervening,
            ):
                matched_idx = i
                break
        if matched_idx >= 0:
            _merge_cue_into_target(merged[matched_idx], cue, cfg)
        else:
            merged.append(cue)

    merged = resolve_simultaneous_multiline_cues(merged, asr_transcript=asr_transcript, thresholds=cfg)

    by_layer: dict[Any, list[LyricCue]] = defaultdict(list)
    for cue in merged:
        by_layer[cue.layer_id].append(cue)
    for lcues in by_layer.values():
        lcues_sorted = sorted(lcues, key=lambda c: (c.start_ms, c.end_ms))
        for idx in range(len(lcues_sorted) - 1):
            curr, next_cue = lcues_sorted[idx], lcues_sorted[idx + 1]
            if next_cue.start_ms > curr.start_ms and curr.end_ms > next_cue.start_ms:
                curr.end_ms = next_cue.start_ms

    return sorted(merged, key=_cue_sort_key)
