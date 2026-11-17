from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any

from lyra_ocr.lang import (
    _SPANISH_ACCENTED_VOWEL_RE,
    _SPANISH_CHARS_RE,
    _SPANISH_UNAMBIGUOUS_WORDS,
    _VIETNAMESE_UNIQUE_RE,
    detect_line_lang,
    romaji_score,
    vote_layer_lang,
)
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


def _layer_spanish_rate(lcues: list[LyricCue]) -> float:
    """Estimate Spanish lexical/orthographic signal rate across a layer's cues."""
    if not lcues:
        return 0.0
    es_hits = 0
    total_words = 0
    accent_hits = 0
    for c in lcues:
        norm = unicodedata.normalize("NFC", (c.text or "").strip())
        words = re.findall(r"[\w]+", norm.lower())
        total_words += len(words)
        es_hits += sum(1 for w in words if w in _SPANISH_UNAMBIGUOUS_WORDS)
        if _SPANISH_CHARS_RE.search(norm) or (
            _SPANISH_ACCENTED_VOWEL_RE.search(norm) and not _VIETNAMESE_UNIQUE_RE.search(norm)
        ):
            accent_hits += 1
    word_rate = es_hits / max(1, total_words)
    accent_rate = accent_hits / len(lcues)
    return max(word_rate, accent_rate)


def _detect_parallel_bilingual_langs(
    dom_cues: list[LyricCue],
    sec_cues: list[LyricCue],
    dom_y: float,
    sec_y: float,
    overlap_ratio: float,
) -> tuple[str, str] | None:
    """Return distinct (dom_lang, sec_lang) when two overlapping bands are bilingual streams."""
    if overlap_ratio < 0.45 or len(dom_cues) < 8 or len(sec_cues) < 8:
        return None
    min_len = min(len(dom_cues), len(sec_cues))
    max_len = max(len(dom_cues), len(sec_cues))
    if min_len / max_len < 0.45 or abs(sec_y - dom_y) < 0.035:
        return None

    dom_ro = sum(romaji_score(c.text) for c in dom_cues) / len(dom_cues)
    sec_ro = sum(romaji_score(c.text) for c in sec_cues) / len(sec_cues)
    dom_es = _layer_spanish_rate(dom_cues)
    sec_es = _layer_spanish_rate(sec_cues)

    if dom_ro >= 0.30 and dom_ro >= 2.0 * sec_ro:
        other_lang = "es" if (sec_es >= 0.10 and sec_es >= 2.0 * dom_es) else "en"
        return ("ja-Latn", other_lang)
    if sec_ro >= 0.30 and sec_ro >= 2.0 * dom_ro:
        other_lang = "es" if (dom_es >= 0.10 and dom_es >= 2.0 * sec_es) else "en"
        return (other_lang, "ja-Latn")
    if dom_es >= 0.10 and dom_es >= 2.0 * sec_es:
        return ("es", "en")
    if sec_es >= 0.10 and sec_es >= 2.0 * dom_es:
        return ("en", "es")
    return None


def consolidate_language_layers(
    cues: list[LyricCue],
    *,
    overlap_threshold: float = 0.25,
    asr_transcript: Any = None,
) -> list[LyricCue]:
    """Consolidate layers so that each distinct language has at most ONE layer.

    - First applies merge_duplicate_layers to collapse duplicate bands of the same text stream.
    - Then for remaining layers of the same language:
      - If they are non-overlapping/complementary (e.g. ad-libs, alternating lines),
        merge the secondary layer into the primary layer.
      - If they overlap significantly, keep vertically stacked companion bands or drop noisy overlays.
    - Resolves simultaneous multi-line cues before returning.
    """
    if not cues:
        return []

    cues = merge_duplicate_layers(cues)

    by_layer: dict[Any, list[LyricCue]] = defaultdict(list)
    for c in cues:
        if c.layer_id is not None:
            by_layer[c.layer_id].append(c)

    if len(by_layer) < 2:
        return resolve_simultaneous_multiline_cues(cues, asr_transcript=asr_transcript)

    # Re-vote each layer's language from fresh detect_line_lang(c.text) across its cues
    layers_by_lang: dict[str, list[Any]] = defaultdict(list)
    for lid, lcues in by_layer.items():
        stored_lang = _layer_lang(lcues)
        fresh_lang = vote_layer_lang([detect_line_lang(c.text) for c in lcues])
        if fresh_lang == "und":
            lang = stored_lang
        elif fresh_lang == "zh" and stored_lang == "ja":
            lang = "ja"
        elif fresh_lang == "en" and stored_lang not in ("und", "en"):
            # 'en' is the generic Latin fallback in detect_line_lang; do not downgrade
            # an already-classified layer (e.g. 'ja-Latn' after user edit) to 'en'.
            lang = stored_lang
        else:
            lang = fresh_lang
        if lang != "und" and lang != stored_lang:
            for c in lcues:
                if (
                    lang in ("ja", "zh", "ko", "vi")
                    and c.lang == "en"
                    and detect_line_lang(c.text) == "en"
                ):
                    continue
                c.lang = lang
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
            sec_y = sum(c.y_center for c in sec_cues) / len(sec_cues)
            dom_y = sum(c.y_center for c in dom_cues) / len(dom_cues)

            bilingual_langs = _detect_parallel_bilingual_langs(
                dom_cues, sec_cues, dom_y, sec_y, overlap_ratio
            )
            if bilingual_langs is not None:
                dom_lang, sec_lang = bilingual_langs
                for dc in dom_cues:
                    dc.lang = dom_lang
                for sc in sec_cues:
                    sc.lang = sec_lang
                continue

            dom_texts = [_normalize_text(c.text) for c in dom_cues]

            if overlap_ratio <= overlap_threshold:
                # Complementary non-overlapping layer: merge completely into dominant layer
                for sc in sec_cues:
                    sc.layer_id = dominant_lid
                    sc.lang = lang
                dom_cues.extend(sec_cues)
            else:
                sec_x = sum(c.x_center for c in sec_cues) / len(sec_cues)
                is_companion_band = (
                    0.02 <= abs(sec_y - dom_y) <= 0.18 and 0.28 <= sec_x <= 0.72
                )

                # Overlapping layer: rescue companion stacked bands or filter noisy duplicates
                for sc in sec_cues:
                    norm = _normalize_text(sc.text)
                    if norm and _text_in(dom_texts, norm):
                        dropped_cues.add(id(sc))
                    elif (not is_companion_band) and (
                        len(sec_cues) < 0.35 * len(dom_cues) or len(sec_cues) < 10
                    ):
                        dropped_cues.add(id(sc))
                    else:
                        sc.layer_id = dominant_lid
                        sc.lang = lang
                        dom_cues.append(sc)
                        dom_texts.append(norm)

    if dropped_cues:
        cues = [c for c in cues if id(c) not in dropped_cues]

    cues = resolve_simultaneous_multiline_cues(cues, asr_transcript=asr_transcript)
    cues.sort(key=lambda c: (c.start_ms, c.end_ms))
    return cues


def _is_cjk_heavy(text: str) -> bool:
    no_space = re.sub(r"\s+", "", text)
    if not no_space:
        return False
    cjk_chars = len(re.findall(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]", no_space))
    return cjk_chars > 0


def resolve_simultaneous_multiline_cues(
    cues: list[LyricCue],
    *,
    asr_transcript: Any = None,
    short_wrap_max_dur_ms: int = 4000,
    short_wrap_max_words: int = 12,
    min_stagger_gap_ms: int = 350,
) -> list[LyricCue]:
    """Resolve simultaneous multi-line cues inside each layer via wrap-merging or de-colliding."""
    if not cues:
        return []

    by_layer: dict[Any, list[LyricCue]] = defaultdict(list)
    for cue in cues:
        by_layer[cue.layer_id].append(cue)

    resolved: list[LyricCue] = []

    for _lid, lcues in by_layer.items():
        ordered = sorted(lcues, key=lambda c: (c.start_ms, c.y_center, c.end_ms))
        groups: list[list[LyricCue]] = []
        for cue in ordered:
            placed = False
            for grp in groups:
                g_start = min(c.start_ms for c in grp)
                g_end = max(c.end_ms for c in grp)
                c_dur = max(1, cue.end_ms - cue.start_ms)
                g_dur = max(1, g_end - g_start)
                ov = min(cue.end_ms, g_end) - max(cue.start_ms, g_start)
                union = g_dur + c_dur - ov if ov > 0 else 0
                iou = ov / union if union > 0 else 0.0
                same_y_contained = any(
                    abs(cue.y_center - gc.y_center) < 0.032
                    and (min(cue.end_ms, gc.end_ms) - max(cue.start_ms, gc.start_ms))
                    / max(1, min(max(1, cue.end_ms - cue.start_ms), max(1, gc.end_ms - gc.start_ms)))
                    >= 0.70
                    for gc in grp
                )
                if (abs(cue.start_ms - g_start) <= 600 and iou >= 0.60) or same_y_contained:
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

            # Collapse same-Y overlapping reads (|Δy| < 0.032) into one stable cue
            y_bands: list[list[LyricCue]] = []
            for cue in grp:
                if y_bands and abs(cue.y_center - y_bands[-1][0].y_center) < 0.032:
                    y_bands[-1].append(cue)
                else:
                    y_bands.append([cue])

            collapsed_grp: list[LyricCue] = []
            for band in y_bands:
                if len(band) == 1:
                    collapsed_grp.append(band[0])
                    continue
                best = max(
                    band,
                    key=lambda c: (
                        len(c.text.strip()) >= 4,
                        c.confidence * len(c.text.strip()),
                        c.confidence,
                        len(c.text.strip()),
                    ),
                )
                tids = sorted(
                    {tid for c in band for tid in c.source_track_ids},
                    key=_track_sort_key,
                )
                sids = sorted(
                    {sid for c in band for sid in c.source_segment_ids},
                    key=_track_sort_key,
                )
                merged_words: list[dict[str, Any]] = []
                for bc in band:
                    if bc.words:
                        merged_words = _merge_cue_words(merged_words, bc.words)
                lang = best.lang
                if not lang or lang == "und":
                    lang = next((c.lang for c in band if c.lang and c.lang != "und"), "und")
                layer_id = best.layer_id
                if layer_id is None:
                    layer_id = next((c.layer_id for c in band if c.layer_id is not None), None)
                collapsed_grp.append(
                    replace(
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
                )

            grp = collapsed_grp
            if len(grp) < 2:
                resolved.extend(grp)
                continue

            t_start = min(c.start_ms for c in grp)
            t_end = max(c.end_ms for c in grp)
            delta_t = max(0, t_end - t_start)

            # Branch 1: Short 2-line wrapped sentence in adjacent vertical bands
            can_merge_wrap = False
            if len(grp) == 2 and delta_t <= short_wrap_max_dur_ms:
                l1, l2 = grp[0], grp[1]
                y_diff = abs(l2.y_center - l1.y_center)
                if 0.015 <= y_diff <= 0.18:
                    combined = f"{l1.text.strip()} {l2.text.strip()}"
                    if l1.lang in ("ja", "zh", "ko") or _is_cjk_heavy(combined):
                        can_merge_wrap = len(re.sub(r"\s+", "", combined)) <= 22
                    else:
                        can_merge_wrap = len(combined.split()) <= short_wrap_max_words

            if can_merge_wrap:
                l1, l2 = grp[0], grp[1]
                lang = l1.lang if l1.lang and l1.lang != "und" else l2.lang
                sep = "" if lang in ("ja", "zh", "ko") else " "
                merged_text = f"{l1.text.strip()}{sep}{l2.text.strip()}"
                raw_1 = (l1.raw_text or l1.text).strip()
                raw_2 = (l2.raw_text or l2.text).strip()
                merged_raw = f"{raw_1}{sep}{raw_2}"
                tids = sorted(set(l1.source_track_ids + l2.source_track_ids), key=_track_sort_key)
                sids = sorted(set(l1.source_segment_ids + l2.source_segment_ids), key=_track_sort_key)
                qa = list(dict.fromkeys(list(l1.qa_flags) + list(l2.qa_flags) + ["multiline_merged"]))
                resolved.append(
                    LyricCue(
                        start_ms=t_start,
                        end_ms=t_end,
                        text=merged_text,
                        raw_text=merged_raw,
                        source_track_ids=tids,
                        source_segment_ids=sids,
                        confidence=round((l1.confidence + l2.confidence) / 2.0, 4),
                        decision=l1.decision,
                        review_required=l1.review_required or l2.review_required,
                        bbox=l1.bbox,
                        x_center=round((l1.x_center + l2.x_center) / 2.0, 4),
                        y_center=round((l1.y_center + l2.y_center) / 2.0, 4),
                        correction=l1.correction,
                        qa_flags=qa,
                        ocr_confidence=l1.ocr_confidence,
                        text_quality=l1.text_quality,
                        layer_id=l1.layer_id,
                        lang=lang,
                        words=_merge_cue_words(l1.words, l2.words),
                    )
                )
                continue

            # Branch 2: Multi-line stanza / simultaneous card — de-collide timestamps
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
                delta_eff = max(delta_t, len(grp) * min_stagger_gap_ms)
                cum_w = 0
                for cue, w in zip(grp, weights):
                    cue.start_ms = t_start + int((delta_eff * cum_w) // total_w)
                    cum_w += w
                    cue.end_ms = t_start + int((delta_eff * cum_w) // total_w)
                    if "multiline_staggered" not in cue.qa_flags:
                        cue.qa_flags = list(cue.qa_flags) + ["multiline_staggered"]

            resolved.extend(grp)

    resolved.sort(
        key=lambda c: (
            c.start_ms,
            c.y_center,
            _track_sort_key(c.source_track_ids[0]) if c.source_track_ids else (0, -1),
        )
    )
    return resolved


def dedupe_and_group(
    cues: list[LyricCue],
    *,
    asr_transcript: Any = None,
) -> list[LyricCue]:
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

            y_diff = abs(existing.y_center - cue.y_center)
            iou = _bbox_iou(existing.bbox, cue.bbox)
            same_band = (y_diff <= 0.08 or iou >= 0.30)
            if not same_band:
                continue

            sim = string_similarity(existing.text, cue.text)
            text_match = (sim >= 0.88)
            if not text_match:
                e_clean = existing.text.strip().lower()
                c_clean = cue.text.strip().lower()
                if (len(e_clean) >= 10 or len(c_clean) >= 10) and sim >= 0.80:
                    if (
                        e_clean.startswith(c_clean)
                        or e_clean.endswith(c_clean)
                        or c_clean.startswith(e_clean)
                        or c_clean.endswith(e_clean)
                    ):
                        text_match = True

            start_delta = abs(existing.start_ms - cue.start_ms)
            gap_ms = cue.start_ms - existing.end_ms
            min_dur = min(
                max(0, existing.end_ms - existing.start_ms),
                max(0, cue.end_ms - cue.start_ms),
            )
            fade_transition_match = (
                y_diff < 0.032
                and gap_ms <= 800
                and min_dur <= 1400
                and sim >= 0.62
            )

            if not text_match and not fade_transition_match:
                continue

            should_merge = False
            if fade_transition_match:
                should_merge = True
            elif start_delta <= 1500 or gap_ms <= 200:
                # Case 1: Overlapping or Simultaneous Duplicate
                if gap_ms > 0:
                    from lyra_ocr.audio.asr import asr_has_new_prefix_onset

                    is_repeat = asr_has_new_prefix_onset(
                        cue.text,
                        cue.start_ms,
                        asr_transcript,
                        first_start_ms=existing.start_ms,
                    )
                    if not is_repeat:
                        should_merge = True
                else:
                    should_merge = True
            elif 200 < gap_ms <= 2500 and start_delta > 1500:
                # Case 2: Long Line Split Across Frames
                from lyra_ocr.audio.asr import asr_has_new_prefix_onset

                is_repeat = asr_has_new_prefix_onset(
                    cue.text,
                    cue.start_ms,
                    asr_transcript,
                    first_start_ms=existing.start_ms,
                )
                if not is_repeat:
                    should_merge = True

            if should_merge:
                matched_idx = i
                break

        if matched_idx >= 0:
            target = merged[matched_idx]
            target_dur = max(0, target.end_ms - target.start_ms)
            cue_dur = max(0, cue.end_ms - cue.start_ms)
            prefer_longer_stable = (
                target_dur <= 1400 < cue_dur
                and len(cue.text.strip()) >= len(target.text.strip())
            )
            avoid_shorter_fade = (
                cue_dur <= 1400 < target_dur
                and len(cue.text.strip()) < len(target.text.strip())
            )
            # Decide whether to upgrade target with cue's text/confidence
            if prefer_longer_stable or (
                not avoid_shorter_fade
                and (
                    cue.confidence > target.confidence
                    or (
                        len(cue.text) >= len(target.text) + 3
                        and cue.confidence >= target.confidence - 0.05
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
            if cue.words:
                target.words = _merge_cue_words(target.words, cue.words)
        else:
            merged.append(cue)

    # 3. Resolve simultaneous multi-line cues before enforcing non-overlap cap
    merged = resolve_simultaneous_multiline_cues(merged, asr_transcript=asr_transcript)

    # 4. Cap end_ms at the next cue's start_ms within the same layer
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


def share_solitary_cross_layer_cues(
    cues: list[LyricCue],
    *,
    min_gap_ms: int = 1200,
    min_cue_dur_ms: int = 900,
) -> list[LyricCue]:
    """Clone solitary English cues from one major layer into global silence gaps of other major layers."""
    if not cues:
        return []

    by_layer: dict[Any, list[LyricCue]] = defaultdict(list)
    for cue in cues:
        if cue.layer_id is not None:
            by_layer[cue.layer_id].append(cue)

    if len(by_layer) < 2:
        return cues

    max_count = max(len(lc) for lc in by_layer.values())
    major_lids = [
        lid
        for lid, lc in by_layer.items()
        if len(lc) >= 8 or (len(lc) >= 2 and len(lc) >= 0.25 * max_count)
    ]
    if len(major_lids) < 2:
        return cues

    layer_spans: dict[Any, tuple[int, int]] = {}
    layer_langs: dict[Any, str] = {}
    for lid in major_lids:
        lc = by_layer[lid]
        layer_spans[lid] = (min(c.start_ms for c in lc), max(c.end_ms for c in lc))
        layer_langs[lid] = _layer_lang(lc)

    added: list[LyricCue] = []

    for src_lid in major_lids:
        for cue in by_layer[src_lid]:
            if (cue.end_ms - cue.start_ms) < min_cue_dur_ms:
                continue
            if "cross_layer_inherited" in cue.qa_flags:
                continue
            if cue.lang != "en" or detect_line_lang(cue.text) != "en":
                continue

            # Strict global-frame solitary check across every other major layer
            if any(
                oc.start_ms < (cue.end_ms + 800) and oc.end_ms > (cue.start_ms - 800)
                for other_lid in major_lids
                if other_lid != src_lid
                for oc in by_layer[other_lid]
            ):
                continue

            for dst_lid in major_lids:
                if dst_lid == src_lid:
                    continue

                dst_lang = layer_langs[dst_lid]

                span_min, span_max = layer_spans[dst_lid]
                if cue.start_ms < span_min - 5000 or cue.end_ms > span_max + 5000:
                    continue

                dst_cues = by_layer[dst_lid]
                # Also avoid duplicating with already queued additions for this dst_lid
                if any(
                    ac.layer_id == dst_lid
                    and ac.start_ms < (cue.end_ms + 800)
                    and ac.end_ms > (cue.start_ms - 800)
                    for ac in added
                ):
                    continue

                prev_ends = [dc.end_ms for dc in dst_cues if dc.end_ms <= cue.start_ms + 200]
                next_starts = [dc.start_ms for dc in dst_cues if dc.start_ms >= cue.end_ms - 200]
                left_bound = max(prev_ends) if prev_ends else (span_min - 5000)
                right_bound = min(next_starts) if next_starts else (span_max + 5000)
                if (right_bound - left_bound) < min_gap_ms:
                    continue

                cloned = replace(
                    cue,
                    source_track_ids=list(cue.source_track_ids),
                    source_segment_ids=list(cue.source_segment_ids),
                    bbox=list(cue.bbox),
                    qa_flags=list(cue.qa_flags) + ["cross_layer_inherited"],
                    words=[dict(w) for w in cue.words],
                    layer_id=dst_lid,
                    lang=dst_lang if dst_lang != "und" else cue.lang,
                )
                added.append(cloned)

    if not added:
        return cues

    combined = list(cues) + added
    combined.sort(
        key=lambda c: (
            c.start_ms,
            c.y_center,
            _track_sort_key(c.source_track_ids[0]) if c.source_track_ids else (0, -1),
        )
    )
    return combined

