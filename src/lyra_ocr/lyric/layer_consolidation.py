"""Layer-level consolidation: duplicate band merging, bilingual separation, and cross-layer sharing."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any

from lyra_ocr.config import DEFAULT_GROUPING_THRESHOLDS, GroupingThresholds
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

__all__ = [
    "consolidate_language_layers",
    "merge_duplicate_layers",
    "share_solitary_cross_layer_cues",
]


def _track_sort_key(tid: Any) -> tuple[int, int | str]:
    s = str(tid)
    return (0, int(s)) if s.isdigit() else (1, s)


def _cue_sort_key(cue: LyricCue) -> tuple[int, float, tuple[int, int | str]]:
    first_tid = _track_sort_key(cue.source_track_ids[0]) if cue.source_track_ids else (0, -1)
    return (cue.start_ms, cue.y_center, first_tid)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _texts_match(
    a: str,
    b: str,
    thresholds: GroupingThresholds = DEFAULT_GROUPING_THRESHOLDS,
) -> bool:
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    # OCR often drops the tail of a line on one of two duplicate bands.
    if len(shorter) >= thresholds.substring_match_min_len and shorter in longer:
        return True
    return string_similarity(a, b) >= thresholds.layer_text_similarity


def _text_in(
    texts: list[str],
    needle: str,
    thresholds: GroupingThresholds = DEFAULT_GROUPING_THRESHOLDS,
) -> bool:
    return any(_texts_match(needle, t, thresholds) for t in texts)


def _layer_lang(cues: list[LyricCue]) -> str:
    counts = Counter(c.lang for c in cues if c.lang and c.lang != "und")
    return counts.most_common(1)[0][0] if counts else "und"


def _group_by_layer(cues: list[LyricCue]) -> dict[Any, list[LyricCue]]:
    by_layer: dict[Any, list[LyricCue]] = defaultdict(list)
    for cue in cues:
        if cue.layer_id is not None:
            by_layer[cue.layer_id].append(cue)
    return by_layer


def merge_duplicate_layers(
    cues: list[LyricCue],
    *,
    min_containment: float | None = None,
    max_y_gap: float | None = None,
    thresholds: GroupingThresholds | None = None,
) -> list[LyricCue]:
    """Collapse spatial bands rendering the same lyric stream into one layer before export."""
    if not cues:
        return []

    cfg = thresholds or DEFAULT_GROUPING_THRESHOLDS
    eff_min_containment = cfg.duplicate_min_containment if min_containment is None else min_containment
    eff_max_y_gap = cfg.duplicate_max_y_gap if max_y_gap is None else max_y_gap

    by_layer = _group_by_layer(cues)
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

    for i, a in enumerate(layer_ids):
        for b in layer_ids[i + 1 :]:
            pa, pb = profiles[a], profiles[b]
            if not pa["texts"] or not pb["texts"]:
                continue
            if pa["lang"] != pb["lang"] and "und" not in (pa["lang"], pb["lang"]):
                continue
            if abs(pa["y"] - pb["y"]) > eff_max_y_gap:
                continue
            smaller, larger = (pa, pb) if len(pa["texts"]) <= len(pb["texts"]) else (pb, pa)
            matched = sum(1 for t in smaller["texts"] if _text_in(larger["texts"], t, cfg))
            if matched / len(smaller["texts"]) >= eff_min_containment:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    components: dict[Any, list[Any]] = defaultdict(list)
    for lid in layer_ids:
        components[find(lid)].append(lid)

    dropped: set[int] = set()
    for members in components.values():
        if len(members) < 2:
            continue
        keeper = max(
            members,
            key=lambda lid: (len(profiles[lid]["texts"]), len(by_layer[lid]), -profiles[lid]["y"]),
        )
        keeper_texts = profiles[keeper]["texts"]
        keeper_lang = profiles[keeper]["lang"]
        for lid in members:
            if lid == keeper:
                continue
            for cue in by_layer[lid]:
                norm = _normalize_text(cue.text)
                if norm and _text_in(keeper_texts, norm, cfg):
                    dropped.add(id(cue))
                    continue
                cue.layer_id = keeper
                if not cue.lang or cue.lang == "und":
                    cue.lang = keeper_lang

    return [c for c in cues if id(c) not in dropped] if dropped else cues


def _layer_spanish_rate(lcues: list[LyricCue]) -> float:
    """Estimate Spanish lexical/orthographic signal rate across a layer's cues."""
    if not lcues:
        return 0.0
    es_hits = 0
    total_words = 0
    accent_hits = 0
    for cue in lcues:
        norm = unicodedata.normalize("NFC", (cue.text or "").strip())
        words = re.findall(r"[\w]+", norm.lower())
        total_words += len(words)
        es_hits += sum(1 for w in words if w in _SPANISH_UNAMBIGUOUS_WORDS)
        if _SPANISH_CHARS_RE.search(norm) or (
            _SPANISH_ACCENTED_VOWEL_RE.search(norm) and not _VIETNAMESE_UNIQUE_RE.search(norm)
        ):
            accent_hits += 1
    return max(es_hits / max(1, total_words), accent_hits / len(lcues))


def _detect_parallel_bilingual_langs(
    dom_cues: list[LyricCue],
    sec_cues: list[LyricCue],
    dom_y: float,
    sec_y: float,
    overlap_ratio: float,
    thresholds: GroupingThresholds = DEFAULT_GROUPING_THRESHOLDS,
) -> tuple[str, str] | None:
    """Return distinct (dom_lang, sec_lang) when two overlapping bands are bilingual streams."""
    if (
        overlap_ratio < thresholds.bilingual_min_overlap_ratio
        or len(dom_cues) < thresholds.bilingual_min_cues
        or len(sec_cues) < thresholds.bilingual_min_cues
    ):
        return None
    min_len = min(len(dom_cues), len(sec_cues))
    max_len = max(len(dom_cues), len(sec_cues))
    if (
        min_len / max_len < thresholds.bilingual_min_len_ratio
        or abs(sec_y - dom_y) < thresholds.bilingual_min_y_separation
    ):
        return None

    dom_ro = sum(romaji_score(c.text) for c in dom_cues) / len(dom_cues)
    sec_ro = sum(romaji_score(c.text) for c in sec_cues) / len(sec_cues)
    dom_es = _layer_spanish_rate(dom_cues)
    sec_es = _layer_spanish_rate(sec_cues)
    ratio = thresholds.bilingual_dominant_ratio
    min_ro = thresholds.bilingual_min_romaji_score
    min_es = thresholds.bilingual_min_spanish_rate

    if dom_ro >= min_ro and dom_ro >= ratio * sec_ro:
        other_lang = "es" if (sec_es >= min_es and sec_es >= ratio * dom_es) else "en"
        return ("ja-Latn", other_lang)
    if sec_ro >= min_ro and sec_ro >= ratio * dom_ro:
        other_lang = "es" if (dom_es >= min_es and dom_es >= ratio * sec_es) else "en"
        return (other_lang, "ja-Latn")
    if dom_es >= min_es and dom_es >= ratio * sec_es:
        return ("es", "en")
    if sec_es >= min_es and sec_es >= ratio * dom_es:
        return ("en", "es")
    return None


def _revote_layer_languages(by_layer: dict[Any, list[LyricCue]]) -> dict[str, list[Any]]:
    """Re-vote each layer's language from fresh line detection without downgrading non-English layers."""
    layers_by_lang: dict[str, list[Any]] = defaultdict(list)
    for lid, lcues in by_layer.items():
        stored_lang = _layer_lang(lcues)
        fresh_lang = vote_layer_lang([detect_line_lang(c.text) for c in lcues])
        if fresh_lang == "und":
            lang = stored_lang
        elif fresh_lang == "zh" and stored_lang == "ja":
            lang = "ja"
        elif fresh_lang == "en" and stored_lang not in ("und", "en"):
            # 'en' is the generic Latin fallback; preserve already-classified non-English layers.
            lang = stored_lang
        else:
            lang = fresh_lang
        if lang != "und" and lang != stored_lang:
            for cue in lcues:
                if (
                    lang in ("ja", "zh", "ko", "vi")
                    and cue.lang == "en"
                    and detect_line_lang(cue.text) == "en"
                ):
                    continue
                cue.lang = lang
        layers_by_lang[lang].append(lid)
    return layers_by_lang


def _merge_or_filter_secondary_layer(
    dominant_lid: Any,
    dom_cues: list[LyricCue],
    sec_cues: list[LyricCue],
    lang: str,
    eff_overlap_threshold: float,
    cfg: GroupingThresholds,
    dropped_cues: set[int],
) -> None:
    """Separate parallel bilingual layers, merge complementary ad-libs, or filter noisy overlays."""
    n_overlapping = sum(
        1
        for sc in sec_cues
        if any(
            max(sc.start_ms, dc.start_ms) < min(sc.end_ms, dc.end_ms) - cfg.consolidate_overlap_margin_ms
            for dc in dom_cues
        )
    )
    overlap_ratio = n_overlapping / len(sec_cues)
    sec_y = sum(c.y_center for c in sec_cues) / len(sec_cues)
    dom_y = sum(c.y_center for c in dom_cues) / len(dom_cues)

    bilingual_langs = _detect_parallel_bilingual_langs(
        dom_cues, sec_cues, dom_y, sec_y, overlap_ratio, cfg
    )
    if bilingual_langs is not None:
        dom_lang, sec_lang = bilingual_langs
        for dc in dom_cues:
            dc.lang = dom_lang
        for sc in sec_cues:
            sc.lang = sec_lang
        return

    if overlap_ratio <= eff_overlap_threshold:
        for sc in sec_cues:
            sc.layer_id = dominant_lid
            sc.lang = lang
        dom_cues.extend(sec_cues)
        return

    dom_texts = [_normalize_text(c.text) for c in dom_cues]
    sec_x = sum(c.x_center for c in sec_cues) / len(sec_cues)
    is_companion_band = (
        cfg.companion_band_y_min <= abs(sec_y - dom_y) <= cfg.companion_band_y_max
        and cfg.companion_band_x_min <= sec_x <= cfg.companion_band_x_max
    )
    for sc in sec_cues:
        norm = _normalize_text(sc.text)
        if norm and _text_in(dom_texts, norm, cfg):
            dropped_cues.add(id(sc))
        elif (not is_companion_band) and (
            len(sec_cues) < cfg.noisy_layer_min_dom_ratio * len(dom_cues)
            or len(sec_cues) < cfg.noisy_layer_min_cues
        ):
            dropped_cues.add(id(sc))
        else:
            sc.layer_id = dominant_lid
            sc.lang = lang
            dom_cues.append(sc)
            dom_texts.append(norm)


def consolidate_language_layers(
    cues: list[LyricCue],
    *,
    overlap_threshold: float | None = None,
    asr_transcript: Any = None,
    thresholds: GroupingThresholds | None = None,
) -> list[LyricCue]:
    """Consolidate layers so each distinct language keeps one primary stream unless parallel bilingual."""
    if not cues:
        return []

    from lyra_ocr.lyric.grouping import resolve_simultaneous_multiline_cues

    cfg = thresholds or DEFAULT_GROUPING_THRESHOLDS
    eff_overlap_threshold = (
        cfg.consolidate_overlap_threshold if overlap_threshold is None else overlap_threshold
    )
    cues = merge_duplicate_layers(cues, thresholds=cfg)

    by_layer = _group_by_layer(cues)
    if len(by_layer) < 2:
        return resolve_simultaneous_multiline_cues(cues, asr_transcript=asr_transcript, thresholds=cfg)

    layers_by_lang = _revote_layer_languages(by_layer)
    dropped_cues: set[int] = set()

    for lang, lids in layers_by_lang.items():
        if len(lids) < 2 or lang == "und":
            continue
        sorted_lids = sorted(lids, key=lambda lid: len(by_layer[lid]), reverse=True)
        dominant_lid = sorted_lids[0]
        dom_cues = by_layer[dominant_lid]
        for sec_lid in sorted_lids[1:]:
            sec_cues = by_layer[sec_lid]
            if sec_cues:
                _merge_or_filter_secondary_layer(
                    dominant_lid, dom_cues, sec_cues, lang, eff_overlap_threshold, cfg, dropped_cues
                )

    if dropped_cues:
        cues = [c for c in cues if id(c) not in dropped_cues]

    cues = resolve_simultaneous_multiline_cues(cues, asr_transcript=asr_transcript, thresholds=cfg)
    cues.sort(key=lambda c: (c.start_ms, c.end_ms))
    return cues


def _padded_overlap(a: LyricCue, b: LyricCue, pad: int) -> bool:
    return a.start_ms < (b.end_ms + pad) and a.end_ms > (b.start_ms - pad)


def share_solitary_cross_layer_cues(
    cues: list[LyricCue],
    *,
    min_gap_ms: int | None = None,
    min_cue_dur_ms: int | None = None,
    thresholds: GroupingThresholds | None = None,
) -> list[LyricCue]:
    """Clone solitary English cues from one major layer into global silence gaps of other major layers."""
    if not cues:
        return []

    cfg = thresholds or DEFAULT_GROUPING_THRESHOLDS
    eff_min_gap_ms = cfg.share_min_gap_ms if min_gap_ms is None else min_gap_ms
    eff_min_cue_dur_ms = cfg.share_min_cue_dur_ms if min_cue_dur_ms is None else min_cue_dur_ms

    by_layer = _group_by_layer(cues)
    if len(by_layer) < 2:
        return cues

    max_count = max(len(lc) for lc in by_layer.values())
    major_lids = [
        lid for lid, lc in by_layer.items()
        if len(lc) >= cfg.share_major_layer_min_cues
        or (len(lc) >= cfg.share_major_layer_min_small_cues and len(lc) >= cfg.share_major_layer_min_ratio * max_count)
    ]
    if len(major_lids) < 2:
        return cues

    layer_spans = {lid: (min(c.start_ms for c in by_layer[lid]), max(c.end_ms for c in by_layer[lid])) for lid in major_lids}
    layer_langs = {lid: _layer_lang(by_layer[lid]) for lid in major_lids}
    pad, margin, slack = cfg.share_solitary_pad_ms, cfg.share_span_margin_ms, cfg.share_neighbor_slack_ms
    added: list[LyricCue] = []

    for src_lid in major_lids:
        for cue in by_layer[src_lid]:
            if (cue.end_ms - cue.start_ms) < eff_min_cue_dur_ms or "cross_layer_inherited" in cue.qa_flags:
                continue
            if cue.lang != "en" or detect_line_lang(cue.text) != "en":
                continue
            # Strict global-frame solitary check across every other major layer
            if any(_padded_overlap(oc, cue, pad) for lid in major_lids if lid != src_lid for oc in by_layer[lid]):
                continue

            for dst_lid in major_lids:
                if dst_lid == src_lid:
                    continue
                span_min, span_max = layer_spans[dst_lid]
                if cue.start_ms < span_min - margin or cue.end_ms > span_max + margin:
                    continue
                if any(ac.layer_id == dst_lid and _padded_overlap(ac, cue, pad) for ac in added):
                    continue

                dst_cues = by_layer[dst_lid]
                prev_ends = [dc.end_ms for dc in dst_cues if dc.end_ms <= cue.start_ms + slack]
                next_starts = [dc.start_ms for dc in dst_cues if dc.start_ms >= cue.end_ms - slack]
                left_bound = max(prev_ends) if prev_ends else (span_min - margin)
                right_bound = min(next_starts) if next_starts else (span_max + margin)
                if (right_bound - left_bound) < eff_min_gap_ms:
                    continue

                dst_lang = layer_langs[dst_lid]
                added.append(
                    replace(
                        cue,
                        source_track_ids=list(cue.source_track_ids),
                        source_segment_ids=list(cue.source_segment_ids),
                        bbox=list(cue.bbox),
                        qa_flags=list(cue.qa_flags) + ["cross_layer_inherited"],
                        words=[dict(w) for w in cue.words],
                        layer_id=dst_lid,
                        lang=dst_lang if dst_lang != "und" else cue.lang,
                    )
                )

    if not added:
        return cues
    combined = list(cues) + added
    combined.sort(key=_cue_sort_key)
    return combined
