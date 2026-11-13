"""Similarity scoring, DP alignment, gap filling, and overlap reconciliation for lyrics."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from lyra_ocr.lyric.models import CorrectionInfo, LyricCue, ReferenceConfig
from lyra_ocr.lyric.ref_models import (
    AlignmentResult,
    GapFillResult,
    MatchStats,
    ReferenceBundle,
    ReferenceLine,
    _WORD_RE,
    _squash,
    _tokenize,
    best_reference_similarity,
    line_match_stats,
    line_similarity,
)

__all__ = [
    "AlignmentResult",
    "GapFillResult",
    "MatchStats",
    "align_reference",
    "best_reference_similarity",
    "fill_reference_gaps",
    "line_match_stats",
    "line_similarity",
    "reconcile_reference_overlaps",
    "repair_romaji_cue_text",
]

_CJK_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+")
_PARTICLE_EQUIV_MAP: dict[tuple[str, str], str] = {
    ("wa", "ha"): "wa",
    ("ha", "wa"): "wa",
    ("e", "he"): "e",
    ("he", "e"): "e",
}


def _match_word_case(replacement: str, original: str) -> str:
    if len(original) > 1 and original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _repair_romaji_pair(
    w_ocr: str,
    w_ref: str,
    *,
    ocr_confidence: float | None = None,
    is_asr: bool = False,
) -> str:
    lo, lr = w_ocr.lower(), w_ref.lower()
    if (lo, lr) in _PARTICLE_EQUIV_MAP:
        return _match_word_case(_PARTICLE_EQUIV_MAP[(lo, lr)], w_ocr)
    if (lo, lr) in {("o", "wo"), ("wo", "o")}:
        return w_ocr
    sim = line_similarity(lo, lr)
    if sim >= 0.55:
        return _match_word_case(w_ref, w_ocr) if sim < 1.0 else w_ocr
    if is_asr and (ocr_confidence is None or ocr_confidence < 0.85):
        return _match_word_case(w_ref, w_ocr)
    return w_ocr


def repair_romaji_cue_text(
    ocr_text: str,
    ref_text: str,
    *,
    ocr_confidence: float | None = None,
    is_asr: bool = False,
) -> str:
    """Token-level Romaji repair against romanized reference, guarding ateji and zero CJK."""
    from lyra_ocr.lyric.text_repair import repair_romaji_confusables

    ocr_work = repair_romaji_confusables((ocr_text or "").strip())
    ocr_work = _CJK_CHAR_RE.sub("", ocr_work).strip()
    ref_clean = _CJK_CHAR_RE.sub("", (ref_text or "").strip()).strip()
    if not ocr_work:
        return ""
    if not ref_clean:
        return re.sub(r"\s+", " ", ocr_work)

    ocr_words = ocr_work.split()
    ref_words = ref_clean.split()
    if not ocr_words or not ref_words:
        return re.sub(r"\s+", " ", ocr_work)

    sm = SequenceMatcher(
        None,
        [w.lower() for w in ocr_words],
        [w.lower() for w in ref_words],
        autojunk=False,
    )
    out_tokens: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "delete"):
            out_tokens.extend(ocr_words[i1:i2])
        elif tag == "insert":
            continue
        elif tag == "replace":
            n_o, n_r = i2 - i1, j2 - j1
            if n_o == n_r:
                for wo, wr in zip(ocr_words[i1:i2], ref_words[j1:j2]):
                    out_tokens.append(
                        _repair_romaji_pair(wo, wr, ocr_confidence=ocr_confidence, is_asr=is_asr)
                    )
            elif n_o == 1 and n_r >= 2:
                wo = ocr_words[i1]
                ref_joined = "".join(ref_words[j1:j2])
                sim = line_similarity(wo.lower(), ref_joined.lower())
                if sim >= 0.55:
                    cased_first = _match_word_case(ref_words[j1], wo)
                    rest = [
                        _match_word_case(w, wo) if wo.isupper() else w
                        for w in ref_words[j1 + 1 : j2]
                    ]
                    out_tokens.extend([cased_first, *rest])
                elif is_asr and (ocr_confidence is None or ocr_confidence < 0.85):
                    out_tokens.extend(ref_words[j1:j2])
                else:
                    out_tokens.append(wo)
            elif n_o >= 2 and n_r == 1:
                ocr_joined = "".join(ocr_words[i1:i2])
                wr = ref_words[j1]
                if line_similarity(ocr_joined.lower(), wr.lower()) >= 0.70:
                    out_tokens.append(_match_word_case(wr, ocr_words[i1]))
                else:
                    out_tokens.extend(ocr_words[i1:i2])
            else:
                out_tokens.extend(ocr_words[i1:i2])

    repaired = _CJK_CHAR_RE.sub("", " ".join(out_tokens))
    return re.sub(r"\s+", " ", repaired).strip()


def _project_ref_onto_cues(cue_texts: list[str], ref_text: str) -> list[str]:
    """Partition ref_text onto k consecutive OCR cue_texts."""
    if len(cue_texts) <= 1:
        return [ref_text]
    if not ref_text.strip():
        return list(cue_texts)

    boundaries: list[int] = []
    curr_len = 0
    for ct in cue_texts[:-1]:
        curr_len += len(ct)
        boundaries.append(curr_len)
        curr_len += 1

    joined_cues = " ".join(cue_texts)
    sm = SequenceMatcher(None, joined_cues, ref_text)
    matching_blocks = sm.get_matching_blocks()

    ref_splits: list[int] = []
    for b in boundaries:
        best_ref_pos = None
        for a_start, b_start, size in matching_blocks:
            if a_start <= b <= a_start + size:
                best_ref_pos = b_start + (b - a_start)
                break
        if best_ref_pos is None:
            best_ref_pos = int(round((b / max(1, len(joined_cues))) * len(ref_text)))
        ref_splits.append(best_ref_pos)

    adjusted_splits: list[int] = []
    for sp in ref_splits:
        best_sp = sp
        min_dist = 999
        for offset in range(-8, 9):
            pos = sp + offset
            if 0 <= pos < len(ref_text) and ref_text[pos].isspace():
                if abs(offset) < min_dist:
                    min_dist = abs(offset)
                    best_sp = pos
        adjusted_splits.append(best_sp)

    parts: list[str] = []
    prev_idx = 0
    for sp in adjusted_splits:
        parts.append(ref_text[prev_idx:sp].strip())
        prev_idx = sp
    parts.append(ref_text[prev_idx:].strip())
    return parts


def _update_cue_text(
    cue: LyricCue,
    replacement: str,
    *,
    is_romaji: bool,
    is_caption: bool,
    is_asr_rom: bool,
) -> bool:
    target = (
        repair_romaji_cue_text(cue.text, replacement, ocr_confidence=cue.ocr_confidence, is_asr=is_asr_rom)
        if is_romaji
        else replacement
    )
    if is_romaji:
        target = _CJK_CHAR_RE.sub("", target).strip()
    if target and target != cue.text:
        cue.correction = CorrectionInfo(
            method="reference_caption" if is_caption else "reference",
            changed=True,
            confidence=cue.confidence,
            diff=f"'{cue.text}' -> '{target}'",
        )
        cue.text = target
        return True
    return False


def _mark_review(cues_to_mark: list[LyricCue], *, is_romaji: bool, repair_romaji_cfg: bool) -> int:
    if is_romaji and not repair_romaji_cfg:
        return 0
    cnt = 0
    for c in cues_to_mark:
        c.review_required = True
        if "reference_uncertain" not in c.qa_flags:
            c.qa_flags.append("reference_uncertain")
        cnt += 1
    return cnt


def align_reference(
    cues: list[LyricCue],
    reference: list[str] | list[ReferenceLine] | ReferenceBundle,
    *,
    min_similarity: float | None = None,
    config: Any = None,
    source: str = "",
) -> AlignmentResult:
    """Replace cue text with the best monotonic reference match."""
    cfg = config or ReferenceConfig()

    ref_lines: list[ReferenceLine] = []
    eff_source = source
    if isinstance(reference, ReferenceBundle):
        ref_lines = reference.lines
        eff_source = source or reference.source
    elif isinstance(reference, list):
        if reference and isinstance(reference[0], ReferenceLine):
            ref_lines = reference  # type: ignore[assignment]
        else:
            ref_lines = [ReferenceLine(text=str(s)) for s in reference]

    if not cues or not ref_lines:
        return AlignmentResult()

    is_caption = "caption" in eff_source.lower()
    is_romaji_ref = eff_source in ("lrclib_romanized", "asr_romanized")
    is_asr_rom = eff_source == "asr_romanized"
    replace_thresh = cfg.caption_replace_similarity if is_caption else cfg.replace_similarity
    review_thresh = 0.75 if is_caption else cfg.review_similarity
    if min_similarity is not None:
        review_thresh = min_similarity

    n, m = len(cues), len(ref_lines)
    from lyra_ocr.lyric.text_repair import repair_romaji_confusables

    cue_cmp_texts = [
        repair_romaji_confusables(c.text) if (c.lang == "ja-Latn" or is_romaji_ref) else c.text
        for c in cues
    ]
    stats_matrix = [
        [line_match_stats(cue_cmp_texts[i], ref_lines[j].text) for j in range(m)]
        for i in range(n)
    ]

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    choice: list[list[tuple[int, int]]] = [[(1, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best_score = dp[i - 1][j]
            best_trans = (1, 0)
            if dp[i][j - 1] > best_score:
                best_score = dp[i][j - 1]
                best_trans = (0, 1)

            for kc, kr in ((1, 1), (1, 2), (1, 3), (2, 1), (3, 1)):
                if i < kc or j < kr:
                    continue
                if kc == 1 and kr == 1:
                    st = stats_matrix[i - 1][j - 1]
                else:
                    cand_cue_text = " ".join(cue_cmp_texts[idx] for idx in range(i - kc, i))
                    cand_ref_text = " ".join(ref_lines[jdx].text for jdx in range(j - kr, j))
                    la, lb = len(cand_cue_text), len(cand_ref_text)
                    if (2.0 * min(la, lb) / max(1, la + lb)) < review_thresh * 0.85:
                        continue
                    st = line_match_stats(cand_cue_text, cand_ref_text)

                if st.similarity >= review_thresh:
                    weight = 1.02 if (kc > 1 or kr > 1) else 1.0
                    match_score = dp[i - kc][j - kr] + st.similarity * weight
                    if match_score > best_score + 1e-9:
                        best_score = match_score
                        best_trans = (kc, kr)

            dp[i][j] = best_score
            choice[i][j] = best_trans

    actions: list[tuple[tuple[int, int], list[int], list[int]]] = []
    i, j = n, m
    while i > 0 and j > 0:
        trans = choice[i][j]
        kc, kr = trans
        if kc == 0 and kr == 1:
            j -= 1
        elif kc == 1 and kr == 0:
            i -= 1
        else:
            actions.append((trans, list(range(i - kc, i)), list(range(j - kr, j))))
            i -= kc
            j -= kr
    actions.reverse()

    pairs: list[tuple[int, int]] = []
    replaced_count = 0
    review_count = 0

    for (kc, kr), cue_idxs, ref_idxs in actions:
        for ci in cue_idxs:
            for ri in ref_idxs:
                pairs.append((ci, ri))

        if kc == 1:
            cue = cues[cue_idxs[0]]
            if cue.correction.method == "human":
                continue
            ref_clean = " ".join(ref_lines[ri].text.strip() for ri in ref_idxs)
            if not ref_clean:
                continue
            st = stats_matrix[cue_idxs[0]][ref_idxs[0]] if kr == 1 else line_match_stats(cue_cmp_texts[cue_idxs[0]], ref_clean)
            cue_is_romaji = cue.lang == "ja-Latn" or is_romaji_ref
            eff_replace = min(replace_thresh, 0.78) if cue_is_romaji else replace_thresh
            eff_cov = min(cfg.min_token_coverage, 0.40) if cue_is_romaji else cfg.min_token_coverage
            ratio_ok = (cfg.min_length_ratio <= st.length_ratio <= cfg.max_length_ratio) if kr == 1 else True
            guards_passed = (
                st.similarity >= eff_replace
                and st.token_coverage >= eff_cov
                and ratio_ok
                and bool(_WORD_RE.search(ref_clean))
            )
            if guards_passed:
                if not (cue_is_romaji and not cfg.repair_romaji):
                    if _update_cue_text(cue, ref_clean, is_romaji=cue_is_romaji, is_caption=is_caption, is_asr_rom=is_asr_rom):
                        replaced_count += 1
            else:
                review_count += _mark_review([cue], is_romaji=cue_is_romaji, repair_romaji_cfg=cfg.repair_romaji)

        elif kc > 1 and kr == 1:
            ocr_cues = [cues[ci] for ci in cue_idxs]
            if any(c.correction.method == "human" for c in ocr_cues):
                continue
            ref_clean = ref_lines[ref_idxs[0]].text.strip()
            cand_cue_text = " ".join(cue_cmp_texts[ci] for ci in cue_idxs)
            st = line_match_stats(cand_cue_text, ref_clean)
            cue_is_romaji = any(c.lang == "ja-Latn" for c in ocr_cues) or is_romaji_ref
            eff_replace = min(replace_thresh, 0.78) if cue_is_romaji else replace_thresh
            eff_cov = min(cfg.min_token_coverage, 0.40) if cue_is_romaji else cfg.min_token_coverage
            guards_passed = (
                st.similarity >= eff_replace
                and st.token_coverage >= eff_cov
                and bool(_WORD_RE.search(ref_clean))
            )
            if guards_passed:
                if not (cue_is_romaji and not cfg.repair_romaji):
                    for ci, part in zip(cue_idxs, _project_ref_onto_cues([c.text for c in ocr_cues], ref_clean)):
                        c = cues[ci]
                        rom = c.lang == "ja-Latn" or is_romaji_ref
                        if _update_cue_text(c, part, is_romaji=rom, is_caption=is_caption, is_asr_rom=is_asr_rom):
                            replaced_count += 1
            else:
                _mark_review(ocr_cues, is_romaji=cue_is_romaji, repair_romaji_cfg=cfg.repair_romaji)

    return AlignmentResult(
        matched_count=len({pair[0] for pair in pairs}),
        replaced_count=replaced_count,
        review_count=review_count,
        matched_pairs=pairs,
    )


def reconcile_reference_overlaps(
    cues: list[LyricCue],
    reference: list[ReferenceLine] | ReferenceBundle,
    *,
    matched_pairs: list[tuple[int, int]] | None = None,
    config: Any = None,
) -> tuple[list[LyricCue], list[str]]:
    """Split overlapping reference-backed cues inside their shared OCR interval."""
    cfg = config or ReferenceConfig()
    ref_lines: list[ReferenceLine] = []
    if isinstance(reference, ReferenceBundle):
        ref_lines = reference.lines
    elif isinstance(reference, list):
        ref_lines = reference
    if not cues or not ref_lines:
        return cues, []

    pairs = matched_pairs if matched_pairs is not None else _infer_anchor_pairs(cues, ref_lines, cfg.replace_similarity)
    ref_by_cue_id = {
        id(cues[ci]): ri for ci, ri in pairs if 0 <= ci < len(cues) and 0 <= ri < len(ref_lines)
    }

    sorted_cues = sorted(cues, key=lambda c: (c.start_ms, c.end_ms))
    groups: list[list[LyricCue]] = []
    group_end = -1
    for cue in sorted_cues:
        if groups and cue.start_ms < group_end:
            groups[-1].append(cue)
            group_end = max(group_end, cue.end_ms)
        else:
            groups.append([cue])
            group_end = cue.end_ms

    warnings: list[str] = []
    for group in groups:
        if len(group) < 2:
            continue
        entries: list[tuple[LyricCue, ReferenceLine]] = []
        for cue in group:
            ref_idx = ref_by_cue_id.get(id(cue))
            if ref_idx is not None:
                ref = ref_lines[ref_idx]
                if ref.start_ms is not None and cue.correction.method != "human":
                    entries.append((cue, ref))

        unres_msg = f"reference_overlap_unresolved at {group[0].start_ms}ms"
        if len(entries) < 2:
            if all(cue.correction.method.startswith("reference") for cue in group):
                warnings.append(unres_msg)
            continue

        if not cfg.split_overlapping_reference_cues:
            warnings.append(unres_msg)
            continue

        entries.sort(key=lambda item: (item[1].start_ms, item[1].text))
        starts = [int(ref.start_ms) for _, ref in entries]
        if any(b <= a for a, b in zip(starts, starts[1:])):
            warnings.append(unres_msg)
            continue

        group_start = min(cue.start_ms for cue in group)
        group_end = max(cue.end_ms for cue in group)
        boundaries = [
            max(group_start + 1, min(group_end - 1, (a + b) // 2))
            for a, b in zip(starts, starts[1:])
        ]
        if any(b <= a for a, b in zip(boundaries, boundaries[1:])) or any(not (group_start < b < group_end) for b in boundaries):
            warnings.append(unres_msg)
            continue

        for pos, (cue, _ref) in enumerate(entries):
            cue.start_ms = group_start if pos == 0 else boundaries[pos - 1]
            cue.end_ms = group_end if pos == len(entries) - 1 else boundaries[pos]
            if "reference_timing_split" not in cue.qa_flags:
                cue.qa_flags.append("reference_timing_split")

    return cues, warnings


def _infer_anchor_pairs(
    cues: list[LyricCue],
    ref_lines: list[ReferenceLine],
    threshold: float,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    next_ref = 0
    for cue_idx, cue in enumerate(cues):
        best_idx = None
        best_sim = threshold
        for ref_idx in range(next_ref, len(ref_lines)):
            sim = line_similarity(cue.text, ref_lines[ref_idx].text)
            if sim >= best_sim:
                best_sim = sim
                best_idx = ref_idx
        if best_idx is not None:
            pairs.append((cue_idx, best_idx))
            next_ref = best_idx + 1
    return pairs


def fill_reference_gaps(
    cues: list[LyricCue],
    reference: list[ReferenceLine] | ReferenceBundle,
    *,
    config: Any = None,
    source: str = "",
    duration_ms: int = 0,
    matched_pairs: list[tuple[int, int]] | None = None,
    allow_edge_gap_fill: bool | None = None,
    vocal_profile: Any = None,
) -> tuple[list[LyricCue], GapFillResult]:
    """Insert synchronized reference lines into anchored gaps between cues."""
    cfg = config or ReferenceConfig()
    if not cfg.enable_gap_fill:
        return list(cues), GapFillResult()

    if (
        any(c.lang == "ja-Latn" for c in cues)
        or (isinstance(reference, ReferenceBundle) and reference.language == "ja-Latn")
    ) and not cfg.repair_romaji:
        return list(cues), GapFillResult()

    eff_source = source
    ref_lines: list[ReferenceLine] = []
    if isinstance(reference, ReferenceBundle):
        ref_lines = reference.lines
        eff_source = source or reference.source
    elif isinstance(reference, list):
        ref_lines = reference

    is_caption = "caption" in eff_source.lower()
    if is_caption and not cfg.enable_caption_gap_fill:
        return list(cues), GapFillResult()

    if eff_source not in ("manual", "lrclib_synced", "youtube_captions") and not is_caption:
        return list(cues), GapFillResult()

    timed_entries = [(idx, ref) for idx, ref in enumerate(ref_lines) if ref.start_ms is not None]
    if not timed_entries:
        return list(cues), GapFillResult()

    timed_sorted = sorted(timed_entries, key=lambda item: (item[1].start_ms, item[0]))
    next_start: dict[int, int | None] = {
        idx: (timed_sorted[pos + 1][1].start_ms if pos + 1 < len(timed_sorted) else None)
        for pos, (idx, _ref) in enumerate(timed_sorted)
    }

    pairs = matched_pairs if matched_pairs is not None else _infer_anchor_pairs(cues, ref_lines, cfg.review_similarity)
    anchors = sorted({j for _, j in pairs if 0 <= j < len(ref_lines)})

    allow_edges = cfg.allow_edge_gap_fill if allow_edge_gap_fill is None else allow_edge_gap_fill
    if anchors:
        allowed: set[int] = set()
        if allow_edges:
            allowed.update(idx for idx, _ in timed_entries if idx < anchors[0] or idx > anchors[-1])
        for left, right in zip(anchors, anchors[1:]):
            allowed.update(idx for idx, _ in timed_entries if left < idx < right)
    elif allow_edges:
        allowed = {idx for idx, _ in timed_entries}
    else:
        allowed = set()

    recovered: list[LyricCue] = []
    warnings: list[str] = []
    existing_signatures = {(cue.start_ms, _squash(cue.text)) for cue in cues}

    for orig_idx, ref in timed_entries:
        if orig_idx not in allowed:
            continue
        ref_clean = ref.text.strip()
        if not ref_clean or len(_tokenize(ref_clean)) < 2:
            continue

        cand_start = ref.start_ms
        if cand_start is None:
            continue

        follow_start = next_start.get(orig_idx)
        if ref.end_ms is not None and ref.end_ms > cand_start:
            cand_end = ref.end_ms
        elif follow_start is not None and follow_start > cand_start:
            cand_end = max(cand_start + 1, follow_start - 100)
        else:
            cand_end = cand_start + 3000

        if duration_ms > 0:
            if cand_start >= duration_ms:
                continue
            cand_end = min(cand_end, duration_ms)

        if cand_end <= cand_start or (cand_start, _squash(ref_clean)) in existing_signatures:
            continue

        if any(max(cue.start_ms, cand_start) < min(cue.end_ms, cand_end) for cue in cues + recovered):
            warnings.append(f"reference_overlap_skipped at {cand_start}ms: {ref_clean[:60]}")
            continue

        if vocal_profile is not None:
            from lyra_ocr.audio.vad import vocal_active_ratio
            if vocal_active_ratio(vocal_profile, cand_start, cand_end) < 0.08:
                warnings.append(f"reference_gap_fill_skipped_silent at {cand_start}ms: {ref_clean[:60]}")
                continue

        confidence = 0.75 if is_caption else 0.90
        recovered.append(
            LyricCue(
                start_ms=cand_start,
                end_ms=cand_end,
                text=ref_clean,
                raw_text=ref_clean,
                confidence=confidence,
                decision="accept",
                review_required=False,
                correction=CorrectionInfo(
                    method="reference_gap_fill",
                    changed=True,
                    confidence=confidence,
                    diff=f"gap_fill: '{ref_clean}'",
                ),
                qa_flags=["caption_gap_fill"] if is_caption else [],
            )
        )
        existing_signatures.add((cand_start, _squash(ref_clean)))

    merged = sorted(cues + recovered, key=lambda c: (c.start_ms, c.end_ms))
    return merged, GapFillResult(filled_cues=recovered, n_filled=len(recovered), warnings=warnings)
