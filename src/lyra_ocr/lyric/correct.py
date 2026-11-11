from __future__ import annotations

from collections import Counter
import re
import unicodedata
from typing import Any, Mapping

from lyra_ocr.lyric.models import CorrectionConfig, CorrectionInfo
from lyra_ocr.lyric.text_repair import RepairConfig, repair_text, text_quality
from lyra_ocr.track import string_similarity


def _levenshtein(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1] + [0] * len(s2)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row[j + 1] = min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            )
        prev_row = curr_row
    return prev_row[-1]


def _normalize_string(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_cjk(s: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]", s))


def _match_casing(candidate: str, original: str) -> str:
    if original.isupper():
        return candidate.upper()
    if original.istitle():
        return candidate.capitalize()
    if original.islower():
        return candidate.lower()
    return candidate


def _spacing_insensitive(s: str) -> str:
    """Case- and whitespace-insensitive key used for variant clustering.

    Missing-space variants such as ``nothinginmy`` must land in the same
    cluster as ``nothing in my`` so the quality-weighted representative can
    choose the better one.
    """
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s)).casefold()


def build_corpus_vocab(
    all_tracks: list[dict[str, Any]],
    *,
    only_accepted: bool = True,
    human_labels: dict[str, Any] | None = None,
) -> Counter[str]:
    vocab: Counter[str] = Counter()
    for tr in all_tracks:
        if not isinstance(tr, dict):
            continue
        tid = str(tr.get("track_id", ""))
        if human_labels:
            hl = human_labels.get(tid)
            if hl:
                h_decision = hl.get("decision") or hl.get("human_decision")
                h_label = hl.get("human_label") or hl.get("label")
                if h_decision == "reject" or h_label == "non_lyric":
                    continue

        if only_accepted:
            label = tr.get("label")
            decision = tr.get("decision")
            # If classification info is present, exclude rejected and non-lyric tracks
            if label == "non_lyric" or decision == "reject":
                continue

        texts: list[str] = []
        tr_features = tr.get("features") or {}
        if not isinstance(tr_features, dict):
            tr_features = {}
        dom = tr_features.get("dominant_text")
        if dom:
            texts.append(dom)
        for seg in tr.get("segments") or []:
            if isinstance(seg, dict) and seg.get("text"):
                texts.append(seg["text"])
        for occ in tr.get("occurrences") or []:
            if isinstance(occ, dict) and occ.get("text_norm"):
                texts.append(occ["text_norm"])

        # The same line appears as dominant_text and as segment text; count it
        # once per track so a single noisy line cannot look corpus-frequent.
        for t in dict.fromkeys(texts):
            cleaned = _normalize_string(t)
            tokens = re.findall(r"\b\w+\b", cleaned)
            for tok in tokens:
                if not _is_cjk(tok) and not re.search(r"\d", tok):
                    vocab[tok.lower()] += 1
    return vocab


def _variant_quality(text: str, vocab: Mapping[str, int] | None, cache: dict[str, float]) -> float:
    if text not in cache:
        cache[text] = text_quality(text, vocab)
    return cache[text]


def _cluster_variants(
    variants: list[tuple[str, float]],
    vocab: Mapping[str, int] | None,
    quality_cache: dict[str, float],
) -> list[list[tuple[str, float]]]:
    clusters: list[list[tuple[str, float]]] = []
    keys: list[str] = []
    for var, conf in variants:
        key = _spacing_insensitive(var)
        assigned = False
        for idx, cluster in enumerate(clusters):
            if string_similarity(key, keys[idx]) >= 0.85:
                cluster.append((var, conf))
                assigned = True
                break
        if not assigned:
            clusters.append([(var, conf)])
            keys.append(key)
    return clusters


def _representative(
    cluster: list[tuple[str, float]],
    vocab: Mapping[str, int] | None,
    quality_cache: dict[str, float],
) -> tuple[str, float]:
    def _score(item: tuple[str, float]) -> tuple[float, int]:
        text, conf = item
        quality = _variant_quality(text, vocab, quality_cache)
        # Confidence dominates, lexical quality breaks OCR ties. Word count is
        # a final tie-break so a spaced variant wins over a merged one.
        return (0.55 * conf + 0.45 * quality, len(text.split()))

    return max(cluster, key=_score)


def correct_text(
    track: dict[str, Any],
    all_tracks: list[dict[str, Any]],
    config: CorrectionConfig,
    *,
    vocab: Counter[str] | None = None,
    segment: dict[str, Any] | None = None,
) -> tuple[str, str, CorrectionInfo]:
    variants: list[tuple[str, float]] = []

    if segment is not None:
        seg_text = (
            segment.get("text")
            or segment.get("corrected_text")
            or (segment.get("text_variants", [""])[0] if segment.get("text_variants") else "")
        )
        seg_conf = float(segment.get("confidence", 0.0))
        if seg_text:
            norm_seg = _normalize_string(seg_text)
            if norm_seg:
                variants.append((norm_seg, seg_conf))

        target_seg_id = segment.get("segment_id")
        for occ in track.get("occurrences") or []:
            if not isinstance(occ, dict):
                continue
            if target_seg_id is not None and occ.get("segment_id") is not None:
                if occ.get("segment_id") != target_seg_id:
                    continue
            t = occ.get("text_norm") or occ.get("text_raw") or occ.get("text", "")
            c = float(occ.get("confidence", 0.0))
            norm_t = _normalize_string(t)
            if norm_t:
                variants.append((norm_t, c))
    else:
        features = track.get("features") or {}
        if not isinstance(features, dict):
            features = {}
        dom = features.get("dominant_text", "")
        dom_conf = float(features.get("avg_confidence", track.get("confidence", 0.0)))
        if dom:
            norm_dom = _normalize_string(dom)
            if norm_dom:
                variants.append((norm_dom, dom_conf))

        for seg in track.get("segments") or []:
            if not isinstance(seg, dict):
                continue
            t = (
                seg.get("text")
                or seg.get("corrected_text")
                or (seg.get("text_variants", [""])[0] if seg.get("text_variants") else "")
            )
            c = float(seg.get("confidence", 0.0))
            norm_t = _normalize_string(t)
            if norm_t:
                variants.append((norm_t, c))

        for occ in track.get("occurrences") or []:
            if not isinstance(occ, dict):
                continue
            t = occ.get("text_norm") or occ.get("text_raw") or occ.get("text", "")
            c = float(occ.get("confidence", 0.0))
            norm_t = _normalize_string(t)
            if norm_t:
                variants.append((norm_t, c))

    if not variants:
        return "", "", CorrectionInfo(method="none", changed=False)

    raw_text = variants[0][0]

    # Method: none
    if config.method == "none":
        best_text, best_conf = max(variants, key=lambda v: v[1])
        return best_text, raw_text, CorrectionInfo(
            method="none",
            changed=False,
            confidence=best_conf,
        )

    # Clustering, quality weighting and repair all need lexical frequencies,
    # regardless of whether the caller selected dictionary or medoid correction.
    effective_vocab: Mapping[str, int] = (
        vocab if vocab is not None else build_corpus_vocab(all_tracks)
    )

    quality_cache: dict[str, float] = {}
    clusters = _cluster_variants(variants, effective_vocab, quality_cache)

    # Weight cluster support by confidence and lexical quality so a repeated
    # merged/garbled variant cannot outvote a single clean variant.
    def _cluster_weight(cluster: list[tuple[str, float]]) -> float:
        return sum(
            conf * (0.5 + 0.5 * _variant_quality(text, effective_vocab, quality_cache))
            for text, conf in cluster
        )

    best_cluster = max(clusters, key=_cluster_weight)
    medoid, medoid_conf = _representative(best_cluster, effective_vocab, quality_cache)

    if config.method == "consensus":
        return medoid, raw_text, CorrectionInfo(
            method="consensus",
            changed=(medoid != raw_text),
            confidence=medoid_conf,
        )

    # Method: dictionary (corpus-derived) + optional conservative NLP repair
    # Pattern for ordinals, short alphanumeric codes, or decade specs (e.g. 1st, 2nd, 3D, 4K, 2U, V8, 90s)
    # that should be preserved as-is rather than being coerced into dictionary words.
    ordinal_or_code_re = re.compile(
        r"^(?:\d+(?:st|nd|rd|th)|\d{1,4}[a-zA-Z]{1,2}|[a-zA-Z]\d{1,4}|\d{2,4}s)$",
        re.IGNORECASE,
    )

    def _replace_token(match: re.Match) -> str:
        tok = match.group(0)
        if _is_cjk(tok) or tok.isdigit() or ordinal_or_code_re.match(tok):
            return tok
        if len(tok) < config.min_token_len:
            return tok

        tok_lower = tok.lower()
        tok_freq = effective_vocab.get(tok_lower, 0)
        if tok_freq >= config.min_vocab_freq:
            return tok

        max_dist = max(1, int(len(tok) * config.max_edit_ratio))
        candidates: list[tuple[str, int]] = []
        min_multiplier = 4 if tok_freq > 0 else 3

        for cand, cand_freq in effective_vocab.items():
            if cand_freq >= min_multiplier * max(1, tok_freq):
                if abs(len(cand) - len(tok_lower)) <= max_dist:
                    dist = _levenshtein(tok_lower, cand)
                    if dist <= max_dist:
                        candidates.append((cand, cand_freq))

        if len(candidates) == 1:
            best_cand = candidates[0][0]
            return _match_casing(best_cand, tok)

        return tok

    corrected = re.sub(r"\b\w+\b", _replace_token, medoid)
    cleaned_final = _normalize_string(corrected)
    if not cleaned_final:
        cleaned_final = medoid

    if config.enable_nlp and not _is_cjk(cleaned_final):
        repaired = repair_text(
            cleaned_final,
            config=RepairConfig(
                min_token_len=config.nlp_min_token_len,
                max_edit=config.nlp_max_edit,
                split_penalty=config.nlp_split_penalty,
            ),
            vocab=effective_vocab,
        )
        repaired = _normalize_string(repaired)
        if repaired:
            cleaned_final = repaired

    diff_str = f"'{raw_text}' -> '{cleaned_final}'" if cleaned_final != raw_text else ""

    return cleaned_final, raw_text, CorrectionInfo(
        method="dictionary",
        changed=(cleaned_final != raw_text),
        confidence=medoid_conf,
        diff=diff_str,
    )
