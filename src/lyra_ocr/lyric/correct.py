from __future__ import annotations

from collections import Counter
import re
import unicodedata
from typing import Any

from lyra_ocr.lyric.models import CorrectionConfig, CorrectionInfo
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


def build_corpus_vocab(all_tracks: list[dict[str, Any]]) -> Counter[str]:
    vocab: Counter[str] = Counter()
    for tr in all_tracks:
        texts: list[str] = []
        dom = tr.get("features", {}).get("dominant_text")
        if dom:
            texts.append(dom)
        for seg in tr.get("segments", []):
            if seg.get("text"):
                texts.append(seg["text"])
        for occ in tr.get("occurrences", []):
            if occ.get("text_norm"):
                texts.append(occ["text_norm"])

        for t in texts:
            cleaned = _normalize_string(t)
            tokens = re.findall(r"\b\w+\b", cleaned)
            for tok in tokens:
                if not _is_cjk(tok) and not re.search(r"\d", tok):
                    vocab[tok.lower()] += 1
    return vocab


def correct_text(
    track: dict[str, Any],
    all_tracks: list[dict[str, Any]],
    config: CorrectionConfig,
    *,
    vocab: Counter[str] | None = None,
) -> tuple[str, str, CorrectionInfo]:
    variants: list[tuple[str, float]] = []

    dom = track.get("features", {}).get("dominant_text", "")
    dom_conf = float(track.get("features", {}).get("avg_confidence", track.get("confidence", 0.0)))
    if dom:
        norm_dom = _normalize_string(dom)
        if norm_dom:
            variants.append((norm_dom, dom_conf))

    for seg in track.get("segments", []):
        t = seg.get("text", "")
        c = float(seg.get("confidence", 0.0))
        norm_t = _normalize_string(t)
        if norm_t:
            variants.append((norm_t, c))

    for occ in track.get("occurrences", []):
        t = occ.get("text_norm", "")
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

    # Consensus clustering
    # Cluster variants with string_similarity >= 0.85
    clusters: list[list[tuple[str, float]]] = []
    for var, conf in variants:
        assigned = False
        for cl in clusters:
            rep = cl[0][0]
            if string_similarity(var, rep) >= 0.85:
                cl.append((var, conf))
                assigned = True
                break
        if not assigned:
            clusters.append([(var, conf)])

    # Select best cluster by sum(confidence)
    best_cluster = max(clusters, key=lambda cl: sum(c for _, c in cl))

    # Medoid of best cluster
    if len(best_cluster) == 1:
        medoid, medoid_conf = best_cluster[0]
    else:
        candidates = [v[0] for v in best_cluster]
        scores: list[tuple[float, float, str]] = []
        for v, c in best_cluster:
            sim_sum = sum(string_similarity(v, other) for other in candidates)
            scores.append((sim_sum, c, v))
        scores.sort(key=lambda s: (s[0], s[1]), reverse=True)
        medoid = scores[0][2]
        medoid_conf = scores[0][1]

    if config.method == "consensus":
        return medoid, raw_text, CorrectionInfo(
            method="consensus",
            changed=(medoid != raw_text),
            confidence=medoid_conf,
        )

    # Method: dictionary (corpus-derived)
    effective_vocab = vocab if vocab is not None else build_corpus_vocab(all_tracks)

    def _replace_token(match: re.Match) -> str:
        tok = match.group(0)
        if _is_cjk(tok) or tok.isdigit():
            return tok
        if len(tok) < config.min_token_len:
            return tok

        tok_lower = tok.lower()
        tok_freq = effective_vocab.get(tok_lower, 0)
        if tok_freq >= config.min_vocab_freq:
            return tok

        max_dist = max(1, int(len(tok) * config.max_edit_ratio))
        candidates: list[tuple[str, int]] = []

        for cand, cand_freq in effective_vocab.items():
            if cand_freq >= 3 * max(1, tok_freq):
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

    return cleaned_final, raw_text, CorrectionInfo(
        method="dictionary",
        changed=(cleaned_final != raw_text),
        confidence=medoid_conf,
    )
