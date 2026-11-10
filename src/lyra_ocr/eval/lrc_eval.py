from __future__ import annotations

import re
import string
import unicodedata
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import theilslopes

from lyra_ocr.lyric.models import LyricCue
from lyra_ocr.track import string_similarity


def _levenshtein_tokens(s1: list[str], s2: list[str]) -> int:
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, tok1 in enumerate(s1):
        curr_row = [i + 1] + [0] * len(s2)
        for j, tok2 in enumerate(s2):
            cost = 0 if tok1 == tok2 else 1
            curr_row[j + 1] = min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
                prev_row[j] + cost,
            )
        prev_row = curr_row
    return prev_row[-1]


def _levenshtein_chars(s1: str, s2: str) -> int:
    return _levenshtein_tokens(list(s1), list(s2))


def normalize_text_for_eval(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    # Strip punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_text_selection(
    cues: list[LyricCue],
    reference_lines: list[str],
    *,
    match_threshold: float = 0.60,
) -> dict[str, Any]:
    norm_gen = [normalize_text_for_eval(c.text) for c in cues]
    norm_gen = [t for t in norm_gen if t]

    norm_ref = [normalize_text_for_eval(l) for l in reference_lines]
    norm_ref = [t for t in norm_ref if t]

    if not norm_ref or not norm_gen:
        return {
            "scorable": False,
            "line_recall": 0.0,
            "line_precision": 0.0,
            "line_f1": 0.0,
            "cer": 0.0 if (not norm_ref and not norm_gen) else 1.0,
            "wer": 0.0 if (not norm_ref and not norm_gen) else 1.0,
            "extra_line_rate": 1.0 if norm_gen else 0.0,
            "order_inversions": 0,
            "order_inversion_rate": 0.0,
        }

    cost_matrix = np.zeros((len(norm_gen), len(norm_ref)), dtype=np.float32)
    for i, g_text in enumerate(norm_gen):
        for j, r_text in enumerate(norm_ref):
            sim = string_similarity(g_text, r_text)
            cost_matrix[i, j] = 1.0 - sim

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs: list[tuple[str, str]] = []
    matched_indices: list[tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        if cost <= (1.0 - match_threshold):
            matched_pairs.append((norm_gen[r], norm_ref[c]))
            matched_indices.append((r, c))

    # Evaluate temporal order consistency
    matched_indices.sort(key=lambda p: p[0])
    order_inversions = 0
    ref_order = [c for _, c in matched_indices]
    for i in range(len(ref_order)):
        for j in range(i + 1, len(ref_order)):
            if ref_order[i] > ref_order[j]:
                order_inversions += 1
    n_possible_pairs = max(1, len(ref_order) * (len(ref_order) - 1) // 2)
    order_inversion_rate = round(order_inversions / n_possible_pairs, 4) if len(ref_order) > 1 else 0.0

    n_matched = len(matched_pairs)
    line_recall = n_matched / len(norm_ref)
    line_precision = n_matched / len(norm_gen)
    if line_recall + line_precision > 0:
        line_f1 = 2.0 * line_precision * line_recall / (line_precision + line_recall)
    else:
        line_f1 = 0.0

    extra_line_rate = max(0.0, (len(norm_gen) - n_matched) / len(norm_gen))

    # Version gate
    scorable = not (line_recall < 0.50 and line_precision < 0.50)

    # Compute CER and WER over matched pairs
    total_char_dist = 0
    total_ref_chars = 0
    total_word_dist = 0
    total_ref_words = 0

    for g_text, r_text in matched_pairs:
        total_char_dist += _levenshtein_chars(g_text, r_text)
        total_ref_chars += max(1, len(r_text))

        g_words = g_text.split()
        r_words = r_text.split()
        total_word_dist += _levenshtein_tokens(g_words, r_words)
        total_ref_words += max(1, len(r_words))

    cer = total_char_dist / max(1, total_ref_chars) if matched_pairs else 1.0
    wer = total_word_dist / max(1, total_ref_words) if matched_pairs else 1.0

    return {
        "scorable": scorable,
        "line_recall": round(line_recall, 4),
        "line_precision": round(line_precision, 4),
        "line_f1": round(line_f1, 4),
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "extra_line_rate": round(extra_line_rate, 4),
        "order_inversions": order_inversions,
        "order_inversion_rate": order_inversion_rate,
        "n_matched": n_matched,
        "n_generated": len(norm_gen),
        "n_reference": len(norm_ref),
    }


def evaluate_synced_reference(
    cues: list[LyricCue],
    reference_lrc: list[tuple[int, str]],
    *,
    min_match_rate: float = 0.60,
) -> dict[str, Any]:
    if not cues or not reference_lrc:
        return {"scorable": False, "status": "empty"}

    norm_cues = [(c.start_ms, normalize_text_for_eval(c.text)) for c in cues if c.text.strip()]
    norm_refs = [(t, normalize_text_for_eval(s)) for t, s in reference_lrc if s.strip()]

    if not norm_cues or not norm_refs:
        return {"scorable": False, "status": "empty"}

    cost_matrix = np.zeros((len(norm_cues), len(norm_refs)), dtype=np.float32)
    for i, (_, g_text) in enumerate(norm_cues):
        for j, (_, r_text) in enumerate(norm_refs):
            sim = string_similarity(g_text, r_text)
            cost_matrix[i, j] = 1.0 - sim

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_times: list[tuple[int, int]] = []
    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        if cost <= 0.40:
            matched_times.append((norm_cues[r][0], norm_refs[c][0]))

    match_rate = len(matched_times) / max(len(norm_cues), len(norm_refs))
    if match_rate < min_match_rate or len(matched_times) < 2:
        return {
            "scorable": False,
            "status": "unmatched",
            "match_rate": round(match_rate, 3),
        }

    t_gen = np.array([m[0] for m in matched_times], dtype=np.float64)
    t_ref = np.array([m[1] for m in matched_times], dtype=np.float64)

    res = theilslopes(t_gen, t_ref)
    scale = float(res.slope)
    offset_ms = float(res.intercept)

    fitted_gen = scale * t_ref + offset_ms
    residuals = np.abs(t_gen - fitted_gen)
    residual_median_ms = float(np.median(residuals))

    return {
        "status": "ok",
        "kind": "lrclib_synced_weak",
        "scorable": True,
        "offset_ms": round(offset_ms, 1),
        "scale": round(scale, 4),
        "residual_median_ms": round(residual_median_ms, 1),
        "n_matched": len(matched_times),
        "match_rate": round(match_rate, 3),
    }
