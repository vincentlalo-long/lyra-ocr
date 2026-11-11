#!/usr/bin/env python3
"""Evaluation-only text-quality metrics for generated LRC cues.

This script never feeds back into LRC generation. It inspects a ``.cues.json``
file (or an artifact directory containing one) and reports:

* cue/skip counts;
* accepted cues with low OCR confidence;
* suspicious text (merged tokens, low alphabetic ratio, repeated punctuation);
* chat/overlay leakage;
* lexical coverage / out-of-vocabulary rate;
* optional CER/WER against a reference lyrics file.

Usage:
    python scripts/eval_lrc_text.py path/to/song.cues.json
    python scripts/eval_lrc_text.py path/to/artifacts/ --reference ref.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from lyra_ocr.lyric.text_repair import get_lexicon

CJK_RANGES = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff"
TOKEN_RE = re.compile(rf"[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff'{CJK_RANGES}]+")
REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")

OVERLAY_PATTERNS = [
    re.compile(r"\blong time no see\b", re.IGNORECASE),
    re.compile(r"\bnew song\b", re.IGNORECASE),
    re.compile(r"\btest stream\b", re.IGNORECASE),
    re.compile(r"\blivestream\b", re.IGNORECASE),
    re.compile(r"\bsubscribe\b", re.IGNORECASE),
    re.compile(r"!{2,}"),
    re.compile(r"\?!|!\?"),
    re.compile(r"[:;=8][\-^o']?[)(\]\[DPpOo/\\|]"),
    re.compile(r"[oO0]\s*\(\s*['\u2019]?\^\s*\)\s*[oO0]"),
]


def load_cues(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load cues and skipped records from a cues.json file or artifact dir."""
    p = Path(path)
    if p.is_dir():
        candidates = sorted(p.glob("*.cues.json"), key=lambda item: item.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No *.cues.json found in {p}")
        p = candidates[-1]
    data = json.loads(p.read_text(encoding="utf-8"))
    cues = data.get("cues", [])
    skipped = data.get("skipped", [])
    return list(cues), list(skipped)


def _is_suspicious(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if REPEATED_PUNCT_RE.search(stripped):
        return True
    tokens = TOKEN_RE.findall(stripped)
    if not tokens:
        return False
    if any(len(token) >= 16 for token in tokens):
        return True
    non_space = sum(1 for ch in stripped if not ch.isspace())
    alpha = sum(1 for ch in stripped if ch.isalpha())
    return alpha / max(1, non_space) < 0.55


def _is_overlay(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in OVERLAY_PATTERNS)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current = [i + 1] + [0] * len(b)
        for j, cb in enumerate(b):
            current[j + 1] = min(
                current[j] + 1,
                previous[j + 1] + 1,
                previous[j] + (0 if ca == cb else 1),
            )
        previous = current
    return previous[-1]


def _normalize_reference(text: str) -> str:
    lowered = (text or "").lower()
    kept = re.sub(rf"[^a-z0-9' {CJK_RANGES}]+", " ", lowered)
    return re.sub(r"\s+", " ", kept).strip()


def _cer(reference: str, hypothesis: str) -> float:
    ref = _normalize_reference(reference)
    hyp = _normalize_reference(hypothesis)
    if not ref:
        return 0.0
    return _levenshtein(ref, hyp) / len(ref)


def _wer(reference: str, hypothesis: str) -> float:
    ref = _normalize_reference(reference).split()
    hyp = _normalize_reference(hypothesis).split()
    if not ref:
        return 0.0
    # Token-level Levenshtein.
    previous = list(range(len(hyp) + 1))
    for i, token in enumerate(ref):
        current = [i + 1] + [0] * len(hyp)
        for j, other in enumerate(hyp):
            current[j + 1] = min(
                current[j] + 1,
                previous[j + 1] + 1,
                previous[j] + (0 if token == other else 1),
            )
        previous = current
    return previous[-1] / len(ref)


def lexical_stats(texts: Iterable[str], vocab: dict[str, int] | None = None) -> dict[str, Any]:
    lexicon = get_lexicon()
    tokens: list[str] = []
    for text in texts:
        tokens.extend(token.lower() for token in TOKEN_RE.findall(text or ""))
    if not tokens:
        return {"tokens": 0, "known_tokens": 0, "oov_rate": 0.0, "lexical_coverage": 0.0}
    known = sum(
        1 for token in tokens if token in lexicon or (vocab and vocab.get(token, 0) > 0)
    )
    return {
        "tokens": len(tokens),
        "known_tokens": known,
        "oov_rate": round(1.0 - known / len(tokens), 4),
        "lexical_coverage": round(known / len(tokens), 4),
    }


def _timestamp_to_ms(match: re.Match[str]) -> int:
    mins = int(match.group(1))
    secs = int(match.group(2))
    millis_str = match.group(3) or "0"
    if len(millis_str) == 2:
        millis = int(millis_str) * 10
    elif len(millis_str) == 1:
        millis = int(millis_str) * 100
    else:
        millis = int(millis_str[:3])
    return mins * 60000 + secs * 1000 + millis


def _extract_ref_lines(ref_text: str) -> list[dict[str, Any]]:
    lines = []
    ts_re = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")
    for raw in ref_text.splitlines():
        s = raw.strip()
        if not s:
            continue
        timestamps = list(ts_re.finditer(s))
        if s.startswith("[") and s.endswith("]") and ":" in s and not timestamps:
            continue
        clean_text = ts_re.sub("", s).strip()
        if not clean_text:
            continue
        if timestamps:
            for match in timestamps:
                lines.append({"text": clean_text, "start_ms": _timestamp_to_ms(match)})
        else:
            lines.append({"text": clean_text, "start_ms": None})
    return lines


def compute_text_metrics(
    cues: list[dict[str, Any]],
    skipped: list[dict[str, Any]] | None = None,
    *,
    reference_text: str | None = None,
    min_confidence: float = 0.70,
    vocab: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compute text-quality metrics for a list of cue dictionaries."""
    texts = [str(cue.get("text") or "") for cue in cues]
    low_confidence = [
        cue for cue in cues if float(cue.get("confidence") or 0.0) < min_confidence
    ]
    suspicious = [text for text in texts if _is_suspicious(text)]
    overlay = [text for text in texts if _is_overlay(text)]

    replaced_count = 0
    suspicious_replacements = 0
    for cue in cues:
        corr = cue.get("correction") or {}
        method = str(corr.get("method") or "")
        changed = bool(corr.get("changed", False))
        if method.startswith("reference") and (changed or method == "reference_gap_fill"):
            replaced_count += 1
            raw_text = str(cue.get("raw_text") or "")
            text = str(cue.get("text") or "")
            raw_tokens = TOKEN_RE.findall(raw_text)
            text_tokens = TOKEN_RE.findall(text)
            ratio = len(text_tokens) / max(1, len(raw_tokens))
            cer_val = _cer(raw_text, text)
            if ratio < 0.60 or ratio > 1.80 or cer_val > 0.40:
                suspicious_replacements += 1

    metrics: dict[str, Any] = {
        "n_cues": len(cues),
        "n_skipped": len(skipped or []),
        "min_confidence": min_confidence,
        "low_confidence_count": len(low_confidence),
        "low_confidence_examples": [str(c.get("text") or "") for c in low_confidence[:10]],
        "suspicious_text_count": len(suspicious),
        "suspicious_examples": suspicious[:10],
        "overlay_leak_count": len(overlay),
        "overlay_examples": overlay[:10],
        "replaced_cue_count": replaced_count,
        "suspicious_replacement_count": suspicious_replacements,
        **lexical_stats(texts, vocab),
    }

    if reference_text is not None:
        ref_lines = _extract_ref_lines(reference_text)
        reference_for_diff = (
            "\n".join(str(item["text"]) for item in ref_lines)
            if ref_lines
            else reference_text
        )
        metrics["cer"] = round(_cer(reference_for_diff, "\n".join(texts)), 4)
        metrics["wer"] = round(_wer(reference_for_diff, "\n".join(texts)), 4)
        metrics["reference_note"] = "evaluation comparison; runtime provenance is recorded in cues.json"

        if ref_lines:
            matched_ref = 0
            for ref_item in ref_lines:
                ref_norm = _normalize_reference(ref_item["text"])
                for cue_text in texts:
                    cue_norm = _normalize_reference(cue_text)
                    if not ref_norm or not cue_norm:
                        continue
                    if ref_norm in cue_norm or cue_norm in ref_norm:
                        matched_ref += 1
                        break
                    elif _wer(ref_norm, cue_norm) <= 0.30 or _cer(ref_norm, cue_norm) <= 0.25:
                        matched_ref += 1
                        break
            metrics["missing_reference_lines"] = len(ref_lines) - matched_ref
            metrics["reference_line_recall"] = round(matched_ref / len(ref_lines), 4)
        else:
            metrics["missing_reference_lines"] = 0
            metrics["reference_line_recall"] = 1.0

    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate LRC text quality")
    parser.add_argument("path", help="Path to *.cues.json or artifact directory")
    parser.add_argument("--reference", default=None, help="Reference lyrics file (evaluation only)")
    parser.add_argument("--json", "-o", default=None, help="Write metrics JSON to this path")
    parser.add_argument("--min-confidence", type=float, default=0.70, help="Low-confidence threshold")
    args = parser.parse_args(argv)

    cues, skipped = load_cues(args.path)
    reference_text = None
    if args.reference:
        reference_text = Path(args.reference).read_text(encoding="utf-8")

    metrics = compute_text_metrics(
        cues,
        skipped,
        reference_text=reference_text,
        min_confidence=args.min_confidence,
    )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Cues              : {metrics['n_cues']} (skipped: {metrics['n_skipped']})")
    print(f"Low OCR confidence: {metrics['low_confidence_count']} (< {args.min_confidence})")
    print(f"Suspicious text   : {metrics['suspicious_text_count']}")
    print(f"Overlay leaks     : {metrics['overlay_leak_count']}")
    print(
        f"Lexical coverage  : {metrics['lexical_coverage'] * 100:.1f}% "
        f"(OOV: {metrics['oov_rate'] * 100:.1f}%)"
    )
    if "cer" in metrics:
        print(f"CER / WER         : {metrics['cer'] * 100:.2f}% / {metrics['wer'] * 100:.2f}%")
    if args.json:
        print(f"Report            : {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
