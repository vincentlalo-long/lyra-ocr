"""Language and script detection with specialized Romaji recognition."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Sequence

# Regex matching valid Japanese Romanization (Hepburn / Kunrei-shiki) mora sequences
_ROMAJI_TOKEN_RE = re.compile(
    r"^(?:(?:ch|sh|ts|[kgsztdnhbpmyrwfjv]y?)?[aeiouāīūēōâîûêô]|n(?![aeiouyāīūēōâîûêô])|(?:kk|ss|tt|pp|cc|dd|gg|zz|bb))+$",
    re.IGNORECASE,
)

# Common short English words that coincidentally match Japanese mora structure
_AMBIGUOUS_ENGLISH_SHORT_WORDS = {
    "a", "i", "to", "do", "so", "no", "go", "me", "he", "be", "we",
    "at", "in", "on", "as", "is", "it", "or", "an", "am", "us", "by", "my",
}

# Vietnamese-specific vowels with diacritics / consonants
_VIETNAMESE_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)


def romaji_score(text: str) -> float:
    """Calculate the romaji confidence score of a Latin text string in [0.0, 1.0]."""
    if not text or not text.strip():
        return 0.0

    # Extract alphabetic words
    tokens = re.findall(r"[A-Za-zāīūēōâîûêô]+", text)
    if not tokens:
        return 0.0

    total_tokens = len(tokens)
    romaji_tokens = 0
    unambiguous_romaji_tokens = 0
    non_romaji_tokens = 0

    for tok in tokens:
        tok_lower = tok.lower()
        if _ROMAJI_TOKEN_RE.fullmatch(tok_lower):
            romaji_tokens += 1
            if tok_lower not in _AMBIGUOUS_ENGLISH_SHORT_WORDS:
                unambiguous_romaji_tokens += 1
        else:
            non_romaji_tokens += 1

    # If there are any clearly non-romaji Latin tokens, heavily penalize
    if non_romaji_tokens > 0:
        return 0.0

    # Fraction of unambiguous romaji words among all tokens
    return unambiguous_romaji_tokens / total_tokens


def detect_line_lang(text: str) -> str:
    """Classify the language/script of a single line of text.

    Returns:
        'ja'      : Japanese containing Hiragana/Katakana
        'ko'      : Korean containing Hangul
        'vi'      : Vietnamese containing Vietnamese diacritics/characters
        'zh'      : Chinese / CJK Unified Ideographs without Kana/Hangul
        'ja-Latn' : Japanese written in Romaji (Latin script)
        'en'      : English or generic Latin text
        'und'     : Undetermined (empty, digits, punctuation only)
    """
    if not text or not text.strip():
        return "und"

    norm = unicodedata.normalize("NFC", text.strip())

    has_kana = False
    has_hangul = False
    has_cjk = False
    has_latin = False

    for ch in norm:
        code = ord(ch)
        # Hiragana: 0x3040-0x309F, Katakana: 0x30A0-0x30FF
        if (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF):
            has_kana = True
        # Hangul Syllables & Jamo: 0xAC00-0xD7AF, 0x1100-0x11FF, 0x3130-0x318F
        elif (0xAC00 <= code <= 0xD7AF) or (0x1100 <= code <= 0x11FF) or (0x3130 <= code <= 0x318F):
            has_hangul = True
        # CJK Unified Ideographs: 0x4E00-0x9FFF, 0x3400-0x4DBF
        elif (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF):
            has_cjk = True
        elif ch.isalpha():
            has_latin = True

    # 1. Asian Scripts
    if has_kana:
        return "ja"
    if has_hangul:
        return "ko"
    if has_cjk and not has_latin:
        return "zh"

    # 2. Vietnamese Diacritics
    if _VIETNAMESE_RE.search(norm):
        return "vi"

    # 3. Latin Script (Romaji vs English/Latin)
    if has_latin:
        score = romaji_score(norm)
        # Require score >= 0.6 and at least 2 tokens, or single high-confidence token
        tokens = re.findall(r"[A-Za-zāīūēōâîûêô]+", norm)
        if score >= 0.6 and len(tokens) >= 2:
            return "ja-Latn"
        return "en"

    return "und"


def vote_layer_lang(line_langs: Sequence[str]) -> str:
    """Aggregate per-line language classifications for an entire spatial text layer.

    Promotes pure CJK lines ('zh') to Japanese ('ja') if the layer contains Japanese Kana lines.
    """
    valid = [l for l in line_langs if l and l != "und"]
    if not valid:
        return "und"

    has_ja = any(l == "ja" for l in valid)
    normalized: list[str] = []
    for l in valid:
        # Contextual promotion: if a track/layer has Kana, pure Kanji lines belong to Japanese
        if l == "zh" and has_ja:
            normalized.append("ja")
        else:
            normalized.append(l)

    counts = Counter(normalized)
    return counts.most_common(1)[0][0]
