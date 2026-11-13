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
_ENGLISH_STOPWORDS = {
    "a", "i", "to", "do", "so", "no", "go", "me", "he", "be", "we", "you",
    "at", "in", "on", "as", "is", "it", "or", "an", "am", "us", "by", "my",
    "one", "more", "time", "take", "home", "make", "like", "life", "love",
    "come", "some", "same", "name", "game", "side", "ride", "hide", "wide",
    "wake", "face", "fade", "late", "gate", "hate", "fate", "date", "rate",
    "hope", "rope", "note", "vote", "rise", "wise", "rule", "pure", "sure",
    "cure", "fire", "wire", "tire", "care", "share", "bare", "rare",
    "mine", "fine", "line", "shine", "wine", "nine", "live", "give", "save",
    "wave", "gave", "have", "here", "there", "where", "were", "gone", "done",
    "none", "bone", "tone", "zone", "alone", "before", "above", "alive",
    "away", "again", "into", "onto", "upon", "over", "ever", "never", "under",
    "eyes", "open", "baby", "nite", "night", "sky", "fly", "cry", "try",
}
_AMBIGUOUS_ENGLISH_SHORT_WORDS = _ENGLISH_STOPWORDS

# English silent-e detector: CVCe pattern
_ENGLISH_SILENT_E_RE = re.compile(r"^[bcdfghjklmnpqrstvwxyz][aeiou][bcdfghjklmnpqrstvwxyz]e$")

# Genuine common 2-mora Romaji words excluded from silent-e detection
_GENUINE_2MORA_ROMAJI_E = {
    "kaze", "koe", "mune", "yume", "ame", "hane", "kage", "tane", "fune", "tame",
    "sore", "kore", "are", "dare", "naze", "toki", "sora", "kimi", "boku",
    "hito", "machi", "hoshi", "hana", "uta", "ima",
}

# Positive Japanese Romaji indicators (particles and core vocabulary)
_ROMAJI_PARTICLES_AND_FUNCTION = {
    "wa", "ga", "wo", "ni", "de", "to", "mo", "no", "ka", "ne", "yo", "na",
    "ze", "zo", "sa", "kara", "made", "dake", "demo", "nara", "kedo", "tachi",
    "koto", "mono", "toki", "sora", "kimi", "boku", "watashi", "kokoro",
    "namida", "hikari", "sekai", "mirai", "ashita", "yume", "kaze", "uta",
    "koi", "ai", "sono", "kono", "ano", "doko", "nani", "itsu", "dare", "tame",
}


# Japanese morphological substrings
_ROMAJI_MORPHOLOGICAL_SUBSTRINGS = (
    "shi", "chi", "tsu", "ky", "gy", "ny", "hy", "my", "ry", "ou", "uu", "ai", "oi", "ei",
)

# Spanish stopwords and distinct characters
_SPANISH_STOPWORDS = {
    "el", "la", "los", "las", "del", "una", "uno", "con", "por", "para", "como",
    "más", "mas", "está", "esta", "son", "ese", "esa", "todo", "toda",
    "día", "dias", "días", "siempre", "nunca", "corazón", "tiempo", "vida",
    "sueño", "mundo", "cielo", "luz", "sin", "mi", "tu", "su", "nos", "qué", "que", "y",
}
_SPANISH_CHARS_RE = re.compile(r"[ñ¿¡]", re.IGNORECASE)

# Vietnamese-specific vowels with diacritics / consonants
_VIETNAMESE_UNIQUE_RE = re.compile(
    r"[ăằắẳẵặâầấẩẫậđêềếểễệơờớởỡợưừứửữựảãạẻẽẹỉĩịỏõọổỗộủũụỷỹỵ]",
    re.IGNORECASE,
)
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
            is_english = (
                tok_lower in _ENGLISH_STOPWORDS
                or (
                    bool(_ENGLISH_SILENT_E_RE.fullmatch(tok_lower))
                    and tok_lower not in _GENUINE_2MORA_ROMAJI_E
                )
            )
            if not is_english:
                unambiguous_romaji_tokens += 1
        else:
            non_romaji_tokens += 1

    # If there are any clearly non-romaji Latin tokens, heavily penalize
    if non_romaji_tokens > 0:
        return 0.0

    # Line with <= 4 tokens that contains zero positive Japanese Romaji indicators/morphology
    # and >= 1 silent-e or English word must score < 0.50 so it never triggers ja-Latn
    has_positive = any(
        tok.lower() in _ROMAJI_PARTICLES_AND_FUNCTION
        or any(sub in tok.lower() for sub in _ROMAJI_MORPHOLOGICAL_SUBSTRINGS)
        for tok in tokens
    )
    has_english = any(
        tok.lower() in _ENGLISH_STOPWORDS
        or (
            bool(_ENGLISH_SILENT_E_RE.fullmatch(tok.lower()))
            and tok.lower() not in _GENUINE_2MORA_ROMAJI_E
        )
        for tok in tokens
    )
    if total_tokens <= 4 and not has_positive and has_english:
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

    # 2. Romance / Spanish Stopwords & Distinct Characters
    lower_norm = norm.lower()
    words = re.findall(r"[\w]+", lower_norm)
    n_es_stops = sum(1 for w in words if w in _SPANISH_STOPWORDS)
    has_es_char = bool(_SPANISH_CHARS_RE.search(norm))
    has_vi_unique = bool(_VIETNAMESE_UNIQUE_RE.search(norm))

    if has_es_char or (n_es_stops >= 2 and not has_vi_unique):
        return "es"

    # 3. Vietnamese Diacritics
    if _VIETNAMESE_RE.search(norm):
        if n_es_stops >= 1 and not has_vi_unique:
            return "es"
        return "vi"

    # 4. Latin Script (Romaji vs English/Spanish/Latin)
    if has_latin:
        if n_es_stops >= 2:
            return "es"
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
