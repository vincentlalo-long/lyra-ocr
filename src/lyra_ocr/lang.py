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
    "been", "seen", "she", "too",
    "feel", "real", "soul", "heal", "deal", "hello", "hero", "zero",
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

# Spanish unambiguous words (not Japanese Romaji particles) and ambiguous short tokens
_SPANISH_UNAMBIGUOUS_WORDS = {
    "el", "los", "las", "del", "una", "uno", "por", "para", "como",
    "más", "mas", "está", "esta", "están", "estan", "estoy", "estuve",
    "ser", "fue", "sido", "han",
    "ese", "esa", "esos", "esas", "este", "estos", "estas",
    "todo", "toda", "todos", "todas", "día", "dias", "días",
    "siempre", "nunca", "corazón", "corazon", "tiempo", "vida",
    "sueño", "sueno", "mundo", "cielo", "luz", "nos", "les",
    "qué", "que", "quien", "cual", "cuando", "donde", "porque",
    "aunque", "aumque", "pero", "sino", "muy", "tanto", "tanta",
    "poco", "poca", "mucho", "mucha", "muchos", "muchas", "vez", "veces",
    "nada", "algo", "alguien", "nadie", "sola", "mismo", "misma",
    "otro", "otra", "hacia", "hasta", "desde", "entre", "sobre", "contra",
    "contigo", "conmigo", "futuro", "destino", "dolor", "doloroso",
    "felicidad", "adios", "adiós", "volverá", "volvera",
    "entenderá", "entendera", "quería", "queria", "olvidar", "respuesta",
}
_SPANISH_AMBIGUOUS_SHORT = {
    "la", "mi", "tu", "su", "me", "te", "se", "lo", "le",
    "ya", "si", "no", "ni", "de", "y", "o", "a", "en",
    "son", "era", "sin", "tan", "ha", "solo", "con",
}
_SPANISH_STOPWORDS = _SPANISH_UNAMBIGUOUS_WORDS
_SPANISH_CHARS_RE = re.compile(r"[ñ¿¡]", re.IGNORECASE)
_SPANISH_ACCENTED_VOWEL_RE = re.compile(r"[áéíóúü]", re.IGNORECASE)

# Vietnamese-specific vowels with diacritics / consonants
_VIETNAMESE_UNIQUE_RE = re.compile(
    r"[ăằắẳẵặâầấẩẫậđêềếểễệơờớởỡợưừứửữựảãạẻẽẹỉĩịỏõọổỗộủũụỷỹỵ]",
    re.IGNORECASE,
)
_VIETNAMESE_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)


def _normalize_romaji_ocr_token(tok: str) -> str:
    """Normalize common Latin OCR confusables in lowercase Romaji tokens for scoring."""
    t = tok.lower()
    t = re.sub(r"(?<=[aeiouāīūēōâîûêô])ll(?=[aeiouāīūēōâîûêô])", "tt", t)
    t = re.sub(r"(?<=[aeiouāīūēōâîûêô])l(?=[aeiouāīūēōâîûêô])", "t", t)
    t = re.sub(r"(^|[^aeiouāīūēōâîûêô])([auoeāūōēâûôê])l$", r"\1\2i", t)
    t = re.sub(r"c(?=k)", "o", t)
    t = re.sub(r"(?<=[bcdfghjklmnpqrstvwxyz])c(?=[aou])", "o", t)
    return t


def romaji_score(text: str) -> float:
    """Calculate the romaji confidence score of a Latin text string in [0.0, 1.0]."""
    if not text or not text.strip():
        return 0.0

    # Extract alphabetic words (keeping apostrophe contractions intact so 've/'re don't score as morae)
    tokens = re.findall(r"[A-Za-zāīūēōâîûêô]+(?:'[A-Za-z]+)*", text)
    if not tokens:
        return 0.0

    total_tokens = len(tokens)
    romaji_tokens = 0
    unambiguous_romaji_tokens = 0
    positive_ja_hits = 0
    has_english = False

    for tok in tokens:
        tok_lower = tok.lower()
        tok_is_english = (
            tok_lower in _ENGLISH_STOPWORDS
            or (
                bool(_ENGLISH_SILENT_E_RE.fullmatch(tok_lower))
                and tok_lower not in _GENUINE_2MORA_ROMAJI_E
            )
        )
        if tok_is_english:
            has_english = True

        norm_tok = (
            tok_lower
            if (tok_is_english or _ROMAJI_TOKEN_RE.fullmatch(tok_lower))
            else _normalize_romaji_ocr_token(tok_lower)
        )
        if _ROMAJI_TOKEN_RE.fullmatch(norm_tok):
            romaji_tokens += 1
            norm_is_english = tok_is_english or (
                norm_tok in _ENGLISH_STOPWORDS
                or (
                    bool(_ENGLISH_SILENT_E_RE.fullmatch(norm_tok))
                    and norm_tok not in _GENUINE_2MORA_ROMAJI_E
                )
            )
            if norm_is_english:
                has_english = True
            else:
                unambiguous_romaji_tokens += 1

            if (
                norm_tok == tok_lower
                and (
                    norm_tok in _ROMAJI_PARTICLES_AND_FUNCTION
                    or any(sub in norm_tok for sub in _ROMAJI_MORPHOLOGICAL_SUBSTRINGS)
                )
            ):
                positive_ja_hits += 1

    valid_ratio = romaji_tokens / total_tokens
    if valid_ratio < 0.50:
        return 0.0

    # Line with <= 4 tokens that contains zero positive Japanese Romaji indicators/morphology
    # and >= 1 silent-e or English word must not trigger ja-Latn
    if total_tokens <= 4 and positive_ja_hits == 0 and has_english:
        return 0.0

    unambiguous_ratio = unambiguous_romaji_tokens / total_tokens
    return min(1.0, unambiguous_ratio + (0.15 if positive_ja_hits >= 2 else 0.0))


def detect_line_lang(text: str) -> str:
    """Classify the language/script of a single line of text.

    Returns:
        'ja'      : Japanese containing Hiragana/Katakana
        'ko'      : Korean containing Hangul
        'vi'      : Vietnamese containing Vietnamese diacritics/characters
        'zh'      : Chinese / CJK Unified Ideographs without Kana/Hangul
        'es'      : Spanish
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

    # 2. Romance / Spanish Disambiguation & Distinct Characters
    lower_norm = norm.lower()
    words = re.findall(r"[\w]+", lower_norm)
    n_es_unambiguous = sum(1 for w in words if w in _SPANISH_UNAMBIGUOUS_WORDS)
    n_es_ambiguous = sum(1 for w in words if w in _SPANISH_AMBIGUOUS_SHORT)
    has_es_char = bool(_SPANISH_CHARS_RE.search(norm))
    has_accented_vowel = bool(_SPANISH_ACCENTED_VOWEL_RE.search(norm))
    has_vi_unique = bool(_VIETNAMESE_UNIQUE_RE.search(norm))

    is_spanish = has_es_char or (
        not has_vi_unique
        and (
            n_es_unambiguous >= 2
            or (n_es_unambiguous >= 1 and (n_es_ambiguous >= 1 or has_accented_vowel))
        )
    )
    if is_spanish:
        return "es"

    # 3. Vietnamese Diacritics
    if _VIETNAMESE_RE.search(norm):
        return "vi"

    # 4. Latin Script (Romaji vs English/Spanish/Latin)
    if has_latin:
        score = romaji_score(norm)
        # Require score >= 0.6 and at least 2 tokens, or single high-confidence token
        tokens = re.findall(r"[A-Za-zāīūēōâîûêô]+(?:'[A-Za-z]+)*", norm)
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
