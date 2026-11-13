"""Conservative English word-boundary and spell repair for OCR lyric text.

Runtime-only and dictionary-based: the bundled SymSpell English lexicon is the
only source of vocabulary. No network calls, no LLM, and no reference lyrics.

The repair is deliberately narrow:

* only Latin alphabetic tokens are touched;
* CJK, digits, ordinals/codes, URLs, apostrophes and hyphens are protected;
* tokens that appear more than once in the accepted-lyrics corpus are protected;
* splits require every piece to be a real lexicon word;
* whole-token spell correction uses edit distance 1 by default and only keeps a
  candidate when it clearly beats the runner-up.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping

_LATIN_RANGE = r"A-Za-z\u00c0-\u024f\u1e00-\u1eff"
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_TOKEN_RE = re.compile(
    rf"[{_LATIN_RANGE}]+"
    rf"(?:['\u2019\-][{_LATIN_RANGE}]+)*"
)
_LATIN_TOKEN_RE = re.compile(rf"^[{_LATIN_RANGE}]+$")
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
#: Accented Latin (Vietnamese, French, …) is not judged by an English lexicon.
_NON_ASCII_LATIN_RE = re.compile(r"[\u00c0-\u024f\u1e00-\u1eff]")
_DIGIT_RE = re.compile(r"\d")

#: Tokens shorter than this are never split or corrected (they are usually
#: ad-libs, codes, or single letters).
_DEFAULT_MIN_TOKEN_LEN = 4

#: Maximum number of pieces accepted from one merged token.
_MAX_PIECES = 5


@dataclass(frozen=True)
class RepairConfig:
    """Tunables for :func:`repair_text`."""

    enabled: bool = True
    min_token_len: int = _DEFAULT_MIN_TOKEN_LEN
    max_edit: int = 1
    split_penalty: float = 4.0
    min_piece_freq: int = 20
    protect_corpus_freq: int = 2
    use_bigram_context: bool = True
    #: A correction must beat the runner-up by this score margin.
    min_correction_margin: float = 0.5


@dataclass
class _Lexicon:
    words: dict[str, int]
    bigrams: dict[str, int]
    symspell: Any | None


_lexicon: _Lexicon | None = None
_lexicon_lock = threading.Lock()


def _dictionary_paths() -> tuple[Path | None, Path | None]:
    try:
        import symspellpy
    except Exception:  # pragma: no cover - dependency present in supported installs
        return None, None
    base = Path(symspellpy.__file__).resolve().parent
    return (
        base / "frequency_dictionary_en_82_765.txt",
        base / "frequency_bigramdictionary_en_243_342.txt",
    )


def _load_lexicon() -> _Lexicon:
    global _lexicon
    if _lexicon is not None:
        return _lexicon
    with _lexicon_lock:
        if _lexicon is not None:
            return _lexicon

        words: dict[str, int] = {}
        bigrams: dict[str, int] = {}
        symspell = None
        word_path, bigram_path = _dictionary_paths()

        if word_path is not None and word_path.is_file():
            for line in word_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    words[parts[0].lower()] = int(parts[1])
                except ValueError:
                    continue

            try:
                from symspellpy import SymSpell

                engine = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                engine.load_dictionary(str(word_path), term_index=0, count_index=1)
                if bigram_path is not None and bigram_path.is_file():
                    engine.load_bigram_dictionary(
                        str(bigram_path), term_index=0, count_index=2
                    )
                    for line in bigram_path.read_text(encoding="utf-8").splitlines():
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                bigrams[f"{parts[0].lower()} {parts[1].lower()}"] = int(parts[2])
                            except ValueError:
                                continue
                symspell = engine
            except Exception:  # pragma: no cover - optional acceleration
                symspell = None

        _lexicon = _Lexicon(words=words, bigrams=bigrams, symspell=symspell)
        return _lexicon


def get_lexicon() -> dict[str, int]:
    """Return the cached lowercase English frequency dictionary."""
    return _load_lexicon().words


def _match_casing(candidate: str, original: str) -> str:
    if len(original) > 1 and original.isupper():
        return candidate.upper()
    if original[:1].isupper():
        return candidate[:1].upper() + candidate[1:]
    return candidate


#: Without bigram evidence, a split is accepted only when every piece is a
#: frequent long word. This rejects accidental word soup such as
#: ``notaweekwent`` -> ``no taw eek went`` while allowing ``othercostume``.
_SPLIT_MIN_PIECE_FREQ = 1_000_000


def _is_protected(token: str, cfg: RepairConfig) -> bool:
    if not _LATIN_TOKEN_RE.match(token):
        return True
    if _NON_ASCII_LATIN_RE.search(token):
        return True
    if _DIGIT_RE.search(token) or "'" in token or "\u2019" in token or "-" in token:
        return True
    if len(token) < cfg.min_token_len:
        return True
    return False


def _piece_freq(piece: str, words: Mapping[str, int], cfg: RepairConfig) -> int:
    freq = words.get(piece, 0)
    # "I" and "a" are legitimate one-letter pieces at the start of a token.
    if len(piece) == 1:
        return max(freq, 10**7)
    if freq < cfg.min_piece_freq:
        return 0
    return freq


def _segment_token(token: str, words: Mapping[str, int], cfg: RepairConfig) -> list[str] | None:
    """Split a lowercase token into dictionary words, or return None."""
    n = len(token)
    if n < 2:
        return None

    best: list[tuple[float, tuple[str, ...]] | None] = [None] * (n + 1)
    best[0] = (0.0, ())

    def _prefer(candidate: tuple[float, tuple[str, ...]], current: tuple[float, tuple[str, ...]] | None) -> bool:
        if current is None:
            return True
        cand_score, cand_pieces = candidate
        cur_score, cur_pieces = current
        # Fewest pieces wins: the repair must not fragment a real word. Then
        # prefer longer pieces, and only use lexicon frequency as a tie-break.
        if len(cand_pieces) != len(cur_pieces):
            return len(cand_pieces) < len(cur_pieces)
        cand_min = min(len(piece) for piece in cand_pieces)
        cur_min = min(len(piece) for piece in cur_pieces)
        if cand_min != cur_min:
            return cand_min > cur_min
        return cand_score > cur_score

    for i in range(n):
        if best[i] is None:
            continue
        base_score, pieces = best[i]
        if len(pieces) >= _MAX_PIECES:
            continue
        for j in range(i + 1, n + 1):
            piece = token[i:j]
            if len(piece) == 1 and (i != 0 or piece not in ("a", "i")):
                continue
            freq = _piece_freq(piece, words, cfg)
            if not freq:
                continue
            score = base_score + math.log(freq) - cfg.split_penalty
            candidate = (score, pieces + (piece,))
            if _prefer(candidate, best[j]):
                best[j] = candidate

    if best[n] is None:
        return None
    pieces = best[n][1]
    if len(pieces) < 2 or "".join(pieces) != token:
        return None
    return list(pieces)


def _bigram_count(bigrams: Mapping[str, int], a: str, b: str) -> int:
    if not a or not b:
        return 0
    return bigrams.get(f"{a} {b}", 0)


#: Two-letter fragments that are almost always remnants of a contraction
#: (``youre`` -> ``you re``), not words a merged lyric should split into.
_CONTRACTION_FRAGMENTS = {"re", "ll", "ve"}


def _split_is_plausible(
    pieces: list[str], words: Mapping[str, int], bigrams: Mapping[str, int]
) -> bool:
    """Reject dictionary-valid but nonsensical splits."""
    if len(pieces) < 2:
        return False
    if any(piece in _CONTRACTION_FRAGMENTS for piece in pieces[1:]):
        return False
    if any(
        _bigram_count(bigrams, pieces[i], pieces[i + 1]) > 0
        for i in range(len(pieces) - 1)
    ):
        return True
    body = pieces[1:] if pieces[0] in ("a", "i") else pieces
    return all(len(piece) >= 4 and words.get(piece, 0) >= _SPLIT_MIN_PIECE_FREQ for piece in body)


def _best_spell_correction(
    token: str,
    *,
    prev: str,
    nxt: str,
    words: Mapping[str, int],
    bigrams: Mapping[str, int],
    symspell: Any | None,
    cfg: RepairConfig,
) -> str | None:
    if symspell is None:
        return None
    try:
        from symspellpy import Verbosity

        suggestions = symspell.lookup(
            token,
            Verbosity.ALL,
            max_edit_distance=cfg.max_edit,
            include_unknown=False,
        )
    except Exception:  # pragma: no cover - lookup must never break export
        return None

    candidates = []
    for sug in suggestions:
        term = str(sug.term).lower()
        if not term or term == token or not _LATIN_TOKEN_RE.match(term):
            continue
        if term not in words and len(term) < cfg.min_token_len:
            continue
        ratio = SequenceMatcher(None, token, term).ratio()
        score = (
            9.0 * ratio
            - 2.0 * abs(len(term) - len(token))
            + math.log10(int(getattr(sug, "count", words.get(term, 0))) + 1)
        )
        context = max(
            _bigram_count(bigrams, prev, term),
            _bigram_count(bigrams, term, nxt),
        )
        candidates.append((score, context, term, int(getattr(sug, "distance", 1))))

    if not candidates:
        return None

    max_context = max(c[1] for c in candidates)
    ranked: list[tuple[float, str]] = []
    for score, context, term, _distance in candidates:
        if cfg.use_bigram_context and max_context > 0 and context > 0:
            score += 2.0 * (math.log10(context + 1) / math.log10(max_context + 1))
        ranked.append((score, term))

    ranked.sort(reverse=True)
    best_score, best_term = ranked[0]
    if len(ranked) > 1 and best_score - ranked[1][0] < cfg.min_correction_margin:
        return None
    return best_term


def _repair_token(
    token: str,
    *,
    prev: str,
    nxt: str,
    words: Mapping[str, int],
    bigrams: Mapping[str, int],
    symspell: Any | None,
    vocab: Mapping[str, int] | None,
    cfg: RepairConfig,
) -> str:
    if _is_protected(token, cfg):
        return token

    lower = token.lower()
    if lower in words:
        return token

    corpus_frequent = bool(vocab and vocab.get(lower, 0) >= cfg.protect_corpus_freq)
    pieces = _segment_token(lower, words, cfg)
    if pieces and _split_is_plausible(pieces, words, bigrams):
        # "I" + "iwalk" must become "I walk", not "I i walk".
        if prev == pieces[0] and pieces[0] in ("a", "i") and len(pieces) > 1:
            joined = " ".join(pieces[1:])
            return joined.upper() if token.isupper() else joined
        return _match_casing(" ".join(pieces), token)
    if corpus_frequent and pieces:
        # A repeated corpus token whose only splits are word soup is more likely
        # a name/stylization than a merge; do not spell-correct it either.
        return token

    correction_cfg = cfg
    if corpus_frequent:
        # Already trusted corpus text: only a clearly dominant correction may
        # replace it (protects names while still fixing repeated OCR errors).
        correction_cfg = RepairConfig(
            enabled=cfg.enabled,
            min_token_len=cfg.min_token_len,
            max_edit=cfg.max_edit,
            split_penalty=cfg.split_penalty,
            min_piece_freq=cfg.min_piece_freq,
            protect_corpus_freq=cfg.protect_corpus_freq,
            use_bigram_context=cfg.use_bigram_context,
            min_correction_margin=max(cfg.min_correction_margin, 1.0),
        )

    corrected = _best_spell_correction(
        lower,
        prev=prev,
        nxt=nxt,
        words=words,
        bigrams=bigrams,
        symspell=symspell,
        cfg=correction_cfg,
    )
    if not corrected:
        return token
    return _match_casing(corrected, token)


_ROMAN_CONJ_RN_RE = re.compile(r"(?<![A-Za-z])([Rr])([Nn])(?=[aeiouAEIOU])|([Rr])([Nn])(?=[aeiouAEIOU])")


_ROMAJI_PROTECTED_ENGLISH = frozenset({
    "clear",
    "clean",
    "close",
    "closed",
    "cloud",
    "clouds",
    "morning",
    "learning",
    "journey",
    "burn",
    "burning",
    "born",
    "return",
    "corner",
    "eternal",
    "eternity",
})


def repair_romaji_confusables(text: str) -> str:
    """Repair reference-free Hepburn Romaji OCR character confusables inside words."""
    if not text or not text.strip():
        return text

    lex = _load_lexicon()
    eng_words = lex.words if lex and lex.words else {}

    def _fix_word(match: re.Match[str]) -> str:
        tok = match.group(0)
        if not any(ch.isalpha() for ch in tok):
            return tok

        lower_tok = tok.lower()
        if lower_tok in _ROMAJI_PROTECTED_ENGLISH or (
            lower_tok.isalpha() and len(lower_tok) >= 3 and eng_words.get(lower_tok, 0) >= 50000
        ):
            return tok

        # Protect all-caps alphanumeric codes (e.g. '00A01', 'OP1', 'ICBM')
        has_lower = any("a" <= ch <= "z" for ch in tok)

        fixed = re.sub(
            r"([Rr])([Nn])(?=[aeiouAEIOU])",
            lambda m: "M" if (m.group(1).isupper() and m.group(2).isupper()) else "m",
            tok,
        )
        fixed = re.sub(
            r"([Vv])([Vv])",
            lambda m: "W" if (m.group(1).isupper() and m.group(2).isupper()) else "w",
            fixed,
        )
        fixed = re.sub(
            r"([Cc])([Ll])(?=[aeiouAEIOU])",
            lambda m: "D" if (m.group(1).isupper() and m.group(2).isupper()) else "d",
            fixed,
        )
        if has_lower:
            fixed = re.sub(r"(?<=[A-Za-z])0|0(?=[A-Za-z])", "o", fixed)
            fixed = re.sub(r"(?<=[A-Za-z])1|1(?=[A-Za-z])", "i", fixed)
        return fixed

    return re.sub(r"[A-Za-z0-9]+", _fix_word, text)


def repair_text(
    text: str,
    *,
    config: RepairConfig | None = None,
    vocab: Mapping[str, int] | None = None,
) -> str:
    """Repair merged words and obvious OCR misspellings in lyric text.

    ``vocab`` is the accepted-lyrics corpus vocabulary; tokens that occur at
    least ``config.protect_corpus_freq`` times in it are left untouched.
    """
    cfg = config or RepairConfig()
    if not cfg.enabled or not text or not text.strip():
        return text
    if _CJK_RE.search(text) or _NON_ASCII_LATIN_RE.search(text):
        # Non-English scripts are preserved as-is; the lexicon is English-only.
        return text

    lexicon = _load_lexicon()
    if not lexicon.words:
        return text

    url_spans = [(m.start(), m.end()) for m in _URL_RE.finditer(text)]

    tokens = list(_TOKEN_RE.finditer(text))
    if not tokens:
        return text

    def _inside_url(start: int) -> bool:
        return any(span_start <= start < span_end for span_start, span_end in url_spans)

    out: list[str] = []
    last = 0
    prev_word = ""
    for idx, match in enumerate(tokens):
        out.append(text[last : match.start()])
        raw_token = match.group(0)
        next_word = tokens[idx + 1].group(0).lower() if idx + 1 < len(tokens) else ""
        if _inside_url(match.start()):
            repaired = raw_token
        else:
            repaired = _repair_token(
                raw_token,
                prev=prev_word,
                nxt=next_word,
                words=lexicon.words,
                bigrams=lexicon.bigrams,
                symspell=lexicon.symspell,
                vocab=vocab,
                cfg=cfg,
            )
        out.append(repaired)
        prev_word = repaired.lower()
        last = match.end()
    out.append(text[last:])

    return "".join(out)


def text_quality(
    text: str,
    vocab: Mapping[str, int] | None = None,
    *,
    config: RepairConfig | None = None,
) -> float:
    """Return a 0..1 readability score for OCR lyric text.

    The score combines lexical coverage (tokens known to the English lexicon or
    the accepted-lyrics corpus) with the alphabetic ratio of the line. A long
    merged token (>= 14 characters) is treated as strong evidence of a
    concatenated/garbled line.
    """
    cfg = config or RepairConfig()
    raw = (text or "").strip()
    if not raw:
        return 0.0
    if _CJK_RE.search(raw) or _NON_ASCII_LATIN_RE.search(raw):
        # CJK/accented-Latin lines are not judged by an English lexicon.
        return 1.0

    clean = _URL_RE.sub(" ", raw)
    tokens = _TOKEN_RE.findall(clean)
    if not tokens:
        return 0.0

    words = get_lexicon()
    known_chars = 0
    total_chars = 0
    for token in tokens:
        total_chars += len(token)
        # Judge readability against the English lexicon only. The accepted-lyrics
        # corpus is for repair protection; letting it validate text would let a
        # repeated garbled variant vouch for itself.
        if token.lower() in words:
            known_chars += len(token)
    # Character-weighted coverage: long garbled tokens cannot be hidden by a few
    # short common words such as "the", "me", "way".
    coverage = min(1.0, known_chars / max(1, total_chars))

    non_space = sum(1 for ch in raw if not ch.isspace())
    alpha = sum(1 for ch in raw if ch.isalpha())
    alpha_ratio = alpha / max(1, non_space)

    long_merged = sum(1 for token in tokens if len(token) >= 14) / len(tokens)
    quality = 0.75 * coverage + 0.25 * alpha_ratio - 0.45 * long_merged
    return max(0.0, min(1.0, quality))


def oov_ratio(
    text: str,
    vocab: Mapping[str, int] | None = None,
    *,
    config: RepairConfig | None = None,
) -> float:
    """Fraction of tokens the English lexicon and corpus do not recognise."""
    cfg = config or RepairConfig()
    raw = (text or "").strip()
    if not raw or _CJK_RE.search(raw) or _NON_ASCII_LATIN_RE.search(raw):
        return 0.0
    tokens = _TOKEN_RE.findall(_URL_RE.sub(" ", raw))
    if not tokens:
        return 0.0
    words = get_lexicon()
    unknown = sum(1 for token in tokens if token.lower() not in words)
    return unknown / len(tokens)
