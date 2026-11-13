"""Reference-lyrics lookup and monotonic alignment for OCR cue text.

Two sources are supported:

* an explicit reference text (``.lrc`` or plain ``.txt``) supplied by the user;
* an online lookup against LRCLIB (https://lrclib.net), a free lyrics database
  with no API key. The lookup is best-effort and never blocks LRC generation.

The reference is used **only to repair the text** of already-timed OCR cues. The
timing stays entirely OCR/visual: no reference timestamp is copied, so the LRC
still describes what is on screen. Alignment is monotonic, so chorus repeats and
skipped lines are handled without scrambling the order.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from lyra_ocr.lyric.models import CorrectionInfo, LyricCue

logger = logging.getLogger(__name__)

LRCLIB_API = "https://lrclib.net/api"
DEFAULT_TIMEOUT = 6.0

_LRC_TIMESTAMP_RE = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")
_META_TAG_RE = re.compile(r"^\[[a-zA-Z#]+:.*\]$")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_ALIGN_TOKEN_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]|[^\W_]+",
    re.UNICODE,
)

_BRACKET_CONTENT_RE = re.compile(r"[『「【\[\(](.*?)[』」】\]\)]")
_NOISE_PATTERNS = [
    re.compile(r"\b(?:official\s+)?lyric(?:s)?(?:\s+video)?\b", re.IGNORECASE),
    re.compile(r"\bofficial\s+(?:music\s+)?video\b", re.IGNORECASE),
    re.compile(r"\blyrics\s+amv\b", re.IGNORECASE),
    re.compile(r"\bamv\b", re.IGNORECASE),
    re.compile(r"\bmv\b", re.IGNORECASE),
    re.compile(r"\b(?:full\s+)?audio\b", re.IGNORECASE),
    re.compile(r"\b(?:op|ed)\s*\d*\s*full\b", re.IGNORECASE),
    re.compile(r"\b(?:nightcore|daycore|sped\s+up|slowed(?:\s*\+\s*reverb)?)\b", re.IGNORECASE),
    re.compile(r"\b(?:english\s+romaji|romaji|english\s+sub(?:title)?s?)\b", re.IGNORECASE),
    re.compile(r"\b(?:1080p|720p|4k|hd|diva|full)\b", re.IGNORECASE),
]
_FEAT_RE = re.compile(r"\s*[\(\[]?\b(?:feat|ft|prod)\.?\s+[^\)\]]+[\)\]]?", re.IGNORECASE)
_HYPHEN_SPLIT_RE = re.compile(r"\s+[-–—]\s+")


@dataclass
class ReferenceLine:
    """A single reference lyric line, optionally bound to an LRC timestamp."""

    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    source: str = ""
    confidence: float = 0.0
    language: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReferenceLine:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ReferenceBundle:
    """A collection of reference lines with provenance metadata."""

    lines: list[ReferenceLine] = field(default_factory=list)
    source: str = ""  # "manual" | "lrclib_synced" | "lrclib_plain" | "youtube_captions" | ""
    provider: str = ""  # "manual" | "lrclib" | "youtube" | ""
    language: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lines": [line.to_dict() for line in self.lines],
            "source": self.source,
            "provider": self.provider,
            "language": self.language,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReferenceBundle:
        lines = [ReferenceLine.from_dict(row) for row in d.get("lines", [])]
        return cls(
            lines=lines,
            source=d.get("source", ""),
            provider=d.get("provider", ""),
            language=d.get("language", ""),
            warnings=list(d.get("warnings", [])),
        )


@dataclass
class ReferenceQuery:
    """Normalized query for looking up reference lyrics."""

    track: str
    artist: str = ""
    duration_ms: int = 0
    raw_title: str = ""


def reference_cache_dir(artifacts_dir: str | Path) -> Path:
    """Return the reference cache directory inside an artifacts folder."""
    p = Path(artifacts_dir) / "reference_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def reference_cache_key(
    query: ReferenceQuery,
    provider: str,
    parser_version: str = "v1",
) -> str:
    """Deterministic hash key for reference lookup caching."""
    duration_bucket = round(query.duration_ms / 5000) if query.duration_ms else 0
    norm_dict = {
        "track": (query.track or "").strip().lower(),
        "artist": (query.artist or "").strip().lower(),
        "raw_title": (query.raw_title or "").strip().lower(),
        "duration_bucket": duration_bucket,
        "provider": provider.strip().lower(),
        "version": parser_version,
    }
    dumped = json.dumps(norm_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(dumped.encode("utf-8")).hexdigest()


def save_cached_reference_bundle(
    cache_dir: Path | str,
    key: str,
    bundle: ReferenceBundle,
    query: ReferenceQuery,
) -> Path:
    """Save ReferenceBundle to disk cache atomically."""
    cd = Path(cache_dir)
    cd.mkdir(parents=True, exist_ok=True)
    target = cd / f"{key}.json"
    tmp = target.with_suffix(".tmp")
    data = {
        "bundle_version": 1,
        "created_at": time.time(),
        "query": asdict(query),
        "bundle": bundle.to_dict(),
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(target)
    return target


def load_cached_reference_bundle(
    cache_dir: Path | str,
    key: str,
    *,
    max_age_seconds: float | None = None,
) -> ReferenceBundle | None:
    """Load ReferenceBundle from disk cache if present and valid."""
    target = Path(cache_dir) / f"{key}.json"
    if not target.is_file():
        return None
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        if max_age_seconds is not None:
            created_at = float(data.get("created_at") or 0.0)
            if created_at and time.time() - created_at > max_age_seconds:
                return None
        b_data = data.get("bundle")
        if isinstance(b_data, dict):
            return ReferenceBundle.from_dict(b_data)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass
    return None




def parse_reference_query(
    title: str,
    channel: str = "",
    duration_ms: int = 0,
) -> ReferenceQuery:
    """Derive clean track name and artist from video title and channel."""
    raw_title = (title or "").strip()
    clean_chan = re.sub(r"\s*-\s*Topic\b", "", channel or "", flags=re.IGNORECASE)
    clean_chan = re.sub(r"\bVEVO\b", "", clean_chan, flags=re.IGNORECASE).strip()

    # 1. Check if there is an explicit bracketed pair like 「sister's noise - fripSide」
    track = ""
    artist = ""
    for match in _BRACKET_CONTENT_RE.findall(raw_title):
        inner = match.strip()
        parts = _HYPHEN_SPLIT_RE.split(inner)
        if len(parts) == 2:
            p0, p1 = parts[0].strip(), parts[1].strip()
            if len(p0) > 1 and len(p1) > 1 and not any(p.search(p0) or p.search(p1) for p in _NOISE_PATTERNS):
                track = p0
                artist = p1
                break

    if not track:
        cleaned_title = raw_title
        for pattern in _NOISE_PATTERNS:
            cleaned_title = pattern.sub("", cleaned_title)

        def _clean_bracket(m):
            content = m.group(1).strip()
            if any(p.search(content) for p in _NOISE_PATTERNS) or not content:
                return ""
            return content

        cleaned_title = _BRACKET_CONTENT_RE.sub(_clean_bracket, cleaned_title).strip()
        cleaned_title = re.sub(r"^\s*[-–—/|]+\s*|\s*[-–—/|]+\s*$", "", cleaned_title).strip()
        cleaned_title = re.sub(r"\s+", " ", cleaned_title).strip()

        parts = [p.strip() for p in _HYPHEN_SPLIT_RE.split(cleaned_title) if p.strip()]
        if len(parts) == 2:
            p0, p1 = parts[0].strip(), parts[1].strip()
            c_lower = clean_chan.lower()
            if c_lower and c_lower in p0.lower():
                artist, track = p0, p1
            elif c_lower and c_lower in p1.lower():
                track, artist = p0, p1
            else:
                artist, track = p0, p1
        elif len(parts) == 1:
            track = parts[0].strip()
            artist = clean_chan
        elif len(parts) > 2:
            track = parts[-1].strip()
            artist = " - ".join(parts[:-1]).strip()

    if track:
        track = _FEAT_RE.sub("", track).strip()
        for pattern in _NOISE_PATTERNS:
            track = pattern.sub("", track).strip()
        track = re.sub(r"[『「【\[\(].*?[』」】\]\)]", "", track).strip()
        track = re.sub(r"^\s*[-–—/|]+\s*|\s*[-–—/|]+\s*$", "", track).strip()
        track = re.sub(r"\s+", " ", track).strip()

    if artist:
        artist = re.sub(r"^\s*[-–—/|]+\s*|\s*[-–—/|]+\s*$", "", artist).strip()
        artist = re.sub(r"\s+", " ", artist).strip()

    if not artist and clean_chan:
        artist = clean_chan

    return ReferenceQuery(
        track=track or raw_title,
        artist=artist,
        duration_ms=duration_ms,
        raw_title=raw_title,
    )



def _parse_ts_ms(m: re.Match) -> int:
    mins = int(m.group(1))
    secs = int(m.group(2))
    sub = m.group(3) or "0"
    if len(sub) == 2:
        ms = int(sub) * 10
    elif len(sub) == 1:
        ms = int(sub) * 100
    else:
        ms = int(sub[:3])
    return mins * 60000 + secs * 1000 + ms


def parse_reference_text(
    text: str,
    *,
    source: str = "",
    language: str = "",
) -> list[ReferenceLine]:
    """Parse reference text (.lrc or plain text) into a list of ReferenceLine objects."""
    result: list[ReferenceLine] = []
    section_re = re.compile(
        r"^\[(?:chorus|verse|intro|outro|bridge|hook|pre-chorus|break)\b[^\]]*\]$",
        re.IGNORECASE,
    )

    for raw in (text or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        # Drop metadata tags like [ti:Song], [ar:Artist], [offset:0]
        if _META_TAG_RE.match(stripped) and not _LRC_TIMESTAMP_RE.search(stripped):
            continue
        # Drop section tags like [Chorus], [Verse 1]
        if section_re.match(stripped):
            continue

        timestamps = list(_LRC_TIMESTAMP_RE.finditer(stripped))
        clean_line = _LRC_TIMESTAMP_RE.sub("", stripped).strip()

        if not clean_line or not _WORD_RE.search(clean_line):
            continue

        confidence = 1.0 if "lrclib" in source or source == "manual" else 0.8

        if timestamps:
            for ts_match in timestamps:
                start_ms = _parse_ts_ms(ts_match)
                result.append(
                    ReferenceLine(
                        text=clean_line,
                        start_ms=start_ms,
                        source=source,
                        confidence=confidence,
                    )
                )
        else:
            result.append(
                ReferenceLine(
                    text=clean_line,
                    start_ms=None,
                    source=source,
                    confidence=confidence,
                )
            )

    return result


def parse_lyrics_text(text: str) -> list[str]:
    """Return lyric lines in order, stripping LRC timestamps and metadata tags."""
    return [line.text for line in parse_reference_text(text)]



def _squash(text: str) -> str:
    """Lowercase, accent-folded, punctuation-free form used for matching."""
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    # Strip apostrophes inside/around words so contractions match OCR text
    folded = re.sub(r"['’‘`]", "", folded)
    return " ".join(_WORD_RE.findall(folded.casefold()))



def _phonetic_skeleton(text: str) -> str:
    """Fast deterministic phonetic skeleton for Latin strings."""
    s = _squash(text).lower()
    if not s:
        return ""
    s = re.sub(r"ph", "f", s)
    s = re.sub(r"ck", "k", s)
    s = re.sub(r"gh", "g", s)
    s = re.sub(r"kn", "n", s)
    s = re.sub(r"wr", "r", s)
    s = re.sub(r"[aeiouy]+", "a", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def line_similarity(a: str, b: str) -> float:
    sa, sb = _squash(a), _squash(b)
    if not sa or not sb:
        return 0.0
    la, lb = len(sa), len(sb)
    if (2.0 * min(la, lb) / (la + lb)) < 0.60:
        return 0.0
    base_ratio = SequenceMatcher(None, sa, sb).ratio()
    # If Latin characters present, blend phonetic similarity
    if re.search(r"[a-z]", sa) and re.search(r"[a-z]", sb):
        psa, psb = _phonetic_skeleton(sa), _phonetic_skeleton(sb)
        if psa and psb:
            p_ratio = SequenceMatcher(None, psa, psb).ratio()
            return 0.85 * base_ratio + 0.15 * p_ratio
    return base_ratio


@dataclass
class MatchStats:
    """Detailed lexical and length alignment statistics between OCR and reference."""

    similarity: float
    token_coverage: float
    length_ratio: float


@dataclass
class GapFillResult:
    """Statistics for recovered reference lines inserted into timeline gaps."""

    filled_cues: list[LyricCue] = field(default_factory=list)
    n_filled: int = 0
    warnings: list[str] = field(default_factory=list)



@dataclass
class AlignmentResult:
    """Result of monotonic reference alignment."""

    matched_count: int = 0
    replaced_count: int = 0
    review_count: int = 0
    matched_pairs: list[tuple[int, int]] = field(default_factory=list)

    def __int__(self) -> int:
        return self.replaced_count

    def __index__(self) -> int:
        return self.replaced_count

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, int):
            return self.replaced_count == other
        return super().__eq__(other)


def _tokenize(text: str) -> list[str]:
    """Tokenize into Latin words and individual CJK characters."""
    cleaned = re.sub(r"['’‘`]", "", text or "")
    return [token.lower() for token in _ALIGN_TOKEN_RE.findall(cleaned)]


def line_match_stats(ocr_text: str, reference_text: str) -> MatchStats:
    """Calculate character similarity, token coverage, and length ratio."""
    sim = line_similarity(ocr_text, reference_text)
    ocr_toks = _tokenize(ocr_text)
    ref_toks = _tokenize(reference_text)

    if not ref_toks or not ocr_toks:
        return MatchStats(
            similarity=sim,
            token_coverage=1.0 if not ref_toks and not ocr_toks else 0.0,
            length_ratio=len(ref_toks) / max(1, len(ocr_toks)),
        )

    matched_ref = 0
    for rt in ref_toks:
        if any(line_similarity(rt, ot) >= 0.80 for ot in ocr_toks):
            matched_ref += 1

    cov = matched_ref / len(ref_toks)
    ratio = len(ref_toks) / max(1, len(ocr_toks))

    return MatchStats(
        similarity=sim,
        token_coverage=cov,
        length_ratio=ratio,
    )


def best_reference_similarity(text: str, reference_lines: list[str] | None) -> float:
    """Highest similarity between ``text`` and any reference line."""
    if not text or not reference_lines:
        return 0.0
    return max((line_similarity(text, ref) for ref in reference_lines), default=0.0)


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
        curr_len += 1  # for the space in join

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

    # Snap split points to closest whitespace in ref_text
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
        part = ref_text[prev_idx:sp].strip()
        parts.append(part)
        prev_idx = sp
    parts.append(ref_text[prev_idx:].strip())

    return parts


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
                    out_tokens.append(_match_word_case(ref_joined, wo))
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


def align_reference(
    cues: list[LyricCue],
    reference: list[str] | list[ReferenceLine] | ReferenceBundle,
    *,
    min_similarity: float | None = None,
    config: Any = None,
    source: str = "",
) -> AlignmentResult:
    """Replace cue text with the best monotonic reference match.

    Supports N:M alignment transitions ({1:1, 1:2, 1:3, 2:1, 3:1}) to handle
    multi-line visual lyric cards and split subtitle phrases.
    """
    from lyra_ocr.lyric.models import ReferenceConfig

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
    if is_caption:
        replace_thresh = cfg.caption_replace_similarity
        review_thresh = 0.75
    else:
        replace_thresh = cfg.replace_similarity
        review_thresh = cfg.review_similarity

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

    # DP alignment with N:M transitions
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    choice: list[list[tuple[int, int]]] = [[(1, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Skip cue (i-1)
            best_score = dp[i - 1][j]
            best_trans = (1, 0)

            # Skip ref (j-1)
            if dp[i][j - 1] > best_score:
                best_score = dp[i][j - 1]
                best_trans = (0, 1)

            # N:M Transitions
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

    # Traceback
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
            cue_indices = list(range(i - kc, i))
            ref_indices = list(range(j - kr, j))
            actions.append((trans, cue_indices, ref_indices))
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

        if kc == 1 and kr == 1:
            cue_idx, ref_idx = cue_idxs[0], ref_idxs[0]
            cue = cues[cue_idx]
            if cue.correction.method == "human":
                continue
            ref = ref_lines[ref_idx]
            st = stats_matrix[cue_idx][ref_idx]
            ref_clean = ref.text.strip()
            if not ref_clean:
                continue

            cue_is_romaji = cue.lang == "ja-Latn" or is_romaji_ref
            eff_replace = min(replace_thresh, 0.78) if cue_is_romaji else replace_thresh
            eff_cov = min(cfg.min_token_coverage, 0.40) if cue_is_romaji else cfg.min_token_coverage
            guards_passed = (
                st.similarity >= eff_replace
                and st.token_coverage >= eff_cov
                and cfg.min_length_ratio <= st.length_ratio <= cfg.max_length_ratio
                and bool(_WORD_RE.search(ref_clean))
            )
            if guards_passed:
                target_text = (
                    repair_romaji_cue_text(
                        cue.text,
                        ref_clean,
                        ocr_confidence=cue.ocr_confidence,
                        is_asr=is_asr_rom,
                    )
                    if cue_is_romaji
                    else ref_clean
                )
                if cue_is_romaji:
                    target_text = _CJK_CHAR_RE.sub("", target_text).strip()
                if target_text and target_text != cue.text:
                    corr_method = "reference_caption" if is_caption else "reference"
                    cue.correction = CorrectionInfo(
                        method=corr_method,
                        changed=True,
                        confidence=cue.confidence,
                        diff=f"'{cue.text}' -> '{target_text}'",
                    )
                    cue.text = target_text
                    replaced_count += 1
            else:
                cue.review_required = True
                if "reference_uncertain" not in cue.qa_flags:
                    cue.qa_flags.append("reference_uncertain")
                review_count += 1

        elif kc == 1 and kr > 1:
            cue = cues[cue_idxs[0]]
            if cue.correction.method == "human":
                continue
            combined_ref = " ".join(ref_lines[ri].text.strip() for ri in ref_idxs)
            st = line_match_stats(cue_cmp_texts[cue_idxs[0]], combined_ref)
            cue_is_romaji = cue.lang == "ja-Latn" or is_romaji_ref
            eff_replace = min(replace_thresh, 0.78) if cue_is_romaji else replace_thresh
            eff_cov = min(cfg.min_token_coverage, 0.40) if cue_is_romaji else cfg.min_token_coverage
            guards_passed = (
                st.similarity >= eff_replace
                and st.token_coverage >= eff_cov
                and bool(_WORD_RE.search(combined_ref))
            )
            if guards_passed:
                target_text = (
                    repair_romaji_cue_text(
                        cue.text,
                        combined_ref,
                        ocr_confidence=cue.ocr_confidence,
                        is_asr=is_asr_rom,
                    )
                    if cue_is_romaji
                    else combined_ref
                )
                if cue_is_romaji:
                    target_text = _CJK_CHAR_RE.sub("", target_text).strip()
                if target_text and target_text != cue.text:
                    corr_method = "reference_caption" if is_caption else "reference"
                    cue.correction = CorrectionInfo(
                        method=corr_method,
                        changed=True,
                        confidence=cue.confidence,
                        diff=f"'{cue.text}' -> '{target_text}'",
                    )
                    cue.text = target_text
                    replaced_count += 1
            else:
                cue.review_required = True
                if "reference_uncertain" not in cue.qa_flags:
                    cue.qa_flags.append("reference_uncertain")
                review_count += 1

        elif kc > 1 and kr == 1:
            ref_clean = ref_lines[ref_idxs[0]].text.strip()
            ocr_cues = [cues[ci] for ci in cue_idxs]
            if any(c.correction.method == "human" for c in ocr_cues):
                continue
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
                projected = _project_ref_onto_cues([c.text for c in ocr_cues], ref_clean)
                for ci, part in zip(cue_idxs, projected):
                    c = cues[ci]
                    target_part = (
                        repair_romaji_cue_text(
                            c.text,
                            part,
                            ocr_confidence=c.ocr_confidence,
                            is_asr=is_asr_rom,
                        )
                        if (c.lang == "ja-Latn" or is_romaji_ref)
                        else part
                    )
                    if c.lang == "ja-Latn" or is_romaji_ref:
                        target_part = _CJK_CHAR_RE.sub("", target_part).strip()
                    if target_part and target_part != c.text:
                        corr_method = "reference_caption" if is_caption else "reference"
                        c.correction = CorrectionInfo(
                            method=corr_method,
                            changed=True,
                            confidence=c.confidence,
                            diff=f"'{c.text}' -> '{target_part}'",
                        )
                        c.text = target_part
                        replaced_count += 1
            else:
                for c in ocr_cues:
                    c.review_required = True
                    if "reference_uncertain" not in c.qa_flags:
                        c.qa_flags.append("reference_uncertain")
                review_count += 1

    matched_count = len({pair[0] for pair in pairs})

    return AlignmentResult(
        matched_count=matched_count,
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
    from lyra_ocr.lyric.models import ReferenceConfig

    cfg = config or ReferenceConfig()
    ref_lines: list[ReferenceLine] = []
    if isinstance(reference, ReferenceBundle):
        ref_lines = reference.lines
    elif isinstance(reference, list):
        ref_lines = reference
    if not cues or not ref_lines:
        return cues, []

    pairs = matched_pairs
    if pairs is None:
        pairs = _infer_anchor_pairs(cues, ref_lines, cfg.replace_similarity)
    ref_by_cue_id = {
        id(cues[cue_idx]): ref_idx
        for cue_idx, ref_idx in pairs
        if 0 <= cue_idx < len(cues) and 0 <= ref_idx < len(ref_lines)
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
            if ref_idx is None:
                continue
            ref = ref_lines[ref_idx]
            if ref.start_ms is None or cue.correction.method == "human":
                continue
            entries.append((cue, ref))
        if len(entries) < 2:
            if all(cue.correction.method.startswith("reference") for cue in group):
                warnings.append(
                    f"reference_overlap_unresolved at {group[0].start_ms}ms"
                )
            continue

        if not cfg.split_overlapping_reference_cues:
            warnings.append(
                f"reference_overlap_unresolved at {group[0].start_ms}ms"
            )
            continue

        entries.sort(key=lambda item: (item[1].start_ms, item[1].text))
        starts = [int(ref.start_ms) for _, ref in entries]
        if any(b <= a for a, b in zip(starts, starts[1:])):
            warnings.append(
                f"reference_overlap_unresolved at {group[0].start_ms}ms"
            )
            continue

        group_start = min(cue.start_ms for cue in group)
        group_end = max(cue.end_ms for cue in group)
        boundaries = [
            max(group_start + 1, min(group_end - 1, (a + b) // 2))
            for a, b in zip(starts, starts[1:])
        ]
        if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
            warnings.append(
                f"reference_overlap_unresolved at {group[0].start_ms}ms"
            )
            continue
        if any(not (group_start < b < group_end) for b in boundaries):
            warnings.append(
                f"reference_overlap_unresolved at {group[0].start_ms}ms"
            )
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
    """Infer monotonic cue/reference anchors when the caller did not align first."""
    pairs: list[tuple[int, int]] = []
    next_ref = 0
    for cue_idx, cue in enumerate(cues):
        best_idx: int | None = None
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
    from lyra_ocr.lyric.models import ReferenceConfig

    cfg = config or ReferenceConfig()
    if not cfg.enable_gap_fill:
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

    timed_entries = [
        (idx, ref)
        for idx, ref in enumerate(ref_lines)
        if ref.start_ms is not None
    ]
    if not timed_entries:
        return list(cues), GapFillResult()

    timed_sorted = sorted(timed_entries, key=lambda item: (item[1].start_ms, item[0]))
    next_start: dict[int, int | None] = {}
    for pos, (idx, _ref) in enumerate(timed_sorted):
        next_start[idx] = (
            timed_sorted[pos + 1][1].start_ms if pos + 1 < len(timed_sorted) else None
        )

    pairs = matched_pairs
    if pairs is None:
        pairs = _infer_anchor_pairs(cues, ref_lines, cfg.review_similarity)
    anchors = sorted({j for _, j in pairs if 0 <= j < len(ref_lines)})

    allow_edges = cfg.allow_edge_gap_fill if allow_edge_gap_fill is None else allow_edge_gap_fill
    if anchors:
        allowed: set[int] = set()
        if allow_edges:
            allowed.update(idx for idx, _ in timed_entries if idx < anchors[0])
            allowed.update(idx for idx, _ in timed_entries if idx > anchors[-1])
        for left, right in zip(anchors, anchors[1:]):
            allowed.update(idx for idx, _ in timed_entries if left < idx < right)
    elif allow_edges:
        allowed = {idx for idx, _ in timed_entries}
    else:
        allowed = set()

    recovered: list[LyricCue] = []
    warnings: list[str] = []
    existing_signatures = {
        (cue.start_ms, _squash(cue.text)) for cue in cues
    }

    for orig_idx, ref in timed_entries:
        if orig_idx not in allowed:
            continue

        ref_clean = ref.text.strip()
        if not ref_clean:
            continue

        toks = _tokenize(ref_clean)
        if len(toks) < 2:
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

        if cand_end <= cand_start:
            continue

        signature = (cand_start, _squash(ref_clean))
        if signature in existing_signatures:
            continue

        has_overlap = any(
            max(cue.start_ms, cand_start) < min(cue.end_ms, cand_end)
            for cue in cues + recovered
        )
        if has_overlap:
            warnings.append(
                f"reference_overlap_skipped at {cand_start}ms: {ref_clean[:60]}"
            )
            continue

        # Vocal Activity Gate: protect silent sections from gap filling
        if vocal_profile is not None:
            from lyra_ocr.audio.vad import vocal_active_ratio
            ratio = vocal_active_ratio(vocal_profile, cand_start, cand_end)
            if ratio < 0.08:
                warnings.append(
                    f"reference_gap_fill_skipped_silent at {cand_start}ms: {ref_clean[:60]}"
                )
                continue

        confidence = 0.75 if is_caption else 0.90
        qa_flags = ["caption_gap_fill"] if is_caption else []

        new_cue = LyricCue(
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
            qa_flags=qa_flags,
        )
        recovered.append(new_cue)
        existing_signatures.add(signature)

    merged = sorted(cues + recovered, key=lambda c: (c.start_ms, c.end_ms))
    return merged, GapFillResult(
        filled_cues=recovered,
        n_filled=len(recovered),
        warnings=warnings,
    )


def fetch_reference_bundle(
    query: ReferenceQuery,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    cache_dir: Path | str | None = None,
    force: bool = False,
    ocr_lines: list[str] | None = None,
) -> ReferenceBundle:
    """Fetch reference lyrics bundle from LRCLIB. Never raises on failure."""
    if not (query.track or "").strip():
        return ReferenceBundle()

    key = ""
    if cache_dir is not None:
        key = reference_cache_key(query, provider="lrclib")
        if not force:
            cached = load_cached_reference_bundle(cache_dir, key)
            if cached is not None:
                return cached

    # Multi-query strategy: try specific track+artist first, then track-only, then full-text q=
    search_strategies: list[dict[str, str]] = []
    if query.artist.strip():
        search_strategies.append({"track_name": query.track.strip(), "artist_name": query.artist.strip()})
    search_strategies.append({"track_name": query.track.strip()})
    if query.artist.strip():
        search_strategies.append({"q": f"{query.track.strip()} {query.artist.strip()}"})
    search_strategies.append({"q": query.track.strip()})

    all_items: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()

    for params in search_strategies:
        url = f"{LRCLIB_API}/search?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "LyraOCR/1.0 (local lyric alignment)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw_bytes = response.read()
                data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LRCLIB lookup failed for %s: %s", params, exc)
            continue

        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    iid = item.get("id")
                    if iid is not None:
                        if iid in seen_ids:
                            continue
                        seen_ids.add(iid)
                    all_items.append(item)
        if all_items:
            break

    if not all_items:
        return ReferenceBundle(warnings=["no_lrclib_results"])

    best_item = None
    best_score = float("-inf")
    best_overlap = 0.0

    for item in all_items:
        if not isinstance(item, dict):
            continue
        synced = str(item.get("syncedLyrics") or "").strip()
        plain = str(item.get("plainLyrics") or "").strip()
        if not synced and not plain:
            continue

        track_sim = line_similarity(query.track, str(item.get("trackName") or ""))
        artist_sim = line_similarity(query.artist, str(item.get("artistName") or ""))
        if track_sim < 0.25 and "q" not in search_strategies[0]:
            continue

        diff: float | None = None
        if query.duration_ms:
            try:
                diff = abs(float(item.get("duration") or 0.0) - query.duration_ms / 1000.0)
            except (TypeError, ValueError):
                diff = None
        if diff is not None and diff > 60.0 and not (track_sim >= 0.85 and artist_sim >= 0.70):
            continue

        score = 2.0 * track_sim + 1.5 * artist_sim
        if diff is not None:
            if diff <= 3.0:
                score += 2.0
            elif diff <= 10.0:
                score += 1.0
            elif diff > 30.0:
                score -= 2.0

        if synced:
            score += 0.5  # Prefer synced over plain

        # Lexical OCR overlap calculation
        cand_overlap = 0.0
        if ocr_lines:
            cand_raw = synced if synced else plain
            cand_parsed = parse_reference_text(cand_raw)
            cand_texts = [l.text for l in cand_parsed]
            if cand_texts:
                matches = sum(
                    1 for ol in ocr_lines
                    if any(line_similarity(ol, ct) >= 0.70 for ct in cand_texts)
                )
                cand_overlap = matches / max(1, len(ocr_lines))
                score += 4.0 * cand_overlap

        if score > best_score:
            best_score = score
            best_item = item
            best_overlap = cand_overlap

    if not best_item:
        return ReferenceBundle(warnings=["no_suitable_lrclib_match"])

    # Overlap gate: if we have >= 5 OCR lines, candidate lyrics must share at least 15% vocabulary
    if ocr_lines and len(ocr_lines) >= 5 and best_overlap < 0.15:
        return ReferenceBundle(warnings=["reference_low_ocr_overlap"])

    synced_lyrics = str(best_item.get("syncedLyrics") or "").strip()
    plain_lyrics = str(best_item.get("plainLyrics") or "").strip()

    bundle = ReferenceBundle()
    if synced_lyrics:
        lines = parse_reference_text(synced_lyrics, source="lrclib_synced")
        if lines:
            bundle = ReferenceBundle(
                lines=lines,
                source="lrclib_synced",
                provider="lrclib",
            )
    elif plain_lyrics:
        lines = parse_reference_text(plain_lyrics, source="lrclib_plain")
        if lines:
            bundle = ReferenceBundle(
                lines=lines,
                source="lrclib_plain",
                provider="lrclib",
            )

    if cache_dir is not None and key and bundle.source:
        save_cached_reference_bundle(cache_dir, key, bundle, query)

    return bundle


def romanize_bundle(bundle: ReferenceBundle) -> ReferenceBundle | None:
    """Convert Japanese reference lines (Kanji/Kana) to Romanized (Hepburn) lyrics using pykakasi."""
    if not bundle or not bundle.lines:
        return None

    try:
        import pykakasi
    except ImportError:
        logger.info("pykakasi not installed; skipping reference romanization")
        return None

    try:
        import jaconv
    except ImportError:
        jaconv = None

    kks = pykakasi.kakasi()
    romanized_lines: list[ReferenceLine] = []
    suffixes_to_attach = {"て", "で", "た", "だ", "ない", "ます", "ました", "たい", "ず", "ぬ"}

    for l in bundle.lines:
        if not l.text.strip():
            continue
        # Convert Katakana okurigana to Hiragana for proper compound verb romanization
        text_norm = jaconv.kata2hira(l.text) if jaconv else l.text
        res = kks.convert(text_norm)
        hepburn_tokens: list[str] = []
        cursor = 0
        for idx, item in enumerate(res):
            orig = str(item.get("orig", ""))
            hira = str(item.get("hira", ""))
            hep = str(item.get("hepburn", "")).strip()

            found_at = text_norm.find(orig, cursor) if orig else -1
            preceded_by_space = False
            if found_at > cursor:
                preceded_by_space = bool(text_norm[cursor:found_at].strip() == "" and found_at > cursor)
            if found_at >= 0:
                cursor = found_at + len(orig)

            if orig == "は" and idx > 0:
                hep = "wa"
            elif orig == "へ" and idx > 0:
                hep = "e"

            hep_clean = _CJK_CHAR_RE.sub("", hep).strip()
            if not hep_clean:
                continue

            if (
                orig in suffixes_to_attach
                and not preceded_by_space
                and hepburn_tokens
                and hira == orig
            ):
                hepburn_tokens[-1] = f"{hepburn_tokens[-1]}{hep_clean}"
            else:
                hepburn_tokens.append(hep_clean)

        romaji_text = _CJK_CHAR_RE.sub("", " ".join(hepburn_tokens)).strip()
        romaji_text = re.sub(r"\s+", " ", romaji_text).strip()
        if not romaji_text:
            continue
        romanized_lines.append(
            ReferenceLine(
                text=romaji_text,
                start_ms=l.start_ms,
                end_ms=l.end_ms,
                source="lrclib_romanized",
                language="ja-Latn",
            )
        )

    if not romanized_lines:
        return None

    return ReferenceBundle(
        lines=romanized_lines,
        source="lrclib_romanized",
        provider=bundle.provider,
        language="ja-Latn",
        warnings=list(bundle.warnings),
    )


def select_reference_lines(
    bundle: ReferenceBundle | None,
    target_lang: str,
) -> ReferenceBundle | None:
    """Select reference lines matching a target layer language, romanizing ja -> ja-Latn."""
    if not bundle or not bundle.lines:
        return None

    from lyra_ocr.lang import detect_line_lang, vote_layer_lang

    bundle_lang = bundle.language or vote_layer_lang(
        [l.language or detect_line_lang(l.text) for l in bundle.lines]
    )

    if target_lang == "ja-Latn":
        if bundle_lang == "ja-Latn":
            return bundle
        ja_lines = [
            l for l in bundle.lines if (l.language or detect_line_lang(l.text)) == "ja"
        ]
        if bundle_lang == "ja" or ja_lines:
            src_bundle = ReferenceBundle(
                lines=ja_lines or bundle.lines,
                source=bundle.source,
                provider=bundle.provider,
                language="ja",
                warnings=list(bundle.warnings),
            )
            return romanize_bundle(src_bundle)
        return None

    # If the bundle as a whole matches the target layer
    if bundle_lang == target_lang or (bundle_lang in ("en", "und") and target_lang in ("en", "und")):
        return bundle

    # Otherwise, check if individual lines match the target language
    matching_lines: list[ReferenceLine] = []
    for line in bundle.lines:
        line_lang = line.language or detect_line_lang(line.text)
        if line_lang == target_lang:
            matching_lines.append(line)

    if not matching_lines:
        return None

    return ReferenceBundle(
        lines=matching_lines,
        source=bundle.source,
        provider=bundle.provider,
        language=target_lang,
        warnings=list(bundle.warnings),
    )



def fetch_reference_lines(
    title: str,
    artist: str = "",
    duration_ms: int = 0,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[str]:
    """Look lyrics up on LRCLIB. Returns [] on any failure (never raises)."""
    q = ReferenceQuery(track=title, artist=artist, duration_ms=duration_ms)
    bundle = fetch_reference_bundle(q, timeout=timeout)
    return [line.text for line in bundle.lines]



def load_reference_file(path_str: str) -> list[str]:
    """Read a local ``.lrc``/``.txt`` reference file. Missing file -> []."""
    try:
        with open(path_str, encoding="utf-8", errors="replace") as handle:
            return parse_lyrics_text(handle.read())
    except OSError as exc:
        logger.warning("Cannot read reference lyrics file %s: %s", path_str, exc)
        return []
