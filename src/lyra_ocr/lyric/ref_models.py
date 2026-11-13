"""Data models, caching, text metrics, and parsers for reference lyrics."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lyra_ocr.lyric.models import LyricCue

logger = logging.getLogger(__name__)

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
    language: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceLine:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ReferenceBundle:
    """Lyrics bundle containing reference lines and provenance metadata."""

    lines: list[ReferenceLine] = field(default_factory=list)
    source: str = ""
    provider: str = ""
    language: str = ""
    synced: bool = False
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.synced and any(line.start_ms is not None for line in self.lines):
            self.synced = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "lines": [l.to_dict() for l in self.lines],
            "source": self.source,
            "provider": self.provider,
            "language": self.language,
            "synced": self.synced,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceBundle:
        lines_data = data.get("lines") or []
        lines = [ReferenceLine.from_dict(ld) for ld in lines_data if isinstance(ld, dict)]
        return cls(
            lines=lines,
            source=str(data.get("source") or ""),
            provider=str(data.get("provider") or ""),
            language=str(data.get("language") or ""),
            synced=bool(data.get("synced", False)),
            warnings=list(data.get("warnings") or []),
        )


@dataclass
class ReferenceQuery:
    """Normalized query for looking up reference lyrics."""

    track: str
    artist: str = ""
    duration_ms: int = 0
    raw_title: str = ""


@dataclass
class MatchStats:
    similarity: float
    token_coverage: float
    length_ratio: float


@dataclass
class GapFillResult:
    filled_cues: list[LyricCue] = field(default_factory=list)
    n_filled: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class AlignmentResult:
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

        def _clean_bracket(m: re.Match) -> str:
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
                        language=language,
                        confidence=confidence,
                    )
                )
        else:
            result.append(
                ReferenceLine(
                    text=clean_line,
                    start_ms=None,
                    source=source,
                    language=language,
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



def load_reference_file(path_str: str) -> list[str]:
    """Read a local ``.lrc``/``.txt`` reference file. Missing file -> []."""
    try:
        with open(path_str, encoding="utf-8", errors="replace") as handle:
            return parse_lyrics_text(handle.read())
    except OSError as exc:
        logger.warning("Cannot read reference lyrics file %s: %s", path_str, exc)
        return []
