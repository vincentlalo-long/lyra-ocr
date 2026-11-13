"""Word-level Automatic Speech Recognition (ASR) engine using faster-whisper."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from lyra_ocr.lyric.models import LyricCue
    from lyra_ocr.lyric.reference import ReferenceBundle

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]+")


@dataclass
class ASRWord:
    word: str
    start_ms: int
    end_ms: int
    probability: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ASRWord:
        return cls(
            word=str(d.get("word", "")),
            start_ms=int(d.get("start_ms", 0)),
            end_ms=int(d.get("end_ms", 0)),
            probability=float(d.get("probability", 1.0)),
        )


@dataclass
class ASRSegment:
    text: str
    start_ms: int
    end_ms: int
    words: list[ASRWord] = field(default_factory=list)
    language: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "language": self.language,
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ASRSegment:
        return cls(
            text=str(d.get("text", "")),
            start_ms=int(d.get("start_ms", 0)),
            end_ms=int(d.get("end_ms", 0)),
            language=str(d.get("language", "")),
            words=[ASRWord.from_dict(w) for w in d.get("words", [])],
        )


@dataclass
class ASRTranscript:
    segments: list[ASRSegment] = field(default_factory=list)
    language: str = ""
    model_size: str = "small"

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "model_size": self.model_size,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ASRTranscript:
        return cls(
            language=str(d.get("language", "")),
            model_size=str(d.get("model_size", "small")),
            segments=[ASRSegment.from_dict(s) for s in d.get("segments", [])],
        )


def save_asr_transcript(transcript: ASRTranscript, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(transcript.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_asr_transcript(path: str | Path) -> ASRTranscript | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return ASRTranscript.from_dict(data)
    except Exception as exc:
        logger.warning("Failed to load ASR transcript cache %s: %s", p, exc)
        return None


def _normalize_whisper_lang(language: str | None) -> str | None:
    if not language:
        return None
    lang = language.strip().lower()
    if lang in ("", "und", "auto", "unknown"):
        return None
    if lang == "ja-latn":
        return "ja"
    if "-" in lang:
        return lang.split("-")[0]
    return lang


def transcribe_audio_words(
    audio_source: str | Path | np.ndarray,
    *,
    cache_dir: str | Path | None = None,
    language: str | None = None,
    model_size: str = "small",
) -> ASRTranscript | None:
    """Transcribe audio into word-level timestamps using faster-whisper, with disk caching."""
    cache_file = Path(cache_dir) / "asr_transcript.json" if cache_dir else None
    if cache_file is not None and cache_file.is_file():
        cached = load_asr_transcript(cache_file)
        if cached is not None:
            return cached

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.info("faster-whisper not installed; skipping word-level ASR")
        return None

    if isinstance(audio_source, np.ndarray):
        audio_input: str | np.ndarray = np.asarray(audio_source, dtype=np.float32).reshape(-1)
        if audio_input.size == 0:
            return None
    else:
        p = Path(audio_source)
        if not p.is_file():
            logger.warning("ASR audio source does not exist: %s", p)
            return None
        audio_input = str(p)

    whisper_lang = _normalize_whisper_lang(language)

    try:
        model = WhisperModel(model_size, device="auto", compute_type="default")
        segments_iter, info = model.transcribe(
            audio_input,
            language=whisper_lang,
            word_timestamps=True,
            vad_filter=True,
        )
        detected_lang = getattr(info, "language", "") or whisper_lang or ""
        segments: list[ASRSegment] = []
        for seg in segments_iter:
            seg_text = str(getattr(seg, "text", "") or "").strip()
            if not seg_text:
                continue
            words: list[ASRWord] = []
            for w in getattr(seg, "words", None) or ():
                w_text = str(getattr(w, "word", "") or "").strip()
                if not w_text:
                    continue
                w_start = int(round(float(getattr(w, "start", 0.0)) * 1000))
                w_end = int(round(float(getattr(w, "end", 0.0)) * 1000))
                w_prob = float(getattr(w, "probability", 1.0))
                words.append(ASRWord(word=w_text, start_ms=w_start, end_ms=max(w_start, w_end), probability=w_prob))

            seg_start = int(round(float(getattr(seg, "start", 0.0)) * 1000))
            seg_end = int(round(float(getattr(seg, "end", 0.0)) * 1000))
            segments.append(
                ASRSegment(
                    text=seg_text,
                    start_ms=seg_start,
                    end_ms=max(seg_start, seg_end),
                    words=words,
                    language=detected_lang,
                )
            )
    except Exception as exc:
        logger.warning("faster-whisper failed (%s); trying CPU int8 fallback", exc)
        try:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            segments_iter, info = model.transcribe(
                audio_input,
                language=whisper_lang,
                word_timestamps=True,
                vad_filter=True,
            )
            detected_lang = getattr(info, "language", "") or whisper_lang or ""
            segments = []
            for seg in segments_iter:
                seg_text = str(getattr(seg, "text", "") or "").strip()
                if not seg_text:
                    continue
                words = [
                    ASRWord(
                        word=str(getattr(w, "word", "") or "").strip(),
                        start_ms=int(round(float(getattr(w, "start", 0.0)) * 1000)),
                        end_ms=int(round(float(getattr(w, "end", 0.0)) * 1000)),
                        probability=float(getattr(w, "probability", 1.0)),
                    )
                    for w in (getattr(seg, "words", None) or ())
                    if str(getattr(w, "word", "") or "").strip()
                ]
                seg_start = int(round(float(getattr(seg, "start", 0.0)) * 1000))
                seg_end = int(round(float(getattr(seg, "end", 0.0)) * 1000))
                segments.append(
                    ASRSegment(
                        text=seg_text,
                        start_ms=seg_start,
                        end_ms=max(seg_start, seg_end),
                        words=words,
                        language=detected_lang,
                    )
                )
        except Exception as exc_cpu:
            logger.warning("ASR transcription unavailable: %s", exc_cpu)
            return None

    transcript = ASRTranscript(segments=segments, language=detected_lang, model_size=model_size)
    if cache_file is not None:
        save_asr_transcript(transcript, cache_file)
    return transcript


def _romanize_ja_text(text: str) -> str:
    """Convert Japanese Kanji/Kana text to Hepburn Romaji with particle normalization."""
    if not text or not text.strip():
        return ""
    try:
        import pykakasi
    except ImportError:
        return text

    try:
        import jaconv
        text_norm = jaconv.kata2hira(text)
    except ImportError:
        text_norm = text

    kks = pykakasi.kakasi()
    items = kks.convert(text_norm)
    tokens: list[str] = []
    for idx, item in enumerate(items):
        orig = str(item.get("orig", ""))
        hira = str(item.get("hira", ""))
        hep = str(item.get("hepburn", "")).strip()
        if orig == "は" or (hira == "は" and idx > 0 and orig == "は"):
            hep = "wa"
        elif orig == "へ":
            hep = "e"
        elif orig == "を":
            hep = "o"
        if hep:
            cleaned = _CJK_RE.sub("", hep).strip()
            if cleaned:
                tokens.append(cleaned)

    out = " ".join(tokens).strip()
    out = _CJK_RE.sub("", out).strip()
    return re.sub(r"\s+", " ", out) if out else ""


def asr_to_reference_bundle(
    transcript: ASRTranscript | None,
    target_lang: str = "",
) -> ReferenceBundle | None:
    """Convert an ASRTranscript into a ReferenceBundle, optionally romanizing ja -> ja-Latn."""
    if not transcript or not transcript.segments:
        return None

    from lyra_ocr.lyric.reference import ReferenceBundle, ReferenceLine, romanize_bundle

    ref_lines: list[ReferenceLine] = []
    for seg in transcript.segments:
        txt = (seg.text or "").strip()
        if not txt:
            continue
        ref_lines.append(
            ReferenceLine(
                text=txt,
                start_ms=int(seg.start_ms),
                end_ms=int(seg.end_ms),
                source="asr_whisper",
                language=transcript.language or seg.language or "und",
            )
        )

    if not ref_lines:
        return None

    bundle = ReferenceBundle(
        lines=ref_lines,
        source="asr_whisper",
        provider="faster_whisper",
        language=transcript.language or "und",
    )

    if target_lang == "ja-Latn" and (transcript.language == "ja" or any(_CJK_RE.search(l.text) for l in ref_lines)):
        rom_bundle = romanize_bundle(bundle)
        if rom_bundle is not None:
            rom_lines = []
            for orig_line, rom_line in zip(ref_lines, rom_bundle.lines):
                # Ensure particle normalization and strict zero-CJK safety
                norm_text = _romanize_ja_text(orig_line.text) or _CJK_RE.sub("", rom_line.text).strip()
                if norm_text:
                    rom_lines.append(
                        ReferenceLine(
                            text=norm_text,
                            start_ms=orig_line.start_ms,
                            end_ms=orig_line.end_ms,
                            source="asr_romanized",
                            language="ja-Latn",
                        )
                    )
            if rom_lines:
                return ReferenceBundle(
                    lines=rom_lines,
                    source="asr_romanized",
                    provider="faster_whisper",
                    language="ja-Latn",
                )

    return bundle


def _clean_match_str(text: str) -> str:
    return re.sub(r"[^a-z0-9\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+", " ", (text or "").lower()).strip()


def align_multiline_cues_with_asr(
    cues_group: list[LyricCue],
    transcript: ASRTranscript | None,
) -> bool:
    """Partition ASR word spans monotonically across simultaneous multi-line cues."""
    if not cues_group or not transcript or not transcript.segments:
        return False

    t_start = min(c.start_ms for c in cues_group)
    t_end = max(c.end_ms for c in cues_group)
    win_start = t_start - 500
    win_end = t_end + 500

    need_romanize = (cues_group[0].lang == "ja-Latn") and (transcript.language == "ja")
    words_in_win: list[ASRWord] = []

    for seg in transcript.segments:
        if seg.end_ms < win_start or seg.start_ms > win_end:
            continue
        if seg.words:
            for w in seg.words:
                if w.end_ms >= win_start and w.start_ms <= win_end and w.word.strip():
                    w_txt = _romanize_ja_text(w.word) if need_romanize else w.word.strip()
                    if not w_txt:
                        w_txt = w.word.strip()
                    words_in_win.append(
                        ASRWord(word=w_txt, start_ms=int(w.start_ms), end_ms=int(w.end_ms), probability=w.probability)
                    )

    words_in_win.sort(key=lambda w: (w.start_ms, w.end_ms))
    k = len(cues_group)
    n = len(words_in_win)
    if n < k:
        return False

    cue_norms = [_clean_match_str(c.text) for c in cues_group]
    if any(not cn for cn in cue_norms):
        return False

    def span_score(cue_idx: int, s: int, e: int) -> float:
        span_txt = _clean_match_str(" ".join(w.word for w in words_in_win[s:e]))
        target = cue_norms[cue_idx]
        if not span_txt or not target:
            return 0.0
        la, lb = len(target), len(span_txt)
        if (2.0 * min(la, lb) / max(1, la + lb)) < 0.30:
            return 0.0
        seq_r = SequenceMatcher(None, target, span_txt).ratio()
        t_toks = set(target.split())
        s_toks = set(span_txt.split())
        tok_r = len(t_toks & s_toks) / max(1, len(t_toks | s_toks))
        return 0.65 * seq_r + 0.35 * tok_r

    # DP[i][j] = best cumulative score aligning cues 0..i-1 using prefixes ending at <= j
    neg_inf = -1e9
    dp = [[neg_inf] * (n + 1) for _ in range(k + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (n + 1) for _ in range(k + 1)]
    for j in range(n + 1):
        dp[0][j] = 0.0

    for i in range(1, k + 1):
        max_span_w = max(6, len(cue_norms[i - 1].split()) * 3 + 4)
        for e in range(i, n + 1):
            # Allow skipping gaps before this line: carry best from earlier s <= prev_e
            s_min = max(i - 1, e - max_span_w)
            best_val = neg_inf
            best_pair: tuple[int, int] | None = None
            for s in range(s_min, e):
                prev_best = dp[i - 1][s]
                if prev_best <= neg_inf / 2:
                    continue
                sc = span_score(i - 1, s, e)
                val = prev_best + sc
                if val > best_val:
                    best_val = val
                    best_pair = (s, e)
            # Also allow advancing e without changing previous choice? No, we propagate max prefix below
            dp[i][e] = best_val
            back[i][e] = best_pair

        # Propagate running prefix max only when we don't overwrite exact spans:
        # Actually compute prefix_max array so dp[i-1][s] already includes skipping noise words!
        # Let's keep track cleanly:
        for e in range(1, n + 1):
            if dp[i][e - 1] > dp[i][e]:
                dp[i][e] = dp[i][e - 1]
                back[i][e] = back[i][e - 1]

    if dp[k][n] <= neg_inf / 2:
        return False

    avg_score = dp[k][n] / k
    if avg_score < 0.50:
        return False

    # Traceback exact assigned spans (s, e)
    spans: list[tuple[int, int]] = []
    curr_e = n
    for i in range(k, 0, -1):
        pair = back[i][curr_e]
        if pair is None:
            return False
        s, e = pair
        spans.append((s, e))
        curr_e = s
    spans.reverse()

    for cue, (s, e) in zip(cues_group, spans):
        span_words = words_in_win[s:e]
        if not span_words:
            return False
        cue.start_ms = int(span_words[0].start_ms)
        cue.end_ms = int(span_words[-1].end_ms)
        cue.words = [
            {
                "word": w.word,
                "start_ms": int(w.start_ms),
                "end_ms": int(w.end_ms),
            }
            for w in span_words
        ]
        if "asr_aligned" not in cue.qa_flags:
            cue.qa_flags = list(cue.qa_flags) + ["asr_aligned"]

    return True


def asr_vocal_gate_score(
    cue: LyricCue,
    transcript: ASRTranscript | None,
) -> float:
    """Return [0.0, 1.0] vocal confidence score from ASR word overlaps around cue window."""
    if not transcript or not transcript.segments:
        return 0.0

    win_start = cue.start_ms - 600
    win_end = cue.end_ms + 600
    cue_dur = max(1, cue.end_ms - cue.start_ms)

    direct_overlap_ms = 0
    window_overlap = False
    probs: list[float] = []

    for seg in transcript.segments:
        if seg.end_ms < win_start or seg.start_ms > win_end:
            continue
        if seg.words:
            for w in seg.words:
                if w.end_ms >= win_start and w.start_ms <= win_end:
                    window_overlap = True
                    probs.append(w.probability)
                    ov = min(cue.end_ms, w.end_ms) - max(cue.start_ms, w.start_ms)
                    if ov > 0:
                        direct_overlap_ms += ov
        else:
            window_overlap = True
            probs.append(0.85)
            ov = min(cue.end_ms, seg.end_ms) - max(cue.start_ms, seg.start_ms)
            if ov > 0:
                direct_overlap_ms += ov

    if not window_overlap:
        return 0.0

    mean_prob = sum(probs) / len(probs) if probs else 0.85
    if direct_overlap_ms > 0:
        ratio = min(1.0, float(direct_overlap_ms) / float(cue_dur))
        return round(min(1.0, max(0.5 * mean_prob, ratio * mean_prob)), 4)

    return round(min(0.5, 0.5 * mean_prob), 4)
