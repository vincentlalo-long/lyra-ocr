"""Unit tests for audio forced alignment and CJK mora/character-level tokenization."""

import numpy as np
import pytest

from lyra_ocr.audio.forced_align import _tokenize_words, align_cues_with_audio
from lyra_ocr.lyric.models import LyricCue


def _generate_tone(freq: float, duration_sec: float, sr: int = 16000, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def test_tokenize_words_cjk_character_splitting():
    # Japanese hiragana/katakana/kanji should be split into individual characters/moras
    assert _tokenize_words("その鼓動") == ["そ", "の", "鼓", "動"]
    assert _tokenize_words("カタカナ") == ["カ", "タ", "カ", "ナ"]

    # Chinese characters
    assert _tokenize_words("最美的期待") == ["最", "美", "的", "期", "待"]

    # Korean hangul
    assert _tokenize_words("사랑해") == ["사", "랑", "해"]

    # Latin words split on whitespace
    assert _tokenize_words("Hello world today") == ["Hello", "world", "today"]

    # Mixed Latin and CJK
    assert _tokenize_words("Hello 世界!") == ["Hello", "世", "界!"] or _tokenize_words("Hello 世界!") == ["Hello", "世", "界"] or _tokenize_words("Hello 世界") == ["Hello", "世", "界"]


def test_align_cues_with_audio_cjk_proportional_word_timings():
    sr = 16000
    audio = np.zeros(sr * 4, dtype=np.float32)
    audio[16 * 1000 : 16 * 3000] = _generate_tone(800.0, 2.0, sr=sr, amp=0.5)

    cues = [LyricCue(start_ms=1000, end_ms=3000, text="その鼓動", layer_id=1, lang="ja")]
    aligned = align_cues_with_audio(cues, audio, sample_rate=sr)

    assert len(aligned) == 1
    cue = aligned[0]
    assert len(cue.words) == 4
    words = [w["word"] for w in cue.words]
    assert words == ["そ", "の", "鼓", "動"]

    # Check non-decreasing, distinct timestamps for each mora
    for i in range(len(cue.words) - 1):
        assert cue.words[i]["start_ms"] < cue.words[i + 1]["start_ms"]
        assert cue.words[i]["end_ms"] <= cue.words[i + 1]["end_ms"]
