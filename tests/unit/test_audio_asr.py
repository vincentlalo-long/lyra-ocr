"""Unit tests for faster-whisper word-level ASR module and alignment helpers."""

from pathlib import Path

from lyra_ocr.audio.asr import (
    ASRSegment,
    ASRTranscript,
    ASRWord,
    align_multiline_cues_with_asr,
    asr_to_reference_bundle,
    asr_vocal_gate_score,
    load_asr_transcript,
    save_asr_transcript,
)
from lyra_ocr.lyric.models import LyricCue


def test_asr_transcript_cache_round_trip(tmp_path: Path):
    transcript = ASRTranscript(
        language="ja",
        model_size="small",
        segments=[
            ASRSegment(
                text="その鼓動を超えて",
                start_ms=1000,
                end_ms=3400,
                language="ja",
                words=[
                    ASRWord(word="その", start_ms=1000, end_ms=1400, probability=0.95),
                    ASRWord(word="鼓動", start_ms=1450, end_ms=2100, probability=0.92),
                    ASRWord(word="を", start_ms=2100, end_ms=2300, probability=0.98),
                    ASRWord(word="超えて", start_ms=2350, end_ms=3400, probability=0.91),
                ],
            )
        ],
    )
    cache_file = tmp_path / "asr_transcript.json"
    save_asr_transcript(transcript, cache_file)
    loaded = load_asr_transcript(cache_file)

    assert loaded is not None
    assert loaded.language == "ja"
    assert loaded.model_size == "small"
    assert len(loaded.segments) == 1
    assert loaded.segments[0].text == "その鼓動を超えて"
    assert len(loaded.segments[0].words) == 4
    assert loaded.segments[0].words[1].start_ms == 1450
    assert loaded.segments[0].words[1].end_ms == 2100


def test_align_multiline_cues_with_asr_extracts_exact_spans():
    transcript = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="In the castle of the sky we fly higher than the sun",
                start_ms=10000,
                end_ms=19000,
                language="en",
                words=[
                    ASRWord("In", 10100, 10300),
                    ASRWord("the", 10300, 10500),
                    ASRWord("castle", 10500, 11800),
                    ASRWord("of", 13100, 13300),
                    ASRWord("the", 13300, 13600),
                    ASRWord("sky", 13600, 15200),
                    ASRWord("we", 16000, 16300),
                    ASRWord("fly", 16300, 17100),
                    ASRWord("higher", 17100, 18700),
                ],
            )
        ],
    )
    cues = [
        LyricCue(start_ms=10000, end_ms=19000, text="In the castle", y_center=0.72, layer_id=1, lang="en"),
        LyricCue(start_ms=10000, end_ms=19000, text="of the sky", y_center=0.80, layer_id=1, lang="en"),
        LyricCue(start_ms=10000, end_ms=19000, text="we fly higher", y_center=0.88, layer_id=1, lang="en"),
    ]

    ok = align_multiline_cues_with_asr(cues, transcript)

    assert ok is True
    assert cues[0].start_ms == 10100
    assert cues[0].end_ms == 11800
    assert cues[1].start_ms == 13100
    assert cues[1].end_ms == 15200
    assert cues[2].start_ms == 16000
    assert cues[2].end_ms == 18700
    assert len(cues[0].words) == 3
    assert cues[0].words[0]["word"] == "In"
    assert len(cues[2].words) == 3


def test_asr_to_reference_bundle_romanizes_ja_to_ja_latn():
    transcript = ASRTranscript(
        language="ja",
        segments=[
            ASRSegment(text="その鼓動は", start_ms=1000, end_ms=3000, language="ja"),
        ],
    )
    bundle = asr_to_reference_bundle(transcript, target_lang="ja-Latn")

    assert bundle is not None
    assert bundle.source == "asr_romanized"
    assert bundle.language == "ja-Latn"
    assert len(bundle.lines) == 1
    assert "sono" in bundle.lines[0].text.lower()
    assert "kodou" in bundle.lines[0].text.lower()
    assert bundle.lines[0].text.lower().endswith("wa")


def test_asr_vocal_gate_score_detects_active_vs_silent():
    transcript = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="fly away",
                start_ms=5000,
                end_ms=7000,
                words=[ASRWord("fly", 5000, 5800), ASRWord("away", 5900, 7000)],
            )
        ],
    )
    active_cue = LyricCue(start_ms=5100, end_ms=6800, text="fly away")
    silent_cue = LyricCue(start_ms=25000, end_ms=27000, text="SUBSCRIBE NOW")

    assert asr_vocal_gate_score(active_cue, transcript) >= 0.8
    assert asr_vocal_gate_score(silent_cue, transcript) == 0.0
