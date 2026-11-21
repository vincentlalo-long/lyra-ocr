"""Unit tests for faster-whisper word-level ASR module and alignment helpers."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from lyra_ocr.audio.asr import (
    ASRSegment,
    ASRTranscript,
    ASRWord,
    align_cues_words_with_asr,
    align_multiline_cues_with_asr,
    asr_has_new_prefix_onset,
    asr_to_reference_bundle,
    asr_vocal_gate_score,
    infer_vocal_language_from_cues,
    load_asr_transcript,
    save_asr_transcript,
    transcribe_audio_words,
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


def test_asr_has_new_prefix_onset_repeated_line():
    transcript = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="Let it go Let it go",
                start_ms=10000,
                end_ms=16000,
                words=[
                    ASRWord("Let", 10100, 10300),
                    ASRWord("it", 10300, 10500),
                    ASRWord("go", 10500, 11200),
                    ASRWord("Let", 13050, 13300),
                    ASRWord("it", 13300, 13500),
                    ASRWord("go", 13500, 14200),
                ],
            )
        ],
    )
    # Check second occurrence at 13000ms
    has_onset = asr_has_new_prefix_onset("Let it go", second_start_ms=13000, transcript=transcript, first_start_ms=10000)
    assert has_onset is True


def test_asr_has_new_prefix_onset_held_line_tail_only():
    transcript = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="I will always love you",
                start_ms=10000,
                end_ms=15500,
                words=[
                    ASRWord("I", 10100, 10300),
                    ASRWord("will", 10300, 10600),
                    ASRWord("always", 10600, 11500),
                    ASRWord("love", 13800, 14200),
                    ASRWord("you", 14200, 14800),
                ],
            )
        ],
    )
    # At 14000ms, ASR only has tail words ("love", "you"), not prefix "I will"
    has_onset = asr_has_new_prefix_onset(
        "I will always love you",
        second_start_ms=14000,
        transcript=transcript,
        first_start_ms=10000,
    )
    assert has_onset is False


def test_asr_has_new_prefix_onset_none_or_empty_transcript():
    assert asr_has_new_prefix_onset("Hello world", 5000, None) is False
    empty = ASRTranscript(language="en", segments=[])
    assert asr_has_new_prefix_onset("Hello world", 5000, empty) is False


def test_asr_has_new_prefix_onset_cjk():
    transcript = ASRTranscript(
        language="ja",
        segments=[
            ASRSegment(
                text="その鼓動は時を超えて",
                start_ms=13000,
                end_ms=16000,
                words=[
                    ASRWord("その", 13000, 13400),
                    ASRWord("鼓動", 13400, 14000),
                    ASRWord("は", 14000, 14200),
                ],
            )
        ],
    )
    assert asr_has_new_prefix_onset("その鼓動は時を超えて", 13000, transcript) is True
    # With only tail word "超えて"
    tail_transcript = ASRTranscript(
        language="ja",
        segments=[
            ASRSegment(
                text="超えて",
                start_ms=13000,
                end_ms=14000,
                words=[ASRWord("超えて", 13000, 13800)],
            )
        ],
    )
    assert asr_has_new_prefix_onset("その鼓動は時を超えて", 13000, tail_transcript) is False


def test_asr_has_new_prefix_onset_first_start_ms_guard():
    transcript = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="Hello world",
                start_ms=1000,
                end_ms=1300,
                words=[ASRWord("Hello", 1020, 1200)],
            )
        ],
    )
    # If second_start_ms is 1400 and first_start_ms is 1000, the onset of first cue at 1020ms
    # must NOT trigger as a new onset for second_start_ms!
    assert asr_has_new_prefix_onset("Hello world", second_start_ms=1400, transcript=transcript, first_start_ms=1000) is False


def test_asr_has_new_prefix_onset_pre_window_tail_word_masking():
    # Cue 1 ends with "away" at 12800ms (within second_start_ms - 450 = 12550ms)
    # Cue 2 onset "Let" is at 13050ms (second_start_ms = 13000ms)
    transcript = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="far away Let it go",
                start_ms=12550,
                end_ms=14500,
                words=[
                    ASRWord("far", 12600, 12750),
                    ASRWord("away", 12800, 12950),  # Cue 1 trailing words in window
                    ASRWord("Let", 13050, 13300),   # Cue 2 onset
                    ASRWord("it", 13300, 13500),
                    ASRWord("go", 13500, 14200),
                ],
            )
        ],
    )
    assert asr_has_new_prefix_onset(
        "Let it go",
        second_start_ms=13000,
        transcript=transcript,
        first_start_ms=10000,
    ) is True


def test_asr_has_new_prefix_onset_continuation_beats_partial_prefix():
    # Romaji/Latin: ASR has a partial/noisy prefix match (>= 0.65, e.g. "ankok" vs "zankoku" = 0.833)
    # alongside a much stronger continuation match ("sagashi motomete ita" = 1.0).
    # best_cont_sim > best_prefix_sim + 0.08 must return False.
    cue_romaji = "Zankoku ni sugi yuku toki no naka de sagashi motomete ita"
    cont_transcript = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="ankok sagashi motomete ita",
                start_ms=36400,
                end_ms=39000,
                words=[
                    ASRWord("ankok", 36450, 36800),
                    ASRWord("sagashi", 36900, 37500),
                    ASRWord("motomete", 37500, 38200),
                    ASRWord("ita", 38200, 38900),
                ],
            )
        ],
    )
    assert (
        asr_has_new_prefix_onset(
            cue_romaji,
            second_start_ms=36440,
            transcript=cont_transcript,
            first_start_ms=30400,
        )
        is False
    )

    # Prefix restart returns True when ASR matches prefix strongly
    restart_transcript = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="zankoku ni sugi",
                start_ms=36400,
                end_ms=38000,
                words=[
                    ASRWord("zankoku", 36450, 37000),
                    ASRWord("ni", 37000, 37300),
                    ASRWord("sugi", 37300, 37800),
                ],
            )
        ],
    )
    assert (
        asr_has_new_prefix_onset(
            cue_romaji,
            second_start_ms=36440,
            transcript=restart_transcript,
            first_start_ms=30400,
        )
        is True
    )

    # CJK: noisy prefix ("残酷の" vs "残酷に" = 0.667 >= 0.65) beaten by tail ("求めていた" = 1.0)
    cue_cjk = "残酷に過ぎゆく時の中で探し求めていた"
    cjk_cont_transcript = ASRTranscript(
        language="ja",
        segments=[
            ASRSegment(
                text="残酷の求めていた",
                start_ms=36400,
                end_ms=39000,
                words=[
                    ASRWord("残酷の", 36450, 37000),
                    ASRWord("求めていた", 37100, 38800),
                ],
            )
        ],
    )
    assert (
        asr_has_new_prefix_onset(
            cue_cjk,
            second_start_ms=36440,
            transcript=cjk_cont_transcript,
            first_start_ms=30400,
        )
        is False
    )

    cjk_restart_transcript = ASRTranscript(
        language="ja",
        segments=[
            ASRSegment(
                text="残酷に過ぎ",
                start_ms=36400,
                end_ms=37800,
                words=[
                    ASRWord("残酷", 36450, 37000),
                    ASRWord("に", 37000, 37200),
                    ASRWord("過ぎ", 37200, 37700),
                ],
            )
        ],
    )
    assert (
        asr_has_new_prefix_onset(
            cue_cjk,
            second_start_ms=36440,
            transcript=cjk_restart_transcript,
            first_start_ms=30400,
        )
        is True
    )


def test_infer_vocal_language_from_cues_prefers_ja_when_romaji_or_ja_present():
    assert infer_vocal_language_from_cues([]) is None
    assert infer_vocal_language_from_cues([LyricCue(start_ms=100, end_ms=500, text="...", lang="und")]) is None

    cues_romaji = [
        LyricCue(start_ms=1000, end_ms=3000, text="In the sky", lang="en"),
        LyricCue(start_ms=1000, end_ms=3000, text="En el cielo", lang="es"),
        LyricCue(start_ms=1000, end_ms=3000, text="sono kodou wa", lang="ja-Latn"),
    ]
    assert infer_vocal_language_from_cues(cues_romaji) == "ja"

    cues_ja = [
        LyricCue(start_ms=1000, end_ms=3000, text="その鼓動", lang="ja"),
        LyricCue(start_ms=1000, end_ms=3000, text="In the sky", lang="en"),
        LyricCue(start_ms=4000, end_ms=6000, text="Fly away", lang="en"),
    ]
    assert infer_vocal_language_from_cues(cues_ja) == "ja"

    cues_vi = [
        LyricCue(start_ms=i * 1000, end_ms=i * 1000 + 800, text="Ngày hôm qua", lang="vi")
        for i in range(1, 3)
    ]
    assert infer_vocal_language_from_cues(cues_vi) == "vi"

    # Non-en language with >= 5 cues beats a larger English translation layer
    cues_vi_with_en = [
        LyricCue(start_ms=i * 1000, end_ms=i * 1000 + 800, text="Ngày hôm qua", lang="vi")
        for i in range(5)
    ] + [
        LyricCue(start_ms=i * 1000, end_ms=i * 1000 + 800, text="Yesterday", lang="en")
        for i in range(8)
    ]
    assert infer_vocal_language_from_cues(cues_vi_with_en) == "vi"


def test_transcribe_audio_words_invalidates_mismatched_cached_language(tmp_path: Path):
    stale_transcript = ASRTranscript(
        language="en",
        model_size="small",
        segments=[
            ASRSegment(
                text="stale english words",
                start_ms=1000,
                end_ms=2500,
                language="en",
                words=[ASRWord("stale", 1000, 1500), ASRWord("english", 1500, 2500)],
            )
        ],
    )
    save_asr_transcript(stale_transcript, tmp_path / "asr_transcript.json")
    audio = np.zeros(16000, dtype=np.float32)

    # Matching language returns cached transcript without invoking faster-whisper
    with patch.dict("sys.modules", {"faster_whisper": None}):
        cached_en = transcribe_audio_words(audio, cache_dir=tmp_path, language="en")
        assert cached_en is not None
        assert cached_en.language == "en"

        # Mismatched language ("ja" vs cached "en") invalidates stale cache and returns None
        # when faster-whisper is unavailable
        invalidated = transcribe_audio_words(audio, cache_dir=tmp_path, language="ja")
        assert invalidated is None


def test_align_cues_words_with_asr_matches_and_interpolates():
    transcript = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="fly high away now",
                start_ms=1000,
                end_ms=4000,
                language="en",
                words=[
                    ASRWord("fly", 1050, 1300),
                    ASRWord("away", 2600, 3400),
                    ASRWord("now", 3450, 3900),
                ],
            )
        ],
    )
    cue = LyricCue(start_ms=1000, end_ms=4000, text="fly high away now", lang="en")
    already_aligned = LyricCue(
        start_ms=5000,
        end_ms=7000,
        text="skip me",
        qa_flags=["asr_aligned"],
        words=[{"word": "skip", "start_ms": 5000, "end_ms": 6000}],
    )

    n = align_cues_words_with_asr([cue, already_aligned], transcript)
    assert n == 1
    assert "asr_word_aligned" in cue.qa_flags
    assert [w["word"] for w in cue.words] == ["fly", "high", "away", "now"]
    # Matched words keep ASR timings
    assert cue.words[0]["start_ms"] == 1050
    assert cue.words[0]["end_ms"] == 1300
    assert cue.words[2]["start_ms"] == 2600
    assert cue.words[2]["end_ms"] == 3400
    # Unmatched word "high" linearly interpolated between 1300 and 2600
    assert cue.words[1]["start_ms"] == 1300
    assert cue.words[1]["end_ms"] == 2600
    # Already asr_aligned cue was left untouched
    assert already_aligned.words == [{"word": "skip", "start_ms": 5000, "end_ms": 6000}]


def test_align_cues_words_with_asr_romanizes_ja_for_ja_latn():
    transcript = ASRTranscript(
        language="ja",
        segments=[
            ASRSegment(
                text="その鼓動は",
                start_ms=1000,
                end_ms=3200,
                language="ja",
                words=[
                    ASRWord("その", 1050, 1400),
                    ASRWord("鼓動", 1450, 2600),
                    ASRWord("は", 2650, 3100),
                ],
            )
        ],
    )
    cue = LyricCue(start_ms=1000, end_ms=3200, text="sono kodou wa", lang="ja-Latn")
    n = align_cues_words_with_asr([cue], transcript)
    assert n == 1
    assert "asr_word_aligned" in cue.qa_flags
    assert [w["word"] for w in cue.words] == ["sono", "kodou", "wa"]
    assert cue.words[1]["start_ms"] == 1450
    assert cue.words[1]["end_ms"] == 2600


def test_align_cues_words_with_asr_overlapping_asr_words_stay_monotonic():
    transcript = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="alpha gamma",
                start_ms=1000,
                end_ms=3000,
                language="en",
                words=[
                    ASRWord("alpha", 1000, 2200),
                    ASRWord("gamma", 1800, 2600),
                ],
            )
        ],
    )
    cue = LyricCue(start_ms=1000, end_ms=3000, text="alpha foo gamma", lang="en")
    n = align_cues_words_with_asr([cue], transcript)
    assert n == 1
    assert [w["word"] for w in cue.words] == ["alpha", "foo", "gamma"]
    for i, w in enumerate(cue.words):
        assert 1000 <= w["start_ms"] <= w["end_ms"] <= 3000
        if i > 0:
            assert cue.words[i - 1]["start_ms"] <= w["start_ms"]
            assert cue.words[i - 1]["end_ms"] <= w["start_ms"]


def test_transcribe_audio_words_retries_without_vad_when_silero_starves_on_music():
    from types import SimpleNamespace

    calls: list[dict] = []

    def _make_seg(text: str, start: float, end: float, words_list: list[tuple[str, float, float]], no_speech_prob: float = 0.1):
        return SimpleNamespace(
            text=text,
            start=start,
            end=end,
            no_speech_prob=no_speech_prob,
            words=[SimpleNamespace(word=w, start=ws, end=we, probability=0.92) for w, ws, we in words_list],
        )

    class _FakeWhisperModel:
        def __init__(self, model_size, device="auto", compute_type="default"):
            pass

        def transcribe(self, audio_input, **kwargs):
            calls.append(dict(kwargs))
            info = SimpleNamespace(language="ja", duration=40.0)
            if kwargs.get("vad_filter", True):
                return iter([]), info
            segs = [
                _make_seg("BGM", 0.5, 2.0, [("BGM", 0.5, 2.0)], no_speech_prob=0.92),
                _make_seg("その鼓動を", 10.0, 12.5, [("その", 10.0, 10.7), ("鼓動を", 10.8, 12.5)], no_speech_prob=0.05),
                _make_seg("超えてゆく", 13.0, 15.5, [("超えて", 13.0, 14.2), ("ゆく", 14.3, 15.5)], no_speech_prob=0.08),
                _make_seg("光の先へ", 16.0, 19.0, [("光の", 16.0, 17.5), ("先へ", 17.6, 19.0)], no_speech_prob=0.04),
            ]
            return iter(segs), info

    audio_40s = np.zeros(16000 * 40, dtype=np.float32)
    fake_module = SimpleNamespace(WhisperModel=_FakeWhisperModel)
    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        transcript = transcribe_audio_words(audio_40s, language="ja")

    assert transcript is not None
    assert len(calls) == 2
    assert calls[0]["vad_filter"] is True
    assert calls[1]["vad_filter"] is False
    assert calls[1]["condition_on_previous_text"] is False
    texts = [s.text for s in transcript.segments]
    assert "BGM" not in texts
    assert texts == ["その鼓動を", "超えてゆく", "光の先へ"]


def test_transcribe_audio_words_invalidates_starved_cache_on_long_audio(tmp_path: Path):
    from types import SimpleNamespace

    stale_transcript = ASRTranscript(
        language="ja",
        model_size="small",
        segments=[
            ASRSegment(
                text="only one segment",
                start_ms=1000,
                end_ms=2500,
                language="ja",
                words=[ASRWord("only", 1000, 2500)],
            )
        ],
    )
    save_asr_transcript(stale_transcript, tmp_path / "asr_transcript.json")

    transcribe_calls = 0

    class _FakeWhisperModel:
        def __init__(self, model_size, device="auto", compute_type="default"):
            pass

        def transcribe(self, audio_input, **kwargs):
            nonlocal transcribe_calls
            transcribe_calls += 1
            info = SimpleNamespace(language="ja", duration=35.0)
            segs = [
                SimpleNamespace(
                    text=f"line {i} here",
                    start=float(i * 5 + 1),
                    end=float(i * 5 + 4),
                    no_speech_prob=0.05,
                    words=[
                        SimpleNamespace(word=f"line{i}", start=float(i * 5 + 1), end=float(i * 5 + 2.5), probability=0.95),
                        SimpleNamespace(word="here", start=float(i * 5 + 2.5), end=float(i * 5 + 4), probability=0.95),
                    ],
                )
                for i in range(3)
            ]
            return iter(segs), info

    audio_35s = np.zeros(16000 * 35, dtype=np.float32)
    fake_module = SimpleNamespace(WhisperModel=_FakeWhisperModel)
    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        transcript = transcribe_audio_words(audio_35s, cache_dir=tmp_path, language="ja")

    assert transcribe_calls == 1
    assert transcript is not None
    assert len(transcript.segments) == 3
    assert [s.text for s in transcript.segments] == ["line 0 here", "line 1 here", "line 2 here"]


def test_transcribe_audio_words_filters_bracketed_and_punctuated_hallucinations(tmp_path: Path):
    from types import SimpleNamespace

    class _FakeWhisperModel:
        def __init__(self, model_size, device="auto", compute_type="default"):
            pass

        def transcribe(self, audio_input, **kwargs):
            info = SimpleNamespace(language="en", duration=30.0)
            segs = [
                SimpleNamespace(text="[BGM]", start=0.5, end=2.0, no_speech_prob=0.1, words=[]),
                SimpleNamespace(text="(Music)", start=2.5, end=4.0, no_speech_prob=0.1, words=[]),
                SimpleNamespace(text="♪ BGM ♪", start=4.5, end=6.0, no_speech_prob=0.1, words=[]),
                SimpleNamespace(text="Thanks for watching!", start=25.0, end=27.0, no_speech_prob=0.1, words=[]),
                SimpleNamespace(
                    text="We are the champions!",
                    start=10.0,
                    end=13.0,
                    no_speech_prob=0.05,
                    words=[
                        SimpleNamespace(word="We", start=10.0, end=10.5, probability=0.95),
                        SimpleNamespace(word="are", start=10.5, end=11.0, probability=0.95),
                        SimpleNamespace(word="the", start=11.0, end=11.5, probability=0.95),
                        SimpleNamespace(word="champions!", start=11.5, end=13.0, probability=0.95),
                    ],
                ),
            ]
            return iter(segs), info

    audio_30s = np.zeros(16000 * 30, dtype=np.float32)
    fake_module = SimpleNamespace(WhisperModel=_FakeWhisperModel)
    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        transcript = transcribe_audio_words(audio_30s, cache_dir=tmp_path, language="en")

    assert transcript is not None
    assert [s.text for s in transcript.segments] == ["We are the champions!"]
