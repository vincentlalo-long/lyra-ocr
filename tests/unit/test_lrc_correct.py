import pytest
from lyra_ocr.lyric.models import CorrectionConfig
from lyra_ocr.lyric.correct import correct_text


def test_correct_text_none():
    track = {
        "features": {"dominant_text": "RAW TEXT"},
        "occurrences": [
            {"text_norm": "RAW TEXT", "confidence": 0.8},
            {"text_norm": "BETTER TEXT", "confidence": 0.99},
        ],
    }
    cfg = CorrectionConfig(method="none")
    final, raw, info = correct_text(track, [track], cfg)
    assert final == "BETTER TEXT"
    assert info.method == "none"
    assert not info.changed


def test_correct_text_consensus():
    track = {
        "features": {"dominant_text": "I FEEL THE NIGHT EXPLODE"},
        "occurrences": [
            {"text_norm": "I FEEL THE NIGHT EXPL0DE", "confidence": 0.9},
            {"text_norm": "I FEEL THE NIGHT EXPLODE", "confidence": 0.98},
            {"text_norm": "I FEEL THE NIGHT EXPLODE", "confidence": 0.97},
            {"text_norm": "I FEEL THE NIGHT EXPLODE", "confidence": 0.99},
        ],
    }
    cfg = CorrectionConfig(method="consensus")
    final, raw, info = correct_text(track, [track], cfg)
    assert final == "I FEEL THE NIGHT EXPLODE"
    assert info.method == "consensus"


def test_correct_text_dictionary_corpus():
    # Vocab contains "NIGHT" 5 times across tracks
    all_tracks = [
        {"features": {"dominant_text": "I FEEL THE NIGHT EXPLODE"}},
        {"features": {"dominant_text": "DARK NIGHT IN THE CITY"}},
        {"features": {"dominant_text": "GOOD NIGHT MY LOVE"}},
        {"features": {"dominant_text": "ONE MORE NIGHT"}},
        {"features": {"dominant_text": "STARRY NIGHT"}},
    ]
    # Current track has typo "NIGTT" (len 5, edit dist 1 to NIGHT)
    track = {
        "features": {"dominant_text": "THE NIGTT IS YOUNG"},
        "occurrences": [
            {"text_norm": "THE NIGTT IS YOUNG", "confidence": 0.95},
        ],
    }
    cfg = CorrectionConfig(method="dictionary", min_vocab_freq=2)
    final, raw, info = correct_text(track, all_tracks, cfg)
    assert "NIGHT" in final
    assert "NIGTT" not in final
    assert info.changed is True


def test_correct_text_preserves_rare_proper_nouns():
    all_tracks = [
        {"features": {"dominant_text": "Hello world"}},
    ]
    track = {
        "features": {"dominant_text": "Xerath"},
        "occurrences": [{"text_norm": "Xerath", "confidence": 0.95}],
    }
    cfg = CorrectionConfig(method="dictionary")
    final, raw, info = correct_text(track, all_tracks, cfg)
    # Should not be changed because no strong candidate in vocab
    assert final == "Xerath"


def test_correct_text_recovers_ocr_digit_substitution():
    all_tracks = [
        {"features": {"dominant_text": "I FEEL THE NIGHT EXPLODE"}},
        {"features": {"dominant_text": "EXPLODE INTO THE NIGHT"}},
        {"features": {"dominant_text": "DON'T EXPLODE"}},
        {"features": {"dominant_text": "HEARTS EXPLODE"}},
    ]
    track = {
        "features": {"dominant_text": "I FEEL THE NIGHT EXPL0DE"},
        "occurrences": [{"text_norm": "I FEEL THE NIGHT EXPL0DE", "confidence": 0.9}],
    }
    cfg = CorrectionConfig(method="dictionary", min_vocab_freq=2)
    final, raw, info = correct_text(track, all_tracks, cfg)
    assert "EXPLODE" in final
    assert "EXPL0DE" not in final
    assert info.changed is True
    assert "EXPL0DE" in info.diff and "EXPLODE" in info.diff


def test_build_corpus_vocab_ignores_rejected_and_non_lyric():
    from lyra_ocr.lyric.correct import build_corpus_vocab

    tracks = [
        {
            "label": "non_lyric",
            "decision": "reject",
            "features": {"dominant_text": "SUBSCRIBE TO CHANNEL"},
        },
        {
            "label": "lyric",
            "decision": "accept",
            "features": {"dominant_text": "SINGING IN THE RAIN"},
        },
    ]
    vocab = build_corpus_vocab(tracks)
    assert "subscribe" not in vocab
    assert "singing" in vocab

