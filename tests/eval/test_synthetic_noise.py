import random
import pytest

from lyra_ocr.lyric.models import CorrectionConfig
from lyra_ocr.lyric.correct import correct_text, _levenshtein


def inject_ocr_noise(text: str, seed: int = 42) -> str:
    """Inject realistic OCR confusion noise into text."""
    rng = random.Random(seed)
    confusion = {
        "O": "0",
        "I": "1",
        "S": "5",
        "B": "8",
    }
    chars = list(text)
    for i, c in enumerate(chars):
        if c in confusion and rng.random() < 0.4:
            chars[i] = confusion[c]
    return "".join(chars)


def test_synthetic_noise_injection_and_recovery():
    # Clean song lyrics representing typical repeated pop song vocabulary
    clean_lines = [
        "I FEEL THE NIGHT EXPLODE",
        "WHEN WE ARE TOGETHER",
        "EMOTION OVERLOAD",
        "IN THE DARK NIGHT",
        "I FEEL THE NIGHT EXPLODE",
        "WHEN WE ARE TOGETHER",
        "HEARTS EXPLODE",
        "EMOTION OVERLOAD",
    ]

    all_tracks = [{"features": {"dominant_text": l}} for l in clean_lines]

    # Corrupt one track with OCR digit substitution (e.g. EXPL0DE)
    noisy_track = {
        "features": {"dominant_text": "I FEEL THE NIGHT EXPL0DE"},
        "occurrences": [{"text_norm": "I FEEL THE NIGHT EXPL0DE", "confidence": 0.95}],
    }

    cfg = CorrectionConfig(method="dictionary", min_vocab_freq=2)
    final, raw, info = correct_text(noisy_track, all_tracks, cfg)

    # Should recover EXPLODE
    assert "EXPLODE" in final
    assert "EXPL0DE" not in final
    assert info.changed is True


def test_synthetic_noise_cer_reduction_and_hallucination():
    clean_lines = [
        "I FEEL THE NIGHT EXPLODE",
        "WHEN WE ARE TOGETHER",
        "EMOTION OVERLOAD",
        "IN THE DARK NIGHT",
        "I FEEL THE NIGHT EXPLODE",
        "WHEN WE ARE TOGETHER",
        "EMOTION OVERLOAD",
    ]
    all_tracks = [{"features": {"dominant_text": l}} for l in clean_lines]
    cfg = CorrectionConfig(method="dictionary", min_vocab_freq=2)

    # 1. Test clean text hallucination: clean text should not be corrupted
    corrupted_chars = 0
    total_clean_chars = 0
    for line in clean_lines:
        tr = {"features": {"dominant_text": line}, "occurrences": [{"text_norm": line, "confidence": 0.95}]}
        final, _, _ = correct_text(tr, all_tracks, cfg)
        corrupted_chars += _levenshtein(line, final)
        total_clean_chars += len(line)

    hallucination_rate = corrupted_chars / total_clean_chars
    # Hallucination rate must be <= 1%
    assert hallucination_rate <= 0.01

    # 2. Test noisy lines CER reduction
    total_pre_err = 0
    total_post_err = 0
    total_ref_len = 0

    target_line = "I FEEL THE NIGHT EXPLODE"
    # Create noisy variant with OCR noise
    noisy_line = inject_ocr_noise(target_line, seed=123)
    if noisy_line == target_line:
        noisy_line = "I FEEL THE NIGHT EXPL0DE"

    tr_noisy = {
        "features": {"dominant_text": noisy_line},
        "occurrences": [{"text_norm": noisy_line, "confidence": 0.95}],
    }
    corrected_line, _, _ = correct_text(tr_noisy, all_tracks, cfg)

    pre_err = _levenshtein(target_line, noisy_line)
    post_err = _levenshtein(target_line, corrected_line)

    assert pre_err > 0
    assert post_err < pre_err
    cer_reduction = (pre_err - post_err) / pre_err
    assert cer_reduction >= 0.50
