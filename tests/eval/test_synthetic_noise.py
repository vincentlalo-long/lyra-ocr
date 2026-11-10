import random
import pytest

from lyra_ocr.lyric.models import CorrectionConfig
from lyra_ocr.lyric.correct import correct_text


def inject_ocr_noise(text: str, seed: int = 42) -> str:
    """Inject realistic OCR confusion noise into text."""
    rng = random.Random(seed)
    confusion = {
        "O": "0", "0": "O",
        "I": "1", "1": "I",
        "S": "5", "5": "S",
        "B": "8", "8": "B",
    }
    chars = list(text)
    for i, c in enumerate(chars):
        if c in confusion and rng.random() < 0.3:
            chars[i] = confusion[c]
    return "".join(chars)


def test_synthetic_noise_injection_and_recovery():
    # Clean song lyrics
    clean_lines = [
        "I FEEL THE NIGHT EXPLODE",
        "WHEN WE ARE TOGETHER",
        "EMOTION OVERLOAD",
        "IN THE DARK NIGHT",
        "I FEEL THE NIGHT EXPLODE",
    ]

    all_tracks = [{"features": {"dominant_text": l}} for l in clean_lines]

    # Corrupt one track with a mild typo
    noisy_track = {
        "features": {"dominant_text": "I FEEL THE NIGTT EXPLODE"},
        "occurrences": [{"text_norm": "I FEEL THE NIGTT EXPLODE", "confidence": 0.95}],
    }

    cfg = CorrectionConfig(method="dictionary", min_vocab_freq=2)
    final, raw, info = correct_text(noisy_track, all_tracks, cfg)

    # Should recover NIGHT
    assert "NIGHT" in final
    assert info.changed is True
