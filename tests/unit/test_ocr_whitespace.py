from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from lyra_ocr.ocr.detect import Detection
from lyra_ocr.ocr.engine import OCREngine, reconstruct_visual_word_spaces


def test_reconstruct_visual_word_spaces_splits_collapsed_romaji():
    # 32px x 240px image with 3 white word blocks on black background
    crop = np.zeros((32, 240), dtype=np.uint8)
    # "sono" (4 chars -> 40px)
    crop[6:26, 20:60] = 255
    # 12px black gap (60:72)
    # "kodou" (5 chars -> 50px)
    crop[6:26, 72:122] = 255
    # 12px black gap (122:134)
    # "wa" (2 chars -> 20px)
    crop[6:26, 134:154] = 255

    reconstructed = reconstruct_visual_word_spaces(crop, "sonokodouwa")
    assert reconstructed == "sono kodou wa"


def test_reconstruct_visual_word_spaces_preserves_continuous_word_with_kerning():
    # 32px x 240px image with single word "shinjita" with 2px kerning gaps
    crop = np.zeros((32, 240), dtype=np.uint8)
    x = 20
    # 7 letters, each 10px wide with 2px gaps
    for _ in range(7):
        crop[6:26, x : x + 10] = 255
        x += 12  # 10px letter + 2px gap

    reconstructed = reconstruct_visual_word_spaces(crop, "shinjita")
    assert reconstructed == "shinjita"


def test_reconstruct_visual_word_spaces_inserts_missing_space_in_partially_spaced_romaji():
    # 4 word blocks: "kimi" (40px), gap (12px), "no" (20px), gap (12px), "na" (20px), gap (12px), "wa" (20px)
    crop = np.zeros((32, 240), dtype=np.uint8)
    crop[6:26, 20:60] = 255    # "kimi"
    crop[6:26, 72:92] = 255    # "no"
    crop[6:26, 104:124] = 255  # "na"
    crop[6:26, 136:156] = 255  # "wa"

    # OCR missed the space between "kimi" and "no"
    reconstructed = reconstruct_visual_word_spaces(crop, "kimino na wa")
    assert reconstructed == "kimi no na wa"


def test_reconstruct_visual_word_spaces_dark_on_bright():
    # White background with dark text
    crop = np.full((32, 240), 255, dtype=np.uint8)
    crop[6:26, 20:60] = 0    # "sono"
    crop[6:26, 72:122] = 0   # "kodou"
    crop[6:26, 134:154] = 0  # "wa"

    reconstructed = reconstruct_visual_word_spaces(crop, "sonokodouwa")
    assert reconstructed == "sono kodou wa"


def test_reconstruct_visual_word_spaces_skips_cjk():
    crop = np.zeros((32, 240), dtype=np.uint8)
    crop[6:26, 20:60] = 255
    crop[6:26, 72:122] = 255

    reconstructed = reconstruct_visual_word_spaces(crop, "千本桜夜ニ紛レ")
    assert reconstructed == "千本桜夜ニ紛レ"


def test_process_detections_dual_recognizer_fallback_and_whitespace():
    engine = OCREngine(default_model="PP-OCRv6_rec_small")
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    det = Detection(
        polygon=[[10, 10], [200, 10], [200, 50], [10, 50]],
        bbox=[10, 10, 190, 40],
        score=0.9,
    )

    # Mock recognize_crops to return collapsed latin text
    with patch.object(
        engine,
        "recognize_crops",
        return_value=[("sonokodouwa", 0.85, "PP-OCRv6_rec_small")],
    ), patch.object(
        engine,
        "recognize_crop",
        return_value=("sono kodou wa", 0.94, "latin_PP-OCRv5_rec_mobile"),
    ) as mock_rec_crop:
        lines = engine.process_detections(img, [det])

        assert len(lines) == 1
        line = lines[0]
        # Verified fallback occurred
        assert mock_rec_crop.called
        assert "sono kodou wa" in line.text_norm
        assert line.confidence == 0.94
        assert line.model_id == "latin_PP-OCRv5_rec_mobile"
