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


def test_get_model_offline_fallback_cached_once():
    engine = OCREngine(default_model="PP-OCRv6_rec_small")
    # latin_PP-OCRv5_rec_mobile not yet in _models
    assert "latin_PP-OCRv5_rec_mobile" not in engine._models

    with patch("rapidocr.RapidOCR", side_effect=RuntimeError("weights unavailable")) as mock_rapid:
        model1 = engine._get_model("latin_PP-OCRv5_rec_mobile")
        assert model1 is engine._models[engine.DEFAULT_MODEL]
        assert mock_rapid.call_count == 1

        # Second call must reuse cached fallback without invoking RapidOCR constructor again
        model2 = engine._get_model("latin_PP-OCRv5_rec_mobile")
        assert model2 is engine._models[engine.DEFAULT_MODEL]
        assert mock_rapid.call_count == 1


def test_reconstruct_visual_word_spaces_splits_one_letter_boundary_word():
    # 1. 1-letter word at line end: "toki o" collapsed as "tokio"
    crop_end = np.zeros((32, 240), dtype=np.uint8)
    crop_end[6:26, 20:60] = 255  # "toki" (40px)
    # 12px gap (60:72)
    crop_end[6:26, 72:82] = 255  # "o" (10px)

    reconstructed_end = reconstruct_visual_word_spaces(crop_end, "tokio")
    assert reconstructed_end == "toki o"

    # 2. 1-letter word at line start: "I know" collapsed as "Iknow"
    crop_start = np.zeros((32, 240), dtype=np.uint8)
    crop_start[6:26, 20:30] = 255  # "I" (10px)
    # 12px gap (30:42)
    crop_start[6:26, 42:82] = 255  # "know" (40px)

    reconstructed_start = reconstruct_visual_word_spaces(crop_start, "Iknow")
    assert reconstructed_start == "I know"

