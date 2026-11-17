from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from lyra_ocr.ocr.detect import Detection
from lyra_ocr.ocr.engine import (
    OCREngine,
    reconstruct_visual_word_spaces,
    to_soft_bw_stroke_crop,
)


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
    # 4 word blocks: "kimi" (40px), gap (12px), "nomonogatari" (120px), gap (12px), "na" (20px), gap (12px), "wa" (20px)
    # Collapsed token "kiminomonogatari" has length 16 >= 15.
    crop = np.zeros((32, 300), dtype=np.uint8)
    crop[6:26, 20:60] = 255    # "kimi" (40px)
    crop[6:26, 72:192] = 255   # "nomonogatari" (120px)
    crop[6:26, 204:224] = 255  # "na" (20px)
    crop[6:26, 236:256] = 255  # "wa" (20px)

    # OCR missed the space between "kimi" and "nomonogatari"
    reconstructed = reconstruct_visual_word_spaces(crop, "kiminomonogatari na wa")
    assert reconstructed == "kimi nomonogatari na wa"


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
    # 1. 1-letter word at line end (len >= 9): "tokimade o" collapsed as "tokimadeo"
    crop_end = np.zeros((32, 240), dtype=np.uint8)
    crop_end[6:26, 20:100] = 255  # "tokimade" (80px)
    # 12px gap (100:112)
    crop_end[6:26, 112:122] = 255  # "o" (10px)

    reconstructed_end = reconstruct_visual_word_spaces(crop_end, "tokimadeo")
    assert reconstructed_end == "tokimade o"

    # 2. 1-letter word at line start (len >= 9): "I remember" collapsed as "Iremember"
    crop_start = np.zeros((32, 240), dtype=np.uint8)
    crop_start[6:26, 20:30] = 255  # "I" (10px)
    # 12px gap (30:42)
    crop_start[6:26, 42:122] = 255  # "remember" (80px)

    reconstructed_start = reconstruct_visual_word_spaces(crop_start, "Iremember")
    assert reconstructed_start == "I remember"


def test_reconstruct_visual_word_spaces_skips_already_spaced_lines_with_short_tokens():
    # Even when crop contains wide visual valleys (12px+), lines that already have
    # spaces and max token length <= 14 must be returned unmodified.
    crop = np.zeros((32, 240), dtype=np.uint8)
    crop[6:26, 20:60] = 255
    crop[6:26, 75:115] = 255
    crop[6:26, 130:170] = 255

    assert reconstruct_visual_word_spaces(crop, "mabataki kara") == "mabataki kara"
    assert reconstruct_visual_word_spaces(crop, "kimi no na wa") == "kimi no na wa"
    assert reconstruct_visual_word_spaces(crop, "abcdefghijklmn op") == "abcdefghijklmn op"


def test_reconstruct_visual_word_spaces_skips_short_single_tokens():
    # Single tokens shorter than 9 chars must never be split even with wide gaps.
    crop = np.zeros((32, 240), dtype=np.uint8)
    crop[6:26, 20:60] = 255
    crop[6:26, 75:115] = 255

    assert reconstruct_visual_word_spaces(crop, "mabataki") == "mabataki"
    assert reconstruct_visual_word_spaces(crop, "hirogaru") == "hirogaru"
    assert reconstruct_visual_word_spaces(crop, "tokio") == "tokio"


def test_reconstruct_visual_word_spaces_preserves_thin_colored_intra_word_strokes():
    # "torawarete" (len=10 >= 9) with a 6px intra-word gap (< min_gap = 9 for h_ink=20)
    # and isoluminant colored strokes must NOT be split.
    crop = np.full((32, 240, 3), (30, 110, 30), dtype=np.uint8)  # green bg (luma ~85)
    # Red letters (luma ~85 in BGR) with a 6px gap at 60:66
    crop[6:26, 20:60] = (30, 30, 220)
    crop[6:26, 66:126] = (30, 30, 220)

    assert reconstruct_visual_word_spaces(crop, "torawarete") == "torawarete"


def test_reconstruct_visual_word_spaces_color_invariant_isoluminant_words():
    # Red text (BGR: 30, 30, 220, luma ~85) on green background (BGR: 30, 110, 30, luma ~85)
    # Grayscale luminance has near-zero contrast, but color distance cleanly detects word valleys.
    crop = np.full((32, 240, 3), (30, 110, 30), dtype=np.uint8)
    crop[6:26, 20:60] = (30, 30, 220)    # "sono"
    crop[6:26, 72:122] = (30, 30, 220)   # "kodou"
    crop[6:26, 134:154] = (30, 30, 220)  # "wa"

    reconstructed = reconstruct_visual_word_spaces(crop, "sonokodouwa")
    assert reconstructed == "sono kodou wa"


def test_to_soft_bw_stroke_crop_upscales_normalizes_and_preserves_antialiasing():
    import cv2

    # 24px tall (< 48px) isoluminant colored crop with anti-aliased blurred edges
    crop = np.full((24, 100, 3), (30, 110, 30), dtype=np.uint8)
    crop[6:18, 20:80] = (30, 30, 220)
    crop = cv2.GaussianBlur(crop, (3, 3), 0.8)

    out = to_soft_bw_stroke_crop(crop)

    # Upscaled from H=24 to >=48, plus 4px top/bottom/left/right padding => H >= 56
    assert out.ndim == 3
    assert out.shape[2] == 3
    assert out.shape[0] >= 56
    assert out.dtype == np.uint8

    # Background maps near 0, stroke core maps near 255, anti-aliased edges stay intermediate
    assert int(out.min()) <= 5
    assert int(out.max()) >= 250
    gray = out[:, :, 0]
    intermediate = gray[(gray > 15) & (gray < 240)]
    assert intermediate.size > 0

    # Empty/None crops return safely
    assert to_soft_bw_stroke_crop(None) is None
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert to_soft_bw_stroke_crop(empty).size == 0


def test_process_detections_retries_soft_bw_when_conf_below_threshold():
    engine = OCREngine(default_model="PP-OCRv6_rec_small")
    img = np.full((80, 240, 3), (30, 110, 30), dtype=np.uint8)
    img[20:50, 20:180] = (30, 30, 220)
    det = Detection(
        polygon=[[10, 10], [200, 10], [200, 60], [10, 60]],
        bbox=[10, 10, 190, 50],
        score=0.95,
    )

    # Case 1: Latin line where mobile fallback still has conf < 0.88 -> Soft B&W retry wins
    seen_crops: list[np.ndarray] = []

    def fake_rec_crop(crop, model_id=None):
        seen_crops.append(crop)
        if len(seen_crops) == 1:
            return ("mobarak", 0.72, "latin_PP-OCRv5_rec_mobile")
        return ("mabataki", 0.93, "latin_PP-OCRv5_rec_mobile")

    with patch.object(
        engine,
        "recognize_crops",
        return_value=[("mobarak", 0.65, "PP-OCRv6_rec_small")],
    ), patch.object(engine, "recognize_crop", side_effect=fake_rec_crop):
        lines = engine.process_detections(img, [det])

    assert len(lines) == 1
    assert len(seen_crops) == 2
    # Second call received the 3-channel soft B&W crop (upscaled/padded, dark bg)
    assert seen_crops[1].shape[2] == 3
    assert int(seen_crops[1][0, 0, 0]) <= 5
    assert lines[0].text_norm == "mabataki"
    assert lines[0].confidence == 0.93


def test_process_detections_retries_soft_bw_for_non_latin_low_conf():
    engine = OCREngine(default_model="PP-OCRv6_rec_small")
    img = np.full((80, 240, 3), 20, dtype=np.uint8)
    img[20:50, 20:180] = 220
    det = Detection(
        polygon=[[10, 10], [200, 10], [200, 60], [10, 60]],
        bbox=[10, 10, 190, 50],
        score=0.95,
    )

    # Non-Latin line with conf < 0.88 must still retry Soft B&W and keep higher-conf result
    with patch.object(
        engine,
        "recognize_crops",
        return_value=[("千本", 0.71, "PP-OCRv6_rec_small")],
    ), patch.object(
        engine,
        "recognize_crop",
        return_value=("千本桜", 0.91, "PP-OCRv6_rec_small"),
    ) as mock_rec_crop:
        lines = engine.process_detections(img, [det])

    assert mock_rec_crop.call_count == 1
    assert lines[0].text_norm == "千本桜"
    assert lines[0].confidence == 0.91
    assert lines[0].script == "japan"


def test_process_detections_skips_soft_bw_when_conf_already_high_or_bw_worse():
    engine = OCREngine(default_model="PP-OCRv6_rec_small")
    img = np.full((80, 240, 3), 20, dtype=np.uint8)
    img[20:50, 20:180] = 220
    det = Detection(
        polygon=[[10, 10], [200, 10], [200, 60], [10, 60]],
        bbox=[10, 10, 190, 50],
        score=0.95,
    )

    # High confidence (>= 0.88): no Soft B&W retry
    with patch.object(
        engine,
        "recognize_crops",
        return_value=[("千本桜", 0.90, "PP-OCRv6_rec_small")],
    ), patch.object(engine, "recognize_crop") as mock_rec_crop:
        lines = engine.process_detections(img, [det])
    assert mock_rec_crop.call_count == 0
    assert lines[0].text_norm == "千本桜"

    # Low confidence where Soft B&W produces lower confidence: keep original
    with patch.object(
        engine,
        "recognize_crops",
        return_value=[("千本桜", 0.82, "PP-OCRv6_rec_small")],
    ), patch.object(
        engine,
        "recognize_crop",
        return_value=("wrong", 0.60, "PP-OCRv6_rec_small"),
    ) as mock_rec_crop:
        lines = engine.process_detections(img, [det])
    assert mock_rec_crop.call_count == 1
    assert lines[0].text_norm == "千本桜"
    assert lines[0].confidence == 0.82
    assert lines[0].script == "japan"

