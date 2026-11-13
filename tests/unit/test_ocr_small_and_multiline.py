"""Unit tests for small-text crop preprocessing and tall 2-line quad splitting."""

import cv2
import numpy as np

from lyra_ocr.ocr.detect import split_multiline_crop_quads
from lyra_ocr.ocr.engine import preprocess_crop_for_recognition


def test_preprocess_crop_upscales_small_text_with_padding():
    # 14px tall x 120px wide synthetic crop with white text strokes on dark background
    crop = np.zeros((14, 120, 3), dtype=np.uint8)
    for x in range(10, 110, 6):
        crop[3:11, x : x + 2] = 240

    out = preprocess_crop_for_recognition(crop)

    # Upscaled height >= 56px + 4px top/bottom border padding => >= 64px
    assert out.shape[0] >= 64
    assert out.shape[1] >= 480 + 8
    assert out.dtype == np.uint8


def test_preprocess_crop_handles_empty_or_none():
    assert preprocess_crop_for_recognition(None) is None
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    res = preprocess_crop_for_recognition(empty)
    assert res.size == 0


def test_split_multiline_crop_quads_splits_two_line_box():
    # 90x260 frame containing a 72px tall 2-line text block at y=[8..80]
    frame = np.zeros((90, 260, 3), dtype=np.uint8)
    # Draw crisp vertical stroke patterns simulating 2 text lines separated by a gap around y=44
    for x in range(20, 220, 8):
        frame[16:34, x : x + 3] = 255  # Top line (y = 16..34)
        frame[54:72, x : x + 3] = 255  # Bottom line (y = 54..72)

    quad = np.array([[10, 8], [230, 8], [230, 80], [10, 80]], dtype=np.float32)
    out_quads, out_scores = split_multiline_crop_quads(frame, [quad], [0.95])

    assert len(out_quads) == 2
    assert out_scores == [0.95, 0.95]

    y_center_0 = float(np.mean(out_quads[0][:, 1]))
    y_center_1 = float(np.mean(out_quads[1][:, 1]))
    assert y_center_0 < 40.0
    assert y_center_1 > 45.0
    assert abs(y_center_1 - y_center_0) >= 24.0


def test_split_multiline_crop_quads_keeps_single_tall_line():
    frame = np.zeros((90, 260, 3), dtype=np.uint8)
    # Continuous strokes across the whole middle (no clear horizontal valley)
    for x in range(20, 220, 8):
        frame[16:72, x : x + 3] = 255

    quad = np.array([[10, 8], [230, 8], [230, 80], [10, 80]], dtype=np.float32)
    out_quads, out_scores = split_multiline_crop_quads(frame, [quad], [0.90])

    assert len(out_quads) == 1
    assert out_scores == [0.90]
