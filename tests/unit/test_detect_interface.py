"""Unit tests for TextDetector interface in detect.py."""

from unittest.mock import MagicMock, patch

import numpy as np
from lyra_ocr.ocr.detect import Detection, TextDetector, polygon_to_bbox


def test_polygon_to_bbox():
    poly = [[10, 25], [150, 20], [145, 80], [15, 85]]
    bbox = polygon_to_bbox(poly)
    assert bbox == [10, 20, 150, 85]


def test_detect_interface_det_only():
    dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)

    mock_boxes = np.array([
        [[10, 10], [50, 10], [50, 30], [10, 30]],
        [[60, 10], [120, 10], [120, 30], [60, 30]],
    ])
    mock_scores = [0.95, 0.88]

    mock_out = MagicMock()
    mock_out.boxes = mock_boxes
    mock_out.scores = mock_scores

    with patch("rapidocr.RapidOCR.__call__", return_value=mock_out):
        detector = TextDetector()
        results = detector.detect(dummy_img)

    assert len(results) == 2
    for item in results:
        assert isinstance(item, Detection)
        assert len(item.polygon) == 4
        assert len(item.bbox) == 4
        assert isinstance(item.score, float)
        # Verify det-only: no text recognition fields on Detection object
        assert not hasattr(item, "text")
        assert not hasattr(item, "text_raw")
        assert not hasattr(item, "text_norm")

    assert results[0].bbox == [10, 10, 50, 30]
    assert results[0].score == 0.95
