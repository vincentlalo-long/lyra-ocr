"""Unit tests for centralized threshold dataclasses and module wiring in config.py."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pytest

from lyra_ocr.classify import classify_track
from lyra_ocr.config import (
    DEFAULT_CLASSIFICATION_THRESHOLDS,
    DEFAULT_GROUPING_THRESHOLDS,
    DEFAULT_LAYER_DISCOVERY_THRESHOLDS,
    DEFAULT_OCR_PREPROCESS_THRESHOLDS,
    ClassificationThresholds,
    GroupingThresholds,
    LayerDiscoveryThresholds,
    OcrPreprocessThresholds,
)
from lyra_ocr.layers import discover_layers
from lyra_ocr.ocr.detect import Detection
from lyra_ocr.ocr.engine import (
    OCREngine,
    preprocess_crop_for_recognition,
    reconstruct_visual_word_spaces,
    to_soft_bw_stroke_crop,
)
from lyra_ocr.track import DetectionOccurrence, TextTrack


def test_config_dataclasses_are_frozen_with_production_defaults():
    assert isinstance(DEFAULT_LAYER_DISCOVERY_THRESHOLDS, LayerDiscoveryThresholds)
    assert isinstance(DEFAULT_GROUPING_THRESHOLDS, GroupingThresholds)
    assert isinstance(DEFAULT_CLASSIFICATION_THRESHOLDS, ClassificationThresholds)
    assert isinstance(DEFAULT_OCR_PREPROCESS_THRESHOLDS, OcrPreprocessThresholds)

    with pytest.raises(FrozenInstanceError):
        DEFAULT_LAYER_DISCOVERY_THRESHOLDS.lyric_min_tracks = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        DEFAULT_GROUPING_THRESHOLDS.dedupe_text_similarity = 0.5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        DEFAULT_CLASSIFICATION_THRESHOLDS.lyric_min_dur_ms = 100  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        DEFAULT_OCR_PREPROCESS_THRESHOLDS.fallback_min_confidence = 0.5  # type: ignore[misc]

    # Spot-check live production values from layers.py, grouping.py, classify.py, ocr/engine.py
    assert DEFAULT_LAYER_DISCOVERY_THRESHOLDS.cluster_cy_tolerance == 0.030
    assert DEFAULT_LAYER_DISCOVERY_THRESHOLDS.cluster_cx_tolerance == 0.25
    assert DEFAULT_LAYER_DISCOVERY_THRESHOLDS.lyric_min_tracks == 4
    assert DEFAULT_LAYER_DISCOVERY_THRESHOLDS.companion_y_dist_min == 0.025
    assert DEFAULT_LAYER_DISCOVERY_THRESHOLDS.companion_y_dist_max == 0.16

    assert DEFAULT_GROUPING_THRESHOLDS.duplicate_min_containment == 0.70
    assert DEFAULT_GROUPING_THRESHOLDS.duplicate_max_y_gap == 0.12
    assert DEFAULT_GROUPING_THRESHOLDS.consolidate_overlap_threshold == 0.25
    assert DEFAULT_GROUPING_THRESHOLDS.same_y_band_tolerance == 0.032
    assert DEFAULT_GROUPING_THRESHOLDS.short_wrap_max_dur_ms == 4000
    assert DEFAULT_GROUPING_THRESHOLDS.dedupe_text_similarity == 0.88
    assert DEFAULT_GROUPING_THRESHOLDS.fade_max_gap_ms == 800
    assert DEFAULT_GROUPING_THRESHOLDS.fade_max_dur_ms == 1400
    assert DEFAULT_GROUPING_THRESHOLDS.share_min_gap_ms == 1200
    assert DEFAULT_GROUPING_THRESHOLDS.share_min_cue_dur_ms == 900

    assert DEFAULT_CLASSIFICATION_THRESHOLDS.max_spatial_variance_non_lyric == 0.04
    assert DEFAULT_CLASSIFICATION_THRESHOLDS.persistent_min_lifetime_ratio == 0.45
    assert DEFAULT_CLASSIFICATION_THRESHOLDS.intro_credit_start_ratio_max == 0.15
    assert DEFAULT_CLASSIFICATION_THRESHOLDS.chorus_min_prior_tracks == 2
    assert DEFAULT_CLASSIFICATION_THRESHOLDS.lyric_min_dur_ms == 500
    assert DEFAULT_CLASSIFICATION_THRESHOLDS.lyric_max_dur_ms == 16000

    assert DEFAULT_OCR_PREPROCESS_THRESHOLDS.upscale_trigger_height_px == 36
    assert DEFAULT_OCR_PREPROCESS_THRESHOLDS.upscale_target_height_px == 56.0
    assert DEFAULT_OCR_PREPROCESS_THRESHOLDS.soft_bw_min_height_px == 48
    assert DEFAULT_OCR_PREPROCESS_THRESHOLDS.spaced_max_token_len_guard == 14
    assert DEFAULT_OCR_PREPROCESS_THRESHOLDS.unspaced_min_len_guard == 9
    assert DEFAULT_OCR_PREPROCESS_THRESHOLDS.fallback_min_confidence == 0.88


def _make_track(track_id: int, start_ms: int, end_ms: int, text: str, cx: float, cy: float) -> TextTrack:
    bbox = [int((cx - 0.2) * 1280), int((cy - 0.02) * 720), int((cx + 0.2) * 1280), int((cy + 0.02) * 720)]
    occ = DetectionOccurrence(
        frame_id=0,
        timestamp_ms=start_ms,
        polygon=[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]],
        bbox=bbox,
        text_raw=text,
        text_norm=text,
        confidence=0.95,
        model_id="test",
        script="latin",
    )
    return TextTrack(
        track_id=track_id,
        occurrences=[occ],
        start_ms=start_ms,
        end_ms=end_ms,
        bbox=bbox,
        features={"dominant_text": text, "mean_centroid": [cx, cy]},
    )


def test_layer_discovery_thresholds_wired_into_discover_layers():
    sample_lines = [
        "Sun rises over quiet hills",
        "Waves crash upon the shore",
        "Shadows dance beneath the trees",
        "Voices echo through the valley",
        "Stars align in silver skies",
    ]
    tracks = [
        _make_track(i + 1, i * 2000, i * 2000 + 1500, text, 0.5, 0.85)
        for i, text in enumerate(sample_lines)
    ]
    default_layers = discover_layers(tracks)
    assert len(default_layers) == 1
    assert default_layers[0].kind == "lyric"

    strict_layers = discover_layers(
        tracks,
        thresholds=LayerDiscoveryThresholds(lyric_min_tracks=10),
    )
    assert len(strict_layers) == 1
    assert strict_layers[0].kind == "sparse"


def test_classification_thresholds_wired_into_classify_track():
    track = TextTrack(
        track_id=1,
        occurrences=[],
        start_ms=10000,
        end_ms=13000,
        bbox=[300, 600, 900, 640],
        features={
            "dominant_text": "Walking down the road",
            "mean_centroid": [0.5, 0.85],
            "spatial_variance": 0.03,
            "aspect_ratio": 6.0,
            "area_ratio": 0.04,
            "is_corner": False,
            "n_text_changes": 0,
            "distinct_text_changes": 0,
            "lifetime_ratio": 0.05,
            "start_time_ratio": 0.15,
            "layer_kind": "sparse",
        },
    )
    default_res = classify_track(track, video_duration_ms=60000)
    assert default_res.label == "lyric"

    strict_res = classify_track(
        track,
        video_duration_ms=60000,
        thresholds=ClassificationThresholds(max_spatial_variance_non_lyric=0.01),
    )
    assert strict_res.label == "non_lyric"
    assert "high_spatial_variance" in strict_res.qa_flags


def test_ocr_preprocess_thresholds_wired_into_helpers_and_engine():
    # 1. to_soft_bw_stroke_crop respects soft_bw_min_height_px
    small_crop = np.full((24, 100, 3), 30, dtype=np.uint8)
    small_crop[6:18, 20:80] = 220
    out_default = to_soft_bw_stroke_crop(small_crop)
    out_custom = to_soft_bw_stroke_crop(
        small_crop,
        thresholds=OcrPreprocessThresholds(soft_bw_min_height_px=72, border_pad_px=2),
    )
    assert out_default.shape[0] == 56  # 48 + 2*4
    assert out_custom.shape[0] == 76   # 72 + 2*2

    # 2. preprocess_crop_for_recognition respects upscale_target_height_px & border_pad_px
    pre_custom = preprocess_crop_for_recognition(
        small_crop,
        thresholds=OcrPreprocessThresholds(upscale_target_height_px=70.0, border_pad_px=2),
    )
    assert pre_custom.shape[0] >= 74

    # 3. reconstruct_visual_word_spaces respects unspaced_min_len_guard
    gap_crop = np.zeros((32, 200), dtype=np.uint8)
    gap_crop[6:26, 20:60] = 255
    gap_crop[6:26, 75:95] = 255
    assert reconstruct_visual_word_spaces(gap_crop, "sonowa") == "sonowa"
    assert (
        reconstruct_visual_word_spaces(
            gap_crop,
            "sonowa",
            thresholds=OcrPreprocessThresholds(unspaced_min_len_guard=5),
        )
        == "sono wa"
    )

    # 4. OCREngine respects preprocess_thresholds.fallback_min_confidence
    class _StubRapidOCR:
        def __init__(self, params=None, **kwargs):
            self.params = params or {}

    with patch("rapidocr.RapidOCR", _StubRapidOCR):
        engine = OCREngine(
            preprocess_thresholds=OcrPreprocessThresholds(fallback_min_confidence=0.95),
        )
    img = np.full((80, 240, 3), 20, dtype=np.uint8)
    img[20:50, 20:180] = 220
    det = Detection(polygon=[[10, 10], [200, 10], [200, 60], [10, 60]], bbox=[10, 10, 190, 50], score=0.95)
    with patch.object(engine, "recognize_crops", return_value=[("千本", 0.90, "PP-OCRv6_rec_small")]), patch.object(
        engine, "recognize_crop", return_value=("千本桜", 0.96, "PP-OCRv6_rec_small")
    ) as mock_retry:
        lines = engine.process_detections(img, [det])
    assert mock_retry.call_count == 1
    assert lines[0].text_norm == "千本桜"
