"""Unit tests for Phase 1 binary lyric classification and evaluation metrics."""

import pytest
from lyra_ocr.eval.metrics import (
    compute_binary_lyric_metrics,
    compute_false_drop_metrics,
    compute_leakage_metrics,
    evaluate_predictions,
    wilson_score_interval,
)
from lyra_ocr.eval.types import EvaluationItem, TrackLabel, TrackPrediction


def make_eval_item(
    video_id: str,
    track_id: str,
    model_label: str,
    human_label: str,
    duration_ms: int = 4000,
    tracking_error: bool = False,
) -> EvaluationItem:
    pred = TrackPrediction(
        video_id=video_id,
        track_id=track_id,
        model_label=model_label,
        model_confidence=0.95,
        model_decision="accept" if model_label == "lyric" else "reject",
        start_ms=1000,
        end_ms=1000 + duration_ms,
        duration_ms=duration_ms,
        dominant_text="sample text",
    )
    lbl = (
        TrackLabel(
            video_id=video_id,
            track_id=track_id,
            human_label=human_label,
            tracking_error=tracking_error,
        )
        if human_label
        else None
    )
    return EvaluationItem(pred=pred, label=lbl)


def test_perfect_predictions():
    items = [
        make_eval_item("v1", "1", "lyric", "lyric"),
        make_eval_item("v1", "2", "non_lyric", "non_lyric"),
        make_eval_item("v1", "3", "non_lyric", "non_lyric"),
    ]
    report = evaluate_predictions(items)

    assert report.coverage_rate == 1.0
    assert report.binary_lyric.precision == 1.0
    assert report.binary_lyric.recall == 1.0
    assert report.binary_lyric.f1 == 1.0
    assert report.leakage.count_rate == 0.0
    assert report.leakage.duration_rate == 0.0
    assert report.false_drop.rate == 0.0
    assert report.tracking_error_tracks == 0
    assert report.gate_passed is True


def test_binary_metric_counts():
    pairs = [
        ("lyric", "lyric"),
        ("lyric", "non_lyric"),
        ("non_lyric", "lyric"),
        ("non_lyric", "non_lyric"),
    ]
    metrics = compute_binary_lyric_metrics(pairs)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.support_lyric == 2
    assert metrics.support_non_lyric == 2


def test_leakage_denominator_exactness():
    """Verify formula: leakage_count_rate = #(model=lyric & human!=lyric) / #(model=lyric)."""
    items = [
        make_eval_item("v1", "1", "lyric", "lyric", duration_ms=2000),
        make_eval_item("v1", "2", "lyric", "non_lyric", duration_ms=4000),
        make_eval_item("v1", "3", "lyric", "non_lyric", duration_ms=4000),
        make_eval_item("v1", "4", "non_lyric", "non_lyric", duration_ms=10000),
    ]
    leak = compute_leakage_metrics(items)

    assert leak.leaked_tracks_count == 2
    assert leak.count_rate == pytest.approx(2.0 / 3.0, abs=1e-4)

    # Duration: total predicted lyric duration = 2000 + 4000 + 4000 = 10000ms.
    # Leaked duration = 4000 + 4000 = 8000ms. Expected = 0.8000
    assert leak.duration_rate == pytest.approx(0.8000, abs=1e-4)


def test_leakage_zero_predictions():
    items = [
        make_eval_item("v1", "1", "non_lyric", "non_lyric"),
        make_eval_item("v1", "2", "non_lyric", "non_lyric"),
    ]
    leak = compute_leakage_metrics(items)
    assert leak.count_rate == 0.0
    assert leak.duration_rate == 0.0


def test_false_drop_denominator_exactness():
    """Verify formula: false_drop_rate = #(human=lyric & model!=lyric) / #(human=lyric)."""
    items = [
        make_eval_item("v1", "1", "lyric", "lyric"),
        make_eval_item("v1", "2", "lyric", "lyric"),
        make_eval_item("v1", "3", "lyric", "lyric"),
        make_eval_item("v1", "4", "non_lyric", "lyric"),  # False drop
        make_eval_item("v1", "5", "non_lyric", "non_lyric"),
    ]
    fdrop = compute_false_drop_metrics(items)
    assert fdrop.total_golden_lyrics == 4
    assert fdrop.count == 1
    assert fdrop.rate == pytest.approx(0.25, abs=1e-4)


def test_tracking_error_excluded_from_classification():
    """Verify that tracks with tracking_error=True are excluded from classification metrics."""
    items = [
        make_eval_item("v1", "1", "lyric", "lyric"),
        make_eval_item("v1", "2", "non_lyric", "non_lyric"),
        # Item 3 is a tracking failure (e.g. merge of logo and lyric), so human flagged tracking_error
        make_eval_item("v1", "3", "lyric", "non_lyric", tracking_error=True),
    ]
    report = evaluate_predictions(items)

    assert report.total_tracks == 3
    assert report.evaluated_tracks == 2
    assert report.tracking_error_tracks == 1
    # Because item 3 was excluded, leakage should be 0.0!
    assert report.leakage.count_rate == 0.0
    assert report.binary_lyric.precision == 1.0


def test_unlabeled_tracks_counted_separately():
    items = [
        make_eval_item("v1", "1", "lyric", "lyric"),
        make_eval_item("v1", "2", "non_lyric", ""),  # Unlabeled
    ]
    report = evaluate_predictions(items)

    assert report.total_tracks == 2
    assert report.evaluated_tracks == 1
    assert report.unlabeled_tracks == 1
    assert report.coverage_rate == 0.5


def test_missing_golden_track_counts_as_false_drop_and_lowers_recall():
    missing_pred = TrackPrediction(
        video_id="v1",
        track_id="m1",
        model_label="missing",
        model_confidence=0.0,
        model_decision="missing",
        start_ms=0,
        end_ms=4000,
        duration_ms=4000,
        dominant_text="lost lyric line",
    )
    missing_label = TrackLabel(video_id="v1", track_id="m1", human_label="lyric")
    items = [
        make_eval_item("v1", "1", "lyric", "lyric"),
        make_eval_item("v1", "2", "non_lyric", "non_lyric"),
        EvaluationItem(pred=missing_pred, label=missing_label, is_missing=True),
    ]

    report = evaluate_predictions(items)

    assert report.evaluated_tracks == 2  # missing tracks are not classified
    assert report.tracking_miss_tracks == 1
    assert report.tracking_miss_rate == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert report.false_drop.count == 1
    assert report.false_drop.rate == pytest.approx(0.5, abs=1e-4)
    assert report.binary_lyric.recall == pytest.approx(0.5, abs=1e-4)


def test_report_stores_gate_thresholds():
    report = evaluate_predictions([], gate_min_coverage=0.5, gate_max_false_drop=0.2)
    assert report.gate_thresholds["min_coverage"] == 0.5
    assert report.gate_thresholds["max_false_drop"] == 0.2
    assert report.binary_lyric_ci["precision"] == [0.0, 0.0]


def test_wilson_score_interval():
    low, high = wilson_score_interval(k=10, n=10)
    assert 0.70 < low < 1.0
    assert high == 1.0

    low_zero, high_zero = wilson_score_interval(k=0, n=0)
    assert (low_zero, high_zero) == (0.0, 0.0)
