"""Unit tests for evaluation reporting, error categorization, and chart generation."""

import json
from pathlib import Path

import pytest
from lyra_ocr.eval.metrics import evaluate_predictions
from lyra_ocr.eval.plots import render_evaluation_charts
from lyra_ocr.eval.report import (
    generate_errors_csv,
    generate_markdown_report,
    generate_predictions_csv,
    save_evaluation_report,
)
from lyra_ocr.eval.types import EvaluationItem, TrackLabel, TrackPrediction


def make_item(
    vid: str,
    tid: str,
    pred_label: str,
    gold_label: str | None,
    tracking_err: bool = False,
) -> EvaluationItem:
    p = TrackPrediction(
        video_id=vid,
        track_id=tid,
        model_label=pred_label,
        model_confidence=0.92,
        model_decision="accept" if pred_label == "lyric" else "reject",
        start_ms=1000,
        end_ms=5000,
        duration_ms=4000,
        dominant_text="sample text line",
    )
    lbl = (
        TrackLabel(
            video_id=vid,
            track_id=tid,
            human_label=gold_label,
            binary="lyric" if gold_label == "lyric" else "non_lyric",
            tracking_error=tracking_err,
        )
        if gold_label
        else None
    )
    return EvaluationItem(pred=p, label=lbl)


def test_generate_predictions_and_errors_csv(tmp_path: Path):
    """Test classification of error_type and filtering into errors.csv."""
    items = [
        make_item("v1", "t1", "lyric", "lyric"),  # correct
        make_item("v1", "t2", "lyric", "channel"),  # leakage
        make_item("v1", "t3", "channel", "lyric"),  # false_drop
        make_item("v1", "t4", "song_title", "channel"),  # misclassification
        make_item("v1", "t5", "channel", "channel", tracking_err=True),  # tracking_error
        make_item("v1", "t6", "lyric", None),  # unlabeled
    ]

    preds_csv = tmp_path / "predictions.csv"
    pred_rows = generate_predictions_csv(items, preds_csv)

    assert preds_csv.exists()
    assert len(pred_rows) == 6

    error_types = {r["track_id"]: r["error_type"] for r in pred_rows}
    assert error_types["t1"] == "correct"
    assert error_types["t2"] == "leakage"
    assert error_types["t3"] == "false_drop"
    assert error_types["t4"] == "misclassification"
    assert error_types["t5"] == "tracking_error"
    assert error_types["t6"] == "unlabeled"

    errs_csv = tmp_path / "errors.csv"
    num_errs = generate_errors_csv(pred_rows, errs_csv)

    assert errs_csv.exists()
    # leakage, false_drop, and misclassification -> 3 errors
    assert num_errs == 3


def test_save_evaluation_report_full(tmp_path: Path):
    """Test saving complete bundle: metrics.json, report.md, predictions.csv, errors.csv."""
    items = [
        make_item("song_a", "1", "lyric", "lyric"),
        make_item("song_a", "2", "channel", "channel"),
        make_item("song_b", "1", "song_title", "song_title"),
    ]

    report = evaluate_predictions(items)
    out_dir = tmp_path / "eval_output"
    saved = save_evaluation_report(report, items, out_dir)

    assert saved["metrics_json"].exists()
    assert saved["report_md"].exists()
    assert saved["predictions_csv"].exists()
    assert saved["errors_csv"].exists()

    with open(saved["metrics_json"], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_tracks"] == 3
    assert data["evaluated_tracks"] == 3
    assert data["gate_passed"] is True
    assert "binary_lyric" in data

    md_content = saved["report_md"].read_text(encoding="utf-8")
    assert "# Phase 1 Evaluation & Quality Gate Report" in md_content
    assert "PHASE 1 QUALITY GATE: PASSED" in md_content
    assert "Per-Class Taxonomy Breakdown" in md_content
    assert "Confusion Matrix" in md_content
    assert "Per-Video Performance" in md_content


def test_custom_gate_threshold_is_reflected_in_markdown(tmp_path: Path):
    items = [
        make_item("song_a", "1", "lyric", "lyric"),
        make_item("song_a", "2", "channel", "channel"),
    ]
    report = evaluate_predictions(items, gate_min_precision=1.01)
    assert report.gate_passed is False

    md_path = tmp_path / "report.md"
    content = generate_markdown_report(report, md_path)

    assert "QUALITY GATE: FAILED" in content
    assert ">= 1.0100" in content
    assert "| **Lyric Precision** | `1.0000` | `>= 1.0100` | **FAIL** |" in content


def test_missing_track_in_predictions_csv(tmp_path: Path):
    from lyra_ocr.eval.types import EvaluationItem, TrackLabel, TrackPrediction

    pred = TrackPrediction(
        video_id="v1",
        track_id="m1",
        model_label="missing",
        model_confidence=0.0,
        model_decision="missing",
        start_ms=0,
        end_ms=1000,
        duration_ms=1000,
        dominant_text="missing lyric",
    )
    item = EvaluationItem(
        pred=pred,
        label=TrackLabel(video_id="v1", track_id="m1", human_label="lyric"),
        is_missing=True,
    )
    rows = generate_predictions_csv([item], tmp_path / "preds.csv")
    assert rows[0]["error_type"] == "tracking_miss"
    assert rows[0]["is_missing"] == "true"


def test_render_evaluation_charts(tmp_path: Path):
    """Test rendering of all 4 charts via matplotlib."""
    items = [
        make_item("song_a", "1", "lyric", "lyric"),
        make_item("song_a", "2", "channel", "channel"),
        make_item("song_a", "3", "song_title", "song_title"),
    ]
    report = evaluate_predictions(items)

    tracker_results = [
        {"tracker": "greedy", "binary_lyric_f1": 0.88, "latency_ms": 12.0},
        {"tracker": "bytetrack", "binary_lyric_f1": 0.98, "latency_ms": 8.0},
    ]

    chart_dir = tmp_path / "charts"
    rendered = render_evaluation_charts(report, chart_dir, tracker_results=tracker_results)

    assert rendered is True
    assert (chart_dir / "confusion_matrix.png").exists()
    assert (chart_dir / "per_class_prf.png").exists()
    assert (chart_dir / "leakage_false_drop.png").exists()
    assert (chart_dir / "tracker_comparison.png").exists()
