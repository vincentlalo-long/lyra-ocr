"""Evaluation framework for Video Text Tracking and Lyric Classification."""

from __future__ import annotations

from lyra_ocr.eval.loader import (
    compute_file_hash,
    load_artifacts_evaluation_items,
    load_golden_dataset,
)
from lyra_ocr.eval.matching import match_predictions_to_golden
from lyra_ocr.eval.metrics import (
    compute_binary_lyric_metrics,
    compute_confusion_matrix,
    compute_false_drop_metrics,
    compute_leakage_metrics,
    compute_macro_micro_f1,
    compute_per_class_metrics,
    evaluate_predictions,
    wilson_score_interval,
)
from lyra_ocr.eval.plots import render_evaluation_charts
from lyra_ocr.eval.report import (
    generate_errors_csv,
    generate_markdown_report,
    generate_predictions_csv,
    save_evaluation_report,
)
from lyra_ocr.eval.types import (
    BinaryLyricMetrics,
    EvaluationItem,
    EvaluationReport,
    FalseDropMetrics,
    LeakageMetrics,
    PerClassMetrics,
    TrackLabel,
    TrackPrediction,
)

__all__ = [
    "EvaluationReport",
    "EvaluationItem",
    "TrackPrediction",
    "TrackLabel",
    "PerClassMetrics",
    "BinaryLyricMetrics",
    "LeakageMetrics",
    "FalseDropMetrics",
    "compute_confusion_matrix",
    "compute_per_class_metrics",
    "compute_macro_micro_f1",
    "compute_binary_lyric_metrics",
    "compute_leakage_metrics",
    "compute_false_drop_metrics",
    "evaluate_predictions",
    "wilson_score_interval",
    "load_artifacts_evaluation_items",
    "load_golden_dataset",
    "match_predictions_to_golden",
    "save_evaluation_report",
    "render_evaluation_charts",
    "generate_markdown_report",
    "generate_predictions_csv",
    "generate_errors_csv",
    "compute_file_hash",
]
