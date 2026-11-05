"""Pure metric calculation functions for Video Text Tracking and Lyric Classification."""

from __future__ import annotations

import math
from typing import Any

from lyra_ocr.eval.types import (
    BinaryLyricMetrics,
    EvaluationItem,
    EvaluationReport,
    FalseDropMetrics,
    LeakageMetrics,
)


def wilson_score_interval(
    k: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    # Standard normal quantile z (1.96 for 95%)
    z = 1.96 if abs(confidence - 0.95) < 0.01 else 1.645
    p = float(k) / float(n)
    denom = 1.0 + (z**2) / float(n)
    center = (p + (z**2) / (2.0 * float(n))) / denom
    spread = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * float(n))) / float(n))) / denom
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return round(lower, 4), round(upper, 4)


def compute_binary_lyric_metrics(pairs: list[tuple[str, str]]) -> BinaryLyricMetrics:
    """Compute Binary Classification metrics: Lyric vs Non-Lyric."""
    tp = 0  # True: lyric, Pred: lyric
    fp = 0  # True: non_lyric, Pred: lyric
    fn = 0  # True: lyric, Pred: non_lyric
    tn = 0  # True: non_lyric, Pred: non_lyric

    for true_lbl, pred_lbl in pairs:
        is_true_lyric = (true_lbl == "lyric")
        is_pred_lyric = (pred_lbl == "lyric")

        if is_true_lyric and is_pred_lyric:
            tp += 1
        elif not is_true_lyric and is_pred_lyric:
            fp += 1
        elif is_true_lyric and not is_pred_lyric:
            fn += 1
        else:
            tn += 1

    prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / float(prec + rec) if (prec + rec) > 0.0 else 0.0

    return BinaryLyricMetrics(
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        support_lyric=tp + fn,
        support_non_lyric=fp + tn,
    )


def compute_leakage_metrics(items: list[EvaluationItem]) -> LeakageMetrics:
    """Compute distractor leakage metrics (non-lyric tracks accepted as lyric)."""
    # Considered only among tracks that have human ground truth
    pred_lyric_items = [it for it in items if it.has_label and it.pred.model_label == "lyric"]
    leaked_items = [
        it for it in pred_lyric_items if it.label and it.label.human_label != "lyric"
    ]

    total_pred_lyric_count = len(pred_lyric_items)
    leaked_count = len(leaked_items)
    count_rate = float(leaked_count) / float(total_pred_lyric_count) if total_pred_lyric_count > 0 else 0.0

    total_pred_duration = sum(it.pred.duration_ms for it in pred_lyric_items)
    leaked_duration = sum(it.pred.duration_ms for it in leaked_items)
    dur_rate = float(leaked_duration) / float(total_pred_duration) if total_pred_duration > 0 else 0.0

    return LeakageMetrics(
        count_rate=round(count_rate, 4),
        duration_rate=round(dur_rate, 4),
        leaked_tracks_count=leaked_count,
    )


def compute_false_drop_metrics(items: list[EvaluationItem]) -> FalseDropMetrics:
    """Compute false drop metrics (true lyrics incorrectly rejected or assigned non-lyric)."""
    golden_lyrics = [it for it in items if it.has_label and it.label and it.label.human_label == "lyric"]
    dropped_lyrics = [it for it in golden_lyrics if it.pred.model_label != "lyric"]

    total_gold = len(golden_lyrics)
    drop_count = len(dropped_lyrics)
    drop_rate = float(drop_count) / float(total_gold) if total_gold > 0 else 0.0

    return FalseDropMetrics(
        rate=round(drop_rate, 4),
        count=drop_count,
        total_golden_lyrics=total_gold,
    )


def _binary_counts(pairs: list[tuple[str, str]]) -> tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    for true_lbl, pred_lbl in pairs:
        is_true_lyric = true_lbl == "lyric"
        is_pred_lyric = pred_lbl == "lyric"
        if is_true_lyric and is_pred_lyric:
            tp += 1
        elif not is_true_lyric and is_pred_lyric:
            fp += 1
        elif is_true_lyric and not is_pred_lyric:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def evaluate_predictions(
    items: list[EvaluationItem],
    gate_min_precision: float = 0.90,
    gate_min_recall: float = 0.85,
    gate_max_leakage: float = 0.00,
    gate_max_false_drop: float = 0.15,
    gate_min_coverage: float = 1.00,
    gate_max_tracking_miss: float = 0.15,
) -> EvaluationReport:
    """Run comprehensive binary lyric evaluation on predicted and labeled items.

    ``items`` may contain ``is_missing=True`` entries produced by track matching:
    these represent golden tracks that no prediction covered. They count as false
    drops and toward the tracking-miss metric.
    """
    source_items = [it for it in items if not it.is_missing]
    missing_items = [it for it in items if it.is_missing]

    total_tracks = len(source_items)
    unlabeled_items = [it for it in source_items if not it.has_label]
    tracking_error_items = [it for it in source_items if it.is_tracking_error]

    # Classification is evaluated on labeled, non-tracking-error predictions.
    eval_items = [it for it in source_items if it.has_label and not it.is_tracking_error]
    missing_eval = [it for it in missing_items if it.has_label and not it.is_tracking_error]
    evaluated_tracks = len(eval_items)

    labeled_source_count = len([it for it in source_items if it.has_label])
    coverage_rate = float(labeled_source_count) / float(max(1, total_tracks))

    binary_pairs = [
        (it.label.human_label, it.pred.model_label) for it in eval_items if it.label
    ] + [
        (it.label.human_label, it.pred.model_label) for it in missing_eval if it.label
    ]

    binary_lyric = compute_binary_lyric_metrics(binary_pairs)
    leakage = compute_leakage_metrics(eval_items)
    false_drop = compute_false_drop_metrics(eval_items + missing_eval)

    tp, fp, fn, _tn = _binary_counts(binary_pairs)
    binary_lyric_ci = {
        "precision": list(wilson_score_interval(tp, tp + fp)),
        "recall": list(wilson_score_interval(tp, tp + fn)),
    }

    tracking_miss_count = len(missing_eval)
    tracking_miss_rate = float(tracking_miss_count) / float(
        max(1, len(eval_items) + len(missing_eval))
    )

    # Per-video breakdowns (missing tracks included so fragmentation is visible).
    videos = sorted({it.pred.video_id for it in items})
    per_video: dict[str, Any] = {}
    for vid in videos:
        v_source = [it for it in source_items if it.pred.video_id == vid]
        v_eval = [it for it in eval_items if it.pred.video_id == vid]
        v_missing = [it for it in missing_eval if it.pred.video_id == vid]
        v_binary_pairs = [
            (it.label.human_label, it.pred.model_label) for it in v_eval if it.label
        ] + [
            (it.label.human_label, it.pred.model_label) for it in v_missing if it.label
        ]
        v_bin = compute_binary_lyric_metrics(v_binary_pairs)
        v_leak = compute_leakage_metrics(v_eval)
        v_fdrop = compute_false_drop_metrics(v_eval + v_missing)
        per_video[vid] = {
            "total_tracks": len(v_source),
            "evaluated_tracks": len(v_eval),
            "tracking_miss_tracks": len(v_missing),
            "binary_lyric_f1": v_bin.f1,
            "binary_lyric_precision": v_bin.precision,
            "binary_lyric_recall": v_bin.recall,
            "leakage_count_rate": v_leak.count_rate,
            "false_drop_rate": v_fdrop.rate,
        }

    gate_thresholds = {
        "min_coverage": gate_min_coverage,
        "min_lyric_precision": gate_min_precision,
        "min_lyric_recall": gate_min_recall,
        "max_leakage": gate_max_leakage,
        "max_false_drop": gate_max_false_drop,
        "max_tracking_miss": gate_max_tracking_miss,
    }

    # Evaluate Phase 1 Quality Gate
    gate_reasons: list[str] = []
    if coverage_rate < gate_min_coverage:
        gate_reasons.append(
            f"Coverage ({coverage_rate*100:.1f}%) < {gate_min_coverage*100:.1f}% required"
        )
    if binary_lyric.precision < gate_min_precision:
        gate_reasons.append(
            f"Lyric Precision ({binary_lyric.precision:.3f}) < {gate_min_precision:.3f}"
        )
    if binary_lyric.recall < gate_min_recall:
        gate_reasons.append(
            f"Lyric Recall ({binary_lyric.recall:.3f}) < {gate_min_recall:.3f}"
        )
    if leakage.count_rate > gate_max_leakage:
        gate_reasons.append(
            f"Distractor Leakage ({leakage.count_rate*100:.1f}%) > {gate_max_leakage*100:.1f}% (Zero tolerance)"
        )
    if false_drop.rate > gate_max_false_drop:
        gate_reasons.append(
            f"False Drop Rate ({false_drop.rate*100:.1f}%) > {gate_max_false_drop*100:.1f}%"
        )
    if tracking_miss_rate > gate_max_tracking_miss:
        gate_reasons.append(
            f"Tracking Miss Rate ({tracking_miss_rate*100:.1f}%) > {gate_max_tracking_miss*100:.1f}%"
        )

    gate_passed = len(gate_reasons) == 0

    return EvaluationReport(
        total_tracks=total_tracks,
        evaluated_tracks=evaluated_tracks,
        unlabeled_tracks=len(unlabeled_items),
        tracking_error_tracks=len(tracking_error_items),
        coverage_rate=round(coverage_rate, 4),
        binary_lyric=binary_lyric,
        leakage=leakage,
        false_drop=false_drop,
        per_video=per_video,
        gate_passed=gate_passed,
        gate_reasons=gate_reasons,
        gate_thresholds=gate_thresholds,
        binary_lyric_ci=binary_lyric_ci,
        tracking_miss_tracks=tracking_miss_count,
        tracking_miss_rate=round(tracking_miss_rate, 4),
    )
