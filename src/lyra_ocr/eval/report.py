"""Generators for JSON, Markdown, and tabular error inspection reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from lyra_ocr.eval.types import EvaluationItem, EvaluationReport


def build_prediction_rows(items: list[EvaluationItem]) -> list[dict[str, Any]]:
    """Build detailed prediction vs ground truth audit rows (in-memory)."""
    rows: list[dict[str, Any]] = []

    for it in items:
        p = it.pred
        hl = it.label.human_label if it.label else ""
        is_err = it.is_tracking_error
        is_missing = it.is_missing

        is_correct = (p.model_label == hl) if hl else False
        error_type = ""
        if not hl:
            error_type = "unlabeled"
        elif is_err:
            error_type = "tracking_error"
        elif is_missing:
            error_type = "tracking_miss" if hl == "lyric" else "missing_track"
        elif p.model_label == hl:
            error_type = "correct"
        elif p.model_label == "lyric" and hl != "lyric":
            error_type = "leakage"
        elif hl == "lyric" and p.model_label != "lyric":
            error_type = "false_drop"
        else:
            error_type = "misclassification"

        rows.append(
            {
                "video_id": p.video_id,
                "track_id": p.track_id,
                "start_ms": p.start_ms,
                "end_ms": p.end_ms,
                "duration_ms": p.duration_ms,
                "text": p.dominant_text,
                "model_label": p.model_label,
                "model_confidence": f"{p.model_confidence:.2f}",
                "model_decision": p.model_decision,
                "human_label": hl,
                "tracking_error": "true" if is_err else "false",
                "is_missing": "true" if is_missing else "false",
                "is_correct": "true" if is_correct else "false",
                "error_type": error_type,
            }
        )

    return rows


PREDICTION_CSV_FIELDS = [
    "video_id",
    "track_id",
    "start_ms",
    "end_ms",
    "duration_ms",
    "text",
    "model_label",
    "model_confidence",
    "model_decision",
    "human_label",
    "tracking_error",
    "is_missing",
    "is_correct",
    "error_type",
]


def generate_predictions_csv(
    items: list[EvaluationItem], out_csv: str | Path
) -> list[dict[str, Any]]:
    """Export detailed prediction vs ground truth audit rows to predictions.csv."""
    rows = build_prediction_rows(items)

    out_p = Path(out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    return rows


def generate_errors_csv(
    prediction_rows: list[dict[str, Any]], out_csv: str | Path
) -> int:
    """Filter misclassified tracks into errors.csv for rapid debugging and inspectability."""
    error_rows = [
        r
        for r in prediction_rows
        if r["error_type"]
        in ("leakage", "false_drop", "misclassification", "tracking_miss", "missing_track")
    ]
    fieldnames = [
        "video_id",
        "track_id",
        "start_ms",
        "end_ms",
        "text",
        "model_label",
        "human_label",
        "error_type",
        "model_confidence",
    ]

    out_p = Path(out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(error_rows)

    return len(error_rows)


def generate_markdown_report(report: EvaluationReport, out_md: str | Path) -> str:
    """Format evaluation metrics into publication-ready GitHub-flavored Markdown."""
    lines: list[str] = []
    lines.append("# Phase 1 Evaluation & Quality Gate Report\n")

    # Gate status alert
    if report.gate_passed:
        lines.append("> [!TIP]\n> **PHASE 1 QUALITY GATE: PASSED**\n> All operational requirements satisfied.\n")
    else:
        reasons_str = "<br>".join(f"- {r}" for r in report.gate_reasons)
        lines.append(f"> [!WARNING]\n> **PHASE 1 QUALITY GATE: FAILED**\n> {reasons_str}\n")

    # High-level overview. Thresholds come from the report so custom gates
    # cannot disagree with the PASS/FAIL banner.
    t = report.gate_thresholds or {}
    min_cov = t.get("min_coverage", 1.0)
    min_prec = t.get("min_lyric_precision", 0.90)
    min_rec = t.get("min_lyric_recall", 0.85)
    max_leak = t.get("max_leakage", 0.00)
    max_drop = t.get("max_false_drop", 0.15)
    max_miss = t.get("max_tracking_miss", 0.15)

    lines.append("## 1. Summary Metrics\n")
    lines.append("| Metric | Value | Threshold Target | Status |")
    lines.append("|---|---|---|---|")
    cov_status = "PASS" if report.coverage_rate >= min_cov else "FAIL"
    lines.append(f"| **Label Coverage** | `{report.coverage_rate*100:.1f}%` ({report.evaluated_tracks}/{report.total_tracks}) | `>= {min_cov*100:.1f}%` | **{cov_status}** |")

    prec_status = "PASS" if report.binary_lyric.precision >= min_prec else "FAIL"
    lines.append(f"| **Lyric Precision** | `{report.binary_lyric.precision:.4f}` | `>= {min_prec:.4f}` | **{prec_status}** |")

    rec_status = "PASS" if report.binary_lyric.recall >= min_rec else "FAIL"
    lines.append(f"| **Lyric Recall** | `{report.binary_lyric.recall:.4f}` | `>= {min_rec:.4f}` | **{rec_status}** |")

    f1_val = report.binary_lyric.f1
    lines.append(f"| **Lyric F1** | `{f1_val:.4f}` | - | INFO |")

    leak_status = "PASS" if report.leakage.count_rate <= max_leak else "FAIL"
    lines.append(f"| **Distractor Leakage Rate** | `{report.leakage.count_rate*100:.1f}%` ({report.leakage.leaked_tracks_count} tracks) | `<= {max_leak*100:.1f}%` | **{leak_status}** |")

    drop_status = "PASS" if report.false_drop.rate <= max_drop else "FAIL"
    lines.append(f"| **False Drop Rate** | `{report.false_drop.rate*100:.1f}%` ({report.false_drop.count} drops) | `<= {max_drop*100:.1f}%` | **{drop_status}** |")

    miss_status = "PASS" if report.tracking_miss_rate <= max_miss else "FAIL"
    lines.append(f"| **Tracking Miss Rate** | `{report.tracking_miss_rate*100:.1f}%` ({report.tracking_miss_tracks} golden tracks) | `<= {max_miss*100:.1f}%` | **{miss_status}** |")
    lines.append(f"| **Tracking Errors (Excluded)** | `{report.tracking_error_tracks}` | - | INFO |")
    lines.append("\n")

    if report.binary_lyric_ci:
        prec_ci = report.binary_lyric_ci.get("precision", [0.0, 0.0])
        rec_ci = report.binary_lyric_ci.get("recall", [0.0, 0.0])
        lines.append(
            f"_95% Wilson CI — lyric precision [{prec_ci[0]:.3f}, {prec_ci[1]:.3f}], "
            f"recall [{rec_ci[0]:.3f}, {rec_ci[1]:.3f}]_\n"
        )

    # Per-video breakdown
    if report.per_video:
        lines.append("## 2. Per-Video Performance\n")
        lines.append("| Video ID | Tracks | Evaluated | Missing | Lyric F1 | Lyric Precision | Lyric Recall | Leakage Rate | False Drop Rate |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for vid, vm in report.per_video.items():
            lines.append(
                f"| `{vid}` | {vm['total_tracks']} | {vm['evaluated_tracks']} | "
                f"{vm.get('tracking_miss_tracks', 0)} | "
                f"{vm['binary_lyric_f1']:.4f} | {vm['binary_lyric_precision']:.4f} | "
                f"{vm['binary_lyric_recall']:.4f} | {vm['leakage_count_rate']*100:.1f}% | "
                f"{vm['false_drop_rate']*100:.1f}% |"
            )
        lines.append("\n")

    content = "\n".join(lines)
    out_p = Path(out_md)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(content)

    return content


def save_evaluation_report(
    report: EvaluationReport,
    items: list[EvaluationItem],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save all evaluation artifacts (metrics.json, report.md, predictions.csv, errors.csv)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    md_path = out / "report.md"
    generate_markdown_report(report, md_path)

    preds_path = out / "predictions.csv"
    pred_rows = generate_predictions_csv(items, preds_path)

    errs_path = out / "errors.csv"
    generate_errors_csv(pred_rows, errs_path)

    return {
        "metrics_json": json_path,
        "report_md": md_path,
        "predictions_csv": preds_path,
        "errors_csv": errs_path,
    }
