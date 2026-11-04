"""Publication-quality evaluation charts generator using optional matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lyra_ocr.eval.types import EvaluationReport
from lyra_ocr.labels import VALID_LABELS


def render_evaluation_charts(
    report: EvaluationReport,
    output_dir: str | Path,
    tracker_results: list[dict[str, Any]] | None = None,
) -> bool:
    """Render PNG evaluation charts. Returns True if rendered, False if matplotlib missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    matrix_arr = np.zeros((len(VALID_LABELS), len(VALID_LABELS)), dtype=int)
    for i, t_lbl in enumerate(VALID_LABELS):
        for j, p_lbl in enumerate(VALID_LABELS):
            matrix_arr[i, j] = report.confusion_matrix.get(t_lbl, {}).get(p_lbl, 0)

    im = plt.imshow(matrix_arr, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Confusion Matrix (Ground Truth vs Predicted)", fontsize=13, pad=12)
    plt.xticks(range(len(VALID_LABELS)), VALID_LABELS, rotation=35, ha="right")
    plt.yticks(range(len(VALID_LABELS)), VALID_LABELS)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Ground Truth (Human)", fontsize=11)

    max_val = matrix_arr.max()
    for i in range(len(VALID_LABELS)):
        for j in range(len(VALID_LABELS)):
            val = matrix_arr[i, j]
            color = "white" if max_val > 0 and val > (max_val / 2.0) else "black"
            plt.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # 2. Per-Class Precision, Recall, F1 Bar Chart
    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(VALID_LABELS))
    width = 0.25

    precs = [report.per_class.get(l, None).precision if report.per_class.get(l) else 0.0 for l in VALID_LABELS]
    recs = [report.per_class.get(l, None).recall if report.per_class.get(l) else 0.0 for l in VALID_LABELS]
    f1s = [report.per_class.get(l, None).f1 if report.per_class.get(l) else 0.0 for l in VALID_LABELS]

    plt.bar(x - width, precs, width, label="Precision", color="#3b82f6")
    plt.bar(x, recs, width, label="Recall", color="#10b981")
    plt.bar(x + width, f1s, width, label="F1-Score", color="#8b5cf6")

    plt.title("Per-Class Classification Performance", fontsize=13, pad=12)
    plt.xticks(x, VALID_LABELS, rotation=25, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score [0.0 - 1.0]", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_prf.png", dpi=200)
    plt.close()

    # 3. Leakage vs False Drop vs Lyric Metrics
    plt.figure(figsize=(7, 4.5))
    metric_names = ["Lyric Prec", "Lyric Recall", "Lyric F1", "Leakage Rate", "False Drop", "Unknown Rate"]
    values = [
        report.binary_lyric.precision,
        report.binary_lyric.recall,
        report.binary_lyric.f1,
        report.leakage.count_rate,
        report.false_drop.rate,
        report.unknown_rate,
    ]
    colors = ["#10b981", "#10b981", "#3b82f6", "#ef4444", "#f59e0b", "#6b7280"]

    bars = plt.bar(metric_names, values, color=colors, width=0.55)
    plt.title("Phase 1 Operational Gate Metrics", fontsize=13, pad=12)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Rate / Score", fontsize=11)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.2f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "leakage_false_drop.png", dpi=200)
    plt.close()

    # 4. Optional Tracker Comparison Chart
    if tracker_results:
        plt.figure(figsize=(8, 4.5))
        t_names = [tr["tracker"] for tr in tracker_results]
        x_tr = np.arange(len(t_names))
        f1_vals = [
            tr.get("binary_lyric_f1", tr.get("lyric_f1", 0.0))
            for tr in tracker_results
        ]
        cov_vals = [
            tr.get("coverage_rate", tr.get("coverage", 0.0)) for tr in tracker_results
        ]
        lat_vals = [tr.get("latency_ms", 0.0) for tr in tracker_results]

        ax1 = plt.gca()
        ax1.bar(x_tr - 0.2, f1_vals, width=0.4, color="#10b981", label="Lyric F1")
        ax1.bar(x_tr + 0.2, cov_vals, width=0.4, color="#3b82f6", label="Gold Coverage")

        for i, (f1, cov, lat) in enumerate(zip(f1_vals, cov_vals, lat_vals)):
            ax1.text(i - 0.2, f1 + 0.02, f"{f1:.2f}", ha="center", fontsize=9)
            ax1.text(i + 0.2, cov + 0.02, f"{cov*100:.0f}%", ha="center", fontsize=9)
            ax1.text(i, 1.05, f"{lat:.0f}ms", ha="center", fontsize=8, color="#6b7280")

        ax1.set_ylabel("Score / Coverage", fontsize=11)
        ax1.set_ylim(0.0, 1.15)
        ax1.set_xticks(x_tr)
        ax1.set_xticklabels(t_names)
        ax1.legend(loc="lower right", fontsize=9)
        ax1.set_title(
            "Tracking Strategy Comparison on Golden Dataset\n(grey labels: CPU latency)",
            fontsize=12,
            pad=12,
        )
        plt.tight_layout()
        plt.savefig(out_dir / "tracker_comparison.png", dpi=200)
        plt.close()

    return True
