"""Publication-quality evaluation charts generator using optional matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lyra_ocr.eval.types import EvaluationReport


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

    # 1. Leakage vs False Drop vs Lyric Metrics
    plt.figure(figsize=(7, 4.5))
    metric_names = ["Lyric Prec", "Lyric Recall", "Lyric F1", "Leakage Rate", "False Drop", "Tracking Miss"]
    values = [
        report.binary_lyric.precision,
        report.binary_lyric.recall,
        report.binary_lyric.f1,
        report.leakage.count_rate,
        report.false_drop.rate,
        report.tracking_miss_rate,
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

    # 2. Optional Tracker Comparison Chart
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
