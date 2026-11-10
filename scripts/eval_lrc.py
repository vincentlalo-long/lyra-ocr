"""LRC Evaluation CLI & Batch Evaluator."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

from lyra_ocr.eval.audio_timing import get_audio_timing_report
from lyra_ocr.eval.lrc_eval import evaluate_synced_reference, evaluate_text_selection
from lyra_ocr.eval.reference import load_plain_lyrics, load_synced_lrc
from lyra_ocr.eval.visual_timing import measure_visual_timing
from lyra_ocr.lyric.grouping import dedupe_and_group
from lyra_ocr.lyric.select import load_artifacts, select_cues

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate_run(
    dataset: str,
    run_id: str,
    out_report: str | Path,
    *,
    base_dir: str | Path = "datasets",
    limit: int | None = None,
    include_review: bool = False,
    correct_method: str = "dictionary",
) -> dict[str, Any]:
    ds_path = Path(base_dir) / dataset
    run_dir = ds_path / "runs" / run_id
    run_json = run_dir / "run.json"

    if not run_json.is_file():
        raise FileNotFoundError(f"Run file not found: {run_json}")

    run_meta = json.loads(run_json.read_text(encoding="utf-8"))
    entries = run_meta.get("videos", [])
    if limit:
        entries = entries[:limit]

    results: list[dict[str, Any]] = []

    for entry in entries:
        vid = entry.get("video_id")
        art_rel = entry.get("artifact_dir")
        if not vid or not art_rel:
            continue

        art_dir = run_dir / art_rel
        if not art_dir.is_dir():
            alt_dir = ds_path / art_rel
            if alt_dir.is_dir():
                art_dir = alt_dir
            else:
                continue

        manifest, tracks, human_labels = load_artifacts(art_dir)
        cues, skipped = select_cues(
            tracks,
            human_labels,
            include_review=include_review,
            video_id=vid,
            correct_method=correct_method,
        )
        cues = dedupe_and_group(cues)

        cand_vpath = manifest.get("video_path")
        if cand_vpath and Path(cand_vpath).is_file():
            video_path = cand_vpath
        else:
            video_path = str(ds_path / "videos" / f"{vid}.mp4")

        # Visual timing
        timing_vis = measure_visual_timing(video_path, cues, str(art_dir))

        # Plain lyrics & text selection
        ref_lyrics = load_plain_lyrics(ds_path, vid)
        text_sel = evaluate_text_selection(cues, ref_lyrics)

        # Synced LRC reference
        ref_lrc_file = ds_path / "lrc" / f"{vid}.lrc"
        ref_lrc = load_synced_lrc(ref_lrc_file) if ref_lrc_file.is_file() else []
        timing_ref = evaluate_synced_reference(cues, ref_lrc)

        # Audio timing
        timing_audio = get_audio_timing_report()

        video_res = {
            "video_id": vid,
            "metric_kind": "automatic_proxy",
            "scorable": text_sel.get("scorable", False),
            "timing_visual": timing_vis,
            "timing_reference": timing_ref,
            "timing_audio": timing_audio,
            "text_selection": text_sel,
            "warnings": [],
        }
        results.append(video_res)

    scorable_results = [r for r in results if r["scorable"]]
    timing_ok = [
        r["timing_visual"]
        for r in scorable_results
        if r["timing_visual"].get("status") == "ok"
        and r["timing_visual"].get("median_abs_error_ms") is not None
    ]

    summary = {
        "dataset": dataset,
        "run_id": run_id,
        "total_evaluated": len(results),
        "n_scorable": len(scorable_results),
        "summary": {
            "mean_line_recall": round(
                float(np.mean([r["text_selection"]["line_recall"] for r in scorable_results])), 4
            ) if scorable_results else 0.0,
            "mean_line_precision": round(
                float(np.mean([r["text_selection"]["line_precision"] for r in scorable_results])), 4
            ) if scorable_results else 0.0,
            "mean_line_f1": round(
                float(np.mean([r["text_selection"]["line_f1"] for r in scorable_results])), 4
            ) if scorable_results else 0.0,
            "mean_cer": round(
                float(np.mean([r["text_selection"]["cer"] for r in scorable_results])), 4
            ) if scorable_results else 0.0,
            "mean_wer": round(
                float(np.mean([r["text_selection"]["wer"] for r in scorable_results])), 4
            ) if scorable_results else 0.0,
            "mean_extra_line_rate": round(
                float(np.mean([r["text_selection"]["extra_line_rate"] for r in scorable_results])), 4
            ) if scorable_results else 0.0,
            "mean_visual_median_err_ms": round(
                float(np.mean([t["median_abs_error_ms"] for t in timing_ok])), 1
            ) if timing_ok else None,
            "mean_visual_within_300ms": round(
                float(np.mean([t["percent_within_300ms"] for t in timing_ok])), 3
            ) if timing_ok else None,
        },
        "videos": results,
    }

    out_p = Path(out_report)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate LRC export metrics against references")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--run-id", required=True, help="Run ID")
    parser.add_argument("--out", "-o", default="./reports/lrc_eval.json", help="Output report JSON")
    parser.add_argument("--base-dir", default="./datasets", help="Datasets base directory")
    parser.add_argument("--limit", type=int, default=None, help="Max videos to evaluate")
    parser.add_argument("--include-review", action="store_true", help="Include review tracks")
    parser.add_argument("--correct", choices=["none", "consensus", "dictionary"], default="dictionary")

    args = parser.parse_args(argv)
    res = evaluate_run(
        dataset=args.dataset,
        run_id=args.run_id,
        out_report=args.out,
        base_dir=args.base_dir,
        limit=args.limit,
        include_review=args.include_review,
        correct_method=args.correct,
    )

    print("\n=== LRC Evaluation Report ===")
    print(f"Dataset          : {res['dataset']}")
    print(f"Run ID           : {res['run_id']}")
    print(f"Scorable Videos  : {res['n_scorable']}/{res['total_evaluated']}")
    s = res["summary"]
    print(f"Mean Line Recall : {s['mean_line_recall'] * 100:.2f}%")
    print(f"Mean Line Prec   : {s['mean_line_precision'] * 100:.2f}%")
    print(f"Mean Line F1     : {s['mean_line_f1'] * 100:.2f}%")
    print(f"Mean CER         : {s['mean_cer'] * 100:.2f}%")
    print(f"Mean WER         : {s['mean_wer'] * 100:.2f}%")
    if s.get("mean_visual_median_err_ms") is not None:
        print(f"Visual Med Error : {s['mean_visual_median_err_ms']:.1f}ms")
    if s.get("mean_visual_within_300ms") is not None:
        print(f"Visual <=300ms   : {s['mean_visual_within_300ms'] * 100:.1f}%")
    print(f"Report saved to  : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
