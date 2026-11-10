"""Batch export synchronized LRC files from Phase 1 run artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

from lyra_ocr.lyric.build import build_lrc_from_artifacts

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def batch_export_lrc(
    dataset: str,
    run_id: str,
    out_dir: str | Path,
    *,
    base_dir: str | Path = "datasets",
    include_review: bool = False,
    correct_method: str = "dictionary",
    refine: bool = False,
    offset_ms: int = 0,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    dataset_path = Path(base_dir) / dataset
    run_path = dataset_path / "runs" / run_id
    run_json_path = run_path / "run.json"

    if not run_json_path.exists():
        raise FileNotFoundError(f"Run file not found: {run_json_path}")

    run_meta = json.loads(run_json_path.read_text(encoding="utf-8"))
    video_entries = run_meta.get("videos", [])

    if limit is not None and limit > 0:
        video_entries = video_entries[:limit]

    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "dataset": dataset,
        "run_id": run_id,
        "total_videos": len(video_entries),
        "processed": 0,
        "total_cues": 0,
        "errors": 0,
        "video_results": [],
    }

    for entry in video_entries:
        vid = entry.get("video_id")
        art_rel = entry.get("artifact_dir")
        if not vid or not art_rel:
            continue

        art_dir = run_path / art_rel
        if not art_dir.exists():
            alt_dir = dataset_path / art_rel
            if alt_dir.exists():
                art_dir = alt_dir
            else:
                logger.warning("Artifact dir not found for video %s: %s", vid, art_dir)
                summary["errors"] += 1
                continue

        out_lrc = out_base / f"{vid}.lrc"
        try:
            res = build_lrc_from_artifacts(
                artifacts_dir=art_dir,
                out_lrc=out_lrc,
                include_review=include_review,
                correct_method=correct_method,
                refine=refine,
                offset_ms=offset_ms,
                force=force,
            )
            summary["processed"] += 1
            summary["total_cues"] += res["n_cues"]
            summary["video_results"].append({
                "video_id": vid,
                "n_cues": res["n_cues"],
                "n_skipped": res["n_skipped"],
                "review_required": res["review_required"],
            })
            logger.info("Exported %s: %d cues -> %s", vid, res["n_cues"], out_lrc)
        except Exception as e:
            logger.error("Failed to export %s: %s", vid, e)
            summary["errors"] += 1

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch export LRC files from dataset runs")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. Fix_lyric_EN)")
    parser.add_argument("--run-id", required=True, help="Run ID (e.g. bytetrack-1794243340-de75c2)")
    parser.add_argument("--out-dir", default="./lrc_out", help="Output directory for .lrc files")
    parser.add_argument("--base-dir", default="./datasets", help="Datasets base directory")
    parser.add_argument("--include-review", action="store_true", help="Include review tracks")
    parser.add_argument(
        "--correct",
        choices=["none", "consensus", "dictionary"],
        default="dictionary",
        help="Correction method",
    )
    parser.add_argument("--refine", action="store_true", help="Dense scan refine")
    parser.add_argument("--offset-ms", type=int, default=0, help="Offset ms")
    parser.add_argument("--limit", type=int, default=None, help="Max videos to process")
    parser.add_argument("--force", action="store_true", help="Force export")

    args = parser.parse_args(argv)
    res = batch_export_lrc(
        dataset=args.dataset,
        run_id=args.run_id,
        out_dir=args.out_dir,
        base_dir=args.base_dir,
        include_review=args.include_review,
        correct_method=args.correct,
        refine=args.refine,
        offset_ms=args.offset_ms,
        limit=args.limit,
        force=args.force,
    )
    print("\n--- Batch Export Summary ---")
    print(f"Processed: {res['processed']}/{res['total_videos']} videos")
    print(f"Total Cues: {res['total_cues']}")
    print(f"Errors: {res['errors']}")
    return 0 if res["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
