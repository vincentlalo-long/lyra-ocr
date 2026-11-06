"""CLI dispatcher for lyra-ocr."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .ocr import TextLine
from .pipeline import run_phase1_pipeline
from .track import merge_fragmented_tracks, track_ocr_results
from .classify import classify_all_tracks
from .provenance import get_tool_versions


def _find_unicode_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Find a system font capable of rendering Latin/Vietnamese/CJK text."""
    candidate_fonts = [
        "/usr/share/fonts/google-noto-sans-mono-cjk-vf-fonts/NotoSansMonoCJK-VF.ttc",
        "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    ]
    for font_path in candidate_fonts:
        if Path(font_path).is_file():
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_roi_preview(
    image_path: str | Path,
    boxes: list[TextLine],
    output_path: str | Path,
) -> None:
    """Render bounding boxes and text labels onto an image frame for visual audit."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _find_unicode_font(size=max(16, int(img.height * 0.025)))

    for item in boxes:
        poly = [(p[0], p[1]) for p in item.polygon]
        draw.polygon(poly, outline="#22c55e", width=2)

        xmin, ymin, xmax, ymax = item.bbox
        display_text = item.text_norm or item.text_raw
        label = f"{display_text} ({item.confidence:.2f})"

        text_bbox = draw.textbbox((xmin, ymin), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        lbl_y0 = max(0, ymin - th - 4)
        lbl_y1 = lbl_y0 + th + 4
        lbl_x1 = min(img.width, xmin + tw + 6)

        draw.rectangle([xmin, lbl_y0, lbl_x1, lbl_y1], fill="#111827")
        draw.text((xmin + 3, lbl_y0 + 2), label, fill="#f9fafb", font=font)

    img.save(output_path, "JPEG", quality=90)


def run_smoke(args: argparse.Namespace) -> int:
    """Execute smoke test / Phase-1 identification on a local video."""
    video_path = Path(args.video).resolve()
    out_dir = Path(args.out).resolve()

    print("=== [Lyra OCR Smoke Test — Phase 0/1] ===")
    print(f"Video : {video_path}")
    print(f"Output: {out_dir}")

    tracker_name = getattr(args, "tracker", "greedy") or "greedy"

    def on_frame(fid: int, t_ms: int, text_lines: list, elapsed_ms: float) -> None:
        print(f"   - Frame {fid} (PTS={t_ms}ms): {len(text_lines)} boxes in {elapsed_ms:.1f}ms")
        for line in text_lines:
            print(f"       [{line.script}] \"{line.text_norm}\" (conf: {line.confidence:.2f}, model: {line.model_id})")

    def on_progress(stage: str, percent: float, message: str) -> None:
        print(f"   [{stage}] {message}")

    result = run_phase1_pipeline(
        video_path,
        out_dir,
        title=getattr(args, "title", "") or "",
        channel=getattr(args, "channel", "") or "",
        fps=args.fps,
        start_sec=args.start_sec,
        max_frames=args.sample_frames,
        tracker=tracker_name,
        progress=on_progress,
        on_frame=on_frame,
        force=True,
        device=getattr(args, "device", "auto") or "auto",
        rec_batch_size=getattr(args, "rec_batch_size", 8) or 8,
        dedup=bool(getattr(args, "dedup", False)),
        frame_source=getattr(args, "frame_source", "pipe") or "pipe",
    )

    tracks = result["tracks"]
    print(f"\nClassified {len(tracks)} track(s):")
    for tr in tracks:
        lbl = tr.classification.label if tr.classification else "non_lyric"
        dec = tr.classification.decision if tr.classification else "review"
        rsn = tr.classification.reason if tr.classification else ""
        txt = tr.features.get("dominant_text", "")
        print(f"   - Track #{tr.track_id} [{lbl.upper()}] ({dec}): \"{txt}\" — {rsn}")

    print(f"\nArtifacts saved to: {out_dir}")
    print("\n=== Lyra OCR Smoke & Identification Completed Successfully! ===")
    return 0


def run_label(args: argparse.Namespace) -> int:
    """Handle human labeling workflows (export, ingest, status)."""
    from lyra_ocr.labels import (
        export_label_batch_csv,
        get_label_status,
        ingest_label_batch_csv,
    )

    action = getattr(args, "label_action", None)
    if action == "export":
        out_csv = args.out or (Path(args.artifacts) / "label_batch.csv")
        blind = not getattr(args, "no_blind", False)
        count = export_label_batch_csv(
            artifacts_dir=args.artifacts,
            out_csv=out_csv,
            blind=blind,
        )
        print(f"Exported {count} tracks to label sheet: {out_csv} (blind={blind})")
        return 0

    elif action == "ingest":
        stats = ingest_label_batch_csv(
            artifacts_dir=args.artifacts,
            csv_path=args.csv,
            annotator=args.annotator,
        )
        print(
            f"Ingested labels from {args.csv}:\n"
            f"  - Appended/Updated : {stats['ingested']}\n"
            f"  - Skipped (empty)  : {stats['skipped_empty']}\n"
            f"  - Errors/Rejected  : {stats['errors']}"
        )
        return 0

    elif action == "status":
        status = get_label_status(args.artifacts)
        print(f"\n=== Label Status for Video [{status['video_id']}] ===")
        print(f"Total Tracks     : {status['total_tracks']}")
        print(f"Labeled Tracks   : {status['labeled_tracks']} ({status['coverage_percent']}%)")
        print(f"Unlabeled Tracks : {status['unlabeled_tracks']}")
        print(f"Tracking Errors  : {status['tracking_error_count']}")
        print("Label Distribution:")
        for k, v in status["distribution"].items():
            print(f"  - {k:<15}: {v}")
        return 0

    else:
        print("Please specify a label action: export, ingest, or status (use -h for help).")
        return 1


def run_eval(args: argparse.Namespace) -> int:
    """Handle evaluation against ground truth labels and Quality Gate verification."""
    from lyra_ocr.eval import (
        evaluate_predictions,
        load_artifacts_evaluation_items,
        load_golden_dataset,
        match_predictions_to_golden,
        render_evaluation_charts,
        save_evaluation_report,
        TrackPrediction,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.golden:
        print(f"\n1. Loading golden dataset from: {args.golden}...")
        golden_items = load_golden_dataset(args.golden)
        eval_items = golden_items
    elif args.artifacts:
        print(f"\n1. Loading evaluation items from artifact: {args.artifacts}...")
        eval_items = load_artifacts_evaluation_items(args.artifacts)
        golden_items = eval_items
    else:
        print("Error: Either --golden or --artifacts must be specified.")
        return 1

    print(f"   Loaded {len(eval_items)} candidate track items.")

    # Multi-tracker comparison on golden dataset
    tracker_results: list[dict[str, Any]] = []
    if getattr(args, "compare_trackers", False) and args.golden:
        print("\n2a. Running multi-tracker comparison on golden OCR raw data...")
        from lyra_ocr.track import track_ocr_results
        from lyra_ocr.classify import classify_all_tracks

        g_manifest_path = Path(args.golden) / "manifest.json"
        with open(g_manifest_path, "r", encoding="utf-8") as f:
            g_manifest = json.load(f)
        video_entries = g_manifest.get("videos", [])

        for tr_name in ["greedy", "bytetrack"]:
            all_tr_pairs = []
            t0 = time.perf_counter()

            for ventry in video_entries:
                vid = ventry.get("video_id")
                art_p = Path(ventry.get("artifact_dir"))
                if not art_p.is_absolute():
                    if not art_p.exists():
                        art_p = Path(args.golden) / ventry.get("artifact_dir")

                ocr_file = art_p / "ocr_raw.json"
                if not ocr_file.exists():
                    continue

                with open(ocr_file, "r", encoding="utf-8") as f:
                    ocr_data = json.load(f)

                # Use the same duration/frame geometry metadata as the original
                # artifact run so all trackers are compared under identical rules.
                meta_file = art_p / "manifest.json"
                art_manifest: dict[str, Any] = {}
                if meta_file.exists():
                    with open(meta_file, "r", encoding="utf-8") as f:
                        art_manifest = json.load(f)
                probe = art_manifest.get("probe", {})
                meta_dict = art_manifest.get("metadata", {})
                v_dur_ms = int(float(probe.get("duration", 0)) * 1000)
                frame_w = int(probe.get("width", 1280))
                frame_h = int(probe.get("height", 720))

                # Re-run tracker on existing OCR frames (no re-OCR!)
                re_tracks = track_ocr_results(
                    frames_ocr_data=ocr_data.get("frames", []),
                    video_duration_ms=v_dur_ms,
                    frame_width=frame_w,
                    frame_height=frame_h,
                    tracker_name=tr_name,
                )
                re_tracks = merge_fragmented_tracks(
                    re_tracks,
                    video_duration_ms=v_dur_ms,
                    frame_width=frame_w,
                    frame_height=frame_h,
                )

                # Re-run rule-based classifier with full geometry metadata
                classify_all_tracks(
                    re_tracks,
                    metadata=meta_dict,
                    video_duration_ms=v_dur_ms,
                    frame_width=frame_w,
                    frame_height=frame_h,
                )

                # Build TrackPredictions
                preds = [
                    TrackPrediction(
                        video_id=vid,
                        track_id=str(t.track_id),
                        model_label=t.classification.label if t.classification else "non_lyric",
                        model_confidence=t.classification.confidence if t.classification else 0.5,
                        model_decision=t.classification.decision if t.classification else "review",
                        start_ms=t.start_ms,
                        end_ms=t.end_ms,
                        duration_ms=max(0, t.end_ms - t.start_ms),
                        dominant_text=t.features.get("dominant_text", ""),
                        bbox=t.bbox,
                    )
                    for t in re_tracks
                ]

                # Match with golden reference tracks
                paired, _, _ = match_predictions_to_golden(preds, golden_items)
                all_tr_pairs.extend(paired)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # Unmatched fragment predictions legitimately dilute label
            # coverage here; the coverage gate only applies to the primary
            # report so comparison gates stay meaningful.
            tr_rep = evaluate_predictions(all_tr_pairs, gate_min_coverage=0.0)
            tracker_results.append({
                "tracker": tr_name,
                "n_tracks": tr_rep.total_tracks,
                "binary_lyric_f1": tr_rep.binary_lyric.f1,
                "binary_lyric_precision": tr_rep.binary_lyric.precision,
                "binary_lyric_recall": tr_rep.binary_lyric.recall,
                "leakage_count_rate": tr_rep.leakage.count_rate,
                "false_drop_rate": tr_rep.false_drop.rate,
                "coverage_rate": tr_rep.coverage_rate,
                "tracking_miss_rate": tr_rep.tracking_miss_rate,
                "gate_passed": tr_rep.gate_passed,
                "latency_ms": round(elapsed_ms, 2),
            })

            print(
                f"   [{tr_name.upper():>9}] Tracks: {tr_rep.total_tracks:2d} | "
                f"Lyric F1: {tr_rep.binary_lyric.f1:.4f} | "
                f"Coverage: {tr_rep.coverage_rate*100:5.1f}% | "
                f"Leakage: {tr_rep.leakage.count_rate*100:.1f}% | "
                f"Gate: {'PASS' if tr_rep.gate_passed else 'FAIL'} | "
                f"Latency: {elapsed_ms:.1f}ms"
            )

    print("\n2. Computing primary evaluation metrics & Quality Gate...")
    report = evaluate_predictions(eval_items)

    print("\n3. Saving evaluation reports...")
    out_files = save_evaluation_report(report, eval_items, out_dir)
    print(f"   - Metrics JSON : {out_files['metrics_json']}")
    print(f"   - Markdown     : {out_files['report_md']}")
    print(f"   - Predictions  : {out_files['predictions_csv']}")
    print(f"   - Errors       : {out_files['errors_csv']}")

    if getattr(args, "charts", False):
        print("\n4. Generating publication-quality charts...")
        charts_dir = out_dir / "charts"
        rendered = render_evaluation_charts(report, charts_dir, tracker_results=tracker_results)
        if rendered:
            print(f"   Saved charts to {charts_dir}")
        else:
            print("   Skipped charts (matplotlib not installed).")

    print(f"\n=== Evaluation Finished: {'GATE PASSED' if report.gate_passed else 'GATE FAILED'} ===")
    print(
        f"Coverage: {report.coverage_rate*100:.1f}% | "
        f"Lyric Precision: {report.binary_lyric.precision:.4f} | "
        f"Lyric Recall: {report.binary_lyric.recall:.4f} | "
        f"Distractor Leakage: {report.leakage.count_rate*100:.1f}%"
    )
    return 0 if report.gate_passed else 2


def run_review_ui(args: argparse.Namespace) -> int:
    """Start local web review UI."""
    from lyra_ocr.web.server import run_server

    run_server(
        host=args.host,
        port=args.port,
        open_browser=args.open,
        workspace_root=args.workspace_root,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lyra-ocr", description="Lyra OCR CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    for cmd_name, cmd_help in [
        ("smoke", "Run OCR smoke test on a local video"),
        ("identify", "Run Phase 1 lyric identification and tracking on a video"),
    ]:
        p = subparsers.add_parser(cmd_name, help=cmd_help)
        p.add_argument("video", help="Path to input video file")
        p.add_argument(
            "--out",
            "-o",
            default="./smoke_output",
            help="Output directory (default: ./smoke_output)",
        )
        p.add_argument(
            "--sample-frames",
            "-n",
            type=int,
            default=5,
            help="Number of frames to sample (default: 5)",
        )
        p.add_argument(
            "--start-sec",
            "-s",
            type=float,
            default=0.0,
            help="Seconds offset to start sampling frames (default: 0.0)",
        )
        p.add_argument(
            "--fps",
            type=float,
            default=2.0,
            help="Sample FPS (default: 2.0)",
        )
        p.add_argument(
            "--title",
            default="",
            help="Song title metadata (used for intro title card discrimination)",
        )
        p.add_argument(
            "--channel",
            default="",
            help="Channel/Artist metadata (used for persistent watermark discrimination)",
        )
        p.add_argument(
            "--tracker",
            choices=["greedy", "bytetrack"],
            default="greedy",
            help="Text tracking algorithm strategy (default: greedy)",
        )
        p.add_argument(
            "--device",
            choices=["auto", "cpu", "cuda"],
            default="auto",
            help="OCR device; auto uses CUDA when available (default: auto)",
        )
        p.add_argument(
            "--rec-batch-size",
            type=int,
            default=8,
            help="Recognition crops per ONNX call (default: 8)",
        )
        p.add_argument(
            "--dedup",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Reuse OCR output for near-identical frames (default: off)",
        )
        p.add_argument(
            "--frame-source",
            choices=["pipe", "disk"],
            default="pipe",
            help="Stream frames from ffmpeg or write/read JPEGs on disk (default: pipe)",
        )

    # Subcommand: label
    p_label = subparsers.add_parser("label", help="Manage ground truth labels")
    label_sub = p_label.add_subparsers(dest="label_action", help="Label action")

    # label export
    p_lexport = label_sub.add_parser("export", help="Export label batch CSV")
    p_lexport.add_argument("--artifacts", required=True, help="Directory containing tracks.json and manifest.json")
    p_lexport.add_argument("--out", "-o", default=None, help="Output CSV path (default: {artifacts}/label_batch.csv)")
    p_lexport.add_argument("--blind", action="store_true", default=True, help="Omit machine predictions (default: True)")
    p_lexport.add_argument("--no-blind", dest="no_blind", action="store_true", help="Include machine predictions in export")
    p_lexport.add_argument("--per-track", action="store_true", default=True, help="One row per track (default: True)")

    # label ingest
    p_lingest = label_sub.add_parser("ingest", help="Ingest reviewed CSV into labels.jsonl")
    p_lingest.add_argument("--artifacts", required=True, help="Directory containing tracks.json")
    p_lingest.add_argument("--csv", required=True, help="Path to reviewed CSV file")
    p_lingest.add_argument("--annotator", default="human", help="Annotator name/handle (default: human)")

    # label status
    p_lstatus = label_sub.add_parser("status", help="Show labeling coverage and status")
    p_lstatus.add_argument("--artifacts", required=True, help="Directory containing tracks.json and labels.jsonl")

    # Subcommand: eval
    p_eval = subparsers.add_parser("eval", help="Evaluate tracking & classification against ground truth")
    p_eval.add_argument("--golden", default=None, help="Path to labeled dataset directory")
    p_eval.add_argument("--artifacts", default=None, help="Path to single artifacts directory")
    p_eval.add_argument(
        "--out",
        "-o",
        default="./reports/phase1_eval",
        help="Output directory for reports (default: ./reports/phase1_eval)",
    )
    p_eval.add_argument("--charts", action="store_true", help="Generate publication-grade PNG charts")
    p_eval.add_argument("--compare-trackers", action="store_true", help="Run multi-tracker comparison on the labeled dataset")

    # Subcommand: review-ui
    p_ui = subparsers.add_parser("review-ui", help="Start interactive web review and labeling UI")
    p_ui.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    p_ui.add_argument("--port", "-p", type=int, default=8000, help="Port (default: 8000)")
    p_ui.add_argument("--open", action="store_true", help="Automatically open browser")
    p_ui.add_argument(
        "--workspace-root",
        default=None,
        help="Workspace root for artifacts/datasets (default: current directory)",
    )

    args = parser.parse_args(argv)
    if args.command in ("smoke", "identify"):
        return run_smoke(args)
    elif args.command == "label":
        return run_label(args)
    elif args.command == "eval":
        return run_eval(args)
    elif args.command == "review-ui":
        return run_review_ui(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
