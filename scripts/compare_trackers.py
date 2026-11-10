#!/usr/bin/env python3
"""Run and compare Greedy vs ByteTrack on the same 30 videos against Ground Truth."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root and scripts dir are on PYTHONPATH
_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lyra_ocr.dataset import ground_truth, registry
from lyra_ocr.eval.dataset_eval import evaluate_run
from lyra_ocr.pipeline import PipelineEngines, run_phase1_pipeline
from auto_ground_truth import get_reference_lyrics, process_video


def run_tracker_batch(
    dataset_name: str,
    tracker_name: str,
    count: int = 30,
    device: str = "auto",
    rec_batch_size: int = 8,
    dedup: bool = True,
    workspace_root: Path = Path.cwd(),
) -> tuple[str, dict, list]:
    dataset_dir = registry.dataset_dir(workspace_root, dataset_name)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    videos = manifest.get("videos", [])[:count]
    lyrics_cache_dir = dataset_dir / "lyrics_cache"

    run_info = registry.create_run(
        workspace_root,
        dataset_name,
        tracker=tracker_name,
        fps=1.0,
        options={
            "device": device,
            "rec_batch_size": rec_batch_size,
            "dedup": dedup,
        },
    )
    run_id = run_info["run_id"]
    run_dir = dataset_dir / "runs" / run_id

    print(f"\n>>> Running {tracker_name.upper()} batch on {len(videos)} videos (run_id: {run_id})...")
    engines = PipelineEngines()
    t0 = time.time()
    results = []

    for idx, v in enumerate(videos, 1):
        vid = v["video_id"]
        vtitle = v.get("title", "")[:32]
        print(f"  [{tracker_name}] ({idx:>2}/{len(videos)}) {vid} [{vtitle}]...", end="", flush=True)
        gt_file = dataset_dir / "ground_truth" / f"{vid}.json"
        res = process_video(
            v,
            dataset_dir,
            run_dir,
            lyrics_cache_dir,
            tracker=tracker_name,
            device=device,
            rec_batch_size=rec_batch_size,
            dedup=dedup,
            engines=engines,
            write_gt=not gt_file.exists(),
        )
        registry.sync_run_entry(
            workspace_root,
            dataset_name,
            run_id,
            video_id=vid,
            artifact_dir=f"videos/{vid}",
            title=v.get("title", ""),
            channel=v.get("channel", ""),
        )
        print(f" Done ({res['tracks_count']} trk, {res['elapsed_sec']}s)")
        results.append(res)

    total_time = time.time() - t0
    print(f"  {tracker_name.upper()} finished in {total_time:.1f}s (avg {total_time/len(videos):.1f}s/video)")

    # Evaluate against canonical Ground Truth
    report, paired = evaluate_run(dataset_dir, run_id, gate_min_coverage=0.0)
    return run_id, asdict(report), paired


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Greedy vs ByteTrack")
    parser.add_argument("--dataset", default="Fix_lyric_EN")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rec-batch", type=int, default=8)
    parser.add_argument("--output", default="comparison_report.md")
    args = parser.parse_args()

    ws = Path.cwd()

    print("==================================================================")
    print(f"   BENCHMARK & COMPARISON: GREEDY vs BYTETRACK ({args.count} videos)")
    print("==================================================================")

    # 1. Run Greedy
    greedy_run_id, greedy_rep, greedy_paired = run_tracker_batch(
        args.dataset, "greedy", count=args.count, device=args.device, rec_batch_size=args.rec_batch, workspace_root=ws
    )

    # 2. Run ByteTrack
    byte_run_id, byte_rep, byte_paired = run_tracker_batch(
        args.dataset, "bytetrack", count=args.count, device=args.device, rec_batch_size=args.rec_batch, workspace_root=ws
    )

    # 3. Build Markdown Comparison Table
    g_bin = greedy_rep.get("binary_lyric", {})
    b_bin = byte_rep.get("binary_lyric", {})
    g_leak = greedy_rep.get("leakage", {})
    b_leak = byte_rep.get("leakage", {})
    g_fdrop = greedy_rep.get("false_drop", {})
    b_fdrop = byte_rep.get("false_drop", {})

    g_tp = sum(1 for it in greedy_paired if it.has_label and it.label.human_label == "lyric" and it.pred.model_label == "lyric")
    g_fp = sum(1 for it in greedy_paired if it.has_label and it.label.human_label == "non_lyric" and it.pred.model_label == "lyric")
    g_fn = sum(1 for it in greedy_paired if it.has_label and it.label.human_label == "lyric" and it.pred.model_label == "non_lyric")
    g_tn = sum(1 for it in greedy_paired if it.has_label and it.label.human_label == "non_lyric" and it.pred.model_label == "non_lyric")

    b_tp = sum(1 for it in byte_paired if it.has_label and it.label.human_label == "lyric" and it.pred.model_label == "lyric")
    b_fp = sum(1 for it in byte_paired if it.has_label and it.label.human_label == "non_lyric" and it.pred.model_label == "lyric")
    b_fn = sum(1 for it in byte_paired if it.has_label and it.label.human_label == "lyric" and it.pred.model_label == "non_lyric")
    b_tn = sum(1 for it in byte_paired if it.has_label and it.label.human_label == "non_lyric" and it.pred.model_label == "non_lyric")

    g_acc = (g_tp + g_tn) / max(1, greedy_rep.get("evaluated_tracks", 1)) * 100
    b_acc = (b_tp + b_tn) / max(1, byte_rep.get("evaluated_tracks", 1)) * 100

    md = []
    md.append(f"# Báo Cáo So Sánh Thuật Toán: Greedy Tracker vs ByteTrack")
    md.append(f"\n*Ngày thực hiện: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*")
    md.append(f"*Dataset: `{args.dataset}` | Quy mô: {args.count} videos*")
    md.append(f"- Greedy Run ID: `{greedy_run_id}`")
    md.append(f"- ByteTrack Run ID: `{byte_run_id}`\n")

    md.append("## 1. Bảng So Sánh Hiệu Suất Tổng Thể")
    md.append("| Chỉ Số Đánh Giá | Greedy Tracker | ByteTrack | Chênh Lệch / Đánh Giá |")
    md.append("| :--- | :---: | :---: | :---: |")
    md.append(f"| **Tổng số Track sinh ra** | **{greedy_rep.get('total_tracks')}** | **{byte_rep.get('total_tracks')}** | {'ByteTrack sinh ít track hơn (ít phân mảnh)' if byte_rep.get('total_tracks') < greedy_rep.get('total_tracks') else 'Tương đương'} |")
    md.append(f"| **Accuracy (Độ chính xác)** | **{g_acc:.2f}%** | **{b_acc:.2f}%** | **{'+' if b_acc >= g_acc else ''}{b_acc - g_acc:.2f}%** |")
    md.append(f"| **Precision (Lyric)** | **{g_bin.get('precision', 0)*100:.2f}%** | **{b_bin.get('precision', 0)*100:.2f}%** | **{'+' if b_bin.get('precision', 0) >= g_bin.get('precision', 0) else ''}{(b_bin.get('precision', 0) - g_bin.get('precision', 0))*100:.2f}%** |")
    md.append(f"| **Recall (Bao phủ Lyric)** | **{g_bin.get('recall', 0)*100:.2f}%** | **{b_bin.get('recall', 0)*100:.2f}%** | **{'+' if b_bin.get('recall', 0) >= g_bin.get('recall', 0) else ''}{(b_bin.get('recall', 0) - g_bin.get('recall', 0))*100:.2f}%** |")
    md.append(f"| **F1-Score** | **{g_bin.get('f1', 0)*100:.2f}%** | **{b_bin.get('f1', 0)*100:.2f}%** | **{'+' if b_bin.get('f1', 0) >= g_bin.get('f1', 0) else ''}{(b_bin.get('f1', 0) - g_bin.get('f1', 0))*100:.2f}%** |")
    md.append(f"| **Leakage Rate (Non-lyric lọt vào)** | **{g_leak.get('count_rate', 0)*100:.2f}%** ({g_leak.get('leaked_tracks_count')} trk) | **{b_leak.get('count_rate', 0)*100:.2f}%** ({b_leak.get('leaked_tracks_count')} trk) | {'ByteTrack ít lọt hơn' if b_leak.get('count_rate', 0) <= g_leak.get('count_rate', 0) else 'Greedy ít lọt hơn'} |")
    md.append(f"| **False Drop Rate (Bỏ sót Lyric)** | **{g_fdrop.get('rate', 0)*100:.2f}%** ({g_fdrop.get('count')} trk) | **{b_fdrop.get('rate', 0)*100:.2f}%** ({b_fdrop.get('count')} trk) | {'ByteTrack bỏ sót ít hơn' if b_fdrop.get('rate', 0) <= g_fdrop.get('rate', 0) else 'Greedy bỏ sót ít hơn'} |")

    md.append("\n## 2. Chi Tiết Ma Trận Nhầm Lẫn (Confusion Matrix)")
    md.append("| Phân Loại | Greedy Tracker | ByteTrack |")
    md.append("| :--- | :---: | :---: |")
    md.append(f"| **True Positives (TP - Bắt đúng lyric)** | **{g_tp}** | **{b_tp}** |")
    md.append(f"| **False Positives (FP - Nhận nhầm non-lyric)** | **{g_fp}** | **{b_fp}** |")
    md.append(f"| **False Negatives (FN - Bỏ sót lyric)** | **{g_fn}** | **{b_fn}** |")
    md.append(f"| **True Negatives (TN - Lọc đúng non-lyric)** | **{g_tn}** | **{b_tn}** |")

    md.append("\n## 3. Nhận Xét & Khuyến Nghị Lựa Chọn Tracker")
    md.append("""
### Đặc tính của Greedy Tracker:
- **Nguyên lý**: Ghép nối bbox liên tiếp dựa trên threshold IoU và text similarity đơn giản giữa các frame liền kề.
- **Ưu điểm**: Thuật toán nhẹ, tốc độ xử lý nhanh, ít overhead tính toán.
- **Nhược điểm**: Dễ bị đứt track (fragmentation) khi lyric bị mờ hoặc hiệu ứng karaoke đổi màu từng chữ giữa các frame.

### Đặc tính của ByteTrack:
- **Nguyên lý**: Sử dụng thuật toán Hungarian kết hợp 2 tầng matching (high-score detections trước, low-score detections sau) để duy trì track continuity qua các frame bị blur/nhiễu mà không sinh ra track rác.
- **Ưu điểm**: Giữ track ổn định hơn khi video có hiệu ứng visualizer, motion blur hoặc chữ chuyển động. Giảm số lượng track vụn vặt.
- **Nhược điểm**: Có thêm overhead tính toán Hungarian assignment (scipy linear_sum_assignment) so với greedy.
""")

    out_p = Path(args.output).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(md), encoding="utf-8")
    print(f"\nComparison report written to: {out_p}")


if __name__ == "__main__":
    main()
