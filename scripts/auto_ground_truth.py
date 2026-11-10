#!/usr/bin/env python3
"""Automated Silver Ground Truth Generation & Evaluation Pipeline.

Extracts official song lyrics from YouTube description / LRCLIB, generates
canonical ground truth segments via fuzzy string & temporal alignment,
runs the Lyra OCR pipeline, and computes evaluation metrics (Precision, Recall,
F1, Discrepancy analysis).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from lyra_ocr.dataset import ground_truth, registry
from lyra_ocr.dataset.ground_truth import GroundTruthSegment
from lyra_ocr.eval.dataset_eval import evaluate_run
from lyra_ocr.pipeline import PipelineEngines, run_phase1_pipeline


def clean_text(text: str) -> str:
    """Normalize text for phonetic/token comparison: lowercase, alphanumeric and spaces only."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", (text or "").lower()).strip()


def parse_lyrics_from_desc(desc: str) -> list[str]:
    """Extract lyrics text block from YouTube video description."""
    lines = [l.strip() for l in desc.splitlines()]
    lyrics: list[str] = []
    in_lyrics = False

    header_patterns = [
        r"^\(?lyrics?\)?[:\s\-]*$",
        r"^\(?lyrics?\)?$",
        r"\[(verse|chorus|intro|hook|bridge|outro)",
        r"^lyrics\s*:",
    ]
    trailer_patterns = [
        r"^---",
        r"^stream\s*/\s*download",
        r"^follow\s+",
        r"^connect\s+with",
        r"^listen\s+to",
        r"^subscribe",
        r"^tags?:",
        r"^all\s+rights\s+belong",
        r"^copyright",
        r"^http",
    ]

    for line in lines:
        if not line:
            continue
        lower = line.lower()
        if not in_lyrics:
            if any(re.search(pat, lower) for pat in header_patterns):
                in_lyrics = True
                if not lower.startswith("["):
                    continue
        if in_lyrics:
            if any(re.search(pat, lower) for pat in trailer_patterns):
                if len(lyrics) >= 8:
                    break
            if lower.startswith("[") and lower.endswith("]"):
                continue
            lyrics.append(line)

    # Fallback if no explicit header: extract longest contiguous block of lyric-like lines
    if len(lyrics) < 8:
        blocks: list[list[str]] = []
        cur_block: list[str] = []
        for line in lines:
            if not line or any(re.search(pat, line.lower()) for pat in trailer_patterns):
                if cur_block:
                    blocks.append(cur_block)
                    cur_block = []
            elif len(line) < 120 and not line.lower().startswith("http"):
                cur_block.append(line)
            else:
                if cur_block:
                    blocks.append(cur_block)
                    cur_block = []
        if cur_block:
            blocks.append(cur_block)
        if blocks:
            longest = max(blocks, key=len)
            if len(longest) >= 8:
                lyrics = longest

    return lyrics


def fetch_lrclib_lyrics(title: str) -> list[str]:
    """Fallback: Query LRCLIB public API for plain lyrics."""
    cleaned = re.sub(r"[\(\[\{].*?[\)\]\}]", "", title)
    cleaned = re.sub(r"\bft\.?.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bfeat\.?.*$", "", cleaned, flags=re.IGNORECASE)
    query = cleaned.strip()
    if " - " in query:
        artist, track = query.split(" - ", 1)
        query = f"{track} {artist}"
    url = f"https://lrclib.net/api/search?q={urllib.parse.quote(query)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LyraOCR/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())
            if data and data[0].get("plainLyrics"):
                return [l.strip() for l in data[0]["plainLyrics"].splitlines() if l.strip()]
    except Exception:
        pass
    return []


def get_reference_lyrics(
    video_id: str,
    title: str,
    cache_dir: Path,
) -> list[str]:
    """Fetch and cache reference lyrics for a given video."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{video_id}.json"
    if cache_file.is_file():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 1. Try YouTube Description
    lyrics: list[str] = []
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        cmd = ["yt-dlp", "--skip-download", "--print", "%(description)s", url]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=15)
        lyrics = parse_lyrics_from_desc(out)
    except Exception:
        pass

    # 2. Try LRCLIB fallback
    if len(lyrics) < 8:
        lrclib_lines = fetch_lrclib_lyrics(title)
        if len(lrclib_lines) >= 8:
            lyrics = lrclib_lines

    if lyrics:
        cache_file.write_text(json.dumps(lyrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return lyrics


def align_tracks_to_lyrics(
    tracks: list[dict[str, Any]],
    lyrics_lines: list[str],
    video_title: str = "",
    video_duration_ms: int = 0,
) -> list[dict[str, Any]]:
    """Determine silver ground truth labels for tracks based on lyrics alignment & card priors."""
    clean_lyrics = [clean_text(l) for l in lyrics_lines if l.strip()]
    all_words = set(" ".join(clean_lyrics).split())
    all_text = " ".join(clean_lyrics)

    # Keywords typical for non-lyric title/credit cards
    card_kws = [
        "official lyric video",
        "official video",
        "lyrics",
        "lyric video",
        "subscribe",
        "vibemusic",
        "7clouds",
        "remix",
        "slowed",
        "audio",
        "visualizer",
        "prod by",
        "directed by",
    ]
    title_words = set(clean_text(video_title).split())

    results: list[dict[str, Any]] = []

    for track in tracks:
        tid = str(track.get("track_id"))
        text = str(track.get("features", {}).get("dominant_text", "")).strip()
        c_text = clean_text(text)
        words = c_text.split()
        start_ms = int(track.get("start_ms", 0))
        end_ms = int(track.get("end_ms", start_ms))
        dur_ms = max(0, end_ms - start_ms)
        bbox = list(track.get("bbox") or [0, 0, 0, 0])

        # 1. Check if track is a stationary title card / channel watermark
        is_card = False
        reason = ""
        card_kw_found = next((kw for kw in card_kws if kw in c_text), None)

        if start_ms <= 12000 and (card_kw_found or (len(words) <= 7 and len(set(words) & title_words) >= 2)):
            is_card = True
            reason = f"intro_title_card({card_kw_found or 'title_match'})"
        elif video_duration_ms > 0 and start_ms >= (video_duration_ms - 10000) and card_kw_found:
            is_card = True
            reason = f"outro_credit_card({card_kw_found})"
        elif dur_ms > 45000 and len(words) <= 3:
            # Watermark lasting > 45s with few words
            is_card = True
            reason = "persistent_watermark"

        if is_card:
            results.append({
                "track_id": tid,
                "label": "non_lyric",
                "decision": "reject",
                "confidence": 0.98,
                "reason": reason,
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "bbox": bbox,
            })
            continue

        # 2. Token overlap and SequenceMatcher ratio with reference lyrics
        token_overlap = (
            len(set(words) & all_words) / max(1, len(set(words))) if words else 0.0
        )
        best_ratio = (
            max([SequenceMatcher(None, c_text, l).ratio() for l in clean_lyrics])
            if clean_lyrics
            else 0.0
        )
        is_sub = bool(c_text and len(c_text) >= 4 and c_text in all_text)

        if token_overlap >= 0.50 or best_ratio >= 0.50 or is_sub:
            lbl = "lyric"
            dec = "accept"
            conf = max(token_overlap, best_ratio, 0.90)
            reason = f"lyrics_matched(overlap={token_overlap:.2f}, ratio={best_ratio:.2f})"
        elif token_overlap <= 0.20 and best_ratio <= 0.25 and not is_sub:
            lbl = "non_lyric"
            dec = "reject"
            conf = 0.85
            reason = f"no_lyrics_match(overlap={token_overlap:.2f}, ratio={best_ratio:.2f})"
        else:
            # Marginal / short phrase fallback: if within reading zone, bias towards lyric
            lbl = "lyric"
            dec = "accept"
            conf = 0.70
            reason = f"marginal_reading_zone(overlap={token_overlap:.2f})"

        results.append({
            "track_id": tid,
            "label": lbl,
            "decision": dec,
            "confidence": conf,
            "reason": reason,
            "text": text,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "bbox": bbox,
        })

    return results


def process_video(
    video_entry: dict[str, Any],
    dataset_dir: Path,
    run_dir: Path,
    lyrics_cache_dir: Path,
    tracker: str = "greedy",
    device: str = "auto",
    rec_batch_size: int = 8,
    dedup: bool = True,
    engines: Optional[PipelineEngines] = None,
    write_gt: bool = True,
    force_gt: bool = False,
) -> dict[str, Any]:
    """Run pipeline and generate ground truth for one video."""
    video_id = video_entry["video_id"]
    title = video_entry.get("title", "")
    video_path = dataset_dir / video_entry.get("video_path", "")
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing video file for {video_id}: {video_path}")

    # 1. Fetch reference lyrics
    lyrics = get_reference_lyrics(video_id, title, lyrics_cache_dir)

    # 2. Run OCR Pipeline
    out_dir = run_dir / "videos" / video_id
    t0 = time.time()
    pipe_res = run_phase1_pipeline(
        video_path,
        out_dir,
        fps=1.0,
        tracker=tracker,
        video_id=video_id,
        title=title,
        device=device,
        rec_batch_size=rec_batch_size,
        dedup=dedup,
        engines=engines,
        force=False,
    )
    elapsed = time.time() - t0
    tracks_json_path = out_dir / "tracks.json"
    if tracks_json_path.is_file():
        tracks = json.loads(tracks_json_path.read_text(encoding="utf-8")).get("tracks", [])
    else:
        tracks = [t.to_dict() if hasattr(t, "to_dict") else t for t in pipe_res.get("tracks", [])]

    # 3. Align tracks with reference lyrics to produce Silver Ground Truth
    aligned = align_tracks_to_lyrics(tracks, lyrics, video_title=title)

    # 4. Write canonical Ground Truth segments
    segments: list[GroundTruthSegment] = []
    now = datetime.now(timezone.utc).isoformat()
    for item in aligned:
        segments.append(
            GroundTruthSegment(
                video_id=video_id,
                segment_id=f"gt_{item['track_id']}",
                start_ms=item["start_ms"],
                end_ms=item["end_ms"],
                human_label=item["label"],
                text=item["text"],
                bbox=item["bbox"],
                human_decision=item["decision"],
                notes=item["reason"],
                annotator="silver_lyrics_alignment",
                reviewed_at=now,
                source_run_id=run_dir.name,
                source_track_id=item["track_id"],
            )
        )
    if write_gt:
        gt_file = dataset_dir / "ground_truth" / f"{video_id}.json"
        if force_gt or not gt_file.exists():
            ground_truth.write_segments(dataset_dir, video_id, segments)

    # 5. Write labels.jsonl in run artifact dir
    labels_file = out_dir / "labels.jsonl"
    with open(labels_file, "w", encoding="utf-8") as f:
        for item in aligned:
            row = {
                "video_id": video_id,
                "track_id": item["track_id"],
                "human_label": item["label"],
                "human_decision": item["decision"],
                "notes": item["reason"],
                "annotator": "silver_lyrics_alignment",
                "reviewed_at": now,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "video_id": video_id,
        "title": title,
        "tracks_count": len(tracks),
        "lyrics_count": len(lyrics),
        "elapsed_sec": round(elapsed, 2),
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Silver Ground Truth & Eval Pipeline")
    parser.add_argument("--dataset", default="Fix_lyric_EN", help="Dataset name")
    parser.add_argument("--count", type=int, default=25, help="Number of videos to process")
    parser.add_argument("--tracker", default="greedy", help="Tracker (greedy or bytetrack)")
    parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--rec-batch", type=int, default=8, help="Recognition batch size")
    parser.add_argument("--no-dedup", action="store_true", help="Disable frame dedup")
    args = parser.parse_args()

    workspace_root = Path.cwd()
    dataset_dir = registry.dataset_dir(workspace_root, args.dataset)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    videos = manifest.get("videos", [])[: args.count]
    if not videos:
        print(f"No videos found in dataset '{args.dataset}'")
        return

    print(f"=== Running Silver Ground Truth Pipeline on {len(videos)} videos ===")
    print(f"Dataset: {args.dataset} | Tracker: {args.tracker} | Device: {args.device} | Batch: {args.rec_batch}")

    lyrics_cache_dir = dataset_dir / "lyrics_cache"

    # Pre-fetch lyrics in parallel
    print(f"\n[1/3] Pre-fetching lyrics for {len(videos)} videos...")
    t_lyr0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_map = {
            executor.submit(get_reference_lyrics, v["video_id"], v.get("title", ""), lyrics_cache_dir): v["video_id"]
            for v in videos
        }
        fetched_count = 0
        for f in concurrent.futures.as_completed(future_map):
            res = f.result()
            if len(res) >= 5:
                fetched_count += 1
    print(f"  Lyrics fetched: {fetched_count}/{len(videos)} in {time.time() - t_lyr0:.1f}s")

    # Create run in dataset
    run_info = registry.create_run(
        workspace_root,
        args.dataset,
        tracker=args.tracker,
        fps=1.0,
        options={
            "device": args.device,
            "rec_batch_size": args.rec_batch,
            "dedup": not args.no_dedup,
        },
    )
    run_id = run_info["run_id"]
    run_dir = dataset_dir / "runs" / run_id
    print(f"\n[2/3] Created run '{run_id}'. Running OCR and building Silver Ground Truth...")

    engines = PipelineEngines()
    t_run0 = time.time()
    results = []
    for idx, v in enumerate(videos, 1):
        vid = v["video_id"]
        vtitle = v.get("title", "")[:35]
        print(f"  ({idx}/{len(videos)}) Processing {vid} [{vtitle}]...", end="", flush=True)
        res = process_video(
            v,
            dataset_dir,
            run_dir,
            lyrics_cache_dir,
            tracker=args.tracker,
            device=args.device,
            rec_batch_size=args.rec_batch,
            dedup=not args.no_dedup,
            engines=engines,
        )
        # Register in run
        registry.sync_run_entry(
            workspace_root,
            args.dataset,
            run_id,
            video_id=vid,
            artifact_dir=f"videos/{vid}",
            title=v.get("title", ""),
            channel=v.get("channel", ""),
        )
        print(f" Done ({res['tracks_count']} tracks, {res['lyrics_count']} lyric lines, {res['elapsed_sec']}s)")
        results.append(res)

    total_run_time = time.time() - t_run0
    total_tracks = sum(r["tracks_count"] for r in results)
    print(f"  Completed {len(videos)} videos ({total_tracks} tracks total) in {total_run_time:.1f}s (avg {total_run_time/len(videos):.1f}s/video)")

    # Run evaluation
    print(f"\n[3/3] Evaluating Run '{run_id}' against Silver Ground Truth...")
    report, paired = evaluate_run(dataset_dir, run_id, gate_min_coverage=0.0)

    print("\n" + "=" * 60)
    print("                EVALUATION SUMMARY REPORT")
    print("=" * 60)
    print(f"Dataset:              {args.dataset}")
    print(f"Run ID:               {run_id}")
    print(f"Total Videos:         {len(videos)}")
    print(f"Total Tracks:         {report.total_tracks}")
    print(f"Evaluated Tracks:     {report.evaluated_tracks}")
    print(f"Unlabeled Tracks:     {report.unlabeled_tracks}")
    print(f"Precision:            {report.binary_lyric.precision * 100:.2f}%")
    print(f"Recall:               {report.binary_lyric.recall * 100:.2f}%")
    print(f"F1 Score:             {report.binary_lyric.f1 * 100:.2f}%")
    print(f"Coverage:             {report.coverage_rate * 100:.2f}%")
    print(f"Leakage Rate:         {report.leakage.count_rate * 100:.2f}% ({report.leakage.leaked_tracks_count} tracks)")
    print(f"False Drop Rate:      {report.false_drop.rate * 100:.2f}% ({report.false_drop.count} tracks)")
    print(f"Gate Passed:          {report.gate_passed}")

    # Inspect discrepancies
    discrepancies = [
        it for it in paired
        if it.has_label and it.label is not None and it.label.human_label != it.pred.model_label
    ]
    print(f"Total Discrepancies:  {len(discrepancies)}")
    if discrepancies:
        print("\nTop 10 Discrepancy Samples (Model vs Silver GT):")
        for idx, d in enumerate(discrepancies[:10], 1):
            txt = (d.pred.dominant_text or "")[:40]
            print(f"  {idx:>2}. [{d.pred.video_id}] '{txt}'")
            print(f"      Model: {d.pred.model_label} (conf: {d.pred.model_confidence:.2f}) | Silver: {d.label.human_label} | Reason: {d.label.notes}")

    # Save JSON report
    report_out = dataset_dir / f"eval_report_{run_id}.json"
    report_dict = {
        "dataset": args.dataset,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report": asdict(report),
        "discrepancies_count": len(discrepancies),
        "discrepancies": [
            {
                "video_id": d.pred.video_id,
                "track_id": d.pred.track_id,
                "text": d.pred.dominant_text,
                "model_label": d.pred.model_label,
                "model_confidence": d.pred.model_confidence,
                "silver_label": d.label.human_label if d.label else None,
                "notes": d.label.notes if d.label else None,
            }
            for d in discrepancies
        ],
    }
    report_out.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetailed evaluation JSON written to: {report_out}")


if __name__ == "__main__":
    main()
