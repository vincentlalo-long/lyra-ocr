#!/usr/bin/env python
"""Benchmark Lyra OCR stages on a real video.

Measures frame extraction, detection, recognition, and (optionally) the full
pipeline across a small matrix of device / batch / dedup / frame-source
settings. Results are written as JSON under ``reports/bench/``.

Examples:
    python scripts/benchmark_pipeline.py --video datasets/.../videos/song.mp4 --frames 100
    python scripts/benchmark_pipeline.py --video song.mp4 --devices cpu,cuda \
        --batches 1,8,16 --full
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from lyra_ocr.ocr.detect import TextDetector
from lyra_ocr.ocr.engine import OCREngine
from lyra_ocr.ocr.runtime import available_providers, resolve_runtime
from lyra_ocr.pipeline import run_phase1_pipeline
from lyra_ocr.video.frames import iter_sampled_frames, sample_frames


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB (Linux: KiB)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def vram_used_mb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        return float(out.stdout.strip().splitlines()[0])
    except ValueError:
        return None


def median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def _parse_list(raw: str, cast=str) -> list:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def bench_extraction(
    video: Path,
    source: str,
    frames: int,
    fps: float,
    repeats: int,
) -> dict:
    """Median wall-clock to extract ``frames`` frames with the given source."""
    timings: list[float] = []
    count = 0
    for _ in range(repeats):
        with tempfile.TemporaryDirectory(prefix="lyra-bench-extract-") as tmp:
            started = time.perf_counter()
            if source == "pipe":
                count = sum(
                    1
                    for _ in iter_sampled_frames(
                        video, tmp, fps=fps, max_frames=frames, save_frames=False
                    )
                )
            else:
                count = len(sample_frames(video, tmp, fps=fps, max_frames=frames))
            timings.append((time.perf_counter() - started) * 1000.0)
    total_ms = median(timings) or 0.0
    return {
        "frame_source": source,
        "frames": count,
        "total_ms": total_ms,
        "ms_per_frame": round(total_ms / max(1, count), 2),
        "peak_rss_mb": peak_rss_mb(),
    }


def _load_frames(video: Path, frames: int, fps: float) -> list:
    with tempfile.TemporaryDirectory(prefix="lyra-bench-frames-") as tmp:
        return [
            packet.image_bgr
            for packet in iter_sampled_frames(
                video, tmp, fps=fps, max_frames=frames, save_frames=False
            )
        ]


def bench_ocr(
    images: list,
    device: str,
    rec_batch_size: int,
    repeats: int,
) -> dict:
    """Detection / recognition latency for one runtime configuration."""
    runtime = resolve_runtime(device, rec_batch_size)
    if runtime.device != device and device == "cuda":
        return {
            "requested_device": device,
            "device": runtime.device,
            "rec_batch_size": rec_batch_size,
            "skipped": "CUDAExecutionProvider unavailable",
        }

    detector = TextDetector(runtime=runtime)
    ocr = OCREngine(runtime=runtime)

    # Warmup: model load, kernel autotune, allocator growth.
    for image in images[: min(3, len(images))]:
        detector.detect(image)
    detections = []
    for image in images[: min(3, len(images))]:
        detections.append(detector.detect(image))
        ocr.process_detections(image, detections[-1])

    detect_runs: list[float] = []
    rec_runs: list[float] = []
    crop_counts: list[int] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        all_dets = [detector.detect(image) for image in images]
        detect_ms = (time.perf_counter() - t0) * 1000.0

        rec_ms = 0.0
        for image, dets in zip(images, all_dets):
            t1 = time.perf_counter()
            ocr.process_detections(image, dets)
            rec_ms += (time.perf_counter() - t1) * 1000.0
            crop_counts.append(len(dets))

        detect_runs.append(detect_ms)
        rec_runs.append(rec_ms)

    providers = (detector.runtime_info() or {}).get("providers") or list(runtime.providers)
    n = max(1, len(images))
    return {
        "requested_device": device,
        "device": runtime.device,
        "providers": providers,
        "rec_batch_size": rec_batch_size,
        "frames": len(images),
        "avg_crops_per_frame": round(sum(crop_counts) / max(1, len(crop_counts)), 2),
        "detect_ms_per_frame": round((median(detect_runs) or 0.0) / n, 2),
        "recognize_ms_per_frame": round((median(rec_runs) or 0.0) / n, 2),
        "total_ms_per_frame": round(
            ((median(detect_runs) or 0.0) + (median(rec_runs) or 0.0)) / n, 2
        ),
        "peak_rss_mb": peak_rss_mb(),
        "vram_used_mb": vram_used_mb(),
    }


def bench_full_pipeline(
    video: Path,
    device: str,
    rec_batch_size: int,
    dedup: bool,
    frame_source: str,
    frames: int,
    fps: float,
    repeats: int,
) -> dict:
    """End-to-end wall-clock for one full pipeline configuration."""
    timings: list[float] = []
    for _ in range(repeats):
        with tempfile.TemporaryDirectory(prefix="lyra-bench-full-") as tmp:
            started = time.perf_counter()
            run_phase1_pipeline(
                video,
                tmp,
                fps=fps,
                max_frames=frames,
                force=True,
                device=device,
                rec_batch_size=rec_batch_size,
                dedup=dedup,
                frame_source=frame_source,
            )
            timings.append((time.perf_counter() - started) * 1000.0)
    elapsed = median(timings) or 0.0
    return {
        "device": device,
        "rec_batch_size": rec_batch_size,
        "dedup": dedup,
        "frame_source": frame_source,
        "frames": frames,
        "total_ms": elapsed,
        "ms_per_frame": round(elapsed / max(1, frames), 2),
        "peak_rss_mb": peak_rss_mb(),
        "vram_used_mb": vram_used_mb(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--frames", type=int, default=100, help="Frames per measurement")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling fps")
    parser.add_argument("--devices", default="cpu,cuda", help="Comma list: cpu,cuda")
    parser.add_argument("--batches", default="1,8,16", help="Comma list of rec batch sizes")
    parser.add_argument("--repeats", type=int, default=3, help="Repetitions (median)")
    parser.add_argument("--out", default="reports/bench", help="Output directory")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also benchmark the full pipeline (slow; device/batch combos × dedup × source)",
    )
    args = parser.parse_args()

    video = Path(args.video).resolve()
    if not video.is_file():
        parser.error(f"video not found: {video}")

    devices = _parse_list(args.devices)
    batches = _parse_list(args.batches, int)

    report: dict = {
        "video": str(video),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sampling_fps": args.fps,
        "frames_target": args.frames,
        "repeats": args.repeats,
        "available_providers": available_providers(),
        "extraction": [],
        "ocr": [],
        "full_pipeline": [],
    }

    for source in ("disk", "pipe"):
        print(f"[extract] {source} ...", flush=True)
        report["extraction"].append(
            bench_extraction(video, source, args.frames, args.fps, args.repeats)
        )

    print(f"[frames] decoding {args.frames} frames for OCR benchmarks ...", flush=True)
    images = _load_frames(video, args.frames, args.fps)
    report["frames_loaded"] = len(images)

    for device in devices:
        for batch in batches:
            print(f"[ocr] device={device} batch={batch} ...", flush=True)
            report["ocr"].append(bench_ocr(images, device, batch, args.repeats))

    if args.full:
        for device in devices:
            for batch in batches:
                for dedup in (False, True):
                    for source in ("disk", "pipe"):
                        print(
                            f"[full] device={device} batch={batch} dedup={dedup} "
                            f"source={source} ...",
                            flush=True,
                        )
                        report["full_pipeline"].append(
                            bench_full_pipeline(
                                video,
                                device,
                                batch,
                                dedup,
                                source,
                                args.frames,
                                args.fps,
                                max(1, args.repeats - 2),
                            )
                        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_file}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
