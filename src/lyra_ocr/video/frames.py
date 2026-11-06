"""Frame extraction: disk mode (legacy) and streaming MJPEG pipe mode.

``sample_frames`` keeps the original ffmpeg-to-disk behaviour and contract.
``iter_sampled_frames`` runs ffmpeg once with ``image2pipe`` so frames are
decoded in a producer thread while the OCR consumer works, avoiding a
write-then-read cycle and overlapping decode with inference.

Streaming frame boundaries use the JPEG EOI marker (``FFD9``). Entropy-coded
segments are byte-stuffed (``FF`` -> ``FF00``), so an EOI marker cannot appear
inside the scan data; ffmpeg's MJPEG output carries no EXIF thumbnails.
"""

from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import numpy as np

_SHOWINFO_RE = re.compile(r"n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*(-?[\d.]+)")
_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"
_READ_CHUNK = 1 << 16
_PTS_WAIT_SEC = 1.0
_STDERR_TAIL_LINES = 40


class FrameStreamError(RuntimeError):
    """Streaming extraction failed before the first frame or mid-stream."""


@dataclass
class FramePacket:
    """One decoded frame travelling from the ffmpeg producer to the consumer."""

    frame_id: int
    timestamp_ms: int
    pts_time: float
    image_path: str
    image_bgr: np.ndarray
    jpeg_bytes: bytes
    pts_source: str = "showinfo"  # showinfo|estimated


def _clear_extracted_frames(frames_dir: Path) -> int:
    """Remove previous JPEGs so a re-run with fewer frames cannot read stale files."""
    removed = 0
    for stale in frames_dir.glob("frame_*.jpg"):
        try:
            stale.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def write_frame_index(output_dir: str | Path, entries: list[dict[str, Any]]) -> Path:
    """Write ``frame_index.json`` for the processed frame list."""
    out_root = Path(output_dir).resolve()
    index_file = out_root / "frame_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    return index_file


def sample_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 2.0,
    max_frames: int | None = None,
    start_sec: float = 0.0,
) -> list[dict[str, Any]]:
    """Extract frames at a given sampling rate, recording accurate Presentation Time Stamps (PTS).

    Args:
        video_path: Path to the video file.
        output_dir: Directory where 'frames/' and 'frame_index.json' are saved.
        fps: Target frame sampling rate (default: 2.0 fps).
        max_frames: Optional upper limit on the number of frames to extract.
        start_sec: Optional start time in seconds (default: 0.0).

    Returns:
        List of frame index entries with frame_id, timestamp_ms, pts_time, and relative image_path.
    """
    video = Path(video_path).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg is not found on PATH")

    out_root = Path(output_dir).resolve()
    frames_dir = out_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    _clear_extracted_frames(frames_dir)

    cmd = [ffmpeg_bin, "-y"]
    if start_sec > 0:
        cmd.extend(["-ss", f"{start_sec:.3f}"])

    cmd.extend([
        "-i",
        str(video),
        "-vf",
        f"fps={fps},showinfo",
        "-q:v",
        "2",
    ])

    if max_frames and max_frames > 0:
        cmd.extend(["-frames:v", str(max_frames)])

    frame_pattern = str(frames_dir / "frame_%06d.jpg")
    cmd.append(frame_pattern)

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr.strip()}")

    # Collect extracted image files in sorted order
    extracted_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not extracted_files:
        raise RuntimeError("ffmpeg completed but no frames were extracted")

    # Parse PTS from ffmpeg showinfo filter output: n: <n> pts: <pts> pts_time:<pts_time>
    pts_matches = re.findall(r"n:\s*(\d+)\s+pts:\s*(\d+)\s+pts_time:([\d.]+)", result.stderr)
    pts_by_idx: dict[int, float] = {}
    for n_str, _pts_str, pts_time_str in pts_matches:
        try:
            pts_by_idx[int(n_str)] = float(pts_time_str)
        except ValueError:
            pass

    index_entries: list[dict[str, Any]] = []
    step_sec = 1.0 / max(0.1, fps)

    for idx, fpath in enumerate(extracted_files):
        # Prefer exact PTS from showinfo filter; fallback to calculated step
        if idx in pts_by_idx:
            rel_pts = pts_by_idx[idx]
        else:
            rel_pts = idx * step_sec

        t_ms = int(round((start_sec + rel_pts) * 1000.0))
        rel_path = f"frames/{fpath.name}"

        index_entries.append(
            {
                "frame_id": idx,
                "timestamp_ms": t_ms,
                "pts_time": round(start_sec + rel_pts, 4),
                "image_path": rel_path,
            }
        )

    write_frame_index(out_root, index_entries)

    return index_entries


# --------------------------------------------------------------------------- #
# Streaming (ffmpeg image2pipe) extraction
# --------------------------------------------------------------------------- #


def _read_showinfo_stderr(
    proc: subprocess.Popen,
    pts_by_idx: dict[int, float],
    cond: threading.Condition,
    stderr_tail: deque[str],
    eof: dict[str, bool],
) -> None:
    """Parse ``showinfo`` PTS lines from ffmpeg stderr as they arrive."""
    stderr = proc.stderr
    if stderr is None:
        with cond:
            eof["done"] = True
            cond.notify_all()
        return

    try:
        for raw_line in stderr:
            line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else str(raw_line)
            stderr_tail.append(line.rstrip())
            match = _SHOWINFO_RE.search(line)
            if not match:
                continue
            try:
                idx = int(match.group(1))
                pts = float(match.group(3))
            except ValueError:
                continue
            with cond:
                pts_by_idx[idx] = pts
                cond.notify_all()
    finally:
        with cond:
            eof["done"] = True
            cond.notify_all()


def _wait_for_pts(
    pts_by_idx: dict[int, float],
    cond: threading.Condition,
    eof: dict[str, bool],
    idx: int,
) -> Optional[float]:
    """Wait briefly for the showinfo line of ``idx`` (stderr lags stdout)."""
    deadline = time.monotonic() + _PTS_WAIT_SEC
    with cond:
        while idx not in pts_by_idx:
            remaining = deadline - time.monotonic()
            if eof["done"] or remaining <= 0:
                return None
            cond.wait(timeout=min(0.05, remaining))
        return pts_by_idx[idx]


def _queue_put(
    frame_queue: "queue.Queue[Optional[FramePacket]]",
    item: Optional[FramePacket],
    cancel_event: Optional[threading.Event],
    stop_event: threading.Event,
) -> bool:
    while True:
        if stop_event.is_set() or (cancel_event is not None and cancel_event.is_set()):
            return False
        try:
            frame_queue.put(item, timeout=0.2)
            return True
        except queue.Full:
            continue


def _produce_packets(
    proc: subprocess.Popen,
    frames_dir: Path,
    frame_queue: "queue.Queue[Optional[FramePacket]]",
    pts_by_idx: dict[int, float],
    cond: threading.Condition,
    eof: dict[str, bool],
    cancel_event: Optional[threading.Event],
    stop_event: threading.Event,
    *,
    fps: float,
    start_sec: float,
    save_frames: bool,
) -> None:
    """Split ffmpeg's MJPEG stdout into frames and push packets onto the queue."""
    step_sec = 1.0 / max(0.1, fps)
    idx = 0
    buffer = b""
    stdout = proc.stdout
    frames_dir.mkdir(parents=True, exist_ok=True)

    def read_chunk() -> bytes:
        if stdout is None:
            return b""
        reader = getattr(stdout, "read1", stdout.read)
        try:
            return reader(_READ_CHUNK)
        except (OSError, ValueError):
            return b""

    try:
        while True:
            if stop_event.is_set() or (cancel_event is not None and cancel_event.is_set()):
                return
            chunk = read_chunk()
            if not chunk:
                break
            buffer += chunk

            while True:
                eoi = buffer.find(_JPEG_EOI)
                if eoi < 0:
                    break
                raw = buffer[: eoi + len(_JPEG_EOI)]
                buffer = buffer[eoi + len(_JPEG_EOI) :]

                soi = raw.find(_JPEG_SOI)
                if soi < 0:
                    continue
                jpeg = raw[soi:]
                image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue  # skip undecodable chunk, keep the stream alive

                pts = _wait_for_pts(pts_by_idx, cond, eof, idx)
                if pts is None:
                    rel_pts = idx * step_sec
                    pts_source = "estimated"
                else:
                    rel_pts = pts
                    pts_source = "showinfo"

                packet = FramePacket(
                    frame_id=idx,
                    timestamp_ms=int(round((start_sec + rel_pts) * 1000.0)),
                    pts_time=round(start_sec + rel_pts, 4),
                    image_path=f"frames/frame_{idx + 1:06d}.jpg",
                    image_bgr=image,
                    jpeg_bytes=jpeg,
                    pts_source=pts_source,
                )
                if save_frames:
                    (frames_dir / f"frame_{idx + 1:06d}.jpg").write_bytes(jpeg)

                if not _queue_put(frame_queue, packet, cancel_event, stop_event):
                    return
                idx += 1
    finally:
        _queue_put(frame_queue, None, cancel_event, stop_event)


def _stop_producer(proc: subprocess.Popen, producer: threading.Thread, stderr_thread: threading.Thread) -> None:
    """Terminate ffmpeg and join helper threads; never leave a zombie process."""
    if proc.poll() is None:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass

    for thread in (producer, stderr_thread):
        if thread.is_alive():
            thread.join(timeout=2.0)

    for stream in (proc.stdout, proc.stderr):
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass


def _build_pipe_command(
    ffmpeg_bin: str,
    video: Path,
    fps: float,
    max_frames: Optional[int],
    start_sec: float,
    jpeg_quality: int,
) -> list[str]:
    cmd = [ffmpeg_bin, "-hide_banner", "-nostdin", "-y"]
    if start_sec > 0:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    cmd.extend(["-i", str(video), "-vf", f"fps={fps},showinfo", "-q:v", str(jpeg_quality)])
    if max_frames and max_frames > 0:
        cmd.extend(["-frames:v", str(max_frames)])
    cmd.extend(["-an", "-f", "image2pipe", "-vcodec", "mjpeg", "-"])
    return cmd


def iter_sampled_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 2.0,
    max_frames: int | None = None,
    start_sec: float = 0.0,
    *,
    jpeg_quality: int = 2,
    queue_size: int = 8,
    save_frames: bool = True,
    cancel_event: Optional[threading.Event] = None,
) -> Iterator[FramePacket]:
    """Yield sampled frames from an ffmpeg MJPEG pipe (producer/consumer).

    Args:
        video_path: Input video.
        output_dir: Artifact root; JPEGs go to ``frames/`` when ``save_frames``.
        fps: Sampling rate passed to ffmpeg's ``fps`` filter.
        max_frames: Optional cap on extracted frames.
        start_sec: Optional seek offset.
        jpeg_quality: ffmpeg ``-q:v`` (2 = high quality).
        queue_size: Bounded queue length (caps RAM at ~queue_size decoded frames).
        save_frames: Write JPEGs immediately; False lets the consumer skip duplicates.
        cancel_event: Cooperative cancellation checked per frame.

    Raises:
        FileNotFoundError: Video missing.
        RuntimeError: ffmpeg not on PATH.
        FrameStreamError: ffmpeg failed before producing any decodable frame.
    """
    video = Path(video_path).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg is not found on PATH")

    out_root = Path(output_dir).resolve()
    frames_dir = out_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    _clear_extracted_frames(frames_dir)

    cmd = _build_pipe_command(ffmpeg_bin, video, fps, max_frames, start_sec, jpeg_quality)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise FrameStreamError(f"failed to start ffmpeg: {exc}") from exc

    pts_by_idx: dict[int, float] = {}
    cond = threading.Condition()
    eof = {"done": False}
    stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
    frame_queue: "queue.Queue[Optional[FramePacket]]" = queue.Queue(maxsize=max(1, queue_size))
    # Set when the consumer stops (close/cancel/error) so the producer can never
    # stay blocked on a full queue, even if cancel_event was never raised.
    stop_event = threading.Event()

    stderr_thread = threading.Thread(
        target=_read_showinfo_stderr,
        args=(proc, pts_by_idx, cond, stderr_tail, eof),
        name="lyra-ffmpeg-pts",
        daemon=True,
    )
    producer = threading.Thread(
        target=_produce_packets,
        args=(proc, frames_dir, frame_queue, pts_by_idx, cond, eof, cancel_event, stop_event),
        kwargs={
            "fps": fps,
            "start_sec": start_sec,
            "save_frames": save_frames,
        },
        name="lyra-ffmpeg-frames",
        daemon=True,
    )
    stderr_thread.start()
    producer.start()

    produced = 0
    cancelled = False
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            try:
                packet = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if packet is None:
                break
            produced += 1
            yield packet
    finally:
        stop_event.set()
        _stop_producer(proc, producer, stderr_thread)

    if not cancelled and proc.returncode not in (0, None):
        tail = "\n".join(stderr_tail).strip()
        if produced == 0:
            raise FrameStreamError(
                f"ffmpeg produced no decodable frames (rc={proc.returncode}): {tail[-800:]}"
            )
        raise FrameStreamError(
            f"ffmpeg failed mid-stream after {produced} frame(s) "
            f"(rc={proc.returncode}): {tail[-800:]}"
        )
