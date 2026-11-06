"""Unit tests for streaming (ffmpeg pipe) frame extraction with a fake process."""

import io
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from lyra_ocr.video import frames


class FakeProcess:
    """Minimal subprocess.Popen stand-in backed by in-memory pipes."""

    def __init__(self, cmd, stdout=b"", stderr=b"", returncode=0):
        self.args = cmd
        self._raw = stdout
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self._rc = returncode
        self.terminated = False

    def poll(self):
        if self.terminated:
            return -15
        return self._rc if self.stdout.tell() >= len(self._raw) else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return -15 if self.terminated else self._rc

    @property
    def returncode(self):
        return -15 if self.terminated else self._rc


def _jpeg(gray: int) -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.full((72, 128, 3), gray, np.uint8))
    assert ok
    return buffer.tobytes()


def _showinfo(pts: list[float]) -> bytes:
    return "".join(
        f"[Parsed_showinfo_1 @ 0x1] n: {i} pts: {i} pts_time:{t}\n"
        for i, t in enumerate(pts)
    ).encode()


def _install(monkeypatch, proc):
    dummy_video = Path("/tmp/lyra-stream-test.mp4")
    dummy_video.write_bytes(b"x")
    monkeypatch.setattr(frames.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def fake_popen(cmd, **kwargs):
        proc.args = cmd
        return proc

    monkeypatch.setattr(frames.subprocess, "Popen", fake_popen)
    return dummy_video


def test_stream_yields_frames_with_showinfo_pts(tmp_path, monkeypatch):
    proc = FakeProcess([], stdout=_jpeg(10) + _jpeg(20), stderr=_showinfo([0.0, 0.5]))
    video = _install(monkeypatch, proc)

    packets = list(
        frames.iter_sampled_frames(video, tmp_path / "art", fps=2.0, save_frames=True)
    )

    assert [p.frame_id for p in packets] == [0, 1]
    assert [p.timestamp_ms for p in packets] == [0, 500]
    assert [p.pts_source for p in packets] == ["showinfo", "showinfo"]
    assert packets[0].image_path == "frames/frame_000001.jpg"
    assert (tmp_path / "art" / "frames" / "frame_000001.jpg").is_file()
    assert (tmp_path / "art" / "frames" / "frame_000002.jpg").is_file()
    assert packets[1].image_bgr.shape == (72, 128, 3)


def test_stream_clears_stale_frames(tmp_path, monkeypatch):
    out = tmp_path / "art"
    (out / "frames").mkdir(parents=True)
    stale = out / "frames" / "frame_999999.jpg"
    stale.write_bytes(b"old")

    proc = FakeProcess([], stdout=_jpeg(10), stderr=_showinfo([0.0]))
    video = _install(monkeypatch, proc)
    list(frames.iter_sampled_frames(video, out, fps=2.0))

    assert not stale.exists()


def test_stream_command_caps_frames_and_uses_pipe(tmp_path, monkeypatch):
    proc = FakeProcess([], stdout=_jpeg(10), stderr=_showinfo([0.0]))
    video = _install(monkeypatch, proc)

    list(frames.iter_sampled_frames(video, tmp_path / "art", fps=1.0, max_frames=2))

    assert "-frames:v" in proc.args and "2" in proc.args
    assert proc.args[-1] == "-"
    assert "image2pipe" in proc.args
    assert "mjpeg" in proc.args


def test_stream_estimates_pts_when_stderr_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(frames, "_PTS_WAIT_SEC", 0.01)
    proc = FakeProcess([], stdout=_jpeg(10) + _jpeg(20), stderr=b"")
    video = _install(monkeypatch, proc)

    packets = list(frames.iter_sampled_frames(video, tmp_path / "art", fps=2.0))

    assert [p.pts_source for p in packets] == ["estimated", "estimated"]
    assert [p.timestamp_ms for p in packets] == [0, 500]


def test_stream_cancel_yields_nothing(tmp_path, monkeypatch):
    proc = FakeProcess([], stdout=_jpeg(10), stderr=_showinfo([0.0]))
    video = _install(monkeypatch, proc)
    cancel = threading.Event()
    cancel.set()

    assert list(frames.iter_sampled_frames(video, tmp_path / "art", cancel_event=cancel)) == []


def test_stream_raises_when_ffmpeg_produces_no_frames(tmp_path, monkeypatch):
    proc = FakeProcess([], stdout=b"", stderr=b"Unknown encoder", returncode=1)
    video = _install(monkeypatch, proc)

    with pytest.raises(frames.FrameStreamError):
        list(frames.iter_sampled_frames(video, tmp_path / "art"))


def test_closing_generator_stops_producer_blocked_on_full_queue(tmp_path, monkeypatch):
    proc = FakeProcess(
        [], stdout=_jpeg(10) * 4, stderr=_showinfo([0.0, 0.5, 1.0, 1.5])
    )
    video = _install(monkeypatch, proc)

    generator = frames.iter_sampled_frames(video, tmp_path / "art", fps=2.0, queue_size=1)
    assert next(generator).frame_id == 0
    generator.close()

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        alive = [t for t in threading.enumerate() if t.name == "lyra-ffmpeg-frames"]
        if not alive:
            break
        alive[0].join(0.05)

    assert not [t for t in threading.enumerate() if t.name == "lyra-ffmpeg-frames"]


def test_stream_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(frames.iter_sampled_frames(tmp_path / "missing.mp4", tmp_path / "art"))


@pytest.mark.slow
def test_pipe_and_disk_pts_parity(tmp_path):
    """Same video, both extraction modes: frame count and PTS must agree."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required")

    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10",
            "-t",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )

    disk = frames.sample_frames(video, tmp_path / "disk", fps=2.0)
    pipe = list(frames.iter_sampled_frames(video, tmp_path / "pipe", fps=2.0, save_frames=True))

    assert len(disk) == len(pipe)
    for disk_entry, packet in zip(disk, pipe):
        assert abs(disk_entry["timestamp_ms"] - packet.timestamp_ms) <= 1
        assert (tmp_path / "pipe" / packet.image_path).is_file()

    index = json.loads((tmp_path / "disk" / "frame_index.json").read_text(encoding="utf-8"))
    assert len(index) == len(disk)
