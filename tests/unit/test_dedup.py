"""Tests for near-duplicate frame reuse in the Phase-1 pipeline (no ffmpeg needed)."""

import json

import cv2
import numpy as np

from lyra_ocr import pipeline
from lyra_ocr.ocr import TextLine
from lyra_ocr.video.dedup import frame_signature, is_duplicate

PROBE = {
    "duration": 2.0,
    "fps": 10.0,
    "width": 128,
    "height": 72,
    "rotation": 0,
    "has_audio": True,
}


class FakeDetector:
    def detect(self, img):
        return [{"polygon": [[10, 10], [90, 10], [90, 40], [10, 40]]}]


class CountingEngine:
    DEFAULT_MODEL = "fake_rec"

    def __init__(self):
        self.calls = 0

    def process_detections(self, img, detections):
        self.calls += 1
        return [
            TextLine(
                polygon=[[10, 10], [90, 10], [90, 40], [10, 40]],
                bbox=[10, 10, 90, 40],
                text_raw="la la",
                text_norm="la la",
                confidence=0.9,
                model_id=self.DEFAULT_MODEL,
                script="latin",
            )
        ]


def _packet(idx: int, frame: np.ndarray, jpeg: bytes) -> dict:
    return {
        "frame_id": idx,
        "timestamp_ms": idx * 500,
        "pts_time": idx * 0.5,
        "image_path": f"frames/frame_{idx + 1:06d}.jpg",
        "image_bgr": frame,
        "jpeg_bytes": jpeg,
        "pts_source": "showinfo",
    }


def _run(tmp_path, monkeypatch, dedup: bool):
    frames = [
        np.full((72, 128, 3), 50, np.uint8),
        np.full((72, 128, 3), 50, np.uint8),
        np.full((72, 128, 3), 180, np.uint8),
    ]
    jpegs = [cv2.imencode(".jpg", frame)[1].tobytes() for frame in frames]
    packets = [_packet(i, frame, jpeg) for i, (frame, jpeg) in enumerate(zip(frames, jpegs))]

    monkeypatch.setattr(pipeline, "_iter_pipe_frames", lambda *a, **k: iter(packets))
    monkeypatch.setattr(pipeline, "probe_video", lambda path: dict(PROBE))

    video = tmp_path / "video.mp4"
    video.write_bytes(b"x")
    engine = CountingEngine()
    engines = pipeline.PipelineEngines(detector=FakeDetector(), ocr_engine=engine)
    out = tmp_path / ("dedup" if dedup else "nodup")

    result = pipeline.run_phase1_pipeline(
        video,
        out,
        fps=2.0,
        max_frames=3,
        video_id="vid",
        engines=engines,
        dedup=dedup,
        frame_source="pipe",
        force=True,
    )
    return out, engine, result


def test_dedup_reuses_previous_ocr_and_frame_file(tmp_path, monkeypatch):
    out, engine, _ = _run(tmp_path, monkeypatch, dedup=True)

    assert engine.calls == 2  # frames 0 and 2 only

    raw = json.loads((out / "ocr_raw.json").read_text(encoding="utf-8"))
    frames = raw["frames"]
    assert len(frames) == 3
    assert "reused" not in frames[0]
    assert frames[1]["reused"] is True
    assert frames[1]["dedup_of_frame"] == 0
    assert frames[1]["image_path"] == frames[0]["image_path"]
    assert frames[1]["elapsed_ms"] == 0.0
    assert frames[1]["boxes"] == frames[0]["boxes"]
    assert raw["total_ocr_calls"] == 2

    assert (out / "frames" / "frame_000001.jpg").is_file()
    assert not (out / "frames" / "frame_000002.jpg").exists()
    assert (out / "frames" / "frame_000003.jpg").is_file()

    index = json.loads((out / "frame_index.json").read_text(encoding="utf-8"))
    assert len(index) == 3
    assert index[1]["reused"] is True
    assert index[1]["image_path"] == index[0]["image_path"]

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime"]["dedup"] is True
    assert manifest["runtime"]["pipeline_version"] == "2"
    assert manifest["runtime"]["frame_source"] == "pipe"


def test_no_dedup_processes_and_writes_every_frame(tmp_path, monkeypatch):
    out, engine, _ = _run(tmp_path, monkeypatch, dedup=False)

    assert engine.calls == 3
    raw = json.loads((out / "ocr_raw.json").read_text(encoding="utf-8"))
    assert all("reused" not in frame for frame in raw["frames"])
    for idx in (1, 2, 3):
        assert (out / "frames" / f"frame_{idx:06d}.jpg").is_file()


def test_config_hash_tracks_performance_options(tmp_path, monkeypatch):
    _out_on, _eng_on, on = _run(tmp_path, monkeypatch, dedup=True)
    _out_off, _eng_off, off = _run(tmp_path, monkeypatch, dedup=False)

    assert on["config_hash"] != off["config_hash"]


def test_dedup_helpers_detect_identical_but_not_local_change():
    base = np.full((72, 128, 3), 50, np.uint8)
    identical = base.copy()
    changed = base.copy()
    changed[20:30, 20:60] = 255

    assert is_duplicate(frame_signature(base), frame_signature(identical))
    assert not is_duplicate(frame_signature(base), frame_signature(changed))
