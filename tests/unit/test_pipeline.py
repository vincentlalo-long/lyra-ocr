"""Unit tests for the extracted Phase-1 pipeline runner (fake OCR engines)."""

import json
import shutil
import threading
from pathlib import Path

import pytest

from lyra_ocr.ocr import TextLine
from lyra_ocr.pipeline import (
    PipelineCancelled,
    PipelineEngines,
    compute_video_id,
    run_phase1_pipeline,
)

FIXTURE_VIDEO = Path(__file__).resolve().parents[1] / "fixtures" / "castle_en.mp4"

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="pipeline test requires ffmpeg"
)


class FakeDetector:
    def detect(self, img):
        return [{"polygon": [[10, 10], [210, 10], [210, 50], [10, 50]]}]


class FakeOcrEngine:
    DEFAULT_MODEL = "fake_rec"

    def process_detections(self, img, detections):
        return [
            TextLine(
                polygon=[[10, 10], [210, 10], [210, 50], [10, 50]],
                bbox=[10, 10, 210, 50],
                text_raw="We are the champions",
                text_norm="we are the champions",
                confidence=0.97,
                model_id="fake_rec",
                script="latin",
            )
        ]


@pytest.fixture()
def engines():
    return PipelineEngines(detector=FakeDetector(), ocr_engine=FakeOcrEngine())


def test_compute_video_id_override_and_legacy():
    assert compute_video_id("/x/y/z.mp4", "dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    legacy = compute_video_id(str(FIXTURE_VIDEO))
    assert legacy.startswith("castle_en_")
    assert legacy == compute_video_id(str(FIXTURE_VIDEO))


def test_pipeline_full_run_writes_artifacts(tmp_path, engines):
    out = tmp_path / "art"
    result = run_phase1_pipeline(
        FIXTURE_VIDEO,
        out,
        title="Castle",
        channel="CLARX",
        fps=2.0,
        max_frames=2,
        video_id="vid_override1",
        engines=engines,
        force=True,
    )
    assert result["skipped"] is False
    assert result["video_id"] == "vid_override1"
    assert result["frames_processed"] >= 1

    for fname in ("manifest.json", "tracks.json", "ocr_raw.json", "review_tracks.csv"):
        assert (out / fname).is_file(), f"missing {fname}"

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["video_id"] == "vid_override1"
    assert manifest["config_hash"] == result["config_hash"]
    assert manifest["models"]["det"] == "PP-OCRv6_det_small"

    tracks = json.loads((out / "tracks.json").read_text(encoding="utf-8"))
    assert tracks["total_tracks"] >= 1


def test_pipeline_skip_on_complete_and_force_rerun(tmp_path, engines):
    out = tmp_path / "art"
    kwargs = dict(title="T", channel="C", fps=2.0, max_frames=1, video_id="vid_sk", engines=engines)
    first = run_phase1_pipeline(FIXTURE_VIDEO, out, **kwargs)
    assert first["skipped"] is False

    second = run_phase1_pipeline(FIXTURE_VIDEO, out, **kwargs)
    assert second["skipped"] is True
    assert second["config_hash"] == first["config_hash"]

    # Different config (fps) must not skip
    third = run_phase1_pipeline(FIXTURE_VIDEO, out, **{**kwargs, "fps": 1.0})
    assert third["skipped"] is False


def test_pipeline_preserves_labels_on_force_rerun(tmp_path, engines):
    out = tmp_path / "art"
    kwargs = dict(fps=2.0, max_frames=1, video_id="vid_lbl", engines=engines)
    run_phase1_pipeline(FIXTURE_VIDEO, out, **kwargs)

    labels_file = out / "labels.jsonl"
    labels_file.write_text('{"video_id": "vid_lbl", "track_id": "1", "human_label": "lyric"}\n', encoding="utf-8")

    run_phase1_pipeline(FIXTURE_VIDEO, out, **kwargs, force=True)
    assert labels_file.is_file()
    assert "human_label" in labels_file.read_text(encoding="utf-8")


def test_pipeline_cancel_event_raises(tmp_path, engines):
    event = threading.Event()
    event.set()
    with pytest.raises(PipelineCancelled):
        run_phase1_pipeline(
            FIXTURE_VIDEO,
            tmp_path / "art",
            fps=2.0,
            max_frames=1,
            engines=engines,
            cancel_event=event,
        )


def test_pipeline_progress_stages_reported(tmp_path, engines):
    stages = []
    run_phase1_pipeline(
        FIXTURE_VIDEO,
        tmp_path / "art",
        fps=2.0,
        max_frames=1,
        video_id="vid_prog",
        engines=engines,
        force=True,
        progress=lambda stage, pct, msg: stages.append(stage),
    )
    for expected in ("probe", "sampling", "ocr", "tracking", "classifying", "exporting"):
        assert expected in stages
