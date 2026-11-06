"""Regression: per-run device/batch options must override a holder's old runtime."""

import json
import shutil
import subprocess

import pytest

from lyra_ocr.pipeline import PipelineEngines, run_phase1_pipeline


class FakeDetector:
    def detect(self, img):
        return [{"polygon": [[10, 10], [60, 10], [60, 30], [10, 30]]}]


class FakeOcrEngine:
    DEFAULT_MODEL = "fake_rec"

    def process_detections(self, img, detections):
        return []


@pytest.fixture()
def engines():
    return PipelineEngines(detector=FakeDetector(), ocr_engine=FakeOcrEngine())


@pytest.fixture()
def video(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required")
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10",
            "-t",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_pipeline_resolves_runtime_from_run_options(tmp_path, engines, video, monkeypatch):
    """A holder carrying an old CPU runtime must not override this run's device."""
    from lyra_ocr.ocr.runtime import RuntimeConfig

    calls = []

    def fake_resolve(request="auto", rec_batch_size=8, **kwargs):
        use_cuda = request == "cuda"
        runtime = RuntimeConfig(
            request=request,
            device="cuda" if use_cuda else "cpu",
            rec_batch_size=rec_batch_size,
            use_cuda=use_cuda,
            providers=("CUDAExecutionProvider",) if use_cuda else ("CPUExecutionProvider",),
        )
        calls.append((request, rec_batch_size))
        return runtime

    monkeypatch.setattr("lyra_ocr.pipeline.resolve_runtime", fake_resolve)
    engines.runtime = fake_resolve("cpu", 1)  # stale runtime from an earlier run
    calls.clear()

    common = dict(
        fps=2.0,
        max_frames=1,
        engines=engines,
        force=True,
        frame_source="disk",
    )
    cpu = run_phase1_pipeline(
        video,
        tmp_path / "cpu",
        video_id="vid_rt_cpu",
        device="cpu",
        rec_batch_size=1,
        **common,
    )
    cuda = run_phase1_pipeline(
        video,
        tmp_path / "cuda",
        video_id="vid_rt_cuda",
        device="cuda",
        rec_batch_size=8,
        **common,
    )

    assert calls == [("cpu", 1), ("cuda", 8)]
    assert cpu["config_hash"] != cuda["config_hash"]
    manifest = json.loads((tmp_path / "cuda" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime"]["device"] == "cuda"
    assert manifest["runtime"]["rec_batch_size"] == 8
    assert manifest["runtime"]["frame_source"] == "disk"
