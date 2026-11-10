from pathlib import Path
import pytest

from lyra_ocr.lyric.models import LyricCue
from lyra_ocr.eval.visual_timing import measure_visual_timing


def test_measure_visual_timing_not_measured():
    cues = [LyricCue(start_ms=1000, end_ms=3000, text="TEST")]
    res = measure_visual_timing("nonexistent.mp4", cues, "/nonexistent")
    assert res["status"] == "not_measured"
    assert res["percent_within_100ms"] == 0.0
    assert res["percent_within_300ms"] == 0.0
    assert res["median_abs_error_ms"] is None or res["median_abs_error_ms"] == 0.0


def test_measure_visual_timing_insufficient_cues(monkeypatch, tmp_path):
    video_file = tmp_path / "dummy.mp4"
    video_file.write_bytes(b"")
    art_dir = tmp_path / "art"
    art_dir.mkdir()

    cues = [
        LyricCue(start_ms=1000, end_ms=3000, text="A"),
        LyricCue(start_ms=4000, end_ms=6000, text="B"),
    ]

    import lyra_ocr.lyric.timing as timing_mod

    def mock_refine_timings(cues_in, *args, **kwargs):
        refined = [
            LyricCue(start_ms=1050, end_ms=3000, text="A"),
            LyricCue(start_ms=4080, end_ms=6000, text="B"),
        ]
        return refined, [0, 1]

    monkeypatch.setattr(timing_mod, "refine_timings", mock_refine_timings)

    res = measure_visual_timing(str(video_file), cues, str(art_dir))
    assert res["status"] == "insufficient_cues"
    assert res["metric_type"] == "refinement_delta"
    assert res["n_measured_cues"] == 2
    assert res["refinement_delta_ms"] is not None


def test_measure_visual_timing_sufficient_cues(monkeypatch, tmp_path):
    video_file = tmp_path / "dummy.mp4"
    video_file.write_bytes(b"")
    art_dir = tmp_path / "art"
    art_dir.mkdir()

    cues = [
        LyricCue(start_ms=1000, end_ms=3000, text="A"),
        LyricCue(start_ms=4000, end_ms=6000, text="B"),
        LyricCue(start_ms=7000, end_ms=9000, text="C"),
    ]

    import lyra_ocr.lyric.timing as timing_mod

    def mock_refine_timings(cues_in, *args, **kwargs):
        refined = [
            LyricCue(start_ms=1050, end_ms=3000, text="A"),
            LyricCue(start_ms=4080, end_ms=6000, text="B"),
            LyricCue(start_ms=7100, end_ms=9000, text="C"),
        ]
        return refined, [0, 1, 2]

    monkeypatch.setattr(timing_mod, "refine_timings", mock_refine_timings)

    res = measure_visual_timing(str(video_file), cues, str(art_dir))
    assert res["status"] == "ok"
    assert res["metric_type"] == "refinement_delta"
    assert res["n_measured_cues"] == 3
    assert res["refinement_delta_ms"] is not None

