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
