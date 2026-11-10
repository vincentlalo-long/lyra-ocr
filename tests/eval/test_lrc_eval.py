from pathlib import Path
import pytest

from lyra_ocr.lyric.models import LyricCue
from lyra_ocr.eval.reference import load_synced_lrc, load_plain_lyrics
from lyra_ocr.eval.lrc_eval import evaluate_text_selection, evaluate_synced_reference


def test_load_synced_lrc(tmp_path: Path):
    lrc_file = tmp_path / "test.lrc"
    content = """[ti:Song]
[00:10.50]First line
[00:20.00][00:25.00]Multi tag line
"""
    lrc_file.write_text(content, encoding="utf-8")
    parsed = load_synced_lrc(lrc_file)
    assert len(parsed) == 3
    assert parsed[0] == (10500, "First line")
    assert parsed[1] == (20000, "Multi tag line")
    assert parsed[2] == (25000, "Multi tag line")


def test_evaluate_text_selection_perfect_match():
    cues = [
        LyricCue(start_ms=1000, end_ms=3000, text="I feel the night explode"),
        LyricCue(start_ms=4000, end_ms=6000, text="When we're together"),
    ]
    reference = [
        "I feel the night explode",
        "When we're together",
    ]

    res = evaluate_text_selection(cues, reference)
    assert res["scorable"] is True
    assert res["line_recall"] == 1.0
    assert res["line_precision"] == 1.0
    assert res["line_f1"] == 1.0
    assert res["cer"] == 0.0
    assert res["wer"] == 0.0


def test_evaluate_text_selection_unscorable_version_mismatch():
    cues = [
        LyricCue(start_ms=1000, end_ms=3000, text="Completely different song lyrics"),
    ]
    reference = [
        "Unrelated text line one",
        "Unrelated text line two",
    ]

    res = evaluate_text_selection(cues, reference)
    assert res["scorable"] is False
    assert res["line_recall"] == 0.0


def test_evaluate_synced_reference():
    # 5 matching cues with a known +1000ms offset
    cues = [
        LyricCue(start_ms=11000, end_ms=13000, text="Line 1"),
        LyricCue(start_ms=21000, end_ms=23000, text="Line 2"),
        LyricCue(start_ms=31000, end_ms=33000, text="Line 3"),
        LyricCue(start_ms=41000, end_ms=43000, text="Line 4"),
        LyricCue(start_ms=51000, end_ms=53000, text="Line 5"),
    ]
    ref_lrc = [
        (10000, "Line 1"),
        (20000, "Line 2"),
        (30000, "Line 3"),
        (40000, "Line 4"),
        (50000, "Line 5"),
    ]

    res = evaluate_synced_reference(cues, ref_lrc)
    assert res["scorable"] is True
    # Estimated offset should be around 1000ms, scale ~ 1.0
    assert abs(res["offset_ms"] - 1000) < 50
    assert abs(res["scale"] - 1.0) < 0.05
