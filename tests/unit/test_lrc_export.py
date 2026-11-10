from pathlib import Path
import json
import pytest

from lyra_ocr.lyric.models import LyricCue, LrcMetadata, CorrectionInfo
from lyra_ocr.lyric.export import (
    format_lrc_timestamp,
    write_lrc,
    write_cues_json,
)


def test_format_lrc_timestamp():
    assert format_lrc_timestamp(0) == "[00:00.00]"
    assert format_lrc_timestamp(61234) == "[01:01.23]"
    assert format_lrc_timestamp(3600000) == "[60:00.00]"
    assert format_lrc_timestamp(-500) == "[00:00.00]"


def test_write_lrc_basic(tmp_path: Path):
    out_file = tmp_path / "test.lrc"
    metadata = LrcMetadata(
        title="Tell It to My Heart",
        artist="Taylor Dayne",
        album="",  # Should be omitted because empty
        by="Lyra OCR",
        offset_ms=0,
    )
    cues = [
        LyricCue(
            start_ms=18000,
            end_ms=23000,
            text="I FEEL THE NIGHT EXPLODE\n",
            confidence=0.99,
        ),
        LyricCue(
            start_ms=26000,
            end_ms=32000,
            text="WHEN WE'RE TOGETHER\r\n",
            confidence=0.97,
        ),
    ]

    write_lrc(out_file, cues, metadata)

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    lines = content.strip().splitlines()

    assert lines[0] == "[ti:Tell It to My Heart]"
    assert lines[1] == "[ar:Taylor Dayne]"
    assert lines[2] == "[by:Lyra OCR]"
    assert lines[3] == "[offset:0]"
    assert "[al:" not in content
    assert lines[4] == "[00:18.00]I FEEL THE NIGHT EXPLODE"
    assert lines[5] == "[00:26.00]WHEN WE'RE TOGETHER"


def test_write_lrc_sanitization_and_sorting(tmp_path: Path):
    out_file = tmp_path / "test_sort.lrc"
    metadata = LrcMetadata(title="Song [Remix]", artist="Artist [feat. B]")
    # Out of order and with brackets in metadata and non-decreasing timestamp check
    cues = [
        LyricCue(start_ms=30000, end_ms=35000, text="Line 2", y_center=0.5),
        LyricCue(start_ms=10000, end_ms=15000, text="Line 1b", y_center=0.6),
        LyricCue(start_ms=10000, end_ms=15000, text="Line 1a", y_center=0.2),
        LyricCue(start_ms=10000, end_ms=15000, text="Line 1a", y_center=0.2),  # exact duplicate
    ]

    write_lrc(out_file, cues, metadata)
    content = out_file.read_text(encoding="utf-8")
    lines = content.strip().splitlines()

    # Sanitized brackets
    assert "[ti:Song  Remix ]" in lines
    assert "[ar:Artist  feat. B ]" in lines

    # Sorted by start_ms, then y_center
    assert lines[-3] == "[00:10.00]Line 1a"
    assert lines[-2] == "[00:10.00]Line 1b"
    assert lines[-1] == "[00:30.00]Line 2"


def test_write_cues_json(tmp_path: Path):
    out_file = tmp_path / "test.cues.json"
    metadata = LrcMetadata(title="Song", artist="Artist")
    cues = [
        LyricCue(
            start_ms=1000,
            end_ms=3000,
            text="Hello world",
            raw_text="Hello world",
            source_track_ids=[1],
            confidence=0.95,
            correction=CorrectionInfo(method="none", changed=False),
        )
    ]
    skipped = [{"track_id": 2, "reason": "non_lyric", "text": "Watermark"}]

    write_cues_json(
        out_file,
        cues,
        metadata,
        video_id="vid_123",
        video_path="/path/to/vid.mp4",
        duration_ms=120000,
        tracker="bytetrack",
        skipped=skipped,
    )

    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["video_id"] == "vid_123"
    assert data["tracker"] == "bytetrack"
    assert data["n_cues"] == 1
    assert data["cues"][0]["text"] == "Hello world"
    assert data["cues"][0]["correction"]["method"] == "none"
    assert data["skipped"][0]["track_id"] == 2
