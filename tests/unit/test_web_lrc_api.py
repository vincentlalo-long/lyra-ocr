"""Unit tests for web LRC API: lossless cue metadata saving and enhanced LRC export."""

import json
from pathlib import Path
import urllib.request

import pytest

from lyra_ocr.lyric.models import CorrectionInfo, LyricCue
from lyra_ocr.web.server import create_server


def http_request(url: str, data: dict | None = None, method: str | None = None):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method=method
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_lyric_cue_from_dict_preserves_all_metadata():
    cue_dict = {
        "start_ms": 1500,
        "end_ms": 4200,
        "text": "  Sample Lyric Line  ",
        "raw_text": "Sample Lyric Line",
        "source_track_ids": [101, 102],
        "source_segment_ids": [1, 2],
        "confidence": 0.98,
        "decision": "accept",
        "review_required": False,
        "bbox": [10, 20, 300, 80],
        "x_center": 0.45,
        "y_center": 0.85,
        "correction": {
            "method": "dictionary",
            "changed": True,
            "confidence": 0.92,
            "diff": "-Line +Line",
        },
        "qa_flags": ["multiline_merged", "high_confidence"],
        "ocr_confidence": 0.96,
        "text_quality": 0.91,
        "layer_id": 1,
        "lang": "en",
        "words": [
            {"word": "Sample", "start_ms": 1500, "end_ms": 2200},
            {"word": "Lyric", "start_ms": 2300, "end_ms": 3100},
            {"word": "Line", "start_ms": 3200, "end_ms": 4200},
        ],
    }

    cue = LyricCue.from_dict(cue_dict)
    assert cue.start_ms == 1500
    assert cue.end_ms == 4200
    assert cue.text == "Sample Lyric Line"
    assert cue.raw_text == "Sample Lyric Line"
    assert cue.source_track_ids == [101, 102]
    assert cue.source_segment_ids == [1, 2]
    assert cue.confidence == 0.98
    assert cue.bbox == [10, 20, 300, 80]
    assert cue.x_center == 0.45
    assert cue.y_center == 0.85
    assert isinstance(cue.correction, CorrectionInfo)
    assert cue.correction.method == "dictionary"
    assert cue.correction.changed is True
    assert cue.qa_flags == ["multiline_merged", "high_confidence"]
    assert cue.ocr_confidence == 0.96
    assert cue.text_quality == 0.91
    assert cue.layer_id == 1
    assert cue.lang == "en"
    assert len(cue.words) == 3
    assert cue.words[0]["word"] == "Sample"

    # Alias track_ids and segment_ids fallback
    cue2 = LyricCue.from_dict({
        "start_ms": 100,
        "end_ms": 200,
        "text": "test",
        "track_ids": [42],
        "segment_ids": [7],
    })
    assert cue2.source_track_ids == [42]
    assert cue2.source_segment_ids == [7]


def test_web_save_preserves_words_qa_flags_and_enhanced_lrc(tmp_path: Path):
    server = create_server("127.0.0.1", 0, workspace_root=tmp_path)
    import threading
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        host, port = server.server_address
        base_url = f"http://{host}:{port}"

        out_lrc = tmp_path / "song.lrc"
        out_cues = tmp_path / "song.cues.json"

        # Prepopulate existing cues.json
        cues_data = {
            "video_id": "vid_test",
            "video_path": "vid_test.mp4",
            "duration_ms": 60000,
            "tracker": "bytetrack",
            "enhanced": True,
            "cues": [
                {
                    "start_ms": 1000,
                    "end_ms": 3000,
                    "text": "Hello world",
                    "bbox": [10, 20, 100, 50],
                    "x_center": 0.5,
                    "y_center": 0.8,
                    "qa_flags": ["verified"],
                    "ocr_confidence": 0.95,
                    "text_quality": 0.92,
                    "words": [
                        {"word": "Hello", "start_ms": 1000, "end_ms": 1800},
                        {"word": "world", "start_ms": 1900, "end_ms": 3000},
                    ],
                }
            ],
            "metadata": {"title": "Test Title", "artist": "Test Artist"},
        }
        out_cues.write_text(json.dumps(cues_data), encoding="utf-8")
        out_lrc.write_text("[ti:Test Title]\n[00:01.00]Hello world\n", encoding="utf-8")

        save_payload = {
            "out_lrc": str(out_lrc),
            "cues": [
                {
                    "start_ms": 1000,
                    "end_ms": 3000,
                    "text": "Hello world EDITED",
                    "bbox": [10, 20, 100, 50],
                    "x_center": 0.5,
                    "y_center": 0.8,
                    "qa_flags": ["verified", "edited"],
                    "ocr_confidence": 0.95,
                    "text_quality": 0.92,
                    "words": [
                        {"word": "Hello", "start_ms": 1000, "end_ms": 1800},
                        {"word": "world", "start_ms": 1900, "end_ms": 2500},
                        {"word": "EDITED", "start_ms": 2600, "end_ms": 3000},
                    ],
                }
            ],
            "metadata": {"title": "Test Title", "artist": "Test Artist"},
            "enhanced": True,
        }

        status, resp = http_request(f"{base_url}/api/lrc/save", data=save_payload, method="POST")
        assert status == 200
        assert resp.get("ok") is True

        # Verify cues.json retained words, bbox, qa_flags
        saved_cues = json.loads(out_cues.read_text(encoding="utf-8"))
        saved_cue = saved_cues["cues"][0]
        assert saved_cue["text"] == "Hello world EDITED"
        assert saved_cue["bbox"] == [10, 20, 100, 50]
        assert saved_cue["qa_flags"] == ["verified", "edited"]
        assert saved_cue["ocr_confidence"] == 0.95
        assert saved_cue["text_quality"] == 0.92
        assert len(saved_cue["words"]) == 3
        assert saved_cue["words"][2]["word"] == "EDITED"

        # Verify enhanced LRC tags were written
        lrc_text = out_lrc.read_text(encoding="utf-8")
        assert "<00:01.00>Hello <00:01.90>world <00:02.60>EDITED" in lrc_text
    finally:
        server.shutdown()
        server.server_close()

def test_web_save_can_disable_enhanced_lrc(tmp_path: Path):
    server = create_server("127.0.0.1", 0, workspace_root=tmp_path)
    import threading
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        host, port = server.server_address
        base_url = f"http://{host}:{port}"

        out_lrc = tmp_path / "song2.lrc"
        out_cues = tmp_path / "song2.cues.json"

        # Prepopulate existing cues.json WITH ENHANCED=TRUE
        cues_data = {
            "video_id": "vid_test2",
            "video_path": "vid_test2.mp4",
            "duration_ms": 60000,
            "tracker": "bytetrack",
            "enhanced": True,
            "options": {"enhanced": True},
            "cues": [
                {
                    "start_ms": 1000,
                    "end_ms": 3000,
                    "text": "Hello world",
                    "words": [
                        {"word": "Hello", "start_ms": 1000, "end_ms": 1800},
                        {"word": "world", "start_ms": 1900, "end_ms": 3000},
                    ],
                }
            ],
            "metadata": {"title": "Test Title", "artist": "Test Artist"},
        }
        out_cues.write_text(json.dumps(cues_data), encoding="utf-8")
        out_lrc.write_text("[ti:Test Title]\n[00:01.00]<00:01.00>Hello <00:01.90>world\n", encoding="utf-8")

        # Explicitly pass enhanced: False in save_payload
        save_payload = {
            "out_lrc": str(out_lrc),
            "cues": [
                {
                    "start_ms": 1000,
                    "end_ms": 3000,
                    "text": "Hello world",
                    "words": [
                        {"word": "Hello", "start_ms": 1000, "end_ms": 1800},
                        {"word": "world", "start_ms": 1900, "end_ms": 3000},
                    ],
                }
            ],
            "metadata": {"title": "Test Title", "artist": "Test Artist"},
            "enhanced": False,
        }

        status, resp = http_request(f"{base_url}/api/lrc/save", data=save_payload, method="POST")
        assert status == 200
        assert resp.get("ok") is True

        # Verify enhanced LRC tags were NOT written because enhanced was explicitly False
        lrc_text = out_lrc.read_text(encoding="utf-8")
        assert "<00:01.00>" not in lrc_text
        assert "[00:01.00]Hello world" in lrc_text
    finally:
        server.shutdown()
        server.server_close()
