from pathlib import Path
import pytest
import json

from lyra_ocr.lyric.select import load_artifacts, select_cues
from lyra_ocr.lyric.models import LyricCue


def test_select_cues_filtering():
    tracks = [
        {
            "track_id": 1,
            "start_ms": 1000,
            "end_ms": 3000,
            "label": "lyric",
            "decision": "accept",
            "confidence": 0.95,
            "features": {"dominant_text": "ACCEPTED LYRIC", "mean_centroid": [0.5, 0.8]},
        },
        {
            "track_id": 2,
            "start_ms": 4000,
            "end_ms": 6000,
            "label": "non_lyric",
            "decision": "reject",
            "confidence": 0.99,
            "features": {"dominant_text": "CHANNEL LOGO"},
        },
        {
            "track_id": 3,
            "start_ms": 7000,
            "end_ms": 9000,
            "label": "lyric",
            "decision": "review",
            "confidence": 0.60,
            "features": {"dominant_text": "UNCERTAIN LYRIC"},
        },
        {
            "track_id": 4,
            "start_ms": 10000,
            "end_ms": 12000,
            # Missing label and decision -> should default to non_lyric/review and be skipped safely
            "features": {"dominant_text": "NO METADATA"},
        },
    ]

    # Without include_review
    cues, skipped = select_cues(tracks, human_labels={}, include_review=False)
    assert len(cues) == 1
    assert cues[0].source_track_ids == [1]
    assert cues[0].text == "ACCEPTED LYRIC"
    assert not cues[0].review_required
    assert len(skipped) == 3
    reasons = {s["track_id"]: s["reason"] for s in skipped}
    assert reasons[2] == "non_lyric"
    assert reasons[3] == "review"

    # With include_review
    cues_review, skipped_review = select_cues(tracks, human_labels={}, include_review=True)
    assert len(cues_review) == 2
    cue_3 = next(c for c in cues_review if c.source_track_ids == [3])
    assert cue_3.review_required is True
    assert cue_3.text == "UNCERTAIN LYRIC"


def test_select_cues_human_labels_override():
    tracks = [
        {
            "track_id": 1,
            "start_ms": 1000,
            "end_ms": 3000,
            "label": "lyric",
            "decision": "accept",
            "features": {"dominant_text": "MACHINE TEXT"},
        },
        {
            "track_id": 2,
            "start_ms": 4000,
            "end_ms": 6000,
            "label": "lyric",
            "decision": "accept",
            "features": {"dominant_text": "MACHINE LYRIC"},
        },
        {
            "track_id": 3,
            "start_ms": 7000,
            "end_ms": 9000,
            "label": "lyric",
            "decision": "accept",
            "features": {"dominant_text": "TRACKING ERROR TRACK"},
        },
    ]

    human_labels = {
        "1": {
            "human_label": "lyric",
            "human_decision": "accept",
            "corrected_text": "HUMAN CORRECTED TEXT",
        },
        "2": {
            "human_label": "non_lyric",
            "human_decision": "reject",
        },
        "3": {
            "human_label": "lyric",
            "tracking_error": True,
        },
    }

    cues, skipped = select_cues(tracks, human_labels=human_labels)
    assert len(cues) == 1
    assert cues[0].source_track_ids == [1]
    assert cues[0].text == "HUMAN CORRECTED TEXT"
    assert cues[0].correction.method == "human"

    skipped_tids = [s["track_id"] for s in skipped]
    assert 2 in skipped_tids
    assert 3 in skipped_tids


def test_load_artifacts(tmp_path: Path):
    manifest = {"video_id": "test_vid", "probe": {"duration": 60}}
    tracks_data = {
        "total_tracks": 1,
        "tracks": [{"track_id": 10, "start_ms": 0, "end_ms": 2000, "label": "lyric"}],
    }
    label_line = {
        "video_id": "test_vid",
        "track_id": "10",
        "human_label": "lyric",
        "human_decision": "accept",
        "corrected_text": "TEST",
    }

    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "tracks.json").write_text(json.dumps(tracks_data), encoding="utf-8")
    (tmp_path / "labels.jsonl").write_text(json.dumps(label_line) + "\n", encoding="utf-8")

    m, tr, hl = load_artifacts(tmp_path)
    assert m["video_id"] == "test_vid"
    assert len(tr) == 1
    assert "10" in hl
    assert hl["10"]["corrected_text"] == "TEST"
