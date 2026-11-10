import json
from pathlib import Path
import pytest

from lyra_ocr.lyric.build import build_lrc_from_artifacts


def test_build_lrc_from_artifacts_success(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    manifest = {
        "video_id": "vid1",
        "video_path": "/tmp/vid1.mp4",
        "metadata": {"title": "Song", "channel": "Artist"},
        "probe": {"duration": 120.0, "width": 1280, "height": 720},
        "config": {"sample_fps": 2.0},
        "tracker": {"name": "bytetrack"},
    }
    tracks = {
        "total_tracks": 3,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 0,
                "end_ms": 120000,
                "bbox": [10, 10, 100, 40],
                "features": {"dominant_text": "Vibelyusic", "mean_centroid": [0.03, 0.03]},
                "segments": [],
                "occurrences": [],
                "label": "non_lyric",
                "decision": "reject",
            },
            {
                "track_id": 2,
                "start_ms": 18000,
                "end_ms": 23000,
                "bbox": [165, 151, 473, 180],
                "features": {"dominant_text": "I FEEL THE NIGHT EXPLODE", "mean_centroid": [0.24, 0.22]},
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": 18000,
                        "end_ms": 23000,
                        "text": "I FEEL THE NIGHT EXPLODE",
                        "confidence": 0.99,
                        "num_occurrences": 10,
                    }
                ],
                "occurrences": [],
                "label": "lyric",
                "confidence": 0.98,
                "decision": "accept",
            },
            {
                "track_id": 3,
                "start_ms": 26000,
                "end_ms": 32000,
                "bbox": [180, 179, 458, 206],
                "features": {"dominant_text": "WHEN WE'RE TOGETHER", "mean_centroid": [0.25, 0.26]},
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": 26000,
                        "end_ms": 32000,
                        "text": "WHEN WE'RE TOGETHER",
                        "confidence": 0.97,
                        "num_occurrences": 8,
                    }
                ],
                "occurrences": [],
                "label": "lyric",
                "confidence": 0.95,
                "decision": "accept",
            },
        ],
    }

    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    out_lrc = tmp_path / "song.lrc"
    res = build_lrc_from_artifacts(artifacts_dir, out_lrc)

    assert out_lrc.exists()
    assert (tmp_path / "song.cues.json").exists()
    assert res["n_cues"] == 2
    assert res["n_skipped"] == 1

    lrc_content = out_lrc.read_text(encoding="utf-8")
    expected_lines = [
        "[ti:Song]",
        "[ar:Artist]",
        "[by:Lyra OCR]",
        "[offset:0]",
        "[00:18.00]I FEEL THE NIGHT EXPLODE",
        "[00:26.00]WHEN WE'RE TOGETHER",
    ]
    for line in expected_lines:
        assert line in lrc_content


def test_build_lrc_from_artifacts_empty_tracks(tmp_path: Path):
    artifacts_dir = tmp_path / "empty_artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "tracks.json").write_text(json.dumps({"total_tracks": 0, "tracks": []}))

    with pytest.raises(ValueError, match="No tracks found"):
        build_lrc_from_artifacts(artifacts_dir, tmp_path / "song.lrc")
