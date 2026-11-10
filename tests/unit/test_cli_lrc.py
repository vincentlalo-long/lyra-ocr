import json
from pathlib import Path
import pytest

from lyra_ocr.cli import main


def test_cli_lrc_with_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    manifest = {
        "video_id": "vid_cli",
        "video_path": "/tmp/vid_cli.mp4",
        "metadata": {"title": "Test Title", "channel": "Test Artist"},
        "probe": {"duration": 60.0, "width": 1280, "height": 720},
        "tracker": {"name": "bytetrack"},
    }
    tracks = {
        "total_tracks": 2,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "features": {"dominant_text": "FIRST LINE", "mean_centroid": [0.5, 0.8]},
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
            },
            {
                "track_id": 2,
                "start_ms": 5000,
                "end_ms": 7000,
                "features": {"dominant_text": "LOGO"},
                "label": "non_lyric",
                "decision": "reject",
                "confidence": 0.99,
            },
        ],
    }

    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    out_lrc = tmp_path / "output.lrc"
    exit_code = main([
        "lrc",
        str(tmp_path / "vid_cli.mp4"),
        "--artifacts",
        str(artifacts_dir),
        "-o",
        str(out_lrc),
    ])

    assert exit_code == 0
    assert out_lrc.exists()
    out = capsys.readouterr().out
    assert "Wrote 1 cues to" in out
    assert "Skipped 1 tracks" in out


def test_cli_lrc_zero_cues(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    artifacts_dir = tmp_path / "artifacts_zero"
    artifacts_dir.mkdir()

    manifest = {"video_id": "vid_zero", "probe": {"duration": 30.0}}
    tracks = {
        "total_tracks": 1,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "features": {"dominant_text": "LOGO"},
                "label": "non_lyric",
                "decision": "reject",
            }
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    out_lrc = tmp_path / "output_zero.lrc"
    exit_code = main([
        "lrc",
        "dummy.mp4",
        "--artifacts",
        str(artifacts_dir),
        "-o",
        str(out_lrc),
    ])

    assert exit_code == 2
