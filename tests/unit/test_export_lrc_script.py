import json
from pathlib import Path
import sys
import pytest

# Ensure repo root is on sys.path for scripts import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.export_lrc import batch_export_lrc


def test_batch_export_lrc(tmp_path: Path):
    ds_dir = tmp_path / "datasets" / "Test_DS"
    run_dir = ds_dir / "runs" / "run_test_1"
    run_dir.mkdir(parents=True)

    # Video 1 artifacts
    v1_art = run_dir / "videos" / "vid1"
    v1_art.mkdir(parents=True)
    (v1_art / "manifest.json").write_text(
        json.dumps({"video_id": "vid1", "metadata": {"title": "Title 1"}}),
        encoding="utf-8",
    )
    (v1_art / "tracks.json").write_text(
        json.dumps({
            "total_tracks": 1,
            "tracks": [{
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "label": "lyric",
                "decision": "accept",
                "features": {"dominant_text": "LINE ONE"},
            }],
        }),
        encoding="utf-8",
    )

    # Video 2 artifacts
    v2_art = run_dir / "videos" / "vid2"
    v2_art.mkdir(parents=True)
    (v2_art / "manifest.json").write_text(
        json.dumps({"video_id": "vid2", "metadata": {"title": "Title 2"}}),
        encoding="utf-8",
    )
    (v2_art / "tracks.json").write_text(
        json.dumps({
            "total_tracks": 1,
            "tracks": [{
                "track_id": 2,
                "start_ms": 2000,
                "end_ms": 4000,
                "label": "lyric",
                "decision": "accept",
                "features": {"dominant_text": "LINE TWO"},
            }],
        }),
        encoding="utf-8",
    )

    run_meta = {
        "run_id": "run_test_1",
        "dataset": "Test_DS",
        "videos": [
            {"video_id": "vid1", "artifact_dir": "videos/vid1", "status": "done"},
            {"video_id": "vid2", "artifact_dir": "videos/vid2", "status": "done"},
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(run_meta), encoding="utf-8")

    out_dir = tmp_path / "lrc_out"
    summary = batch_export_lrc(
        dataset="Test_DS",
        run_id="run_test_1",
        out_dir=out_dir,
        base_dir=tmp_path / "datasets",
    )

    assert summary["processed"] == 2
    assert summary["errors"] == 0
    assert (out_dir / "vid1.lrc").exists()
    assert (out_dir / "vid2.lrc").exists()
