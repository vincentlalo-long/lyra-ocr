import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_lrc import evaluate_run


def test_evaluate_run_basic(tmp_path: Path):
    ds_dir = tmp_path / "datasets" / "Eval_DS"
    run_dir = ds_dir / "runs" / "run_eval_1"
    v1_art = run_dir / "videos" / "vid1"
    v1_art.mkdir(parents=True)

    manifest = {"video_id": "vid1", "video_path": "/fake/vid1.mp4"}
    tracks = {
        "total_tracks": 1,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "label": "lyric",
                "decision": "accept",
                "features": {"dominant_text": "I feel the night explode"},
            }
        ],
    }
    (v1_art / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (v1_art / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    run_meta = {
        "run_id": "run_eval_1",
        "dataset": "Eval_DS",
        "videos": [{"video_id": "vid1", "artifact_dir": "videos/vid1", "status": "done"}],
    }
    (run_dir / "run.json").write_text(json.dumps(run_meta), encoding="utf-8")

    # Add plain lyrics cache
    cache_dir = ds_dir / "lyrics_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "vid1.json").write_text(json.dumps(["I feel the night explode"]), encoding="utf-8")

    out_json = tmp_path / "eval_report.json"
    res = evaluate_run(
        dataset="Eval_DS",
        run_id="run_eval_1",
        out_report=out_json,
        base_dir=tmp_path / "datasets",
    )

    assert out_json.exists()
    assert res["n_scorable"] == 1
    assert res["summary"]["mean_line_recall"] == 1.0
