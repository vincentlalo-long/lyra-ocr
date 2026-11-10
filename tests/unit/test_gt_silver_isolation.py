from pathlib import Path
import json
import pytest

from lyra_ocr.dataset.ground_truth import GroundTruthSegment, promote_run
from lyra_ocr.lyric.select import load_artifacts, select_cues


def test_promote_run_skips_silver_labels(tmp_path: Path):
    run_dir = tmp_path / "runs" / "test_run" / "videos" / "v1"
    run_dir.mkdir(parents=True)
    (run_dir / "tracks.json").write_text(json.dumps({"tracks": [{"track_id": 1, "start_ms": 0, "end_ms": 1000}]}))
    silver_label = {
        "video_id": "v1",
        "track_id": "1",
        "human_label": "lyric",
        "human_decision": "accept",
        "annotator": "silver_lyrics_alignment",
    }
    (run_dir / "labels.jsonl").write_text(json.dumps(silver_label) + "\n")

    run_meta = {
        "run_id": "test_run",
        "videos": [{"video_id": "v1", "artifact_dir": "videos/v1"}],
    }
    res = promote_run(tmp_path, run_meta)
    assert len(res["promoted"]) == 0
    gt_file = tmp_path / "ground_truth" / "v1.json"
    assert not gt_file.exists()


def test_select_cues_ignores_silver_labels_in_artifacts(tmp_path: Path):
    art_dir = tmp_path / "art"
    art_dir.mkdir(parents=True)
    (art_dir / "manifest.json").write_text(json.dumps({"video_id": "v1"}))
    (art_dir / "tracks.json").write_text(json.dumps({
        "tracks": [{
            "track_id": 1,
            "start_ms": 1000,
            "end_ms": 3000,
            "label": "non_lyric",
            "decision": "reject",
            "features": {"dominant_text": "WATERMARK"},
            "segments": [],
        }]
    }))
    silver_label = {
        "video_id": "v1",
        "track_id": "1",
        "human_label": "lyric",
        "human_decision": "accept",
        "annotator": "silver_lyrics_alignment",
    }
    (art_dir / "labels.jsonl").write_text(json.dumps(silver_label) + "\n")

    manifest, tracks, human_labels = load_artifacts(art_dir)
    assert "1" not in human_labels

    cues, skipped = select_cues(tracks, human_labels)
    assert len(cues) == 0
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "non_lyric"
