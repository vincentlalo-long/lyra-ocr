"""Tests for dataset ground-truth promotion and per-run evaluation."""

import json

import pytest

from lyra_ocr.dataset import ground_truth, registry
from lyra_ocr.eval.dataset_eval import (
    evaluate_all_runs,
    evaluate_run,
    list_run_ids,
    load_run_predictions,
)


def _track(track_id: str, label: str, start_ms: int, end_ms: int, text: str) -> dict:
    return {
        "track_id": track_id,
        "label": label,
        "confidence": 0.9,
        "decision": "accept",
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "bbox": [10, 10, 200, 40],
        "features": {"dominant_text": text},
    }


def _label(video_id: str, track_id: str, human_label: str) -> dict:
    return {
        "video_id": video_id,
        "track_id": track_id,
        "human_label": human_label,
        "human_decision": "accept",
        "annotator": "tester",
        "reviewed_at": "2026-11-05T00:00:00Z",
    }


def _make_run(workspace, ds_dir, run_id: str, tracks, labels) -> None:
    registry.create_run(workspace, "demo", tracker="greedy", run_id=run_id)
    art = ds_dir / "runs" / run_id / "videos" / "vid1"
    art.mkdir(parents=True, exist_ok=True)
    (art / "tracks.json").write_text(
        json.dumps({"video_id": "vid1", "total_tracks": len(tracks), "tracks": tracks}),
        encoding="utf-8",
    )
    (art / "labels.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8"
    )
    registry.sync_run_entry(
        workspace, "demo", run_id, video_id="vid1", artifact_dir="videos/vid1"
    )


@pytest.fixture()
def workspace(tmp_path):
    registry.create_dataset(tmp_path, "demo")
    ds_dir = tmp_path / "datasets" / "demo"
    registry.upsert_video(tmp_path, "demo", video_id="vid1", video_path="videos/vid1.mp4")
    _make_run(
        tmp_path,
        ds_dir,
        "label-run",
        [
            _track("1", "lyric", 1000, 5000, "hello world"),
            _track("2", "non_lyric", 6000, 9000, "channel logo"),
        ],
        [_label("vid1", "1", "lyric"), _label("vid1", "2", "non_lyric")],
    )
    return tmp_path, ds_dir


def test_promote_run_freezes_segments_and_counts(workspace):
    ws, ds_dir = workspace
    result = ground_truth.promote_run(ds_dir, registry.get_run(ws, "demo", "label-run"))

    assert result["total"] == 2
    assert result["skipped"] == []
    segments = ground_truth.load_segments(ds_dir, "vid1")
    assert [s.segment_id for s in segments] == ["gt_1", "gt_2"]
    assert [s.human_label for s in segments] == ["lyric", "non_lyric"]
    assert segments[0].text == "hello world"
    assert segments[0].source_run_id == "label-run"

    listing = registry.list_datasets(ws)
    assert listing[0]["ground_truth_count"] == 1


def test_evaluate_run_against_promoted_ground_truth(workspace):
    ws, ds_dir = workspace
    ground_truth.promote_run(ds_dir, registry.get_run(ws, "demo", "label-run"))
    _make_run(
        ws,
        ds_dir,
        "candidate-run",
        [
            _track("7", "lyric", 1000, 5000, "hello world"),
            _track("8", "non_lyric", 6000, 9000, "channel logo"),
        ],
        [],
    )

    predictions = load_run_predictions(ds_dir, "candidate-run")
    assert [p.track_id for p in predictions] == ["7", "8"]

    report, paired = evaluate_run(ds_dir, "candidate-run")
    assert report.total_tracks == 2
    assert report.evaluated_tracks == 2
    assert report.binary_lyric.f1 == 1.0
    assert report.coverage_rate == 1.0
    assert report.gate_passed
    assert len(paired) == 2


def test_evaluate_run_reports_false_drop_and_missing_tracks(workspace):
    ws, ds_dir = workspace
    ground_truth.promote_run(ds_dir, registry.get_run(ws, "demo", "label-run"))
    # Candidate rejects the lyric segment and drops the distractor entirely.
    _make_run(
        ws,
        ds_dir,
        "bad-run",
        [_track("9", "non_lyric", 1000, 5000, "hello world")],
        [],
    )

    report, _ = evaluate_run(ds_dir, "bad-run")
    assert report.false_drop.count == 1
    assert report.false_drop.total_golden_lyrics == 1
    assert report.tracking_miss_tracks == 1
    assert not report.gate_passed


def test_evaluate_all_runs_uses_dataset_runs_directory(workspace):
    ws, ds_dir = workspace
    ground_truth.promote_run(ds_dir, registry.get_run(ws, "demo", "label-run"))

    assert list_run_ids(ds_dir) == ["label-run"]
    results = evaluate_all_runs(ds_dir)
    assert [r["run_id"] for r in results] == ["label-run"]
    assert results[0]["report"].gate_passed


def test_evaluate_run_without_ground_truth_raises(workspace):
    _, ds_dir = workspace
    with pytest.raises(ValueError, match="no ground truth"):
        evaluate_run(ds_dir, "label-run")


def test_promote_skips_unlabeled_tracks(workspace):
    ws, ds_dir = workspace
    _make_run(
        ws,
        ds_dir,
        "partial-run",
        [
            _track("1", "lyric", 1000, 5000, "hello world"),
            _track("3", "lyric", 7000, 8000, "unreviewed"),
        ],
        [_label("vid1", "1", "lyric")],
    )

    result = ground_truth.promote_run(ds_dir, registry.get_run(ws, "demo", "partial-run"))
    assert result["promoted"] == {"vid1": 1}
    assert len(ground_truth.load_segments(ds_dir, "vid1")) == 1
