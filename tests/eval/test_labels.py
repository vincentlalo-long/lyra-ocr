"""Unit tests for human label schema, persistence, and batch CSV export/ingest."""

import csv
import json
from pathlib import Path

import pytest
from lyra_ocr.labels import (
    VALID_DECISIONS,
    VALID_LABELS,
    LabelRecord,
    append_label_record,
    export_label_batch_csv,
    get_label_status,
    ingest_label_batch_csv,
    load_labels_jsonl,
)


def test_label_record_valid_derivation():
    """Verify LabelRecord properly sets defaults, binary labels, and decisions."""
    rec = LabelRecord(video_id="vid1", track_id=1, human_label="lyric")
    assert rec.track_id == "1"
    assert rec.binary == "lyric"
    assert rec.human_decision == "accept"
    assert rec.reviewed_at != ""

    rec_channel = LabelRecord(video_id="vid1", track_id="2", human_label="channel")
    assert rec_channel.binary == "non_lyric"
    assert rec_channel.human_decision == "reject"

    rec_unknown = LabelRecord(video_id="vid1", track_id="3", human_label="unknown")
    assert rec_unknown.binary == "non_lyric"
    assert rec_unknown.human_decision == "review"


def test_label_record_invalid_label_rejection():
    """Verify LabelRecord rejects labels outside VALID_LABELS."""
    with pytest.raises(ValueError, match="Invalid human_label"):
        LabelRecord(video_id="vid1", track_id="1", human_label="invalid_label")


def test_append_and_load_latest_wins(tmp_path: Path):
    """Verify load_labels_jsonl preserves the latest record for (video_id, track_id)."""
    labels_file = tmp_path / "labels.jsonl"

    rec1 = LabelRecord(
        video_id="v1",
        track_id="t1",
        human_label="unknown",
        notes="initial uncertain label",
    )
    rec2 = LabelRecord(
        video_id="v1",
        track_id="t1",
        human_label="lyric",
        notes="corrected to lyric after audit",
    )
    append_label_record(labels_file, rec1)
    append_label_record(labels_file, rec2)

    loaded = load_labels_jsonl(labels_file)
    assert len(loaded) == 1
    assert ("v1", "t1") in loaded
    assert loaded[("v1", "t1")]["human_label"] == "lyric"
    assert loaded[("v1", "t1")]["notes"] == "corrected to lyric after audit"


def test_load_multi_video_isolation(tmp_path: Path):
    """Verify that identical track_ids across different videos are isolated."""
    labels_file = tmp_path / "labels.jsonl"

    rec_v1 = LabelRecord(video_id="video_A", track_id="track_1", human_label="lyric")
    rec_v2 = LabelRecord(video_id="video_B", track_id="track_1", human_label="channel")

    append_label_record(labels_file, rec_v1)
    append_label_record(labels_file, rec_v2)

    loaded = load_labels_jsonl(labels_file)
    assert len(loaded) == 2
    assert loaded[("video_A", "track_1")]["human_label"] == "lyric"
    assert loaded[("video_B", "track_1")]["human_label"] == "channel"


@pytest.fixture
def mock_artifacts(tmp_path: Path) -> Path:
    """Create a mock artifact directory with manifest.json and tracks.json."""
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir(parents=True)

    manifest = {
        "video_id": "test_song",
        "video_path": "path/test_song.mp4",
        "config_hash": "cfg12345",
    }
    with open(art_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    tracks_data = {
        "tracks": [
            {
                "track_id": 1,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
                "start_ms": 1000,
                "end_ms": 5000,
                "duration_ms": 4000,
                "features": {"dominant_text": "Hello world"},
            },
            {
                "track_id": 2,
                "label": "channel",
                "decision": "reject",
                "confidence": 0.88,
                "start_ms": 0,
                "end_ms": 60000,
                "duration_ms": 60000,
                "features": {"dominant_text": "V-POP Official"},
            },
        ]
    }
    with open(art_dir / "tracks.json", "w", encoding="utf-8") as f:
        json.dump(tracks_data, f)

    return art_dir


def test_export_label_batch_csv_blind_and_non_blind(mock_artifacts: Path, tmp_path: Path):
    """Test exporting CSV in blind mode (no machine predictions) vs non-blind mode."""
    blind_csv = tmp_path / "blind.csv"
    count_blind = export_label_batch_csv(mock_artifacts, blind_csv, blind=True)
    assert count_blind == 2

    with open(blind_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["machine_label"] == ""
    assert rows[0]["machine_decision"] == ""
    assert rows[0]["confidence"] == ""
    assert rows[0]["video_id"] == "test_song"

    non_blind_csv = tmp_path / "non_blind.csv"
    count_nb = export_label_batch_csv(mock_artifacts, non_blind_csv, blind=False)
    assert count_nb == 2

    with open(non_blind_csv, "r", encoding="utf-8") as f:
        rows_nb = list(csv.DictReader(f))
    assert rows_nb[0]["machine_label"] == "lyric"
    assert rows_nb[0]["machine_decision"] == "accept"
    assert rows_nb[0]["confidence"] == "0.95"


def test_ingest_populates_corrections_and_provenance(mock_artifacts: Path, tmp_path: Path):
    from lyra_ocr.labels import sha256_prefix

    csv_file = tmp_path / "with_correction.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "track_id",
                "human_label",
                "human_decision",
                "corrected_text",
                "tracking_error",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_id": "test_song",
                "track_id": "1",
                "human_label": "lyric",
                "human_decision": "accept",
                "corrected_text": "Hello, world!",
                "tracking_error": "false",
                "notes": "fixed OCR comma",
            }
        )

    stats = ingest_label_batch_csv(mock_artifacts, csv_file)
    assert stats["ingested"] == 1

    loaded = load_labels_jsonl(mock_artifacts / "labels.jsonl")
    rec = loaded[("test_song", "1")]
    assert rec["corrected_text"] == "Hello, world!"
    assert rec["artifact_hash"] == sha256_prefix(mock_artifacts / "tracks.json")
    assert rec["config_hash"] == "cfg12345"


def test_ingest_rejects_video_id_mismatch(mock_artifacts: Path, tmp_path: Path):
    csv_file = tmp_path / "wrong_video.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["video_id", "track_id", "human_label"]
        )
        writer.writeheader()
        writer.writerow(
            {"video_id": "another_video", "track_id": "1", "human_label": "lyric"}
        )

    stats = ingest_label_batch_csv(mock_artifacts, csv_file)
    assert stats["errors"] == 1
    assert stats["ingested"] == 0


def test_ingest_label_batch_csv(mock_artifacts: Path, tmp_path: Path):
    """Test batch CSV ingestion with valid labels, invalid labels, and tracking errors."""
    csv_file = tmp_path / "batch_input.csv"
    fieldnames = [
        "video_id",
        "track_id",
        "human_label",
        "human_decision",
        "tracking_error",
        "notes",
    ]
    rows = [
        {
            "video_id": "test_song",
            "track_id": "1",
            "human_label": "lyric",
            "human_decision": "accept",
            "tracking_error": "false",
            "notes": "clear lyric",
        },
        {
            "video_id": "test_song",
            "track_id": "2",
            "human_label": "channel",
            "human_decision": "reject",
            "tracking_error": "1",
            "notes": "watermark with split track",
        },
        {
            "video_id": "test_song",
            "track_id": "1",
            "human_label": "",  # Empty -> skipped
            "human_decision": "",
            "tracking_error": "",
            "notes": "",
        },
        {
            "video_id": "test_song",
            "track_id": "999",  # Non-existent track -> error
            "human_label": "lyric",
            "human_decision": "accept",
            "tracking_error": "0",
            "notes": "",
        },
        {
            "video_id": "test_song",
            "track_id": "1",
            "human_label": "invalid_category",  # Invalid enum -> error
            "human_decision": "",
            "tracking_error": "0",
            "notes": "",
        },
    ]

    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stats = ingest_label_batch_csv(mock_artifacts, csv_file, annotator="expert_test")
    assert stats["total"] == 5
    assert stats["ingested"] == 2
    assert stats["skipped_empty"] == 1
    assert stats["errors"] == 2

    # Verify status reflects the ingested labels
    status = get_label_status(mock_artifacts)
    assert status["total_tracks"] == 2
    assert status["labeled_tracks"] == 2
    assert status["coverage_percent"] == 100.0
    assert status["tracking_error_count"] == 1
    assert status["distribution"]["lyric"] == 1
    assert status["distribution"]["channel"] == 1
