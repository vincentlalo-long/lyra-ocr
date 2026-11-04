"""Tests for golden/artifact loading integrity and label precedence."""

import json
from pathlib import Path

import pytest
from lyra_ocr.eval.loader import load_golden_dataset
from lyra_ocr.labels import LabelRecord, append_label_record, sha256_prefix


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def make_artifact(root: Path, video_id: str = "vid1") -> Path:
    art = root / "outputs" / video_id
    write_json(
        art / "manifest.json",
        {"video_id": video_id, "config_hash": "cfg123"},
    )
    write_json(
        art / "tracks.json",
        {
            "tracks": [
                {
                    "track_id": 1,
                    "label": "lyric",
                    "confidence": 0.9,
                    "decision": "accept",
                    "start_ms": 0,
                    "end_ms": 4000,
                    "duration_ms": 4000,
                    "bbox": [0, 0, 100, 50],
                    "features": {"dominant_text": "hello world"},
                }
            ]
        },
    )
    # Deliberately wrong local label: golden master must take precedence.
    append_label_record(
        art / "labels.jsonl",
        LabelRecord(video_id=video_id, track_id="1", human_label="channel"),
    )
    return art


def make_golden(root: Path, art: Path, artifact_hash: str | None = None) -> Path:
    golden = root / "golden" / "v1"
    write_json(
        golden / "manifest.json",
        {
            "version": "test",
            "videos": [
                {
                    "video_id": "vid1",
                    "artifact_dir": "outputs/vid1",
                    "artifact_hash": artifact_hash or sha256_prefix(art / "tracks.json"),
                }
            ],
        },
    )
    append_label_record(
        golden / "labels.jsonl",
        LabelRecord(video_id="vid1", track_id="1", human_label="lyric"),
    )
    return golden


def test_golden_labels_override_artifact_local_labels(tmp_path: Path):
    art = make_artifact(tmp_path)
    golden = make_golden(tmp_path, art)

    items = load_golden_dataset(golden)

    assert len(items) == 1
    assert items[0].label is not None
    assert items[0].label.human_label == "lyric"
    assert items[0].label.artifact_hash == sha256_prefix(art / "tracks.json")
    assert items[0].label.config_hash == "cfg123"


def test_hash_mismatch_is_rejected(tmp_path: Path):
    art = make_artifact(tmp_path)
    golden = make_golden(tmp_path, art, artifact_hash="deadbeef1234")

    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        load_golden_dataset(golden)


def test_missing_artifact_dir_is_rejected(tmp_path: Path):
    golden = tmp_path / "golden" / "v1"
    write_json(
        golden / "manifest.json",
        {
            "videos": [
                {"video_id": "vid1", "artifact_dir": "outputs/does_not_exist"}
            ]
        },
    )

    with pytest.raises(FileNotFoundError, match="missing artifact directory"):
        load_golden_dataset(golden)


def test_video_id_mismatch_is_rejected(tmp_path: Path):
    art = make_artifact(tmp_path, video_id="vid1")
    golden = make_golden(tmp_path, art)

    # Rewrite the golden manifest to declare a different video_id.
    manifest_path = golden / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["videos"][0]["video_id"] = "other_video"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match artifact"):
        load_golden_dataset(golden)
