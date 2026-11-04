"""Tests for the golden-compatible dataset registry."""

import json
from pathlib import Path

import pytest

from lyra_ocr.dataset import registry
from lyra_ocr.dataset.models import VideoSource

GOLDEN_MANIFEST = Path(__file__).resolve().parents[2] / "golden" / "v1" / "manifest.json"

pytestmark = pytest.mark.skipif(
    not GOLDEN_MANIFEST.is_file(), reason="golden/v1 manifest not present"
)


@pytest.fixture()
def golden_entry_keys():
    data = json.loads(GOLDEN_MANIFEST.read_text(encoding="utf-8"))
    return set(data.keys()), set(data["videos"][0].keys())


def test_create_dataset_schema_matches_golden(tmp_path, golden_entry_keys):
    top_keys, entry_keys = golden_entry_keys
    registry.create_dataset(tmp_path, "demo", description="unit test")
    manifest = json.loads((tmp_path / "datasets" / "demo" / "manifest.json").read_text(encoding="utf-8"))

    assert set(manifest.keys()) == top_keys
    assert (tmp_path / "datasets" / "demo" / "videos").is_dir()
    assert (tmp_path / "datasets" / "demo" / "artifacts").is_dir()

    registry.sync_entry(tmp_path, "demo", video_id="abc", artifact_dir="artifacts/abc")
    manifest = json.loads((tmp_path / "datasets" / "demo" / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["videos"][0].keys()) == entry_keys


def test_invalid_dataset_names_rejected(tmp_path):
    for bad in ("", "_lead", "-lead", "a" * 65, "bad name", "bad/slash", "bäd"):
        with pytest.raises(ValueError):
            registry.dataset_dir(tmp_path, bad)


def test_create_duplicate_raises(tmp_path):
    registry.create_dataset(tmp_path, "dup")
    with pytest.raises(FileExistsError):
        registry.create_dataset(tmp_path, "dup")


def test_sync_entry_upsert_preserves_split_and_language(tmp_path):
    registry.create_dataset(tmp_path, "ds1")
    registry.sync_entry(tmp_path, "ds1", video_id="v1", artifact_dir="artifacts/v1", title="T1")
    registry.set_split(tmp_path, "ds1", "v1", "train")

    registry.sync_entry(tmp_path, "ds1", video_id="v1", artifact_dir="artifacts/v1", title="T1 updated")
    data = registry.get_dataset(tmp_path, "ds1")
    assert len(data["videos"]) == 1
    entry = data["videos"][0]
    assert entry["split"] == "train"
    assert entry["language"] == "en"
    assert entry["title"] == "T1 updated"


def test_set_split_validates(tmp_path):
    registry.create_dataset(tmp_path, "ds2")
    registry.sync_entry(tmp_path, "ds2", video_id="v", artifact_dir="artifacts/v")
    with pytest.raises(ValueError):
        registry.set_split(tmp_path, "ds2", "v", "validation")
    with pytest.raises(KeyError):
        registry.set_split(tmp_path, "ds2", "missing", "train")


def _write_fake_artifact(ds_dir: Path, video_id: str, config_hash: str) -> None:
    art = ds_dir / "artifacts" / video_id
    art.mkdir(parents=True, exist_ok=True)
    (art / "tracks.json").write_text(
        json.dumps({"video_id": video_id, "total_tracks": 3, "tracks": []}), encoding="utf-8"
    )
    (art / "manifest.json").write_text(
        json.dumps({"video_id": video_id, "config_hash": config_hash}), encoding="utf-8"
    )


def test_freeze_dataset_recomputes_hashes(tmp_path):
    registry.create_dataset(tmp_path, "dsf")
    _write_fake_artifact(tmp_path / "datasets" / "dsf", "v1", "cfg123")
    registry.sync_entry(tmp_path, "dsf", video_id="v1", artifact_dir="artifacts/v1")

    first = registry.freeze_dataset(tmp_path, "dsf")
    assert first["changed"] == 1
    data = registry.get_dataset(tmp_path, "dsf")
    entry = data["videos"][0]
    assert len(entry["artifact_hash"]) == 12
    assert entry["config_hash"] == "cfg123"
    assert data["frozen_at"]

    # Idempotent: second freeze finds no changes
    second = registry.freeze_dataset(tmp_path, "dsf")
    assert second["changed"] == 0


def test_freeze_missing_tracks_raises(tmp_path):
    registry.create_dataset(tmp_path, "dsx")
    registry.sync_entry(tmp_path, "dsx", video_id="gone", artifact_dir="artifacts/gone")
    with pytest.raises(FileNotFoundError):
        registry.freeze_dataset(tmp_path, "dsx")


def test_get_dataset_enriches_track_counts(tmp_path):
    ds_dir = tmp_path / "datasets" / "dse"
    registry.create_dataset(tmp_path, "dse")
    _write_fake_artifact(ds_dir, "v1", "cfg1")
    art = ds_dir / "artifacts" / "v1"
    (art / "labels.jsonl").write_text(
        '{"video_id": "v1", "track_id": "1", "human_label": "lyric"}\n'
        '{"video_id": "v1", "track_id": "2", "human_label": "channel"}\n',
        encoding="utf-8",
    )
    registry.sync_entry(tmp_path, "dse", video_id="v1", artifact_dir="artifacts/v1")

    data = registry.get_dataset(tmp_path, "dse")
    entry = data["videos"][0]
    assert entry["total_tracks"] == 3
    assert entry["labeled_tracks"] == 2


def test_list_datasets_summary(tmp_path):
    registry.create_dataset(tmp_path, "aa", description="first")
    registry.create_dataset(tmp_path, "bb")
    _write_fake_artifact(tmp_path / "datasets" / "aa", "v1", "c")
    registry.sync_entry(tmp_path, "aa", video_id="v1", artifact_dir="artifacts/v1")
    (tmp_path / "datasets" / "aa" / "artifacts" / "v1" / "labels.jsonl").write_text(
        '{"video_id": "v1", "track_id": "1", "human_label": "lyric"}\n', encoding="utf-8"
    )

    listing = registry.list_datasets(tmp_path)
    assert [d["name"] for d in listing] == ["aa", "bb"]
    assert listing[0]["video_count"] == 1
    assert listing[0]["labeled_count"] == 1
    assert listing[1]["video_count"] == 0


def test_derive_video_id_youtube_and_local():
    yt = VideoSource(kind="youtube", url="https://youtu.be/dQw4w9WgXcQ", yt_id="dQw4w9WgXcQ")
    assert registry.derive_video_id(yt) == "dQw4w9WgXcQ"

    this_file = Path(__file__)
    local = VideoSource(kind="local", path=str(this_file))
    vid = registry.derive_video_id(local)
    assert vid.startswith(this_file.stem)
    assert len(vid.split("_")[-1]) == 8
