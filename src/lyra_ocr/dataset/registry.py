"""Golden-compatible dataset registry.

A dataset lives at ``<workspace>/datasets/<name>/`` and its ``manifest.json``
uses the exact schema of ``golden/v1/manifest.json`` so ``lyra-ocr eval
--golden datasets/<name>``, scripts/freeze_golden.py and the labeling tooling
work without modification.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import date
from pathlib import Path
from typing import Any

from lyra_ocr.dataset.models import VideoSource
from lyra_ocr.labels import sha256_prefix
from lyra_ocr.pipeline import compute_video_id

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# Manifests are read-modify-written from both HTTP handler threads and the
# job worker thread; serialize mutations.
MANIFEST_LOCK = threading.Lock()


def datasets_root(workspace_root: Path) -> Path:
    return Path(workspace_root) / "datasets"


def dataset_dir(workspace_root: Path, name: str) -> Path:
    if not NAME_RE.match(name):
        raise ValueError(
            f"Invalid dataset name '{name}'. Allowed: letters, digits, '_', '-'; max 64 chars."
        )
    return datasets_root(workspace_root) / name


def _manifest_path(workspace_root: Path, name: str) -> Path:
    return dataset_dir(workspace_root, name) / "manifest.json"


def _read_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def create_dataset(workspace_root: Path, name: str, description: str = "") -> Path:
    d = dataset_dir(workspace_root, name)
    manifest = d / "manifest.json"
    with MANIFEST_LOCK:
        if manifest.exists():
            raise FileExistsError(f"Dataset '{name}' already exists at {d}")
        (d / "videos").mkdir(parents=True, exist_ok=True)
        (d / "artifacts").mkdir(parents=True, exist_ok=True)
        _write_manifest(
            manifest,
            {
                "version": name,
                "frozen_at": "",
                "description": description,
                "videos": [],
            },
        )
    return d


def list_datasets(workspace_root: Path) -> list[dict[str, Any]]:
    root = datasets_root(workspace_root)
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        manifest = child / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            data = _read_manifest(manifest)
        except (json.JSONDecodeError, OSError):
            continue
        videos = data.get("videos", [])
        out.append(
            {
                "name": child.name,
                "description": data.get("description", ""),
                "frozen_at": data.get("frozen_at", ""),
                "video_count": len(videos),
                "labeled_count": sum(
                    1 for v in videos if _artifact_labels_count(child, v) > 0
                ),
            }
        )
    return out


def _artifact_labels_count(ds_dir: Path, entry: dict[str, Any]) -> int:
    labels = ds_dir / entry.get("artifact_dir", "__missing__") / "labels.jsonl"
    if not labels.is_file():
        return 0
    try:
        return sum(1 for line in labels.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def get_dataset(workspace_root: Path, name: str) -> dict[str, Any]:
    manifest = _manifest_path(workspace_root, name)
    if not manifest.exists():
        raise FileNotFoundError(f"Dataset '{name}' not found ({manifest})")
    data = _read_manifest(manifest)
    ds_dir = manifest.parent
    enriched = []
    for entry in data.get("videos", []):
        art_dir = ds_dir / entry.get("artifact_dir", "")
        tracks = art_dir / "tracks.json"
        n_tracks = 0
        if tracks.is_file():
            try:
                n_tracks = _read_manifest(tracks).get("total_tracks", 0)
            except (json.JSONDecodeError, OSError):
                n_tracks = 0
        e = dict(entry)
        e["total_tracks"] = n_tracks
        e["labeled_tracks"] = _artifact_labels_count(ds_dir, entry)
        enriched.append(e)
    data["videos"] = enriched
    data["name"] = name
    return data


def sync_entry(
    workspace_root: Path,
    name: str,
    *,
    video_id: str,
    artifact_dir: str,
    title: str = "",
    channel: str = "",
    language: str = "en",
    split: str = "test",
) -> dict[str, Any]:
    """Upsert a manifest entry for a completed video; hashes are filled by freeze."""
    manifest = _manifest_path(workspace_root, name)
    with MANIFEST_LOCK:
        data = _read_manifest(manifest)
        entry = {
            "video_id": video_id,
            "artifact_dir": artifact_dir,
            "language": language,
            "split": split,
            "artifact_hash": "",
            "title": title,
            "channel": channel,
            "config_hash": "",
        }
        videos = data.get("videos", [])
        for i, v in enumerate(videos):
            if v.get("video_id") == video_id:
                entry["split"] = v.get("split", split)
                entry["language"] = v.get("language", language)
                videos[i] = entry
                break
        else:
            videos.append(entry)
        data["videos"] = videos
        _write_manifest(manifest, data)
    return entry


def set_split(workspace_root: Path, name: str, video_id: str, split: str) -> None:
    if split not in ("train", "test"):
        raise ValueError(f"Invalid split '{split}' (expected 'train' or 'test')")
    manifest = _manifest_path(workspace_root, name)
    data = _read_manifest(manifest)
    for v in data.get("videos", []):
        if v.get("video_id") == video_id:
            v["split"] = split
            break
    else:
        raise KeyError(f"video_id '{video_id}' not in dataset '{name}'")
    _write_manifest(manifest, data)


def freeze_dataset(workspace_root: Path, name: str) -> dict[str, Any]:
    """Recompute artifact_hash/config_hash for every entry (golden-manifest freeze)."""
    manifest = _manifest_path(workspace_root, name)
    data = _read_manifest(manifest)
    ds_dir = manifest.parent
    changed = 0
    for video in data.get("videos", []):
        art_dir = ds_dir / video.get("artifact_dir", "")
        tracks_path = art_dir / "tracks.json"
        if not tracks_path.exists():
            raise FileNotFoundError(f"missing {tracks_path} for video {video.get('video_id')}")
        new_hash = sha256_prefix(tracks_path)
        if new_hash != video.get("artifact_hash"):
            video["artifact_hash"] = new_hash
            changed += 1
        art_manifest = art_dir / "manifest.json"
        if art_manifest.exists():
            video["config_hash"] = _read_manifest(art_manifest).get("config_hash", "")
    data["frozen_at"] = date.today().isoformat()
    _write_manifest(manifest, data)
    return {"frozen_at": data["frozen_at"], "changed": changed, "videos": len(data.get("videos", []))}


def derive_video_id(source: VideoSource) -> str:
    """Stable id: raw YouTube id when available, else legacy path-hash rule."""
    if source.kind == "youtube" and source.yt_id:
        return re.sub(r"[^A-Za-z0-9_-]", "", source.yt_id)[:64]
    return compute_video_id(source.path)
