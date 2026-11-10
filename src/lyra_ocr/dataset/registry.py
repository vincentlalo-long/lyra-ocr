"""Dataset registry.

A dataset lives at ``<workspace>/datasets/<name>/`` and its ``manifest.json``
is compatible with the evaluation loader, so ``lyra-ocr eval --golden
datasets/<name>``, scripts/freeze_golden.py and the labeling tooling work
without modification.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from lyra_ocr.dataset.models import RunRecord, VideoSource
from lyra_ocr.labels import sha256_prefix
from lyra_ocr.pipeline import compute_video_id

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")

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
        (d / "ground_truth").mkdir(parents=True, exist_ok=True)
        (d / "runs").mkdir(parents=True, exist_ok=True)
        (d / "artifacts").mkdir(parents=True, exist_ok=True)
        _write_manifest(
            manifest,
            {
                "version": name,
                "schema_version": 2,
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
                "ground_truth_count": sum(
                    1 for v in videos if _ground_truth_count(child, v.get("video_id", "")) > 0
                ),
                "runs_count": len(list_runs(workspace_root, child.name)),
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


def _ground_truth_count(ds_dir: Path, video_id: str) -> int:
    if not video_id:
        return 0
    path = ds_dir / "ground_truth" / f"{video_id}.jsonl"
    if not path.is_file():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
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
        e = dict(entry)
        art_dir = ds_dir / e.get("artifact_dir", "") if e.get("artifact_dir") else None
        n_tracks = 0
        if art_dir and (art_dir / "tracks.json").is_file():
            try:
                n_tracks = _read_manifest(art_dir / "tracks.json").get("total_tracks", 0)
            except (json.JSONDecodeError, OSError):
                n_tracks = 0
        e["total_tracks"] = n_tracks
        e["labeled_tracks"] = _artifact_labels_count(ds_dir, e) if e.get("artifact_dir") else 0
        e["ground_truth_segments"] = _ground_truth_count(ds_dir, str(e.get("video_id", "")))
        video_path = e.get("video_path", "")
        e["downloaded"] = bool(video_path and (ds_dir / video_path).is_file())
        enriched.append(e)
    data["videos"] = enriched
    data["name"] = name
    data["runs"] = list_runs(workspace_root, name)
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


def upsert_video(
    workspace_root: Path,
    name: str,
    *,
    video_id: str,
    video_path: str,
    source: dict[str, Any] | None = None,
    title: str = "",
    channel: str = "",
    language: str = "en",
    split: str = "test",
    duration: float | None = None,
) -> dict[str, Any]:
    """Register or update a downloaded video without tying it to a pipeline run."""
    manifest = _manifest_path(workspace_root, name)
    with MANIFEST_LOCK:
        data = _read_manifest(manifest)
        videos = data.get("videos", [])
        existing = next((v for v in videos if v.get("video_id") == video_id), {})
        entry = {
            "video_id": video_id,
            "video_path": video_path,
            "source": source or existing.get("source") or {},
            "title": title or existing.get("title", ""),
            "channel": channel or existing.get("channel", ""),
            "language": existing.get("language", language),
            "split": existing.get("split", split),
            "duration": duration if duration is not None else existing.get("duration"),
        }
        for i, v in enumerate(videos):
            if v.get("video_id") == video_id:
                videos[i] = {**v, **entry}
                break
        else:
            videos.append(entry)
        data["videos"] = videos
        _write_manifest(manifest, data)
    return entry


def get_video(workspace_root: Path, name: str, video_id: str) -> dict[str, Any]:
    data = get_dataset(workspace_root, name)
    for video in data.get("videos", []):
        if video.get("video_id") == video_id:
            return video
    raise KeyError(f"video_id '{video_id}' not in dataset '{name}'")


def runs_root(workspace_root: Path, name: str) -> Path:
    return dataset_dir(workspace_root, name) / "runs"


def run_dir(workspace_root: Path, name: str, run_id: str) -> Path:
    if not RUN_RE.match(run_id):
        raise ValueError(f"Invalid run id '{run_id}'")
    return runs_root(workspace_root, name) / run_id


def _run_path(workspace_root: Path, name: str, run_id: str) -> Path:
    return run_dir(workspace_root, name, run_id) / "run.json"


def create_run(
    workspace_root: Path,
    name: str,
    *,
    tracker: str = "greedy",
    fps: float = 1.0,
    options: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Create a tracker/config run; outputs live under runs/<run_id>/."""
    from lyra_ocr.trackers import resolve_tracker_name

    canonical_tracker = resolve_tracker_name(tracker)
    manifest_path = _manifest_path(workspace_root, name)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset '{name}' not found")
    run_id = run_id or f"{canonical_tracker}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    target = run_dir(workspace_root, name, run_id)
    run_file = target / "run.json"
    with MANIFEST_LOCK:
        if run_file.exists():
            raise FileExistsError(f"Run '{run_id}' already exists")
        (target / "videos").mkdir(parents=True, exist_ok=True)
        record = RunRecord(
            run_id=run_id,
            dataset=name,
            tracker=canonical_tracker,
            fps=float(fps),
            options=dict(options or {}),
            created_at=int(time.time()),
            videos=[],
        )
        _write_manifest(run_file, record.to_dict())
    return record.to_dict()


def get_run(workspace_root: Path, name: str, run_id: str) -> dict[str, Any]:
    path = _run_path(workspace_root, name, run_id)
    if not path.is_file():
        raise FileNotFoundError(f"Run '{run_id}' not found in dataset '{name}'")
    return _read_manifest(path)


def list_runs(workspace_root: Path, name: str) -> list[dict[str, Any]]:
    root = runs_root(workspace_root, name)
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        run_file = child / "run.json"
        if not run_file.is_file():
            continue
        try:
            run = _read_manifest(run_file)
        except (json.JSONDecodeError, OSError):
            continue
        run["video_count"] = len(run.get("videos", []))
        runs.append(run)
    return runs


def delete_run(workspace_root: Path, name: str, run_id: str) -> None:
    """Delete a run and all its output artifacts from disk."""
    target = run_dir(workspace_root, name, run_id)
    with MANIFEST_LOCK:
        if not target.exists():
            raise FileNotFoundError(f"Run '{run_id}' not found in dataset '{name}'")
        shutil.rmtree(target)


def delete_all_runs(workspace_root: Path, name: str) -> list[str]:
    """Delete all runs and their artifacts for a given dataset."""
    root = runs_root(workspace_root, name)
    deleted: list[str] = []
    with MANIFEST_LOCK:
        if not root.exists():
            return []
        for child in sorted(root.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
                deleted.append(child.name)
    return deleted


def sync_run_entry(
    workspace_root: Path,
    name: str,
    run_id: str,
    *,
    video_id: str,
    artifact_dir: str,
    title: str = "",
    channel: str = "",
    status: str = "done",
) -> dict[str, Any]:
    """Record one completed pipeline artifact inside a run."""
    path = _run_path(workspace_root, name, run_id)
    with MANIFEST_LOCK:
        run = _read_manifest(path)
        artifact_path = run_dir(workspace_root, name, run_id) / artifact_dir / "tracks.json"
        entry = {
            "video_id": video_id,
            "artifact_dir": artifact_dir,
            "artifact_hash": sha256_prefix(artifact_path) if artifact_path.is_file() else "",
            "config_hash": "",
            "title": title,
            "channel": channel,
            "status": status,
        }
        manifest_path = run_dir(workspace_root, name, run_id) / artifact_dir / "manifest.json"
        if manifest_path.is_file():
            entry["config_hash"] = _read_manifest(manifest_path).get("config_hash", "")
        videos = run.get("videos", [])
        for i, v in enumerate(videos):
            if v.get("video_id") == video_id:
                videos[i] = entry
                break
        else:
            videos.append(entry)
        run["videos"] = videos
        _write_manifest(path, run)
    return entry


def set_split(workspace_root: Path, name: str, video_id: str, split: str) -> None:
    if split not in ("train", "test"):
        raise ValueError(f"Invalid split '{split}' (expected 'train' or 'test')")
    manifest = _manifest_path(workspace_root, name)
    with MANIFEST_LOCK:
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
    with MANIFEST_LOCK:
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
