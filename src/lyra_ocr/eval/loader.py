"""Data loading and alignment of artifacts, labels, and golden manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

from lyra_ocr.eval.types import EvaluationItem, TrackLabel, TrackPrediction
from lyra_ocr.labels import load_labels_jsonl, normalize_label, sha256_prefix


def compute_file_hash(path: str | Path) -> str:
    """Compute SHA-256 hash prefix (12 chars) of a file for artifact tracking."""
    return sha256_prefix(path)


def _resolve_artifact_dir(golden_dir: Path, artifact_dir: str | Path) -> Path:
    """Resolve a manifest artifact_dir relative to CWD, repo root, then golden dir."""
    p = Path(artifact_dir)
    if p.is_absolute():
        return p

    candidates = [
        Path.cwd() / p,
        golden_dir.parent.parent / p,
        golden_dir / p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _manifest_video_id(art_dir: Path) -> str:
    manifest_file = art_dir / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        return manifest_data.get("video_id") or manifest_data.get("video_path", "unknown_video")
    return "unknown_video"


def load_artifacts_evaluation_items(
    artifacts_dir: str | Path,
    labels_path: Optional[str | Path] = None,
    *,
    master_labels: Optional[dict[tuple[str, str], dict[str, Any]]] = None,
    artifact_hash: Optional[str] = None,
    config_hash: Optional[str] = None,
) -> list[EvaluationItem]:
    """Load predictions from tracks.json and match with human ground truth labels.

    Args:
        artifacts_dir: Artifact directory containing tracks.json/manifest.json.
        labels_path: Optional explicit labels.jsonl path (defaults to artifact-local).
        master_labels: Authoritative golden labels keyed by (video_id, track_id).
        artifact_hash: Expected hash of tracks.json for provenance verification.
        config_hash: Config hash to stamp on transferred labels.
    """
    art_dir = Path(artifacts_dir)
    tracks_file = art_dir / "tracks.json"

    if not tracks_file.exists():
        raise FileNotFoundError(f"Missing tracks.json in {art_dir}")

    actual_hash = compute_file_hash(tracks_file)
    if artifact_hash and actual_hash and actual_hash != artifact_hash:
        raise ValueError(
            f"Artifact hash mismatch for {tracks_file}: golden manifest declares "
            f"{artifact_hash}, current file is {actual_hash}. Re-freeze the golden "
            f"dataset or point the manifest at the correct artifact."
        )

    with open(tracks_file, "r", encoding="utf-8") as f:
        tracks_data = json.load(f)
    tracks = tracks_data.get("tracks", [])

    video_id = _manifest_video_id(art_dir)
    if config_hash is None:
        manifest_file = art_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                config_hash = json.load(f).get("config_hash")

    lbl_file = Path(labels_path) if labels_path else art_dir / "labels.jsonl"
    labels_map = load_labels_jsonl(lbl_file) if lbl_file.exists() else {}
    if master_labels:
        # Golden labels are authoritative and must override artifact-local labels.
        labels_map = {**labels_map, **master_labels}

    items: list[EvaluationItem] = []
    for tr in tracks:
        tid_str = str(tr.get("track_id"))
        pred = TrackPrediction(
            video_id=video_id,
            track_id=tid_str,
            model_label=normalize_label(tr.get("label")),
            model_confidence=float(tr.get("confidence", 0.0)),
            model_decision=tr.get("decision", "review"),
            start_ms=int(tr.get("start_ms", 0)),
            end_ms=int(tr.get("end_ms", 0)),
            duration_ms=int(tr.get("duration_ms", 0)),
            dominant_text=tr.get("features", {}).get("dominant_text", ""),
            bbox=tr.get("bbox", [0, 0, 0, 0]),
        )

        raw_lbl = labels_map.get((video_id, tid_str))
        label_obj: Optional[TrackLabel] = None
        if raw_lbl:
            label_obj = TrackLabel(
                video_id=video_id,
                track_id=tid_str,
                human_label=normalize_label(raw_lbl.get("human_label")) if raw_lbl.get("human_label") else "",
                tracking_error=bool(raw_lbl.get("tracking_error", False)),
                corrected_text=raw_lbl.get("corrected_text"),
                notes=raw_lbl.get("notes", ""),
                annotator=raw_lbl.get("annotator", "human"),
                reviewed_at=raw_lbl.get("reviewed_at", ""),
                artifact_hash=artifact_hash or actual_hash,
                config_hash=config_hash,
            )

        items.append(EvaluationItem(pred=pred, label=label_obj))

    return items


def load_golden_dataset(
    golden_dir: str | Path,
    *,
    strict_hashes: bool = True,
) -> list[EvaluationItem]:
    """Load all video evaluation items declared in a golden dataset manifest.

    Golden labels in ``<golden_dir>/labels.jsonl`` are authoritative: they are
    merged over artifact-local labels rather than only filling missing rows.
    """
    g_dir = Path(golden_dir)
    manifest_path = g_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Golden dataset manifest not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    master_labels_path = g_dir / "labels.jsonl"
    master_labels = load_labels_jsonl(master_labels_path) if master_labels_path.exists() else {}

    all_items: list[EvaluationItem] = []
    videos = manifest_data.get("videos", [])

    for v_entry in videos:
        vid = v_entry.get("video_id")
        art_path = _resolve_artifact_dir(g_dir, v_entry.get("artifact_dir", ""))

        if not art_path.exists():
            raise FileNotFoundError(
                f"Golden manifest references missing artifact directory: "
                f"{v_entry.get('artifact_dir')} (resolved to {art_path})"
            )

        expected_hash = v_entry.get("artifact_hash")
        try:
            video_items = load_artifacts_evaluation_items(
                art_path,
                master_labels=master_labels,
                artifact_hash=expected_hash,
                config_hash=v_entry.get("config_hash"),
            )
        except ValueError as exc:
            if strict_hashes:
                raise
            print(f"WARNING: {exc}", file=sys.stderr)
            video_items = load_artifacts_evaluation_items(
                art_path,
                master_labels=master_labels,
                artifact_hash=None,
                config_hash=v_entry.get("config_hash"),
            )

        for it in video_items:
            if vid and it.pred.video_id != vid:
                raise ValueError(
                    f"Golden manifest video_id '{vid}' does not match artifact "
                    f"manifest video_id '{it.pred.video_id}' in {art_path}"
                )

        all_items.extend(video_items)

    return all_items
