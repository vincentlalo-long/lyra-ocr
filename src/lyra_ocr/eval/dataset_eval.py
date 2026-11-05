"""Evaluate pipeline runs against dataset-level ground truth segments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lyra_ocr.dataset import ground_truth, registry
from lyra_ocr.eval.matching import match_predictions_to_golden
from lyra_ocr.eval.metrics import evaluate_predictions
from lyra_ocr.eval.types import EvaluationItem, EvaluationReport, TrackLabel, TrackPrediction


def load_dataset_ground_truth(dataset_dir: str | Path) -> list[EvaluationItem]:
    """Load canonical ground-truth segments declared by a dataset manifest."""
    ds_dir = Path(dataset_dir)
    manifest_path = ds_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items: list[EvaluationItem] = []
    for video in manifest.get("videos", []):
        video_id = str(video.get("video_id", ""))
        if not video_id:
            continue
        for segment in ground_truth.load_segments(ds_dir, video_id):
            pred = TrackPrediction(
                video_id=video_id,
                track_id=segment.segment_id,
                model_label="ground_truth",
                model_confidence=1.0,
                model_decision="accept",
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                duration_ms=max(0, segment.end_ms - segment.start_ms),
                dominant_text=segment.text,
                bbox=list(segment.bbox),
            )
            label = TrackLabel(
                video_id=video_id,
                track_id=segment.segment_id,
                human_label=segment.human_label,
                tracking_error=segment.tracking_error,
                notes=segment.notes,
                annotator=segment.annotator,
                reviewed_at=segment.reviewed_at,
            )
            items.append(EvaluationItem(pred=pred, label=label))
    return items


def _run_file(ds_dir: Path, run_id: str) -> Path:
    if not registry.RUN_RE.match(run_id):
        raise ValueError(f"Invalid run id '{run_id}'")
    return Path(ds_dir) / "runs" / run_id / "run.json"


def _load_run(ds_dir: Path, run_id: str) -> dict[str, Any]:
    """Read one run record straight from the dataset dir (no workspace lookup)."""
    path = _run_file(ds_dir, run_id)
    if not path.is_file():
        raise FileNotFoundError(f"Run '{run_id}' not found in dataset '{ds_dir.name}'")
    return json.loads(path.read_text(encoding="utf-8"))


def list_run_ids(dataset_dir: str | Path) -> list[str]:
    """Run ids present under ``<dataset>/runs/`` in sorted order."""
    root = Path(dataset_dir) / "runs"
    if not root.is_dir():
        return []
    return sorted(p.parent.name for p in root.glob("*/run.json"))


def load_run_predictions(dataset_dir: str | Path, run_id: str) -> list[TrackPrediction]:
    """Load all track predictions produced by one run."""
    ds_dir = Path(dataset_dir)
    run = _load_run(ds_dir, run_id)
    run_dir = ds_dir / "runs" / run_id
    predictions: list[TrackPrediction] = []
    for entry in run.get("videos", []):
        video_id = str(entry.get("video_id", ""))
        artifact_dir = run_dir / str(entry.get("artifact_dir", ""))
        tracks_path = artifact_dir / "tracks.json"
        if not video_id or not tracks_path.is_file():
            continue
        data = json.loads(tracks_path.read_text(encoding="utf-8"))
        for track in data.get("tracks", []):
            start_ms = int(track.get("start_ms", 0))
            end_ms = int(track.get("end_ms", start_ms))
            predictions.append(
                TrackPrediction(
                    video_id=video_id,
                    track_id=str(track.get("track_id")),
                    model_label=str(track.get("label", "non_lyric")),
                    model_confidence=float(track.get("confidence", 0.0)),
                    model_decision=str(track.get("decision", "review")),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=int(track.get("duration_ms", max(0, end_ms - start_ms))),
                    dominant_text=str(track.get("features", {}).get("dominant_text", "") or ""),
                    bbox=list(track.get("bbox") or [0, 0, 0, 0]),
                )
            )
    return predictions


def evaluate_run(
    dataset_dir: str | Path,
    run_id: str,
    *,
    gate_min_coverage: float = 1.0,
) -> tuple[EvaluationReport, list[EvaluationItem]]:
    """Match one run's predictions to canonical GT and evaluate them."""
    gt_items = load_dataset_ground_truth(dataset_dir)
    if not gt_items:
        raise ValueError("Dataset has no ground truth segments. Promote a labeled run first.")
    predictions = load_run_predictions(dataset_dir, run_id)
    paired, _, _ = match_predictions_to_golden(predictions, gt_items)
    report = evaluate_predictions(paired, gate_min_coverage=gate_min_coverage)
    return report, paired


def evaluate_all_runs(
    dataset_dir: str | Path,
    run_ids: list[str] | None = None,
    *,
    gate_min_coverage: float = 0.0,
) -> list[dict[str, Any]]:
    """Evaluate every (or selected) run against the same ground truth."""
    ds_dir = Path(dataset_dir)
    available = list_run_ids(ds_dir)
    selected = run_ids or available
    results: list[dict[str, Any]] = []
    for run_id in selected:
        report, paired = evaluate_run(ds_dir, run_id, gate_min_coverage=gate_min_coverage)
        results.append(
            {
                "run_id": run_id,
                "report": report,
                "paired": paired,
            }
        )
    return results
