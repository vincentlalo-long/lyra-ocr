"""Dataset-level ground truth segments, independent of tracker track IDs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from lyra_ocr.labels import load_labels_jsonl, normalize_label


@dataclass
class GroundTruthSegment:
    """One canonical labeled text segment for a video.

    Segments use time/text/bbox rather than a tracker-specific track id, so the
    same labels can be matched against any tracker output.
    """

    video_id: str
    segment_id: str
    start_ms: int
    end_ms: int
    human_label: str
    text: str = ""
    bbox: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    human_decision: str = "accept"
    tracking_error: bool = False
    notes: str = ""
    annotator: str = "human"
    reviewed_at: str = ""
    source_run_id: str = ""
    source_track_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["human_label"] = normalize_label(d["human_label"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GroundTruthSegment":
        values = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        values["human_label"] = normalize_label(values.get("human_label"))
        values["bbox"] = list(values.get("bbox") or [0, 0, 0, 0])
        return cls(**values)


def ground_truth_dir(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / "ground_truth"


def ground_truth_path(dataset_dir: Path, video_id: str) -> Path:
    return ground_truth_dir(dataset_dir) / f"{video_id}.jsonl"


def load_segments(dataset_dir: Path, video_id: str) -> list[GroundTruthSegment]:
    path = ground_truth_path(dataset_dir, video_id)
    if not path.is_file():
        return []
    records: list[GroundTruthSegment] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not raw.get("human_label"):
                continue
            records.append(GroundTruthSegment.from_dict(raw))
    return records


def write_segments(dataset_dir: Path, video_id: str, segments: Iterable[GroundTruthSegment]) -> Path:
    path = ground_truth_path(dataset_dir, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(json.dumps(segment.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def promote_run(dataset_dir: Path, run: dict[str, Any], video_ids: set[str] | None = None) -> dict[str, Any]:
    """Freeze one run's human labels into dataset ground truth segments.

    The run is used as the labelling reference. Later runs can be evaluated
    against these segments without re-labelling.
    """
    dataset_dir = Path(dataset_dir)
    run_dir = dataset_dir / "runs" / str(run.get("run_id", ""))
    promoted: dict[str, int] = {}
    skipped: list[str] = []

    for entry in run.get("videos", []):
        video_id = str(entry.get("video_id", ""))
        if not video_id:
            continue
        if video_ids and video_id not in video_ids:
            continue
        artifact_dir = run_dir / str(entry.get("artifact_dir", ""))
        tracks_path = artifact_dir / "tracks.json"
        labels_path = artifact_dir / "labels.jsonl"
        if not tracks_path.is_file():
            skipped.append(f"{video_id}: missing tracks.json")
            continue
        if not labels_path.is_file():
            skipped.append(f"{video_id}: no human labels")
            continue

        tracks = json.loads(tracks_path.read_text(encoding="utf-8")).get("tracks", [])
        labels = load_labels_jsonl(labels_path)
        now = datetime.now(timezone.utc).isoformat()
        segments: list[GroundTruthSegment] = []
        for track in tracks:
            track_id = str(track.get("track_id"))
            label = labels.get((video_id, track_id))
            if not label or not label.get("human_label"):
                continue
            features = track.get("features", {})
            start_ms = int(track.get("start_ms", 0))
            end_ms = int(track.get("end_ms", start_ms))
            text = str(features.get("dominant_text", "") or "")
            segment_id = f"gt_{track_id}"
            segments.append(
                GroundTruthSegment(
                    video_id=video_id,
                    segment_id=segment_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    human_label=normalize_label(label.get("human_label")),
                    text=text,
                    bbox=list(track.get("bbox") or [0, 0, 0, 0]),
                    human_decision=str(label.get("human_decision", "accept") or "accept"),
                    tracking_error=bool(label.get("tracking_error", False)),
                    notes=str(label.get("notes", "") or ""),
                    annotator=str(label.get("annotator", "human") or "human"),
                    reviewed_at=str(label.get("reviewed_at", "") or now),
                    source_run_id=str(run.get("run_id", "")),
                    source_track_id=track_id,
                )
            )

        if not segments:
            skipped.append(f"{video_id}: no labeled tracks")
            continue
        write_segments(dataset_dir, video_id, segments)
        promoted[video_id] = len(segments)

    return {"promoted": promoted, "skipped": skipped, "total": sum(promoted.values())}
