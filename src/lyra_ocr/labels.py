"""Human label schema, persistence, validation, and batch CSV export/ingest."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


VALID_LABELS = ("lyric", "non_lyric")

VALID_DECISIONS = ("accept", "reject", "review")

# Read-time migration map for artifacts/labels produced by the old 6-class
# taxonomy. Only ``lyric`` survives as a positive label; every distractor
# subtype collapses to ``non_lyric``.
LEGACY_LABEL_MAP = {
    "lyric": "lyric",
    "non_lyric": "non_lyric",
    "channel": "non_lyric",
    "song_title": "non_lyric",
    "supplementary": "non_lyric",
    "scene_text": "non_lyric",
    "unknown": "non_lyric",
}


def normalize_label(value: str | None) -> str:
    """Map any legacy taxonomy value onto the binary lyric/non-lyric schema."""
    return LEGACY_LABEL_MAP.get(str(value or "").strip().lower(), "non_lyric")


def sha256_prefix(path: str | Path, length: int = 12) -> str:
    """Compute a short SHA-256 digest of a file for artifact provenance checks."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:length]


@dataclass
class LabelRecord:
    video_id: str
    track_id: str | int
    human_label: str
    tracking_error: bool = False
    human_decision: str = ""
    corrected_text: Optional[str] = None
    notes: str = ""
    annotator: str = "human"
    reviewed_at: str = ""
    model_label: Optional[str] = None
    model_confidence: Optional[float] = None
    artifact_hash: Optional[str] = None
    config_hash: Optional[str] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        self.track_id = str(self.track_id)
        hl = LEGACY_LABEL_MAP.get(self.human_label.strip().lower())
        if hl is None:
            raise ValueError(
                f"Invalid human_label '{self.human_label}'. Allowed: {VALID_LABELS}"
            )
        self.human_label = hl

        # Derive human_decision if empty
        if not self.human_decision:
            self.human_decision = "accept" if self.human_label == "lyric" else "reject"
        elif self.human_decision not in VALID_DECISIONS:
            raise ValueError(
                f"Invalid human_decision '{self.human_decision}'. Allowed: {VALID_DECISIONS}"
            )

        # Timestamp in ISO 8601 UTC
        if not self.reviewed_at:
            self.reviewed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_labels_jsonl(labels_path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load latest records from append-only labels.jsonl indexed by (video_id, track_id)."""
    p = Path(labels_path)
    if not p.exists():
        return {}

    records: dict[tuple[str, str], dict[str, Any]] = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                vid = str(row.get("video_id", ""))
                tid = str(row.get("track_id", ""))
                if vid and tid:
                    if "human_label" in row:
                        row["human_label"] = normalize_label(row["human_label"])
                    row.pop("binary", None)
                    records[(vid, tid)] = row
            except json.JSONDecodeError:
                continue
    return records


def append_label_record(labels_path: str | Path, record: LabelRecord | dict[str, Any]) -> None:
    """Append a validated label record to the append-only labels.jsonl store."""
    if isinstance(record, dict):
        rec_obj = LabelRecord(**record)
    else:
        rec_obj = record

    p = Path(labels_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec_obj.to_dict(), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def export_label_batch_csv(
    artifacts_dir: str | Path,
    out_csv: str | Path,
    blind: bool = True,
) -> int:
    """Export review/label CSV for human inspection or offline spreadsheet labeling.

    Args:
        artifacts_dir: Path to directory containing tracks.json and manifest.json.
        out_csv: Output CSV filepath.
        blind: When True, omits machine predictions to prevent reviewer bias.

    Returns:
        Number of candidate rows exported.
    """
    art_dir = Path(artifacts_dir)
    tracks_path = art_dir / "tracks.json"
    manifest_path = art_dir / "manifest.json"
    labels_jsonl_path = art_dir / "labels.jsonl"

    if not tracks_path.exists():
        raise FileNotFoundError(f"Missing tracks.json in {art_dir}")

    with open(tracks_path, "r", encoding="utf-8") as f:
        tracks_data = json.load(f)
    tracks = tracks_data.get("tracks", [])

    video_id = "unknown_video"
    config_hash = ""
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            video_id = manifest_data.get("video_id") or manifest_data.get("video_path", "unknown_video")
            config_hash = manifest_data.get("config_hash", "")

    existing_labels = load_labels_jsonl(labels_jsonl_path) if labels_jsonl_path.exists() else {}

    fieldnames = [
        "video_id",
        "track_id",
        "start_ms",
        "end_ms",
        "duration_ms",
        "dominant_text",
        "machine_label",
        "machine_decision",
        "confidence",
        "human_label",
        "human_decision",
        "corrected_text",
        "tracking_error",
        "notes",
        "annotator",
        "reviewed_at",
    ]

    rows = []
    for tr in tracks:
        tid_str = str(tr.get("track_id"))
        key = (video_id, tid_str)
        existing = existing_labels.get(key, {})

        m_label = normalize_label(tr.get("label", "non_lyric"))
        m_decision = tr.get("decision", "review")
        m_conf = tr.get("confidence", 0.0)

        rows.append(
            {
                "video_id": video_id,
                "track_id": tid_str,
                "start_ms": tr.get("start_ms", 0),
                "end_ms": tr.get("end_ms", 0),
                "duration_ms": tr.get("duration_ms", 0),
                "dominant_text": tr.get("features", {}).get("dominant_text", ""),
                "machine_label": "" if blind else m_label,
                "machine_decision": "" if blind else m_decision,
                "confidence": "" if blind else f"{m_conf:.2f}",
                "human_label": existing.get("human_label", ""),
                "human_decision": existing.get("human_decision", ""),
                "corrected_text": existing.get("corrected_text") or "",
                "tracking_error": "true" if existing.get("tracking_error") else "false",
                "notes": existing.get("notes", ""),
                "annotator": existing.get("annotator", ""),
                "reviewed_at": existing.get("reviewed_at", ""),
            }
        )

    out_p = Path(out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def ingest_label_batch_csv(
    artifacts_dir: str | Path,
    csv_path: str | Path,
    annotator: str = "human",
) -> dict[str, int]:
    """Ingest annotated batch CSV, validate schemas, and append records into labels.jsonl."""
    art_dir = Path(artifacts_dir)
    labels_jsonl_path = art_dir / "labels.jsonl"
    tracks_path = art_dir / "tracks.json"
    manifest_path = art_dir / "manifest.json"

    if not tracks_path.exists():
        raise FileNotFoundError(f"Missing tracks.json in {art_dir}")

    with open(tracks_path, "r", encoding="utf-8") as f:
        tracks_data = json.load(f)
    tracks_map = {str(t.get("track_id")): t for t in tracks_data.get("tracks", [])}

    manifest_video_id = "unknown_video"
    config_hash = ""
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            manifest_video_id = manifest_data.get("video_id") or manifest_data.get("video_path") or "unknown_video"
            config_hash = manifest_data.get("config_hash", "")

    artifact_hash = sha256_prefix(tracks_path)

    csv_p = Path(csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    stats = {"total": 0, "ingested": 0, "skipped_empty": 0, "errors": 0}

    with open(csv_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total"] += 1
            tid = str(row.get("track_id", "")).strip()
            hl_raw = str(row.get("human_label", "")).strip().lower()

            if not hl_raw:
                stats["skipped_empty"] += 1
                continue

            hl = LEGACY_LABEL_MAP.get(hl_raw)
            if hl is None:
                stats["errors"] += 1
                continue

            track_info = None
            if tid in tracks_map:
                track_info = tracks_map[tid]
            else:
                base_tid = tid.split(".")[0] if "." in tid else tid
                if base_tid in tracks_map:
                    track_info = tracks_map[base_tid]

            if track_info is None:
                stats["errors"] += 1
                continue

            row_video_id = str(row.get("video_id", "")).strip()
            if row_video_id and manifest_video_id != "unknown_video" and row_video_id != manifest_video_id:
                stats["errors"] += 1
                continue
            vid = manifest_video_id if manifest_video_id != "unknown_video" else (row_video_id or "unknown_video")
            err_raw = str(row.get("tracking_error", "")).strip().lower()
            is_tracking_error = err_raw in ("true", "1", "yes", "y")
            corrected_raw = str(row.get("corrected_text", "")).strip()

            try:
                rec = LabelRecord(
                    video_id=vid,
                    track_id=tid,
                    human_label=hl,
                    human_decision=str(row.get("human_decision", "")).strip(),
                    corrected_text=corrected_raw or None,
                    tracking_error=is_tracking_error,
                    notes=str(row.get("notes", "")).strip(),
                    annotator=str(row.get("annotator", "")).strip() or annotator,
                    reviewed_at=str(row.get("reviewed_at", "")).strip(),
                    model_label=track_info.get("label"),
                    model_confidence=track_info.get("confidence"),
                    artifact_hash=artifact_hash,
                    config_hash=config_hash,
                )
            except ValueError:
                stats["errors"] += 1
                continue
            append_label_record(labels_jsonl_path, rec)
            stats["ingested"] += 1

    return stats


def get_label_status(artifacts_dir: str | Path) -> dict[str, Any]:
    """Compute labeling status and coverage summary for an artifacts directory."""
    art_dir = Path(artifacts_dir)
    tracks_path = art_dir / "tracks.json"
    manifest_path = art_dir / "manifest.json"
    labels_jsonl_path = art_dir / "labels.jsonl"

    if not tracks_path.exists():
        raise FileNotFoundError(f"Missing tracks.json in {art_dir}")

    with open(tracks_path, "r", encoding="utf-8") as f:
        tracks_data = json.load(f)
    tracks = tracks_data.get("tracks", [])
    total_tracks = len(tracks)

    video_id = "unknown_video"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
            video_id = m.get("video_id") or m.get("video_path", "unknown_video")

    labels_map = load_labels_jsonl(labels_jsonl_path) if labels_jsonl_path.exists() else {}

    labeled_count = 0
    tracking_error_count = 0
    distribution: dict[str, int] = {k: 0 for k in VALID_LABELS}

    for tr in tracks:
        tid_str = str(tr.get("track_id"))
        key = (video_id, tid_str)
        lbl = labels_map.get(key)
        if lbl and lbl.get("human_label"):
            labeled_count += 1
            hl = normalize_label(lbl["human_label"])
            distribution[hl] = distribution.get(hl, 0) + 1
            if lbl.get("tracking_error"):
                tracking_error_count += 1

    coverage = (labeled_count / max(1, total_tracks)) * 100.0

    return {
        "video_id": video_id,
        "total_tracks": total_tracks,
        "labeled_tracks": labeled_count,
        "unlabeled_tracks": total_tracks - labeled_count,
        "coverage_percent": round(coverage, 2),
        "tracking_error_count": tracking_error_count,
        "distribution": distribution,
    }
