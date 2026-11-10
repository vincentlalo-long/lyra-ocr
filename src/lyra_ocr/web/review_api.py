"""Artifact browsing, frame streaming, and human labeling endpoints."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from lyra_ocr.labels import (
    VALID_DECISIONS,
    VALID_LABELS,
    LabelRecord,
    append_label_record,
    get_label_status,
    load_labels_jsonl,
    normalize_label,
)
from lyra_ocr.web.base import _safe_resolve

_logger = logging.getLogger(__name__)


class ReviewAPIMixin:
    """Review tab: artifacts, tracks, frames, and labels."""

    def _handle_get_artifacts(self) -> None:
        """Scan known artifact roots for directories with tracks.json + manifest.json."""
        root = self._workspace_root()
        started = time.perf_counter()
        candidates = 0
        seen: set[Path] = set()
        candidate_dirs: list[Path] = []

        # Never glob the whole workspace: `.venv` (or any other tree) would be walked.
        search_roots = [root / name for name in ("outputs", "golden", "datasets", "artifacts")]

        for s_root in search_roots:
            if not s_root.is_dir():
                continue
            for item in s_root.glob("**/tracks.json"):
                candidates += 1
                art_dir = item.parent
                if not (art_dir / "manifest.json").exists():
                    continue
                resolved = art_dir.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidate_dirs.append(art_dir)

        artifacts_list = []
        for d in candidate_dirs:
            rel_path = str(d.relative_to(root)) if d.is_relative_to(root) else str(d)
            try:
                status = get_label_status(d)
                manifest_path = d / "manifest.json"
                v_title = ""
                v_channel = ""
                if manifest_path.exists():
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                        meta = m_data.get("metadata", {})
                        v_title = meta.get("title", "")
                        v_channel = meta.get("channel", "")

                artifacts_list.append({
                    "name": d.name,
                    "path": rel_path,
                    "video_id": status.get("video_id", d.name),
                    "title": v_title,
                    "channel": v_channel,
                    "total_tracks": status.get("total_tracks", 0),
                    "labeled_tracks": status.get("labeled_tracks", 0),
                    "coverage_percent": status.get("coverage_percent", 0.0),
                })
            except Exception:
                continue

        artifacts_list.sort(key=lambda x: x["name"])
        _logger.debug(
            "artifacts scan: %d candidate dir(s), %d listed, %.1f ms",
            candidates,
            len(artifacts_list),
            (time.perf_counter() - started) * 1000.0,
        )
        self._send_json({"artifacts": artifacts_list})

    def _handle_get_data(self, art_dir_str: str) -> None:
        """Load manifest, tracks, frame_index, and latest labels for an artifact directory."""
        art_path = _safe_resolve(art_dir_str, self._workspace_root())
        if not art_path or not art_path.exists():
            self._send_error(f"Artifact directory not found: {art_dir_str}", 404)
            return

        tracks_file = art_path / "tracks.json"
        manifest_file = art_path / "manifest.json"
        frames_file = art_path / "frame_index.json"
        labels_file = art_path / "labels.jsonl"

        if not tracks_file.exists():
            self._send_error("tracks.json missing in artifacts directory", 400)
            return

        with open(tracks_file, "r", encoding="utf-8") as f:
            tracks_data = json.load(f)

        manifest_data = {}
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

        frames_data = []
        if frames_file.exists():
            with open(frames_file, "r", encoding="utf-8") as f:
                raw_f = json.load(f)
                if isinstance(raw_f, list):
                    frames_data = raw_f
                elif isinstance(raw_f, dict):
                    frames_data = raw_f.get("frames", [])

        # Normalize frame fields for frontend
        normalized_frames = []
        for idx, fr in enumerate(frames_data):
            img_p = fr.get("image_path") or fr.get("filename") or f"frame_{idx:06d}.jpg"
            normalized_frames.append({
                "frame_id": fr.get("frame_id", idx),
                "pts_ms": fr.get("timestamp_ms") or fr.get("pts_ms", 0),
                "filename": Path(img_p).name,
                "image_path": img_p,
            })

        # Load existing labels
        labels_map = {}
        if labels_file.exists():
            raw_labels = load_labels_jsonl(labels_file)
            for (vid, tid), row in raw_labels.items():
                labels_map[tid] = row

        status = get_label_status(art_path)

        tracks_payload = tracks_data.get("tracks", [])
        for tr in tracks_payload:
            tr["label"] = normalize_label(tr.get("label"))

        self._send_json({
            "artifact_dir": str(art_dir_str),
            "manifest": manifest_data,
            "tracks": tracks_payload,
            "frame_index": normalized_frames,
            "labels": labels_map,
            "status": status,
            "valid_labels": VALID_LABELS,
        })

    def _handle_get_frame(self, art_dir_str: str, filename: str) -> None:
        """Serve extracted JPEG frame image."""
        art_path = _safe_resolve(art_dir_str, self._workspace_root())
        if not art_path:
            self._send_error("Invalid artifact directory", 400)
            return

        # Sanitize filename (no path components)
        clean_name = Path(filename).name
        frame_path = art_path / "frames" / clean_name
        if not frame_path.exists():
            # Check directly in art_path
            frame_path = art_path / clean_name

        if not frame_path.exists() or not frame_path.is_file():
            self._send_error(f"Frame image not found: {filename}", 404)
            return

        with open(frame_path, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(content)

    def _handle_get_status(self, art_dir_str: str) -> None:
        art_path = _safe_resolve(art_dir_str, self._workspace_root())
        if not art_path or not art_path.exists():
            self._send_error(f"Artifact directory not found: {art_dir_str}", 404)
            return
        status = get_label_status(art_path)
        self._send_json(status)

    def _handle_post_label(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            self._send_error("Empty request body", 400)
            return

        body_raw = self.rfile.read(content_len).decode("utf-8")
        try:
            payload = json.loads(body_raw)
        except json.JSONDecodeError:
            self._send_error("Invalid JSON body", 400)
            return

        art_dir_str = payload.get("artifact_dir")
        if not art_dir_str:
            self._send_error("Missing required field: artifact_dir", 400)
            return

        art_path = _safe_resolve(art_dir_str, self._workspace_root())
        if not art_path or not art_path.exists():
            self._send_error(f"Artifact directory not found: {art_dir_str}", 404)
            return

        video_id = payload.get("video_id")
        track_id = payload.get("track_id")
        human_label_raw = payload.get("human_label")
        if not isinstance(human_label_raw, str) or not human_label_raw.strip():
            self._send_error("human_label is required and must be a string", 400)
            return
        human_label = human_label_raw.strip().lower()

        if not video_id or track_id is None:
            self._send_error("video_id and track_id are required", 400)
            return

        if human_label not in VALID_LABELS:
            self._send_error(f"Invalid human_label '{human_label}'. Allowed: {VALID_LABELS}", 400)
            return

        human_decision = str(payload.get("human_decision", "") or "").strip()
        if human_decision and human_decision not in VALID_DECISIONS:
            self._send_error(
                f"Invalid human_decision '{human_decision}'. Allowed: {VALID_DECISIONS}", 400
            )
            return

        # Validate that track_id exists in tracks.json
        tracks_file = art_path / "tracks.json"
        if not tracks_file.exists():
            self._send_error(f"tracks.json not found in artifact directory: {art_dir_str}", 404)
            return

        target_tid = str(track_id).strip()
        base_tid = target_tid.split(".", 1)[0] if "." in target_tid else target_tid

        model_label = None
        model_confidence = None
        track_found = False
        try:
            with open(tracks_file, "r", encoding="utf-8") as f:
                td = json.load(f)
                for tr in td.get("tracks", []):
                    tr_id = str(tr.get("track_id"))
                    if tr_id == target_tid or tr_id == base_tid:
                        track_found = True
                        model_label = normalize_label(tr.get("label"))
                        model_confidence = tr.get("confidence")
                        break
        except (json.JSONDecodeError, OSError) as e:
            self._send_error(f"Failed to read tracks.json: {e}", 500)
            return

        if not track_found:
            self._send_error(f"Track '{track_id}' not found in tracks.json", 404)
            return

        labels_path = art_path / "labels.jsonl"
        record = LabelRecord(
            video_id=str(video_id),
            track_id=str(track_id),
            human_label=human_label,
            tracking_error=bool(payload.get("tracking_error", False)),
            human_decision=human_decision,
            corrected_text=payload.get("corrected_text") or None,
            notes=str(payload.get("notes", "")),
            annotator=str(payload.get("annotator", "web_reviewer")),
            model_label=model_label,
            model_confidence=model_confidence,
        )

        append_label_record(labels_path, record)
        updated_status = get_label_status(art_path)

        self._send_json({
            "ok": True,
            "record": record.to_dict(),
            "status": updated_status,
        })
