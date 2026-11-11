"""LRC generation, cue inspection, and download endpoints."""

from __future__ import annotations

import json
import logging
import mimetypes
import re
from http import HTTPStatus
from pathlib import Path
from typing import Any

from lyra_ocr.lyric.build import build_lrc_from_artifacts, build_lrc_from_video
from lyra_ocr.lyric.export import format_lrc_timestamp, write_cues_json, write_lrc
from lyra_ocr.lyric.models import LrcMetadata, LyricCue
from lyra_ocr.web.base import _safe_resolve

logger = logging.getLogger(__name__)


class LrcAPIMixin:
    """API endpoints for LRC Studio."""

    def _handle_post_lrc_generate(self) -> None:
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
        if not isinstance(payload, dict):
            self._send_error("Invalid JSON body: expected an object", 400)
            return

        video_path = str(payload.get("video_path", "")).strip()
        art_dir_str = str(payload.get("artifact_dir", "")).strip()
        out_lrc_str = str(payload.get("out_lrc", "")).strip()

        fps = float(payload.get("fps", 1.0))
        tracker = str(payload.get("tracker", "bytetrack")).strip()
        correct_method = str(payload.get("correct_method", "dictionary")).strip()
        include_review = bool(payload.get("include_review", False))
        refine = bool(payload.get("refine", False))
        offset_ms = int(payload.get("offset_ms", 0))
        title = str(payload.get("title", "")).strip()
        artist = str(payload.get("artist", "")).strip()
        force = bool(payload.get("force", False))
        use_reference = bool(payload.get("use_reference", False))
        no_caption_fallback = bool(payload.get("no_caption_fallback", False))
        reference_path_str = str(payload.get("reference_path", "")).strip()
        source_url = str(payload.get("source_url", "")).strip()

        root = self._workspace_root()

        reference_text = ""
        if reference_path_str:
            ref_target = _safe_resolve(reference_path_str, root)
            if not ref_target or not ref_target.is_file():
                self._send_error(f"Reference lyrics file not found: {reference_path_str}", 404)
                return
            reference_text = ref_target.read_text(encoding="utf-8", errors="replace")

        try:
            # Case 1: Build from existing artifacts
            if art_dir_str:
                resolved_art = _safe_resolve(art_dir_str, root)
                if not resolved_art or not resolved_art.is_dir():
                    self._send_error(f"Artifact directory not found: {art_dir_str}", 404)
                    return
                if not out_lrc_str:
                    out_lrc = resolved_art / f"{resolved_art.name}.lrc"
                else:
                    out_lrc = _safe_resolve(out_lrc_str, root) or (root / out_lrc_str)

                result = build_lrc_from_artifacts(
                    resolved_art,
                    out_lrc,
                    include_review=include_review,
                    correct_method=correct_method,
                    offset_ms=offset_ms,
                    refine=refine,
                    title=title,
                    artist=artist,
                    video_path=video_path or None,
                    force=True,
                    reference_text=reference_text or None,
                    use_online_reference=use_reference,
                    no_caption_fallback=no_caption_fallback,
                    source_url=source_url or None,
                )
            # Case 2: Build from video path or URL
            elif video_path:
                if video_path.startswith(("http://", "https://")):
                    source_url = source_url or video_path
                    from lyra_ocr.video.downloader import download_video

                    dl_dir = root / "outputs" / "downloads"
                    dl_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("Downloading video from URL: %s", video_path)
                    dl_info = download_video(video_path, str(dl_dir), max_height=720)
                    if not isinstance(dl_info, dict) or not dl_info.get("video_path"):
                        raise RuntimeError("Video download did not return a file path")
                    resolved_vid = Path(dl_info["video_path"])
                    if not title and dl_info.get("title"):
                        title = dl_info["title"]
                else:
                    resolved_vid = _safe_resolve(video_path, root)
                    if not resolved_vid or not resolved_vid.is_file():
                        self._send_error(f"Video file not found: {video_path}", 404)
                        return
                if not out_lrc_str:
                    out_dir = root / "outputs" / resolved_vid.stem
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_lrc = out_dir / f"{resolved_vid.stem}.lrc"
                else:
                    out_lrc = _safe_resolve(out_lrc_str, root) or (root / out_lrc_str)

                result = build_lrc_from_video(
                    resolved_vid,
                    out_lrc,
                    artifacts_dir=out_lrc.parent,
                    fps=fps,
                    tracker=tracker,
                    include_review=include_review,
                    correct_method=correct_method,
                    offset_ms=offset_ms,
                    refine=refine,
                    title=title,
                    artist=artist,
                    force=force,
                    reference_text=reference_text or None,
                    use_online_reference=use_reference,
                    no_caption_fallback=no_caption_fallback,
                    source_url=source_url or None,
                )
            else:
                self._send_error("Either 'video_path' or 'artifact_dir' is required", 400)
                return

            # Read back generated files
            out_lrc_path = Path(result["out_lrc"])
            lrc_text = out_lrc_path.read_text(encoding="utf-8") if out_lrc_path.is_file() else ""
            cues_json_path = Path(result["out_cues_json"])
            cues_data = json.loads(cues_json_path.read_text(encoding="utf-8")) if cues_json_path.is_file() else {}

            self._send_json({
                "ok": True,
                "video_id": result.get("video_id", ""),
                "out_lrc": str(out_lrc_path),
                "out_cues_json": str(cues_json_path),
                "video_path": result.get("video_path", ""),
                "lrc_text": lrc_text,
                "cues": cues_data.get("cues", []),
                "metadata": cues_data.get("metadata", {}),
                "n_cues": result.get("n_cues", 0),
                "n_skipped": result.get("n_skipped", 0),
                "review_required": result.get("review_required", 0),
                "reference_source": result.get("reference_source", ""),
                "n_reference_matches": result.get("n_reference_matches", 0),
                "n_reference_gap_fills": result.get("n_reference_gap_fills", 0),
                "reference_warnings": result.get("reference_warnings", []),
                "cached": bool(result.get("cached", False)),
                "skipped": result.get("skipped", []),
            })
        except Exception as exc:
            logger.exception("Error generating LRC")
            self._send_error(f"LRC generation failed: {exc}", 500)

    def _handle_get_lrc(self, path_str: str) -> None:
        root = self._workspace_root()
        target = _safe_resolve(path_str, root)
        if not target or not target.is_file():
            self._send_error(f"File not found: {path_str}", 404)
            return

        if target.suffix == ".lrc":
            text = target.read_text(encoding="utf-8")
            cues_json = target.with_suffix(".cues.json")
            cues_data = json.loads(cues_json.read_text(encoding="utf-8")) if cues_json.is_file() else {}
            self._send_json({
                "path": str(target),
                "lrc_text": text,
                "cues": cues_data.get("cues", []),
                "metadata": cues_data.get("metadata", {}),
                "video_path": cues_data.get("video_path", ""),
            })
        elif target.name.endswith(".cues.json"):
            cues_data = json.loads(target.read_text(encoding="utf-8"))
            lrc_file = target.with_suffix("").with_suffix(".lrc")
            lrc_text = lrc_file.read_text(encoding="utf-8") if lrc_file.is_file() else ""
            self._send_json({
                "path": str(target),
                "lrc_text": lrc_text,
                "cues": cues_data.get("cues", []),
                "metadata": cues_data.get("metadata", {}),
                "video_path": cues_data.get("video_path", ""),
            })
        else:
            self._send_error("Expected .lrc or .cues.json file", 400)

    def _handle_get_media(self, path_str: str) -> None:
        """Stream a local media file with HTTP Range support for <video>."""
        root = self._workspace_root()
        target = _safe_resolve(path_str, root)
        if not target or not target.is_file():
            self._send_error(f"Media not found: {path_str}", 404)
            return

        mime_type, _ = mimetypes.guess_type(str(target))
        mime_type = mime_type or "application/octet-stream"
        size = target.stat().st_size
        start, end = 0, max(0, size - 1)
        status = 200

        range_match = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", "").strip())
        if range_match and size > 0:
            raw_start, raw_end = range_match.group(1), range_match.group(2)
            if raw_start:
                start = int(raw_start)
                if raw_end:
                    end = min(int(raw_end), size - 1)
            elif raw_end:
                start = max(0, size - int(raw_end))
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        with open(target, "rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def _handle_post_lrc_save(self) -> None:
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
        if not isinstance(payload, dict):
            self._send_error("Invalid JSON body: expected an object", 400)
            return

        lrc_path_str = str(payload.get("out_lrc", "")).strip()
        if not lrc_path_str:
            self._send_error("Missing 'out_lrc' filepath", 400)
            return

        root = self._workspace_root()
        out_lrc = _safe_resolve(lrc_path_str, root) or (root / lrc_path_str)
        lrc_text = payload.get("lrc_text")
        raw_cues = payload.get("cues")

        try:
            # If raw lrc_text provided, save directly
            if lrc_text is not None:
                out_lrc.parent.mkdir(parents=True, exist_ok=True)
                out_lrc.write_text(str(lrc_text), encoding="utf-8")

            # If cues list provided, rebuild and update .cues.json and .lrc
            if isinstance(raw_cues, list):
                cues: list[LyricCue] = []
                for c in raw_cues:
                    if not isinstance(c, dict):
                        continue
                    cues.append(
                        LyricCue(
                            start_ms=int(c.get("start_ms", 0)),
                            end_ms=int(c.get("end_ms", 0)),
                            text=str(c.get("text", "")).strip(),
                            raw_text=str(c.get("raw_text", c.get("text", ""))),
                            confidence=float(c.get("confidence", 1.0)),
                            decision=str(c.get("decision", "accept")),
                            review_required=bool(c.get("review_required", False)),
                            source_track_ids=list(c.get("source_track_ids") or []),
                            source_segment_ids=list(c.get("source_segment_ids") or []),
                        )
                    )
                meta_dict = payload.get("metadata") or {}
                meta = LrcMetadata(
                    title=str(meta_dict.get("title", "")),
                    artist=str(meta_dict.get("artist", "")),
                    album=str(meta_dict.get("album", "")),
                    by=str(meta_dict.get("by", "Lyra OCR")),
                    offset_ms=int(meta_dict.get("offset_ms", 0)),
                )
                write_lrc(out_lrc, cues, meta)
                cues_json_path = out_lrc.with_suffix(".cues.json")
                write_cues_json(
                    cues_json_path,
                    cues,
                    meta,
                    video_id=str(payload.get("video_id", out_lrc.stem)),
                    video_path=str(payload.get("video_path", "")),
                    duration_ms=int(payload.get("duration_ms", 0)),
                    tracker=str(payload.get("tracker", "")),
                    skipped=[s for s in (payload.get("skipped") or []) if isinstance(s, dict)],
                    options=dict(payload.get("options") or {}),
                )

            self._send_json({"ok": True, "saved_to": str(out_lrc)})
        except Exception as exc:
            logger.exception("Error saving LRC")
            self._send_error(f"Failed to save LRC: {exc}", 500)
