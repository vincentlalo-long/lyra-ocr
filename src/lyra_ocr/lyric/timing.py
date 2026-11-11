from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lyra_ocr.lyric.models import LyricCue
from lyra_ocr.video.frames import iter_sampled_frames

logger = logging.getLogger(__name__)


def refine_timings(
    cues: list[LyricCue],
    video_path: str,
    manifest: dict[str, Any],
    artifacts_dir: str,
    *,
    fps: float = 25.0,
    window_ms: int = 1000,
    enabled: bool = True,
    return_meta: bool = False,
) -> list[LyricCue] | tuple[list[LyricCue], set[int]]:
    if not enabled:
        return (cues, set()) if return_meta else cues

    v_path = Path(video_path)
    art_path = Path(artifacts_dir)
    if not v_path.is_file() or not art_path.is_dir():
        logger.warning(
            "refine_timings skipped: video_path '%s' or artifacts_dir '%s' not accessible",
            video_path,
            artifacts_dir,
        )
        return (cues, set()) if return_meta else cues

    frame_index_path = art_path / "frame_index.json"
    if not frame_index_path.is_file():
        logger.warning("refine_timings skipped: frame_index.json not found in '%s'", artifacts_dir)
        return (cues, set()) if return_meta else cues

    try:
        raw_fi = json.loads(frame_index_path.read_text(encoding="utf-8"))
        if isinstance(raw_fi, list):
            frame_index: dict[str, dict[str, Any]] = {
                str(item.get("frame_id")): item for item in raw_fi if isinstance(item, dict)
            }
        elif isinstance(raw_fi, dict):
            frame_index = raw_fi
        else:
            frame_index = {}
    except Exception as e:
        logger.warning("Failed to read frame_index.json: %s", e)
        return cues

    tracks_by_id: dict[int, dict[str, Any]] = {}
    tracks_path = art_path / "tracks.json"
    if tracks_path.is_file():
        try:
            raw_tr = json.loads(tracks_path.read_text(encoding="utf-8"))
            tr_list = raw_tr.get("tracks", []) if isinstance(raw_tr, dict) else raw_tr
            if not isinstance(tr_list, list):
                tr_list = []
            for tr in tr_list:
                if not isinstance(tr, dict):
                    continue
                tid = tr.get("track_id")
                if tid is not None:
                    tracks_by_id[int(tid)] = tr
        except Exception as e:
            logger.warning("Failed to load tracks.json for refine: %s", e)

    frame_interval_ms = int(1000.0 / fps)
    tmp_refine_dir = art_path / "refine_tmp"
    refined_indices: set[int] = set()

    for cue_idx, cue in enumerate(cues):
        # Find best occurrence to extract visual template
        best_occ = None
        best_conf = -1.0
        for tid in cue.source_track_ids:
            tr = tracks_by_id.get(int(tid))
            if tr:
                for occ in tr.get("occurrences", []):
                    c = float(occ.get("confidence", 0.0))
                    if c > best_conf:
                        best_conf = c
                        best_occ = occ

        if not best_occ:
            continue

        fid = str(best_occ.get("frame_id", ""))
        frame_rec = frame_index.get(fid)
        if not frame_rec:
            continue

        img_rel = frame_rec.get("image_path")
        if not img_rel and frame_rec.get("dedup_of_frame") is not None:
            dedup_fid = str(frame_rec["dedup_of_frame"])
            dedup_rec = frame_index.get(dedup_fid)
            if dedup_rec:
                img_rel = dedup_rec.get("image_path")
        if not img_rel:
            continue

        img_path = art_path / img_rel
        if not img_path.is_file():
            continue

        frame_img = cv2.imread(str(img_path))
        if frame_img is None:
            continue

        bbox = best_occ.get("bbox") or cue.bbox
        if not bbox or len(bbox) < 4:
            continue

        fh, fw = frame_img.shape[:2]
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        pad_x = int(0.10 * bw)
        pad_y = int(0.10 * bh)

        tx1 = max(0, bbox[0] - pad_x)
        ty1 = max(0, bbox[1] - pad_y)
        tx2 = min(fw, bbox[2] + pad_x)
        ty2 = min(fh, bbox[3] + pad_y)

        template = frame_img[ty1:ty2, tx1:tx2]
        if template.shape[0] < 5 or template.shape[1] < 5:
            continue

        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Dense frame sampling around [start_ms - window_ms, end_ms + window_ms]
        start_sec = max(0.0, (cue.start_ms - window_ms) / 1000.0)
        duration_scan = (cue.end_ms - cue.start_ms + 2 * window_ms) / 1000.0
        max_frames = int(math.ceil(duration_scan * fps))

        scores: list[tuple[int, float]] = []
        try:
            for packet in iter_sampled_frames(
                v_path,
                tmp_refine_dir,
                fps=fps,
                start_sec=start_sec,
                max_frames=max_frames,
                save_frames=False,
            ):
                bgr = packet.image_bgr
                if bgr is None and packet.image_path and Path(packet.image_path).is_file():
                    bgr = cv2.imread(packet.image_path)
                if bgr is None:
                    continue

                rfh, rfw = bgr.shape[:2]
                rx1 = max(0, tx1 - pad_x)
                ry1 = max(0, ty1 - pad_y)
                rx2 = min(rfw, tx2 + pad_x)
                ry2 = min(rfh, ty2 + pad_y)

                roi = bgr[ry1:ry2, rx1:rx2]
                if roi.shape[0] < template_gray.shape[0] or roi.shape[1] < template_gray.shape[1]:
                    continue

                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                score = float(res.max())
                scores.append((packet.timestamp_ms, score))

            if scores:
                max_score = max(s for _, s in scores)
                threshold = max(0.70, 0.90 * max_score)
                matching = [t for t, s in scores if s >= threshold]
                if matching:
                    visual_start = matching[0]
                    visual_end = matching[-1]
                    cue.start_ms = visual_start
                    cue.end_ms = max(visual_start + 300, visual_end + frame_interval_ms)
                    refined_indices.add(cue_idx)
        except Exception as e:
            logger.warning("Error during dense scan for cue '%s': %s", cue.text, e)

    return (cues, refined_indices) if return_meta else cues
