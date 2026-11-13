"""Spatial text layer discovery and dynamic lyric stream vs static overlay classification."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from lyra_ocr.lang import detect_line_lang, vote_layer_lang
from lyra_ocr.track import TextTrack, string_similarity


@dataclass
class TextLayer:
    """Represents a spatially coherent text band across the video timeline."""

    layer_id: int
    kind: str  # "lyric", "overlay", "sparse"
    lang: str  # e.g. "en", "ja", "ja-Latn", "zh", "und"
    cy_range: tuple[float, float]
    track_ids: list[int]
    n_distinct_texts: int


def _count_distinct_texts(texts: Sequence[str], sim_threshold: float = 0.70) -> int:
    """Count distinct texts using string similarity clustering."""
    cleaned = [re.sub(r"\s+", " ", t.lower().strip()) for t in texts if t and t.strip()]
    if not cleaned:
        return 0

    clusters: list[str] = []
    for txt in cleaned:
        matched = False
        for rep in clusters:
            if string_similarity(txt, rep) >= sim_threshold:
                matched = True
                break
        if not matched:
            clusters.append(txt)
    return len(clusters)


def discover_layers(
    tracks: list[TextTrack],
    frame_w: int = 1280,
    frame_h: int = 720,
    video_duration_ms: int = 0,
) -> list[TextLayer]:
    """Discover spatial text layers and classify them into dynamic lyrics vs overlays.

    Mutates each track's ``features`` dict with:
    - ``layer_id``: int
    - ``layer_kind``: "lyric" | "overlay" | "sparse"
    - ``lang``: layer-voted dominant language code
    - ``line_lang``: individual line language code
    """
    if not tracks:
        return []

    # 1. Extract spatial centroids and dominant text per track
    track_info = []
    for tr in tracks:
        feat = tr.features or {}
        mc = feat.get("mean_centroid")
        if mc and len(mc) >= 2:
            cx, cy = float(mc[0]), float(mc[1])
        elif tr.bbox and len(tr.bbox) >= 4:
            cx = ((tr.bbox[0] + tr.bbox[2]) / 2.0) / max(1, frame_w)
            cy = ((tr.bbox[1] + tr.bbox[3]) / 2.0) / max(1, frame_h)
        else:
            cx, cy = 0.5, 0.5

        dom_text = feat.get("dominant_text", "")
        if not dom_text:
            if tr.segments and tr.segments[0].text:
                dom_text = tr.segments[0].text
            elif tr.occurrences and tr.occurrences[0].text_norm:
                dom_text = tr.occurrences[0].text_norm
            else:
                dom_text = ""

        line_lang = detect_line_lang(dom_text)
        tr.features["line_lang"] = line_lang
        track_info.append({"track": tr, "cx": cx, "cy": cy, "text": dom_text, "line_lang": line_lang})

    # 2. Cluster tracks into 2D spatial groups (primarily by Y band, secondarily by horizontal span)
    # Band tolerance: cy delta <= 0.030, cx delta <= 0.25
    clusters: list[list[dict]] = []
    for info in track_info:
        best_cluster = None
        best_dist = float("inf")
        for cl in clusters:
            avg_cy = sum(item["cy"] for item in cl) / len(cl)
            avg_cx = sum(item["cx"] for item in cl) / len(cl)
            cy_diff = abs(info["cy"] - avg_cy)
            cx_diff = abs(info["cx"] - avg_cx)
            if cy_diff <= 0.030 and cx_diff <= 0.25:
                dist = cy_diff * 4.0 + cx_diff
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = cl

        if best_cluster is not None:
            best_cluster.append(info)
        else:
            clusters.append([info])

    # 3. Analyze each cluster to determine layer kind and dominant language
    layers: list[TextLayer] = []
    cluster_meta: list[dict[str, Any]] = []
    # Sort clusters by mean Y from top to bottom
    clusters.sort(key=lambda cl: sum(item["cy"] for item in cl) / len(cl))

    for layer_id, cl in enumerate(clusters, start=1):
        cys = [item["cy"] for item in cl]
        cxs = [item["cx"] for item in cl]
        cy_min, cy_max = min(cys), max(cys)
        avg_cx = sum(cxs) / len(cxs)
        avg_cy = sum(cys) / len(cys)

        durs = [item["track"].end_ms - item["track"].start_ms for item in cl]
        import statistics

        med_dur = statistics.median(durs) if durs else 0
        std_cx = statistics.pstdev(cxs) if len(cxs) > 1 else 0.0

        texts = [item["text"] for item in cl if item["text"]]
        n_distinct = _count_distinct_texts(texts)
        total_tracks = len(cl)

        # Off-center check: lyrics are horizontally centered (0.30 <= avg_cx <= 0.70)
        # Left-aligned anime staff credits have avg_cx < 0.30; right credits have avg_cx > 0.70.
        # Scattered HUD overlays have wide spread (std_cx > 0.14).
        is_off_center = avg_cx < 0.30 or avg_cx > 0.70 or std_cx > 0.14

        # Reject clusters dominated by short uppercase/digit HUD codes (e.g. '00A01', 'CODE-3')
        n_code_noise = sum(
            1 for t in texts if len(t) <= 6 and bool(re.match(r"^[A-Z0-9_\-\.:# ]+$", t))
        )
        is_code_dominated = len(texts) > 0 and (n_code_noise / len(texts)) >= 0.60

        # Distinguish dynamic lyric stream vs static watermark overlay
        # Dynamic lyric: multiple distinct lines appearing over time, centered, reasonable duration
        if (
            total_tracks >= 4
            and not is_off_center
            and not is_code_dominated
            and (n_distinct >= 4 or total_tracks >= 8)
            and (n_distinct / max(1, total_tracks)) >= 0.30
            and med_dur >= 500
        ):
            kind = "lyric"
        elif total_tracks >= 3 and (n_distinct <= 2 or is_off_center or is_code_dominated or med_dur < 500):
            kind = "overlay"
        else:
            kind = "sparse"

        layer_lang = vote_layer_lang([item["line_lang"] for item in cl])

        layer = TextLayer(
            layer_id=layer_id,
            kind=kind,
            lang=layer_lang,
            cy_range=(round(cy_min, 4), round(cy_max, 4)),
            track_ids=[item["track"].track_id for item in cl],
            n_distinct_texts=n_distinct,
        )
        layers.append(layer)
        cl_tracks = [item["track"] for item in cl]
        cluster_meta.append(
            {
                "layer": layer,
                "avg_cx": avg_cx,
                "avg_cy": avg_cy,
                "is_code_dominated": is_code_dominated,
                "tracks": cl_tracks,
            }
        )

        # Assign layer metadata back to each track
        for t in cl_tracks:
            t.features["layer_id"] = layer_id
            t.features["layer_kind"] = kind
            t.features["lang"] = layer_lang

    # 4. Rescue sparse companion multi-line wrap bands adjacent to confirmed lyric layers
    lyric_metas = [m for m in cluster_meta if m["layer"].kind == "lyric"]
    for s_meta in cluster_meta:
        s_layer = s_meta["layer"]
        if s_layer.kind != "sparse":
            continue
        if not (0.30 <= s_meta["avg_cx"] <= 0.70) or s_meta["is_code_dominated"]:
            continue

        for l_meta in lyric_metas:
            l_layer = l_meta["layer"]
            if s_layer.lang != l_layer.lang and "und" not in (s_layer.lang, l_layer.lang):
                continue
            y_dist = abs(s_meta["avg_cy"] - l_meta["avg_cy"])
            if not (0.025 <= y_dist <= 0.16):
                continue

            has_overlap = False
            for ts in s_meta["tracks"]:
                s_dur = max(1, ts.end_ms - ts.start_ms)
                for tl in l_meta["tracks"]:
                    l_dur = max(1, tl.end_ms - tl.start_ms)
                    ov = min(ts.end_ms, tl.end_ms) - max(ts.start_ms, tl.start_ms)
                    if ov > 0 and (ov / min(s_dur, l_dur)) >= 0.50:
                        has_overlap = True
                        break
                if has_overlap:
                    break

            if has_overlap:
                s_layer.kind = "lyric"
                if s_layer.lang == "und" and l_layer.lang != "und":
                    s_layer.lang = l_layer.lang
                for t in s_meta["tracks"]:
                    t.features["layer_kind"] = "lyric"
                    t.features["lang"] = s_layer.lang
                break

    return layers
