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
    # Band tolerance: cy delta <= 0.045, cx delta <= 0.25
    clusters: list[list[dict]] = []
    for info in track_info:
        best_cluster = None
        best_dist = float("inf")
        for cl in clusters:
            avg_cy = sum(item["cy"] for item in cl) / len(cl)
            avg_cx = sum(item["cx"] for item in cl) / len(cl)
            cy_diff = abs(info["cy"] - avg_cy)
            cx_diff = abs(info["cx"] - avg_cx)
            if cy_diff <= 0.045 and cx_diff <= 0.28:
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
    # Sort clusters by mean Y from top to bottom
    clusters.sort(key=lambda cl: sum(item["cy"] for item in cl) / len(cl))

    for layer_id, cl in enumerate(clusters, start=1):
        cys = [item["cy"] for item in cl]
        cy_min, cy_max = min(cys), max(cys)
        texts = [item["text"] for item in cl if item["text"]]
        n_distinct = _count_distinct_texts(texts)
        total_tracks = len(cl)

        # Distinguish dynamic lyric stream vs static watermark overlay
        # Dynamic lyric: multiple distinct lines appearing over time
        if n_distinct >= 4 and (n_distinct / max(1, total_tracks)) >= 0.30:
            kind = "lyric"
        elif total_tracks >= 3 and n_distinct <= 2:
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

        # Assign layer metadata back to each track
        for item in cl:
            t = item["track"]
            t.features["layer_id"] = layer_id
            t.features["layer_kind"] = kind
            t.features["lang"] = layer_lang

    return layers
