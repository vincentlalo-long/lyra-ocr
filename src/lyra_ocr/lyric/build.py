from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from lyra_ocr.lang import vote_layer_lang
from lyra_ocr.lyric.export import write_cues_json, write_lrc
from lyra_ocr.lyric.grouping import dedupe_and_group
from lyra_ocr.lyric.models import CorrectionConfig, LrcMetadata, ReferenceConfig
from lyra_ocr.lyric.reference import (
    ReferenceBundle,
    ReferenceLine,
    ReferenceQuery,
    align_reference,
    fetch_reference_bundle,
    fetch_reference_lines,
    fill_reference_gaps,
    load_cached_reference_bundle,
    parse_lyrics_text,
    parse_reference_query,
    parse_reference_text,
    reconcile_reference_overlaps,
    reference_cache_dir,
    reference_cache_key,
    save_cached_reference_bundle,
    select_reference_lines,
)

from lyra_ocr.lyric.select import load_artifacts, select_cues

logger = logging.getLogger(__name__)
CAPTION_CACHE_TTL_SECONDS = 3600.0
BUILD_CACHE_VERSION = 2


def _reference_signature(
    text: str | None,
    online: bool,
    title: str | None = None,
    artist: str | None = None,
    no_caption_fallback: bool = False,
) -> dict[str, Any]:
    """Cache key for the reference-text option (content-addressed)."""
    cleaned = (text or "").strip()
    if cleaned:
        return {"mode": "manual", "sha1": hashlib.sha1(cleaned.encode("utf-8")).hexdigest()}
    if online:
        q_str = f"title={title or ''}&artist={artist or ''}&no_caption={no_caption_fallback}"
        return {"mode": "online", "sha1": hashlib.sha1(q_str.encode("utf-8")).hexdigest()}
    return {"mode": "off", "sha1": ""}


def build_lrc_from_artifacts(
    artifacts_dir: str | Path,
    out_lrc: str | Path,
    *,
    video_path: str | Path | None = None,
    include_review: bool = False,
    correct_method: str = "dictionary",
    refine: bool = False,
    offset_ms: int = 0,
    title: str | None = None,
    artist: str | None = None,
    force: bool = False,
    correction_config: CorrectionConfig | None = None,
    reference_text: str | None = None,
    use_online_reference: bool = False,
    no_caption_fallback: bool = False,
    reference_config: ReferenceConfig | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    art_path = Path(artifacts_dir)
    out_path = Path(out_lrc)

    correction_cfg = correction_config or CorrectionConfig(method=correct_method)
    reference_cfg = reference_config or ReferenceConfig()

    target_options = {
        "cache_version": BUILD_CACHE_VERSION,
        "include_review": bool(include_review),
        "correct_method": str(correct_method),
        "refine": bool(refine),
        "offset_ms": int(offset_ms),
        "title": title or "",
        "artist": artist or "",
        # Gate/repair thresholds are part of the cache signature so changing a
        # default or passing a custom config invalidates cached cues.
        "correction_config": asdict(correction_cfg),
        "reference_config": asdict(reference_cfg),
        "source_url": source_url or "",
        "reference": _reference_signature(
            reference_text,
            use_online_reference,
            title,
            artist,
            no_caption_fallback,
        ),
        "no_caption_fallback": bool(no_caption_fallback),
    }

    ref_cache = reference_cache_dir(art_path) if use_online_reference else None
    cues_json_path = out_path.with_suffix(".cues.json")
    if not force and out_path.exists() and cues_json_path.exists():
        try:
            cached_cues = json.loads(cues_json_path.read_text(encoding="utf-8"))
            cached_opts = cached_cues.get("options")
            is_valid = (cached_opts == target_options)
            if video_path and cached_cues.get("video_path"):
                if str(video_path) != str(cached_cues.get("video_path")):
                    is_valid = False

            cues_mtime = cues_json_path.stat().st_mtime
            tracks_file = art_path / "tracks.json"
            if tracks_file.exists() and tracks_file.stat().st_mtime > cues_mtime:
                is_valid = False
            labels_file = art_path / "labels.jsonl"
            cached_had_labels = bool(cached_cues.get("has_labels", False))
            current_has_labels = labels_file.is_file()
            if cached_had_labels != current_has_labels:
                is_valid = False
            elif labels_file.exists() and labels_file.stat().st_mtime > cues_mtime:
                is_valid = False

            # If the user cleared reference_cache, do not silently reuse an LRC
            # that was derived from the now-deleted provider result.
            if (
                is_valid
                and use_online_reference
                and ref_cache is not None
                and not any(ref_cache.glob("*.json"))
            ):
                is_valid = False

            if is_valid:
                return {
                    "video_id": cached_cues.get("video_id", out_path.stem),
                    "out_lrc": str(out_path),
                    "out_cues_json": str(cues_json_path),
                    "n_cues": len(cached_cues.get("cues", [])),
                    "n_skipped": len(cached_cues.get("skipped", [])),
                    "review_required": sum(1 for c in cached_cues.get("cues", []) if c.get("review_required")),
                    "skipped": cached_cues.get("skipped", []),
                    "video_path": cached_cues.get("video_path", ""),
                    "n_reference_matches": cached_cues.get("n_reference_matches", 0),
                    "n_reference_gap_fills": cached_cues.get("n_reference_gap_fills", 0),
                    "reference_source": cached_cues.get("reference_source", ""),
                    "reference_warnings": cached_cues.get("reference_warnings", []),
                    "cached": True,
                }
        except Exception:
            pass

    manifest, tracks, human_labels = load_artifacts(art_path)

    if not tracks:
        raise ValueError(
            f"No tracks found in artifacts directory '{art_path}'. "
            "Please run the Lyra OCR pipeline first or provide valid artifacts."
        )

    video_id = manifest.get("video_id") or art_path.name
    eff_video_path = str(video_path) if video_path else (manifest.get("video_path") or "")
    probe = manifest.get("probe") if isinstance(manifest.get("probe"), dict) else {}
    try:
        duration_ms = int(float(probe.get("duration") or 0.0) * 1000)
    except (TypeError, ValueError):
        duration_ms = 0

    tracker_raw = manifest.get("tracker")
    tracker_name = tracker_raw.get("name", "bytetrack") if isinstance(tracker_raw, dict) else (tracker_raw or "bytetrack")

    meta = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    m_title = title if title is not None else (meta.get("title") or "")
    m_artist = artist if artist is not None else (meta.get("channel") or "")

    metadata = LrcMetadata(
        title=m_title or "",
        artist=m_artist or "",
        offset_ms=offset_ms,
    )

    reference_lines: list[str] = []
    reference_source = ""
    reference_warnings: list[str] = []
    ref_bundle: ReferenceBundle | None = None
    if reference_text and reference_text.strip():
        lines = parse_reference_text(reference_text, source="manual")
        ref_bundle = ReferenceBundle(lines=lines, source="manual", provider="manual")
        reference_lines = [l.text for l in lines]
        reference_source = "manual"
    elif use_online_reference:
        ref_q = parse_reference_query(
            title=m_title,
            channel=meta.get("channel", "") or manifest.get("channel", ""),
            duration_ms=duration_ms,
        )
        lookup_title = m_title if (title and title.strip()) else ref_q.track
        lookup_artist = m_artist if (artist and artist.strip()) else ref_q.artist
        ref_q.track = lookup_title
        ref_q.artist = lookup_artist
        ref_cache = ref_cache or reference_cache_dir(art_path)
        ref_bundle = fetch_reference_bundle(ref_q, cache_dir=ref_cache)
        if (not ref_bundle or not ref_bundle.lines) and not no_caption_fallback:
            from lyra_ocr.lyric.captions import (
                fetch_youtube_caption_bundle,
                resolve_caption_video_ref,
            )

            caption_ref = resolve_caption_video_ref(
                source_url or "",
                video_id,
                manifest,
            )
            if caption_ref:
                caption_query = ReferenceQuery(
                    track=caption_ref,
                    duration_ms=duration_ms,
                    raw_title=caption_ref,
                )
                caption_key = reference_cache_key(
                    caption_query,
                    provider="youtube_captions",
                )
                caption_bundle = load_cached_reference_bundle(
                    ref_cache,
                    caption_key,
                    max_age_seconds=CAPTION_CACHE_TTL_SECONDS,
                )
                if caption_bundle is None:
                    caption_bundle = fetch_youtube_caption_bundle(caption_ref)
                    if caption_bundle is None:
                        caption_bundle = ReferenceBundle(
                            provider="youtube",
                            warnings=["caption_lookup_empty"],
                        )
                    save_cached_reference_bundle(
                        ref_cache,
                        caption_key,
                        caption_bundle,
                        caption_query,
                    )
                if caption_bundle.lines:
                    ref_bundle = caption_bundle
                else:
                    reference_warnings.extend(caption_bundle.warnings)
            else:
                reference_warnings.append("caption_source_unresolved")
        if ref_bundle:
            reference_lines = [l.text for l in ref_bundle.lines]
            reference_source = ref_bundle.source or ("lrclib" if reference_lines else "")

    frame_width = probe.get("width") or 1280
    frame_height = probe.get("height") or 720
    cues, skipped = select_cues(
        tracks,
        human_labels,
        include_review=include_review,
        video_id=video_id,
        correct_method=correct_method,
        frame_width=frame_width,
        frame_height=frame_height,
        config=correction_cfg,
        reference_lines=reference_lines or None,
        reference_bundle=ref_bundle,
        reference_source=reference_source,
        reference_config=reference_cfg,
    )

    if refine:
        try:
            from lyra_ocr.lyric.timing import refine_timings
            cues = refine_timings(cues, eff_video_path, manifest, str(art_path), enabled=True)
        except Exception as e:
            logger.warning("Refine timings failed, falling back to coarse: %s", e)

    # Optional reference lyrics: repair cue text and fill reference gaps without touching OCR timing.
    n_reference_matches = 0
    n_reference_gap_fills = 0

    if cues and not ref_bundle and reference_lines:
        ref_bundle = ReferenceBundle(
            lines=[ReferenceLine(text=t) for t in reference_lines],
            source=reference_source,
            provider="manual" if reference_source == "manual" else "",
        )

    if cues and ref_bundle and ref_bundle.lines:
        from collections import defaultdict

        layer_groups: dict[Any, list[LyricCue]] = defaultdict(list)
        for c in cues:
            layer_groups[c.layer_id].append(c)

        processed_cues: list[LyricCue] = []
        for layer_id, layer_cues in layer_groups.items():
            layer_lang = vote_layer_lang([c.lang for c in layer_cues if c.lang])
            ref_for_layer = select_reference_lines(ref_bundle, layer_lang)

            if ref_for_layer and ref_for_layer.lines and layer_lang != "ja-Latn":
                align_res = align_reference(
                    layer_cues,
                    ref_for_layer,
                    source=reference_source,
                    config=reference_cfg,
                )
                n_reference_matches += int(align_res.replaced_count)
                layer_cues, overlap_warnings = reconcile_reference_overlaps(
                    layer_cues,
                    ref_for_layer,
                    matched_pairs=align_res.matched_pairs,
                    config=reference_cfg,
                )
                layer_cues, gap_res = fill_reference_gaps(
                    layer_cues,
                    ref_for_layer,
                    config=reference_cfg,
                    source=reference_source,
                    duration_ms=duration_ms,
                    matched_pairs=align_res.matched_pairs,
                )
                for gc in layer_cues:
                    if gc.layer_id is None and layer_id is not None:
                        gc.layer_id = layer_id
                    if not gc.lang or gc.lang == "und":
                        gc.lang = layer_lang

                n_reference_gap_fills += int(gap_res.n_filled)
                reference_warnings.extend(overlap_warnings)
                reference_warnings.extend(gap_res.warnings)

            processed_cues.extend(layer_cues)

        cues = processed_cues

    cues = dedupe_and_group(cues)

    if not n_reference_matches and not n_reference_gap_fills:
        reference_source = ""

    write_lrc(out_path, cues, metadata)

    ref_warnings = list(reference_warnings)
    if ref_bundle is not None:
        ref_warnings.extend(ref_bundle.warnings)
    cues_json_path = out_path.with_suffix(".cues.json")
    write_cues_json(
        cues_json_path,
        cues,
        metadata,
        video_id=video_id,
        video_path=eff_video_path,
        duration_ms=duration_ms,
        tracker=tracker_name,
        skipped=skipped,
        options=target_options,
        has_labels=(art_path / "labels.jsonl").is_file(),
        reference_source=reference_source,
        n_reference_matches=n_reference_matches,
        n_reference_gap_fills=n_reference_gap_fills,
        reference_warnings=ref_warnings,
    )

    return {
        "video_id": video_id,
        "out_lrc": str(out_path),
        "out_cues_json": str(cues_json_path),
        "n_cues": len(cues),
        "n_skipped": len(skipped),
        "review_required": sum(1 for c in cues if c.review_required),
        "skipped": skipped,
        "video_path": eff_video_path,
        "reference_source": reference_source,
        "n_reference_matches": n_reference_matches,
        "n_reference_gap_fills": n_reference_gap_fills,
        "reference_warnings": ref_warnings,
        "cached": False,
    }


def build_lrc_from_video(
    video_path: str | Path,
    out_lrc: str | Path,
    *,
    artifacts_dir: str | Path | None = None,
    tracker: str = "bytetrack",
    fps: float = 2.0,
    start_sec: float = 0.0,
    source_url: str = "",
    max_frames: int | None = None,
    title: str = "",
    channel: str = "",
    device: str = "auto",
    rec_batch_size: int = 8,
    dedup: bool = False,
    frame_source: str = "pipe",
    force: bool = False,
    **build_kwargs: Any,
) -> dict[str, Any]:
    v_path = Path(video_path)
    if artifacts_dir is None:
        art_dir = Path("./lyra_output") / v_path.stem
    else:
        art_dir = Path(artifacts_dir)

    tracks_json_path = art_dir / "tracks.json"
    manifest_json_path = art_dir / "manifest.json"

    needs_pipeline = force or not (tracks_json_path.exists() and manifest_json_path.exists())

    if not needs_pipeline:
        try:
            manifest_data = json.loads(manifest_json_path.read_text(encoding="utf-8"))
            if not isinstance(manifest_data, dict):
                manifest_data = {}
            # 1. Tracker check
            tracker_raw = manifest_data.get("tracker")
            manifest_tracker = tracker_raw.get("name") if isinstance(tracker_raw, dict) else None
            if manifest_tracker is not None and manifest_tracker != tracker:
                needs_pipeline = True

            # 2. Config check
            manifest_cfg = manifest_data.get("config")
            if not isinstance(manifest_cfg, dict):
                manifest_cfg = {}
            if "sample_fps" in manifest_cfg and abs(float(manifest_cfg["sample_fps"]) - float(fps)) > 1e-4:
                needs_pipeline = True
            if "start_sec" in manifest_cfg and abs(float(manifest_cfg.get("start_sec", 0.0)) - float(start_sec)) > 1e-4:
                needs_pipeline = True
            if "sample_frames" in manifest_cfg and manifest_cfg.get("sample_frames") != max_frames:
                needs_pipeline = True
            if "dedup" in manifest_cfg and bool(manifest_cfg.get("dedup")) != dedup:
                needs_pipeline = True

            # 3. Config hash check if available
            manifest_hash = manifest_data.get("config_hash")
            if manifest_hash and not needs_pipeline:
                try:
                    from lyra_ocr.pipeline import _config_hash, resolve_runtime
                    runtime = resolve_runtime(device, rec_batch_size)
                    expected_hash = _config_hash(
                        fps=fps,
                        start_sec=start_sec,
                        max_frames=max_frames,
                        tracker=tracker,
                        title=title,
                        channel=channel,
                        device=runtime.device,
                        rec_batch_size=runtime.rec_batch_size,
                        dedup=dedup,
                        frame_source=frame_source,
                    )
                    if manifest_hash != expected_hash:
                        needs_pipeline = True
                except Exception:
                    pass

            # 4. Video identity check
            manifest_vpath = manifest_data.get("video_path")
            if manifest_vpath and v_path.exists():
                try:
                    if Path(manifest_vpath).resolve() != v_path.resolve():
                        needs_pipeline = True
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Manifest config check failed: %s", e)
            needs_pipeline = True

    if needs_pipeline:
        if not v_path.exists():
            raise FileNotFoundError(f"Video file not found: {v_path}")

        from lyra_ocr.pipeline import run_phase1_pipeline

        run_phase1_pipeline(
            video_path=str(v_path),
            out_dir=str(art_dir),
            fps=fps,
            start_sec=start_sec,
            max_frames=max_frames,
            device=device,
            rec_batch_size=rec_batch_size,
            tracker=tracker,
            title=title,
            channel=channel,
            dedup=dedup,
            frame_source=frame_source,
            source_url=source_url,
            force=force,
        )

    effective_force = force or needs_pipeline
    eff_artist = channel or build_kwargs.pop("artist", None)
    eff_title = title or build_kwargs.pop("title", None)
    build_kwargs.pop("force", None)
    return build_lrc_from_artifacts(
        artifacts_dir=art_dir,
        out_lrc=out_lrc,
        video_path=v_path,
        title=eff_title or None,
        artist=eff_artist or None,
        source_url=source_url,
        force=True,
        **build_kwargs,
    )
