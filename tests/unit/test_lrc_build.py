import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from lyra_ocr.lyric.build import build_lrc_from_artifacts, build_lrc_from_video
from lyra_ocr.lyric.models import CorrectionConfig


def test_build_lrc_from_artifacts_success(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    manifest = {
        "video_id": "vid1",
        "video_path": "/tmp/vid1.mp4",
        "metadata": {"title": "Song", "artist": "Artist", "channel": "Channel"},
        "probe": {"duration": 120.0, "width": 1280, "height": 720},
        "config": {"sample_fps": 2.0},
        "tracker": {"name": "bytetrack"},
    }
    tracks = {
        "total_tracks": 3,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 0,
                "end_ms": 120000,
                "bbox": [10, 10, 100, 40],
                "features": {"dominant_text": "Vibelyusic", "mean_centroid": [0.03, 0.03]},
                "segments": [],
                "occurrences": [],
                "label": "non_lyric",
                "decision": "reject",
            },
            {
                "track_id": 2,
                "start_ms": 18000,
                "end_ms": 23000,
                "bbox": [165, 151, 473, 180],
                "features": {"dominant_text": "I FEEL THE NIGHT EXPLODE", "mean_centroid": [0.24, 0.22]},
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": 18000,
                        "end_ms": 23000,
                        "text": "I FEEL THE NIGHT EXPLODE",
                        "confidence": 0.99,
                        "num_occurrences": 10,
                    }
                ],
                "occurrences": [],
                "label": "lyric",
                "confidence": 0.98,
                "decision": "accept",
            },
            {
                "track_id": 3,
                "start_ms": 26000,
                "end_ms": 32000,
                "bbox": [180, 179, 458, 206],
                "features": {"dominant_text": "WHEN WE'RE TOGETHER", "mean_centroid": [0.25, 0.26]},
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": 26000,
                        "end_ms": 32000,
                        "text": "WHEN WE'RE TOGETHER",
                        "confidence": 0.97,
                        "num_occurrences": 8,
                    }
                ],
                "occurrences": [],
                "label": "lyric",
                "confidence": 0.95,
                "decision": "accept",
            },
        ],
    }

    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    out_lrc = tmp_path / "song.lrc"
    res = build_lrc_from_artifacts(artifacts_dir, out_lrc)

    assert out_lrc.exists()
    assert (tmp_path / "song.cues.json").exists()
    assert res["n_cues"] == 2
    assert res["n_skipped"] == 1

    lrc_content = out_lrc.read_text(encoding="utf-8")
    expected_lines = [
        "[ti:Song]",
        "[ar:Artist]",
        "[by:Lyra OCR]",
        "[offset:0]",
        "[00:18.00]I FEEL THE NIGHT EXPLODE",
        "[00:26.00]WHEN WE'RE TOGETHER",
    ]
    for line in expected_lines:
        assert line in lrc_content


def test_build_lrc_from_artifacts_empty_tracks(tmp_path: Path):
    artifacts_dir = tmp_path / "empty_artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "tracks.json").write_text(json.dumps({"total_tracks": 0, "tracks": []}))

    with pytest.raises(ValueError, match="No tracks found"):
        build_lrc_from_artifacts(artifacts_dir, tmp_path / "song.lrc")


def test_build_lrc_from_video_reuse_existing_artifacts(tmp_path: Path):
    artifacts_dir = tmp_path / "existing_art"
    artifacts_dir.mkdir()
    manifest = {"video_id": "test_reuse", "probe": {"duration": 10.0}}
    tracks = {
        "total_tracks": 1,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "label": "lyric",
                "decision": "accept",
                "features": {"dominant_text": "EXISTING"},
            }
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    out_lrc = tmp_path / "reused.lrc"
    with patch("lyra_ocr.pipeline.run_phase1_pipeline") as mock_pipeline:
        res = build_lrc_from_video(
            video_path=tmp_path / "dummy.mp4",
            out_lrc=out_lrc,
            artifacts_dir=artifacts_dir,
            force=False,
        )
        assert mock_pipeline.call_count == 0
        assert res["n_cues"] == 1
        assert out_lrc.exists()


def test_build_lrc_from_video_runs_pipeline_when_missing(tmp_path: Path):
    artifacts_dir = tmp_path / "new_art"
    video_file = tmp_path / "test_vid.mp4"
    video_file.write_bytes(b"fake mp4 content")

    out_lrc = tmp_path / "new.lrc"

    def fake_pipeline(*args, **kwargs):
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"video_id": "new_vid", "probe": {"duration": 10.0}}
        tracks = {
            "total_tracks": 1,
            "tracks": [
                {
                    "track_id": 1,
                    "start_ms": 500,
                    "end_ms": 2500,
                    "label": "lyric",
                    "decision": "accept",
                    "features": {"dominant_text": "GENERATED BY PIPELINE"},
                }
            ],
        }
        (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    with patch("lyra_ocr.pipeline.run_phase1_pipeline", side_effect=fake_pipeline) as mock_pipeline:
        res = build_lrc_from_video(
            video_path=video_file,
            out_lrc=out_lrc,
            artifacts_dir=artifacts_dir,
            tracker="bytetrack",
            force=False,
        )
        assert mock_pipeline.call_count == 1
        assert res["n_cues"] == 1
        assert out_lrc.exists()


def test_build_lrc_from_video_reruns_pipeline_when_config_mismatch(tmp_path: Path):
    artifacts_dir = tmp_path / "existing_art_mismatch"
    artifacts_dir.mkdir()
    video_file = tmp_path / "vid.mp4"
    video_file.write_bytes(b"content")

    # Existing manifest was run with 1.0 fps and greedy tracker
    manifest = {
        "video_id": "vid",
        "probe": {"duration": 10.0},
        "tracker": {"name": "greedy"},
        "config": {"sample_fps": 1.0, "start_sec": 0.0},
    }
    tracks = {
        "total_tracks": 1,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "label": "lyric",
                "decision": "accept",
                "features": {"dominant_text": "OLD CONFIG"},
            }
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    out_lrc = tmp_path / "out.lrc"

    with patch("lyra_ocr.pipeline.run_phase1_pipeline") as mock_pipeline:
        # Requesting fps=4.0 should detect mismatch and call run_phase1_pipeline
        build_lrc_from_video(
            video_path=video_file,
            out_lrc=out_lrc,
            artifacts_dir=artifacts_dir,
            tracker="greedy",
            fps=4.0,
            force=False,
        )
        assert mock_pipeline.call_count == 1


def test_build_lrc_from_artifacts_invalidates_cache_on_option_change(tmp_path: Path):
    artifacts_dir = tmp_path / "art"
    artifacts_dir.mkdir()
    manifest = {"video_id": "v1", "probe": {"duration": 10.0}}
    tracks = {
        "total_tracks": 2,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "label": "lyric",
                "decision": "accept",
                "features": {"dominant_text": "ACCEPTED"},
                "segments": [],
            },
            {
                "track_id": 2,
                "start_ms": 4000,
                "end_ms": 6000,
                "label": "lyric",
                "decision": "review",
                "features": {"dominant_text": "UNDER REVIEW"},
                "segments": [],
            },
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    out_lrc = tmp_path / "song.lrc"

    # Run 1: without include_review -> 1 cue
    res1 = build_lrc_from_artifacts(artifacts_dir, out_lrc, include_review=False)
    assert res1["n_cues"] == 1

    # Run 2: with include_review=True -> must NOT return cached 1 cue, must return 2 cues!
    res2 = build_lrc_from_artifacts(artifacts_dir, out_lrc, include_review=True)
    assert res2["n_cues"] == 2


def test_build_lrc_from_artifacts_cache_invalidates_on_labels_change(tmp_path: Path):
    artifacts_dir = tmp_path / "art_lbl"
    artifacts_dir.mkdir()
    manifest = {"video_id": "v1", "probe": {"duration": 10.0}}
    tracks = {
        "total_tracks": 1,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "label": "lyric",
                "decision": "accept",
                "features": {"dominant_text": "ORIGINAL TEXT"},
                "segments": [],
            }
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    out_lrc = tmp_path / "song.lrc"

    # 1. First build without labels.jsonl
    res1 = build_lrc_from_artifacts(artifacts_dir, out_lrc)
    assert res1["n_cues"] == 1
    assert "ORIGINAL TEXT" in out_lrc.read_text(encoding="utf-8")

    # 2. Add labels.jsonl with human correction
    labels_file = artifacts_dir / "labels.jsonl"
    labels_file.write_text(
        json.dumps({
            "video_id": "v1",
            "track_id": "1",
            "human_label": "lyric",
            "human_decision": "accept",
            "corrected_text": "CORRECTED BY HUMAN",
        }) + "\n",
        encoding="utf-8",
    )

    # 3. Second build: cache must invalidate because labels.jsonl was added
    res2 = build_lrc_from_artifacts(artifacts_dir, out_lrc)
    assert res2["n_cues"] == 1
    assert "CORRECTED BY HUMAN" in out_lrc.read_text(encoding="utf-8")

    # 4. Remove labels.jsonl
    labels_file.unlink()

    # 5. Third build: cache must invalidate because labels.jsonl was removed
    res3 = build_lrc_from_artifacts(artifacts_dir, out_lrc)
    assert res3["n_cues"] == 1
    assert "ORIGINAL TEXT" in out_lrc.read_text(encoding="utf-8")


def test_build_lrc_frequency_method_without_explicit_vocab(tmp_path: Path):
    artifacts_dir = tmp_path / "freq_art"
    artifacts_dir.mkdir()
    (artifacts_dir / "manifest.json").write_text(json.dumps({"video_id": "freq"}), encoding="utf-8")
    tracks = {
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
                "features": {"dominant_text": "HELLO WORLD", "avg_confidence": 0.95},
                "occurrences": [
                    {"timestamp_ms": 1000, "text_norm": "HELLO WORLD", "confidence": 0.95},
                    {"timestamp_ms": 2000, "text_norm": "HELLO WORD", "confidence": 0.90},
                ],
            }
        ]
    }
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    res = build_lrc_from_artifacts(
        artifacts_dir, tmp_path / "freq.lrc", correct_method="frequency", force=True
    )
    assert res["n_cues"] == 1
    assert "HELLO WORLD" in (tmp_path / "freq.lrc").read_text(encoding="utf-8")


def test_build_lrc_tolerates_null_manifest_labels_and_features(tmp_path: Path):
    artifacts_dir = tmp_path / "null_art"
    artifacts_dir.mkdir()
    (artifacts_dir / "manifest.json").write_text("null", encoding="utf-8")
    (artifacts_dir / "labels.jsonl").write_text("null\n", encoding="utf-8")
    tracks = {
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
                "features": None,
                "segments": [
                    {"segment_id": 0, "start_ms": 1000, "end_ms": 3000, "text": "HELLO WORLD", "confidence": 0.95}
                ],
            }
        ]
    }
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    res = build_lrc_from_artifacts(artifacts_dir, tmp_path / "null.lrc", force=True)
    assert res["n_cues"] == 1
    assert "HELLO WORLD" in (tmp_path / "null.lrc").read_text(encoding="utf-8")


def test_build_lrc_tolerates_list_manifest(tmp_path: Path):
    artifacts_dir = tmp_path / "list_manifest"
    artifacts_dir.mkdir()
    (artifacts_dir / "manifest.json").write_text("[]", encoding="utf-8")
    tracks = {
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
                "features": {"dominant_text": "HELLO AGAIN", "mean_centroid": [0.5, 0.8]},
            }
        ]
    }
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    res = build_lrc_from_artifacts(artifacts_dir, tmp_path / "list.lrc", force=True)
    assert res["n_cues"] == 1
    assert "HELLO AGAIN" in (tmp_path / "list.lrc").read_text(encoding="utf-8")


def test_build_lrc_cache_invalidates_on_gate_config_change(tmp_path: Path):
    artifacts_dir = tmp_path / "art_gate"
    artifacts_dir.mkdir()
    manifest = {"video_id": "v1", "probe": {"duration": 10.0}}
    tracks = {
        "total_tracks": 1,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.90,
                "features": {"dominant_text": "READABLE LYRIC LINE", "mean_centroid": [0.5, 0.8]},
                "segments": [
                    {"segment_id": 0, "start_ms": 1000, "end_ms": 3000, "text": "READABLE LYRIC LINE", "confidence": 0.90}
                ],
            }
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
    out_lrc = tmp_path / "gate.lrc"

    lenient = CorrectionConfig(method="dictionary", min_ocr_confidence=0.55)
    res1 = build_lrc_from_artifacts(artifacts_dir, out_lrc, correction_config=lenient)
    assert res1["n_cues"] == 1

    strict = CorrectionConfig(method="dictionary", min_ocr_confidence=0.95)
    res2 = build_lrc_from_artifacts(artifacts_dir, out_lrc, correction_config=strict)
    assert res2["n_cues"] == 0


def test_build_lrc_provider_priority_and_caption_fallback(tmp_path: Path):
    from lyra_ocr.lyric.reference import ReferenceLine, ReferenceBundle

    artifacts_dir = tmp_path / "art_prio"
    artifacts_dir.mkdir()
    manifest = {
        "video_id": "vid_priority",
        "source_url": "https://www.youtube.com/watch?v=mock123abc",
        "metadata": {"title": "Test Song", "channel": "Test Artist"},
        "probe": {"duration": 60.0},
    }
    tracks = {
        "total_tracks": 1,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 10000,
                "end_ms": 15000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
                "features": {"dominant_text": "YOUVE BEEN ROAMIN ROUND", "mean_centroid": [0.5, 0.8]},
                "segments": [
                    {"segment_id": 0, "start_ms": 10000, "end_ms": 15000, "text": "YOUVE BEEN ROAMIN ROUND", "confidence": 0.95}
                ],
            }
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
    out_lrc = tmp_path / "prio.lrc"

    # 1. Manual reference overrides online
    with patch("lyra_ocr.lyric.build.fetch_reference_bundle") as mock_fetch_lrclib:
        res = build_lrc_from_artifacts(
            artifacts_dir,
            out_lrc,
            reference_text="[00:10.00]You've been runnin' round",
            use_online_reference=True,
            force=True,
        )
        assert mock_fetch_lrclib.call_count == 0
        assert res["reference_source"] == "manual"
        assert res["n_reference_matches"] == 1

    # 2. LRCLIB wins over captions
    lrclib_bundle = ReferenceBundle(
        lines=[
            ReferenceLine(text="You've been runnin' round", start_ms=10000, end_ms=15000),
            ReferenceLine(text="I'm filling a gap line", start_ms=20000, end_ms=25000),
        ],
        source="lrclib_synced",
        provider="lrclib",
    )
    with patch("lyra_ocr.lyric.build.fetch_reference_bundle", return_value=lrclib_bundle) as mock_lrclib, \
         patch("lyra_ocr.lyric.captions.fetch_youtube_caption_bundle") as mock_captions:
        res2 = build_lrc_from_artifacts(
            artifacts_dir,
            out_lrc,
            use_online_reference=True,
            force=True,
        )
        assert mock_lrclib.call_count == 1
        assert mock_captions.call_count == 0
        assert res2["reference_source"] == "lrclib_synced"
        assert res2["n_reference_matches"] == 1
        # The extra reference line is after the last matched anchor; conservative
        # edge filling is disabled by default.
        assert res2["n_reference_gap_fills"] == 0
        assert res2["n_cues"] == 1

        # Check provenance written to .cues.json
        cues_json = json.loads((tmp_path / "prio.cues.json").read_text(encoding="utf-8"))
        assert cues_json["reference_source"] == "lrclib_synced"
        assert cues_json["n_reference_matches"] == 1
        assert cues_json["n_reference_gap_fills"] == 0

    # 3. Caption fallback when LRCLIB returns empty
    empty_bundle = ReferenceBundle(lines=[], source="", provider="lrclib")
    caption_bundle = ReferenceBundle(
        lines=[ReferenceLine(text="You've been runnin' round", start_ms=10000, end_ms=15000)],
        source="youtube_captions",
        provider="youtube_caption",
    )
    # Give the track a text with >= 0.90 similarity to satisfy the stricter caption threshold
    tracks_caption = {
        "total_tracks": 1,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 10000,
                "end_ms": 15000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
                "features": {"dominant_text": "YOUVE BEEN RUNING ROUND", "mean_centroid": [0.5, 0.8]},
                "segments": [
                    {"segment_id": 0, "start_ms": 10000, "end_ms": 15000, "text": "YOUVE BEEN RUNING ROUND", "confidence": 0.95}
                ],
            }
        ],
    }
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks_caption), encoding="utf-8")

    with patch("lyra_ocr.lyric.build.fetch_reference_bundle", return_value=empty_bundle) as mock_lrclib, \
         patch("lyra_ocr.lyric.captions.fetch_youtube_caption_bundle", return_value=caption_bundle) as mock_captions:
        res3 = build_lrc_from_artifacts(
            artifacts_dir,
            out_lrc,
            use_online_reference=True,
            force=True,
        )
        assert mock_lrclib.call_count == 1
        assert mock_captions.call_count == 1
        assert res3["reference_source"] == "youtube_captions"
        assert res3["n_reference_matches"] == 1

    # 4. No-caption-fallback prevents caption fallback
    with patch("lyra_ocr.lyric.build.fetch_reference_bundle", return_value=empty_bundle) as mock_lrclib, \
         patch("lyra_ocr.lyric.captions.fetch_youtube_caption_bundle") as mock_captions:
        res4 = build_lrc_from_artifacts(
            artifacts_dir,
            out_lrc,
            use_online_reference=True,
            no_caption_fallback=True,
            force=True,
        )
        assert mock_lrclib.call_count == 1
        assert mock_captions.call_count == 0
        assert res4["reference_source"] == ""
        assert res4["n_reference_matches"] == 0






def test_caption_fallback_is_cached_between_builds(tmp_path: Path):
    from lyra_ocr.lyric.reference import ReferenceBundle, ReferenceLine

    artifacts_dir = tmp_path / "art_caption_cache"
    artifacts_dir.mkdir()
    manifest = {
        "video_id": "Attention [6QYcd7RggNU]",
        "metadata": {"title": "Charlie Puth - Attention", "channel": ""},
        "probe": {"duration": 30.0},
    }
    tracks = {
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 10000,
                "end_ms": 14000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
                "features": {"dominant_text": "YOUVE BEEN RUNNIN ROUND", "mean_centroid": [0.5, 0.8]},
                "segments": [
                    {"segment_id": 0, "start_ms": 10000, "end_ms": 14000, "text": "YOUVE BEEN RUNNIN ROUND", "confidence": 0.95}
                ],
            }
        ]
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
    out_lrc = tmp_path / "caption_cache.lrc"
    empty = ReferenceBundle(lines=[], source="", provider="lrclib")
    caption = ReferenceBundle(
        lines=[ReferenceLine(text="You've been runnin' round", start_ms=10000, end_ms=14000)],
        source="youtube_captions",
        provider="youtube",
    )

    with patch("lyra_ocr.lyric.build.fetch_reference_bundle", return_value=empty), \
         patch("lyra_ocr.lyric.captions.fetch_youtube_caption_bundle", return_value=caption) as mock_captions:
        first = build_lrc_from_artifacts(
            artifacts_dir, out_lrc, use_online_reference=True, force=True
        )
        assert first["reference_source"] == "youtube_captions"
        assert mock_captions.call_count == 1

        second = build_lrc_from_artifacts(
            artifacts_dir, out_lrc, use_online_reference=True, force=True
        )
        assert second["reference_source"] == "youtube_captions"
        assert mock_captions.call_count == 1


def test_build_lrc_per_layer_reference_alignment_protects_romaji(tmp_path: Path):
    from lyra_ocr.lyric.reference import ReferenceBundle, ReferenceLine

    artifacts_dir = tmp_path / "art_multilang"
    artifacts_dir.mkdir()
    manifest = {
        "video_id": "v_multi",
        "probe": {"duration": 30.0},
    }
    tracks = {
        "total_tracks": 2,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
                "features": {
                    "dominant_text": "senbonzakura yoru ni magire",
                    "layer_id": 0,
                    "lang": "ja-Latn",
                    "mean_centroid": [0.5, 0.5],
                },
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": 1000,
                        "end_ms": 3000,
                        "text": "senbonzakura yoru ni magire",
                        "confidence": 0.95,
                    }
                ],
            },
            {
                "track_id": 2,
                "start_ms": 1000,
                "end_ms": 3000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
                "features": {
                    "dominant_text": "千本桜 夜ニ紛レ",
                    "layer_id": 1,
                    "lang": "ja",
                    "mean_centroid": [0.5, 0.8],
                },
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": 1000,
                        "end_ms": 3000,
                        "text": "千本桜 夜ニ紛レ",
                        "confidence": 0.95,
                    }
                ],
            },
            {
                "track_id": 3,
                "start_ms": 8000,
                "end_ms": 10000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
                "features": {
                    "dominant_text": "此処ハ宴 鋼ノ檻",
                    "layer_id": 1,
                    "lang": "ja",
                    "mean_centroid": [0.5, 0.8],
                },
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": 8000,
                        "end_ms": 10000,
                        "text": "此処ハ宴 鋼ノ檻",
                        "confidence": 0.95,
                    }
                ],
            },
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    kanji_bundle = ReferenceBundle(
        lines=[
            ReferenceLine(text="千本桜 夜ニ紛レ", start_ms=1000, end_ms=3000, language="ja"),
            ReferenceLine(text="君ノ声モ届カナイヨ", start_ms=4000, end_ms=6000, language="ja"),
            ReferenceLine(text="此処ハ宴 鋼ノ檻", start_ms=8000, end_ms=10000, language="ja"),
        ],
        source="lrclib_synced",
        provider="lrclib",
        language="ja",
    )

    out_lrc = tmp_path / "multi.lrc"
    with patch("lyra_ocr.lyric.build.fetch_reference_bundle", return_value=kanji_bundle):
        res = build_lrc_from_artifacts(
            artifacts_dir,
            out_lrc,
            use_online_reference=True,
            force=True,
        )

    cues_json = json.loads((tmp_path / "multi.cues.json").read_text(encoding="utf-8"))
    cues = cues_json["cues"]

    # Layer files generated
    assert (tmp_path / "multi.ja-Latn.lrc").exists()
    assert (tmp_path / "multi.ja.lrc").exists()
    # Primary layer is ja because ja has more cues (2) than ja-Latn (1)
    assert res["primary_lang"] == "ja"
    assert len(res["layers"]) == 2

    # Layer 0 (ja-Latn) must NEVER be replaced by Kanji reference

    romaji_cues = [c for c in cues if c.get("layer_id") == 0 or c.get("lang") == "ja-Latn"]
    assert len(romaji_cues) == 1
    assert romaji_cues[0]["text"] == "senbonzakura yoru ni magire"

    # Layer 1 (ja) was matched and gap filled
    ja_cues = [c for c in cues if c.get("layer_id") == 1]
    assert len(ja_cues) == 3
    # Gap fill line belongs to layer 1 and lang ja
    gap_cue = [c for c in ja_cues if "君ノ声モ届カナイヨ" in c["text"]][0]
    assert gap_cue["layer_id"] == 1
    assert gap_cue["lang"] == "ja"


def test_build_lrc_gates_low_conf_unvoiced_overlay_using_asr_vocal_gate(tmp_path: Path):
    artifacts_dir = tmp_path / "art_asr_gate"
    artifacts_dir.mkdir()
    manifest = {
        "video_id": "vid_asr_gate",
        "probe": {"duration": 30.0},
    }
    tracks = {
        "total_tracks": 2,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.90,
                "features": {"dominant_text": "hello singing world", "mean_centroid": [0.5, 0.8]},
                "segments": [
                    {"segment_id": 0, "start_ms": 1000, "end_ms": 3000, "text": "hello singing world", "confidence": 0.90}
                ],
            },
            {
                "track_id": 2,
                "start_ms": 20000,
                "end_ms": 23000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.58,
                "features": {"dominant_text": "subscribe channel watermark", "mean_centroid": [0.5, 0.8]},
                "segments": [
                    {"segment_id": 0, "start_ms": 20000, "end_ms": 23000, "text": "subscribe channel watermark", "confidence": 0.58}
                ],
            },
        ],
    }
    asr_payload = {
        "language": "en",
        "model_size": "small",
        "segments": [
            {
                "text": "hello singing world",
                "start_ms": 1000,
                "end_ms": 3000,
                "language": "en",
                "words": [
                    {"word": "hello", "start_ms": 1000, "end_ms": 1600, "probability": 0.95},
                    {"word": "singing", "start_ms": 1600, "end_ms": 2300, "probability": 0.95},
                    {"word": "world", "start_ms": 2300, "end_ms": 3000, "probability": 0.95},
                ],
            }
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
    (artifacts_dir / "asr_transcript.json").write_text(json.dumps(asr_payload), encoding="utf-8")

    out_lrc = tmp_path / "asr_gate.lrc"
    cfg = CorrectionConfig(method="dictionary", min_ocr_confidence=0.50)
    res = build_lrc_from_artifacts(
        artifacts_dir,
        out_lrc,
        include_review=False,
        correction_config=cfg,
        force=True,
    )
    assert res["n_cues"] == 1
    assert any(s.get("reason") == "asr_unvoiced" for s in res["skipped"])


def test_build_lrc_channel_not_blindly_copied_to_artist(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts_channel"
    artifacts_dir.mkdir()

    manifest = {
        "video_id": "vid_chan",
        "video_path": "/tmp/vid_chan.mp4",
        "metadata": {"title": "Castle", "channel": "Clarx & Harddope"},
        "probe": {"duration": 60.0},
        "tracker": {"name": "bytetrack"},
    }
    tracks = {
        "total_tracks": 1,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "bbox": [10, 800, 500, 850],
                "features": {"dominant_text": "In the castle", "mean_centroid": [0.5, 0.8]},
                "segments": [{"segment_id": 0, "start_ms": 1000, "end_ms": 3000, "text": "In the castle", "confidence": 0.95}],
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
            }
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    out_lrc = tmp_path / "channel.lrc"
    build_lrc_from_artifacts(artifacts_dir, out_lrc, force=True)

    content = out_lrc.read_text(encoding="utf-8")
    assert "[ti:Castle]" in content
    # Channel was NOT copied into artist
    assert "[ar:" not in content


def test_build_lrc_parses_artist_from_hyphenated_title(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts_hyphen"
    artifacts_dir.mkdir()

    manifest = {
        "video_id": "vid_hyphen",
        "video_path": "/tmp/vid_hyphen.mp4",
        "metadata": {"title": "Clarx & Harddope - Castle", "channel": "NCS Release"},
        "probe": {"duration": 60.0},
        "tracker": {"name": "bytetrack"},
    }
    tracks = {
        "total_tracks": 1,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "bbox": [10, 800, 500, 850],
                "features": {"dominant_text": "In the castle", "mean_centroid": [0.5, 0.8]},
                "segments": [{"segment_id": 0, "start_ms": 1000, "end_ms": 3000, "text": "In the castle", "confidence": 0.95}],
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
            }
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    out_lrc = tmp_path / "hyphen.lrc"
    build_lrc_from_artifacts(artifacts_dir, out_lrc, force=True)

    content = out_lrc.read_text(encoding="utf-8")
    assert "[ar:Clarx & Harddope]" in content


def test_build_lrc_enhanced_plumbing(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts_enhanced"
    artifacts_dir.mkdir()

    v_file = tmp_path / "vid_enh.mp4"
    v_file.touch()

    manifest = {
        "video_id": "vid_enh",
        "video_path": str(v_file),
        "metadata": {"title": "Song", "artist": "Artist"},
        "probe": {"duration": 60.0},
        "tracker": {"name": "bytetrack"},
    }
    tracks = {
        "total_tracks": 1,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "bbox": [10, 800, 500, 850],
                "features": {"dominant_text": "Hello world", "mean_centroid": [0.5, 0.8]},
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": 1000,
                        "end_ms": 3000,
                        "text": "Hello world",
                        "confidence": 0.95,
                    }
                ],
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
            }
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    out_lrc = tmp_path / "enhanced.lrc"
    from unittest.mock import patch
    from lyra_ocr.lyric.models import ReferenceConfig

    # Test passing enhanced=True
    with patch("lyra_ocr.lyric.build.extract_audio_16k") as mock_extract, patch("lyra_ocr.lyric.build.align_cues_with_audio") as mock_align:
        mock_extract.return_value = np.zeros(16000, dtype=np.float32)
        def _add_words(cues, audio):
            for c in cues:
                c.words = [
                    {"word": "Hello", "start_ms": 1000, "end_ms": 1800},
                    {"word": "world", "start_ms": 1900, "end_ms": 3000},
                ]
            return cues
        mock_align.side_effect = _add_words
        build_lrc_from_artifacts(
            artifacts_dir,
            out_lrc,
            video_path=str(v_file),
            force=True,
            enhanced=True,
            reference_config=ReferenceConfig(enable_audio_align=True),
        )

    content = out_lrc.read_text(encoding="utf-8")
    assert "<00:01.00>Hello <00:01.90>world" in content


def test_build_lrc_from_video_channel_not_leaked_as_artist(tmp_path: Path):
    artifacts_dir = tmp_path / "vid_chan_art"
    artifacts_dir.mkdir()
    manifest = {
        "video_id": "test_chan",
        "probe": {"duration": 10.0},
        "tracker": {"name": "bytetrack"},
        "config": {"sample_fps": 2.0},
        "metadata": {"title": "Just Title", "channel": "MyChannel"},
    }
    tracks = {
        "total_tracks": 1,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 3000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
                "features": {"dominant_text": "LINE ONE"},
                "segments": [{"segment_id": 0, "start_ms": 1000, "end_ms": 3000, "text": "LINE ONE", "confidence": 0.95}],
            }
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    out_lrc = tmp_path / "vid_chan.lrc"
    with patch("lyra_ocr.pipeline.run_phase1_pipeline"):
        build_lrc_from_video(
            video_path=tmp_path / "dummy.mp4",
            out_lrc=out_lrc,
            artifacts_dir=artifacts_dir,
            title="Just Title",
            channel="MyChannel",
            force=False,
        )

    content = out_lrc.read_text(encoding="utf-8")
    assert "[ti:Just Title]" in content
    # Channel MUST NOT be leaked into artist tag
    assert "[ar:" not in content


def test_build_lrc_refreshes_stale_cached_layer_lang(tmp_path: Path):
    # Cached tracks.json from an older run with stale lang="en" on Romaji and unaccented Spanish
    # lines, plus missing layer_id/layer_kind on one track. build_lrc_from_artifacts() must
    # refresh layer discovery and language voting without dropping existing features.
    artifacts_dir = tmp_path / "stale_lang_art"
    artifacts_dir.mkdir()
    manifest = {
        "video_id": "vid_stale_lang",
        "probe": {"duration": 30.0, "width": 1280, "height": 720},
    }
    romaji_lines = [
        "ille kcko ni nal kaze wo kanjite",
        "walashi no kolae wo kikoe nal yo",
        "fulari de tckiakasu ycake no sora",
        "kono mune ni hikari wo tomoshite",
    ]
    spanish_lines = [
        "Aumque llore ya nada volvera a ser como antes",
        "el futuro contigo siempre sera mejor",
        "solo quiero olvidar este dolor",
        "nunca voy a olvidar tu voz",
    ]
    raw_tracks = []
    for i, text in enumerate(romaji_lines):
        raw_tracks.append(
            {
                "track_id": i + 1,
                "start_ms": i * 3000 + 1000,
                "end_ms": i * 3000 + 3000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.90,
                "features": {
                    "dominant_text": text,
                    "mean_centroid": [0.5, 0.85],
                    "lang": "en",
                    "line_lang": "en",
                    "custom_tag": f"ro-{i}",
                },
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": i * 3000 + 1000,
                        "end_ms": i * 3000 + 3000,
                        "text": text,
                        "confidence": 0.90,
                    }
                ],
            }
        )
    for j, text in enumerate(spanish_lines):
        raw_tracks.append(
            {
                "track_id": 10 + j,
                "start_ms": j * 3000 + 1000,
                "end_ms": j * 3000 + 3000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.90,
                "features": {
                    "dominant_text": text,
                    "mean_centroid": [0.5, 0.12],
                    "lang": "en",
                    "line_lang": "en",
                },
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": j * 3000 + 1000,
                        "end_ms": j * 3000 + 3000,
                        "text": text,
                        "confidence": 0.90,
                    }
                ],
            }
        )

    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(
        json.dumps({"total_tracks": len(raw_tracks), "tracks": raw_tracks}),
        encoding="utf-8",
    )

    out_lrc = tmp_path / "stale_lang.lrc"
    res = build_lrc_from_artifacts(artifacts_dir, out_lrc, force=True)

    cues_data = json.loads((tmp_path / "stale_lang.cues.json").read_text(encoding="utf-8"))
    langs = {c["lang"] for c in cues_data["cues"]}
    assert "ja-Latn" in langs
    assert "es" in langs
    assert res["n_cues"] == 8
    assert res["n_skipped"] == 0

    from lyra_ocr.layers import refresh_dict_tracks_layers

    refresh_dict_tracks_layers(raw_tracks, frame_w=1280, frame_h=720, duration_ms=30000)
    assert raw_tracks[0]["features"]["custom_tag"] == "ro-0"
    assert raw_tracks[0]["features"]["lang"] == "ja-Latn"
    assert raw_tracks[0]["features"]["layer_kind"] == "lyric"
    assert raw_tracks[4]["features"]["lang"] == "es"
    assert raw_tracks[4]["features"]["layer_kind"] == "lyric"


def test_build_lrc_dedupes_per_layer_after_reference_alignment(tmp_path: Path):
    from unittest.mock import patch
    from lyra_ocr.lyric.models import ReferenceConfig

    artifacts_dir = tmp_path / "post_ref_dedupe"
    artifacts_dir.mkdir()
    manifest = {
        "video_id": "vid_post_ref",
        "probe": {"duration": 20.0, "width": 1280, "height": 720},
    }
    # Two adjacent tracks at the same Y; OCR texts differ enough that pre-reference dedupe keeps both,
    # then reference alignment repairs both to the identical text and per-layer dedupe collapses them
    # BEFORE consolidate_language_layers runs.
    tracks = {
        "total_tracks": 2,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 2600,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.90,
                "features": {"dominant_text": "I feel the night", "mean_centroid": [0.5, 0.85]},
                "segments": [
                    {"segment_id": 0, "start_ms": 1000, "end_ms": 2600, "text": "I feel the night", "confidence": 0.90}
                ],
            },
            {
                "track_id": 2,
                "start_ms": 2700,
                "end_ms": 4300,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.92,
                "features": {"dominant_text": "I feel the dark night", "mean_centroid": [0.5, 0.85]},
                "segments": [
                    {"segment_id": 0, "start_ms": 2700, "end_ms": 4300, "text": "I feel the dark night", "confidence": 0.92}
                ],
            },
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    observed_pre_consolidate_counts: list[int] = []
    from lyra_ocr.lyric import grouping as real_grouping

    def spy_consolidate(cues_in, **kwargs):
        observed_pre_consolidate_counts.append(len(cues_in))
        return real_grouping.consolidate_language_layers(cues_in, **kwargs)

    out_lrc = tmp_path / "post_ref.lrc"
    with patch("lyra_ocr.lyric.build.consolidate_language_layers", side_effect=spy_consolidate):
        res = build_lrc_from_artifacts(
            artifacts_dir,
            out_lrc,
            force=True,
            reference_text="[00:01.00]I feel the night\n[00:02.70]I feel the night\n",
            reference_config=ReferenceConfig(replace_similarity=0.85, enable_gap_fill=False),
        )

    assert res["n_cues"] == 1
    assert observed_pre_consolidate_counts == [1]


def test_refresh_dict_tracks_layers_when_all_have_layer_id(tmp_path: Path):
    from lyra_ocr.layers import refresh_dict_tracks_layers

    romaji_lines = [
        "ille kcko ni nal kaze wo kanjite",
        "walashi no kolae wo kikoe nal yo",
        "fulari de tckiakasu ycake no sora",
        "kono mune ni hikari wo tomoshite",
    ]
    spanish_lines = [
        "Aunque llore ya nada volvera a ser como antes",
        "el futuro contigo siempre sera mejor",
        "solo quiero olvidar este dolor",
        "para siempre contigo",
    ]
    companion_lines = [
        "sono kodou wa toki wo koete",
        "mirai no hikari wo sagashite",
    ]

    raw_tracks: list[dict[str, object]] = []
    # Primary Romaji layer with custom pre-existing layer_id=42 and stale lang="en"
    for i, text in enumerate(romaji_lines):
        raw_tracks.append(
            {
                "track_id": i + 1,
                "start_ms": i * 3000 + 1000,
                "end_ms": i * 3000 + 3000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.92,
                "features": {
                    "dominant_text": text,
                    "mean_centroid": [0.5, 0.85],
                    "layer_id": 42,
                    "layer_kind": "lyric",
                    "lang": "en",
                    "line_lang": "en",
                    "custom_tag": f"primary-{i}",
                },
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": i * 3000 + 1000,
                        "end_ms": i * 3000 + 3000,
                        "text": text,
                        "confidence": 0.92,
                    }
                ],
            }
        )

    # Companion sparse band at y=0.78 (within [0.025, 0.16] of y=0.85) overlapping primary layer in time,
    # carrying pre-existing layer_id=99, layer_kind="sparse", and stale lang="en".
    for k, text in enumerate(companion_lines):
        raw_tracks.append(
            {
                "track_id": 20 + k,
                "start_ms": k * 3000 + 1000,
                "end_ms": k * 3000 + 2800,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.90,
                "features": {
                    "dominant_text": text,
                    "mean_centroid": [0.5, 0.78],
                    "layer_id": 99,
                    "layer_kind": "sparse",
                    "lang": "en",
                    "line_lang": "en",
                    "custom_tag": f"companion-{k}",
                },
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": k * 3000 + 1000,
                        "end_ms": k * 3000 + 2800,
                        "text": text,
                        "confidence": 0.90,
                    }
                ],
            }
        )

    # Spanish layer at y=0.12 with pre-existing layer_id=77 and stale lang="en"
    for j, text in enumerate(spanish_lines):
        raw_tracks.append(
            {
                "track_id": 40 + j,
                "start_ms": j * 3000 + 1000,
                "end_ms": j * 3000 + 3000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.91,
                "features": {
                    "dominant_text": text,
                    "mean_centroid": [0.5, 0.12],
                    "layer_id": 77,
                    "layer_kind": "lyric",
                    "lang": "en",
                    "line_lang": "en",
                },
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": j * 3000 + 1000,
                        "end_ms": j * 3000 + 3000,
                        "text": text,
                        "confidence": 0.91,
                    }
                ],
            }
        )

    # 1. Direct refresh_dict_tracks_layers() check when all_have_layer_id == True
    import copy

    tracks_copy = copy.deepcopy(raw_tracks)
    refresh_dict_tracks_layers(tracks_copy, frame_w=1280, frame_h=720, duration_ms=30000)

    # Existing layer_ids must be preserved (not overwritten with 1/2/3 from discover_layers)
    assert [t["features"]["layer_id"] for t in tracks_copy[:4]] == [42, 42, 42, 42]
    assert [t["features"]["layer_id"] for t in tracks_copy[4:6]] == [99, 99]
    assert [t["features"]["layer_id"] for t in tracks_copy[6:]] == [77, 77, 77, 77]

    # Stale lang="en" refreshed dynamically to "ja-Latn" / "es"
    assert all(t["features"]["lang"] == "ja-Latn" for t in tracks_copy[:6])
    assert all(t["features"]["lang"] == "es" for t in tracks_copy[6:])
    # Companion "sparse" band promoted to "lyric"
    assert all(t["features"]["layer_kind"] == "lyric" for t in tracks_copy[4:6])
    # Unrelated custom feature keys preserved
    assert tracks_copy[0]["features"]["custom_tag"] == "primary-0"
    assert tracks_copy[4]["features"]["custom_tag"] == "companion-0"

    # 2. End-to-end build_lrc_from_artifacts() check with all_have_layer_id == True
    artifacts_dir = tmp_path / "all_layer_id_art"
    artifacts_dir.mkdir()
    manifest = {
        "video_id": "vid_all_layer_id",
        "probe": {"duration": 30.0, "width": 1280, "height": 720},
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(
        json.dumps({"total_tracks": len(raw_tracks), "tracks": raw_tracks}),
        encoding="utf-8",
    )

    observed_select_layers: list[tuple[int, int, str, str]] = []
    from lyra_ocr.lyric import select as real_select

    def spy_select_cues(tracks_in, human_labels, **kwargs):
        for tr in tracks_in:
            f = tr.get("features") or {}
            observed_select_layers.append(
                (tr["track_id"], f.get("layer_id"), f.get("layer_kind"), f.get("lang"))
            )
        return real_select.select_cues(tracks_in, human_labels, **kwargs)

    out_lrc = tmp_path / "all_layer_id.lrc"
    with patch("lyra_ocr.lyric.build.select_cues", side_effect=spy_select_cues):
        build_lrc_from_artifacts(artifacts_dir, out_lrc, force=True)

    assert (20, 99, "lyric", "ja-Latn") in observed_select_layers
    assert (21, 99, "lyric", "ja-Latn") in observed_select_layers
    assert (1, 42, "lyric", "ja-Latn") in observed_select_layers
    assert (40, 77, "lyric", "es") in observed_select_layers

    cues_data = json.loads((tmp_path / "all_layer_id.cues.json").read_text(encoding="utf-8"))
    cue_langs = {c["lang"] for c in cues_data["cues"]}
    assert cue_langs == {"ja-Latn", "es"}


def test_build_lrc_routes_vocal_lang_to_asr_and_preserves_asr_words(tmp_path: Path):
    from unittest.mock import patch
    from lyra_ocr.audio.asr import ASRSegment, ASRTranscript, ASRWord
    from lyra_ocr.lyric.models import ReferenceConfig

    artifacts_dir = tmp_path / "art_asr_lang_words"
    artifacts_dir.mkdir()
    v_file = tmp_path / "vid_ja.mp4"
    v_file.touch()

    manifest = {
        "video_id": "vid_ja",
        "video_path": str(v_file),
        "probe": {"duration": 20.0, "width": 1280, "height": 720},
    }
    tracks = {
        "total_tracks": 1,
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 4000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
                "features": {"dominant_text": "sono kodou wa", "mean_centroid": [0.5, 0.85]},
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": 1000,
                        "end_ms": 4000,
                        "text": "sono kodou wa",
                        "confidence": 0.95,
                    }
                ],
            }
        ],
    }
    (artifacts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    # Stale "en" cache on disk must not block calling transcribe_audio_words with language="ja"
    (artifacts_dir / "asr_transcript.json").write_text(
        json.dumps({"language": "en", "model_size": "small", "segments": []}),
        encoding="utf-8",
    )

    ja_transcript = ASRTranscript(
        language="ja",
        segments=[
            ASRSegment(
                text="その鼓動は",
                start_ms=1000,
                end_ms=4000,
                language="ja",
                words=[
                    ASRWord("その", 1050, 1300),
                    ASRWord("鼓動", 1350, 3200),
                    ASRWord("は", 3300, 3900),
                ],
            )
        ],
    )

    observed_asr_langs: list[str | None] = []

    def fake_transcribe(audio_source, *, cache_dir=None, language=None, model_size="small"):
        observed_asr_langs.append(language)
        return ja_transcript

    audio = np.zeros(16000 * 6, dtype=np.float32)
    out_lrc = tmp_path / "ja_words.lrc"
    with (
        patch("lyra_ocr.lyric.build.extract_audio_16k", return_value=audio),
        patch("lyra_ocr.lyric.build.transcribe_audio_words", side_effect=fake_transcribe),
    ):
        build_lrc_from_artifacts(
            artifacts_dir,
            out_lrc,
            video_path=str(v_file),
            force=True,
            enhanced=True,
            reference_config=ReferenceConfig(enable_audio_align=True, enable_asr=True),
        )

    assert observed_asr_langs == ["ja"]
    cues_data = json.loads((tmp_path / "ja_words.cues.json").read_text(encoding="utf-8"))
    assert len(cues_data["cues"]) == 1
    cue = cues_data["cues"][0]
    assert "asr_word_aligned" in cue["qa_flags"]
    assert cue["words"] == [
        {"word": "sono", "start_ms": 1050, "end_ms": 1300},
        {"word": "kodou", "start_ms": 1350, "end_ms": 3200},
        {"word": "wa", "start_ms": 3300, "end_ms": 3900},
    ]


def test_filter_boundary_unvoiced_and_distractor_cues_strips_silent_intro_outro_cues():
    from lyra_ocr.audio.asr import ASRSegment, ASRTranscript, ASRWord
    from lyra_ocr.lyric.build import filter_boundary_unvoiced_and_distractor_cues
    from lyra_ocr.lyric.models import LyricCue

    duration_ms = 240000
    # 50ms hops for 240s = 4800 frames; silent in [0..20s] and [220..240s], voiced in [20..220s]
    vocal_profile = np.zeros(4800, dtype=np.float32)
    vocal_profile[400:4400] = 0.9

    asr_transcript = ASRTranscript(
        language="ja",
        segments=[
            ASRSegment("line1", 22000, 26000, words=[ASRWord("line1", 22000, 26000)]),
            ASRSegment("line2", 40000, 44000, words=[ASRWord("line2", 40000, 44000)]),
            ASRSegment("line3", 80000, 84000, words=[ASRWord("line3", 80000, 84000)]),
            ASRSegment("line4", 120000, 124000, words=[ASRWord("line4", 120000, 124000)]),
            ASRSegment("line5", 214000, 218000, words=[ASRWord("line5", 214000, 218000)]),
        ],
    )

    cues = [
        LyricCue(start_ms=2000, end_ms=6000, text="Silent intro line", ocr_confidence=0.98, source_track_ids=[1]),
        LyricCue(start_ms=4000, end_ms=8000, text="my", ocr_confidence=0.99, source_track_ids=[2]),
        LyricCue(start_ms=40000, end_ms=44000, text="Hanatsu hikari ima koko kara", ocr_confidence=0.97, source_track_ids=[3]),
        LyricCue(start_ms=226000, end_ms=230000, text="Silent outro lyric", ocr_confidence=0.96, source_track_ids=[4]),
        LyricCue(start_ms=232000, end_ms=236000, text="Thanks for watching", ocr_confidence=0.99, source_track_ids=[5]),
    ]
    skipped: list[dict] = []

    kept = filter_boundary_unvoiced_and_distractor_cues(
        cues,
        skipped,
        duration_ms=duration_ms,
        title="only my railgun",
        artist="fripSide",
        asr_transcript=asr_transcript,
        vocal_profile=vocal_profile,
        include_review=False,
    )

    assert [c.text for c in kept] == ["Hanatsu hikari ima koko kara"]
    assert len(skipped) == 4
    assert all(item["reason"] == "boundary_unvoiced_or_credit" for item in skipped)

    # Also verify ASR-only boundary gate when vocal_profile is None
    cues_asr_only = [
        LyricCue(start_ms=5000, end_ms=9000, text="Unvoiced by ASR gate", ocr_confidence=0.98, source_track_ids=[10]),
        LyricCue(start_ms=22000, end_ms=26000, text="Sung opening line", ocr_confidence=0.98, source_track_ids=[11]),
    ]
    skipped_asr: list[dict] = []
    kept_asr = filter_boundary_unvoiced_and_distractor_cues(
        cues_asr_only,
        skipped_asr,
        duration_ms=duration_ms,
        title="only my railgun",
        artist="fripSide",
        asr_transcript=asr_transcript,
        vocal_profile=None,
        include_review=False,
    )
    assert [c.text for c in kept_asr] == ["Sung opening line"]
    assert len(skipped_asr) == 1
    assert skipped_asr[0]["reason"] == "boundary_unvoiced_or_credit"





