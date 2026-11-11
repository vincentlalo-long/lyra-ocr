import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from lyra_ocr.lyric.build import build_lrc_from_artifacts, build_lrc_from_video
from lyra_ocr.lyric.models import CorrectionConfig


def test_build_lrc_from_artifacts_success(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    manifest = {
        "video_id": "vid1",
        "video_path": "/tmp/vid1.mp4",
        "metadata": {"title": "Song", "channel": "Artist"},
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
    assert res["primary_lang"] == "ja-Latn"
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


