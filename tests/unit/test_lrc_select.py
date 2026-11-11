from pathlib import Path
import pytest
import json

from lyra_ocr.lyric.select import load_artifacts, select_cues
from lyra_ocr.lyric.models import LyricCue


def test_select_cues_filtering():
    tracks = [
        {
            "track_id": 1,
            "start_ms": 1000,
            "end_ms": 3000,
            "label": "lyric",
            "decision": "accept",
            "confidence": 0.95,
            "features": {"dominant_text": "ACCEPTED LYRIC", "mean_centroid": [0.5, 0.8]},
        },
        {
            "track_id": 2,
            "start_ms": 4000,
            "end_ms": 6000,
            "label": "non_lyric",
            "decision": "reject",
            "confidence": 0.99,
            "features": {"dominant_text": "CHANNEL LOGO"},
        },
        {
            "track_id": 3,
            "start_ms": 7000,
            "end_ms": 9000,
            "label": "lyric",
            "decision": "review",
            "confidence": 0.60,
            "features": {"dominant_text": "UNCERTAIN LYRIC"},
        },
        {
            "track_id": 4,
            "start_ms": 10000,
            "end_ms": 12000,
            # Missing label and decision -> should default to non_lyric/review and be skipped safely
            "features": {"dominant_text": "NO METADATA"},
        },
    ]

    # Without include_review
    cues, skipped = select_cues(tracks, human_labels={}, include_review=False)
    assert len(cues) == 1
    assert cues[0].source_track_ids == [1]
    assert cues[0].text == "ACCEPTED LYRIC"
    assert not cues[0].review_required
    assert len(skipped) == 3
    reasons = {s["track_id"]: s["reason"] for s in skipped}
    assert reasons[2] == "non_lyric"
    assert reasons[3] == "review"

    # With include_review
    cues_review, skipped_review = select_cues(tracks, human_labels={}, include_review=True)
    assert len(cues_review) == 2
    cue_3 = next(c for c in cues_review if c.source_track_ids == [3])
    assert cue_3.review_required is True
    assert cue_3.text == "UNCERTAIN LYRIC"


def test_select_cues_human_labels_override():
    tracks = [
        {
            "track_id": 1,
            "start_ms": 1000,
            "end_ms": 3000,
            "label": "lyric",
            "decision": "accept",
            "features": {"dominant_text": "MACHINE TEXT"},
        },
        {
            "track_id": 2,
            "start_ms": 4000,
            "end_ms": 6000,
            "label": "lyric",
            "decision": "accept",
            "features": {"dominant_text": "MACHINE LYRIC"},
        },
        {
            "track_id": 3,
            "start_ms": 7000,
            "end_ms": 9000,
            "label": "lyric",
            "decision": "accept",
            "features": {"dominant_text": "TRACKING ERROR TRACK"},
        },
    ]

    human_labels = {
        "1": {
            "human_label": "lyric",
            "human_decision": "accept",
            "corrected_text": "HUMAN CORRECTED TEXT",
        },
        "2": {
            "human_label": "non_lyric",
            "human_decision": "reject",
        },
        "3": {
            "human_label": "lyric",
            "tracking_error": True,
        },
    }

    cues, skipped = select_cues(tracks, human_labels=human_labels)
    assert len(cues) == 1
    assert cues[0].source_track_ids == [1]
    assert cues[0].text == "HUMAN CORRECTED TEXT"
    assert cues[0].correction.method == "human"

    skipped_tids = [s["track_id"] for s in skipped]
    assert 2 in skipped_tids
    assert 3 in skipped_tids


def test_select_cues_multi_segment_track():
    tracks = [
        {
            "track_id": 42,
            "start_ms": 1000,
            "end_ms": 6000,
            "label": "lyric",
            "decision": "accept",
            "confidence": 0.98,
            "features": {"dominant_text": "I FEEL THE NIGHT EXPLODE", "mean_centroid": [0.5, 0.8]},
            "segments": [
                {
                    "segment_id": 0,
                    "start_ms": 1000,
                    "end_ms": 3000,
                    "text": "I FEEL THE NIGHT EXPLODE",
                    "confidence": 0.99,
                },
                {
                    "segment_id": 1,
                    "start_ms": 3500,
                    "end_ms": 6000,
                    "text": "WHEN WE ARE TOGETHER",
                    "confidence": 0.97,
                },
            ],
            "occurrences": [
                {"segment_id": 0, "timestamp_ms": 1500, "text_norm": "I FEEL THE NIGHT EXPLODE", "confidence": 0.99},
                {"segment_id": 1, "timestamp_ms": 4000, "text_norm": "WHEN WE ARE TOGETHER", "confidence": 0.97},
            ],
        }
    ]

    cues, skipped = select_cues(tracks, human_labels={})
    assert len(cues) == 2
    assert cues[0].source_track_ids == [42]
    assert cues[0].source_segment_ids == [0]
    assert cues[0].start_ms == 1000
    assert cues[0].end_ms == 3000
    assert cues[0].text == "I FEEL THE NIGHT EXPLODE"

    assert cues[1].source_track_ids == [42]
    assert cues[1].source_segment_ids == [1]
    assert cues[1].start_ms == 3500
    assert cues[1].end_ms == 6000
    assert cues[1].text == "WHEN WE ARE TOGETHER"


def test_select_cues_chorus_rescue():
    tracks = [
        {
            "track_id": 10,
            "start_ms": 45000,
            "end_ms": 48000,
            "label": "lyric",
            "decision": "review",
            "confidence": 0.85,
            "qa_flags": ["metadata_title_match", "title_match_outside_intro", "potential_chorus"],
            "features": {"dominant_text": "CASTLE", "mean_centroid": [0.5, 0.85]},
            "segments": [{"segment_id": 0, "start_ms": 45000, "end_ms": 48000, "text": "CASTLE", "confidence": 0.85}],
        }
    ]
    # Without include_review, chorus in reading zone is still rescued
    cues, skipped = select_cues(tracks, human_labels={})
    assert len(cues) == 1
    assert cues[0].text == "CASTLE"
    assert cues[0].review_required is True


def test_load_artifacts(tmp_path: Path):
    manifest = {"video_id": "test_vid", "probe": {"duration": 60}}
    tracks_data = {
        "total_tracks": 1,
        "tracks": [{"track_id": 10, "start_ms": 0, "end_ms": 2000, "label": "lyric"}],
    }
    label_line = {
        "video_id": "test_vid",
        "track_id": "10",
        "human_label": "lyric",
        "human_decision": "accept",
        "corrected_text": "TEST",
    }

    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "tracks.json").write_text(json.dumps(tracks_data), encoding="utf-8")
    (tmp_path / "labels.jsonl").write_text(json.dumps(label_line) + "\n", encoding="utf-8")

    m, tr, hl = load_artifacts(tmp_path)
    assert m["video_id"] == "test_vid"
    assert len(tr) == 1
    assert "10" in hl
    assert hl["10"]["corrected_text"] == "TEST"


def test_include_review_does_not_leak_non_lyric():
    # Track is non-lyric, under review with title_match_outside_intro flag
    tracks = [
        {
            "track_id": 99,
            "start_ms": 50000,
            "end_ms": 53000,
            "label": "non_lyric",
            "decision": "review",
            "qa_flags": ["metadata_title_match", "title_match_outside_intro"],
            "features": {"dominant_text": "CORNER TITLE CARD", "mean_centroid": [0.05, 0.05]},
            "segments": [{"segment_id": 0, "start_ms": 50000, "end_ms": 53000, "text": "CORNER TITLE CARD"}],
        }
    ]
    # Even with include_review=True, non_lyric must NEVER be accepted as lyric cue
    cues, skipped = select_cues(tracks, human_labels={}, include_review=True)
    assert len(cues) == 0
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "non_lyric"


def test_human_reject_always_wins_over_chorus_rescue():
    tracks = [
        {
            "track_id": 10,
            "start_ms": 45000,
            "end_ms": 48000,
            "label": "lyric",
            "decision": "review",
            "qa_flags": ["potential_chorus"],
            "features": {"dominant_text": "CASTLE", "mean_centroid": [0.5, 0.85]},
            "segments": [{"segment_id": 0, "start_ms": 45000, "end_ms": 48000, "text": "CASTLE"}],
        }
    ]
    human_labels = {
        "10": {
            "human_label": "non_lyric",
            "human_decision": "reject",
        }
    }
    cues, skipped = select_cues(tracks, human_labels=human_labels, include_review=True)
    assert len(cues) == 0
    assert len(skipped) == 1
    assert skipped[0]["decision"] == "reject"


def test_segment_level_human_labels_enforced():
    tracks = [
        {
            "track_id": 42,
            "label": "lyric",
            "decision": "accept",
            "features": {"dominant_text": "MULTI SEGMENT LINE"},
            "segments": [
                {"segment_id": 0, "start_ms": 1000, "end_ms": 3000, "text": "LINE ONE"},
                {"segment_id": 1, "start_ms": 3500, "end_ms": 5500, "text": "LINE TWO"},
            ],
        }
    ]
    # Human rejects segment 1 specifically
    human_labels = {
        "42.1": {
            "human_label": "non_lyric",
            "human_decision": "reject",
        }
    }
    cues, skipped = select_cues(tracks, human_labels=human_labels)
    assert len(cues) == 1
    assert cues[0].text == "LINE ONE"
    assert cues[0].source_segment_ids == [0]

    assert len(skipped) == 1
    assert skipped[0]["track_id"] == 42
    assert skipped[0]["segment_id"] == 1
    assert skipped[0]["reason"] == "non_lyric"


def test_multisegment_track_level_correction_not_repeated():
    tracks = [
        {
            "track_id": 42,
            "label": "lyric",
            "decision": "accept",
            "features": {"dominant_text": "RAW DOMINANT"},
            "segments": [
                {"segment_id": 0, "start_ms": 1000, "end_ms": 3000, "text": "FIRST ORIGINAL"},
                {"segment_id": 1, "start_ms": 3500, "end_ms": 5500, "text": "SECOND ORIGINAL"},
            ],
        }
    ]
    # Human sets corrected_text at TRACK level only
    human_labels = {
        "42": {
            "human_label": "lyric",
            "human_decision": "accept",
            "corrected_text": "GLOBAL TRACK TEXT",
        }
    }
    cues, skipped = select_cues(tracks, human_labels=human_labels, correct_method="none")
    assert len(cues) == 2
    # Neither segment should be overwritten by track-level corrected_text
    assert cues[0].text == "FIRST ORIGINAL"
    assert cues[1].text == "SECOND ORIGINAL"


def test_low_confidence_garbage_is_gated_to_review():
    garbage = "W.ihnit. warchme nowand"
    tracks = [
        {
            "track_id": 7,
            "start_ms": 1000,
            "end_ms": 3000,
            "label": "lyric",
            "decision": "accept",
            "confidence": 0.62,
            "features": {"dominant_text": garbage, "mean_centroid": [0.5, 0.8]},
            "segments": [
                {"segment_id": 0, "start_ms": 1000, "end_ms": 3000, "text": garbage, "confidence": 0.62}
            ],
        }
    ]
    cues, skipped = select_cues(tracks, human_labels={})
    assert cues == []
    assert len(skipped) == 1
    assert skipped[0]["reason"] in {"low_ocr_confidence", "low_text_quality"}
    assert skipped[0]["ocr_confidence"] == pytest.approx(0.62)
    assert skipped[0]["text_quality"] is not None

    cues_review, _ = select_cues(tracks, human_labels={}, include_review=True)
    assert len(cues_review) == 1
    assert cues_review[0].review_required is True
    assert cues_review[0].text_quality is not None


def test_high_confidence_garbled_multiword_line_is_gated():
    garbage = "Wntctomeyatot ine now and"
    tracks = [
        {
            "track_id": 9,
            "start_ms": 95000,
            "end_ms": 97500,
            "label": "lyric",
            "decision": "accept",
            "confidence": 0.93,
            "features": {"dominant_text": garbage, "mean_centroid": [0.5, 0.9]},
            "segments": [
                {"segment_id": 0, "start_ms": 95000, "end_ms": 97500, "text": garbage, "confidence": 0.93}
            ],
        }
    ]
    cues, skipped = select_cues(tracks, human_labels={})
    assert cues == []
    assert skipped[0]["reason"] == "low_text_quality"


def test_high_confidence_readable_lyric_still_accepted():
    tracks = [
        {
            "track_id": 8,
            "start_ms": 1000,
            "end_ms": 3000,
            "label": "lyric",
            "decision": "accept",
            "confidence": 0.95,
            "features": {"dominant_text": "I walk in like thunder", "mean_centroid": [0.5, 0.8]},
            "segments": [
                {"segment_id": 0, "start_ms": 1000, "end_ms": 3000, "text": "I walk in like thunder", "confidence": 0.95}
            ],
        }
    ]
    cues, skipped = select_cues(tracks, human_labels={})
    assert len(cues) == 1
    assert skipped == []
    assert cues[0].text == "I walk in like thunder"
    assert cues[0].text_quality is not None and cues[0].text_quality > 0.9


def test_select_cues_conservative_rescue_with_trusted_reference():
    tracks = [
        {
            "track_id": 9,
            "start_ms": 1000,
            "end_ms": 4000,
            "label": "non_lyric",
            "decision": "reject",
            "confidence": 0.85,
            "features": {"dominant_text": "YOUVE BEEN RUNNIN ROUND", "mean_centroid": [0.5, 0.8]},
            "segments": [
                {"segment_id": 0, "start_ms": 1000, "end_ms": 4000, "text": "YOUVE BEEN RUNNIN ROUND", "confidence": 0.85}
            ],
        }
    ]
    ref_lines = ["You've been runnin' round"]

    # 1. Trusted LRCLIB reference rescues a classifier-rejected track
    cues1, skipped1 = select_cues(
        tracks,
        human_labels={},
        reference_lines=ref_lines,
        reference_source="lrclib_synced",
    )
    assert len(cues1) == 1
    assert "RUNNIN" in cues1[0].text or "RUNNING" in cues1[0].text
    assert "reference_rescue" in cues1[0].qa_flags

    # 2. Captions NEVER rescue a non_lyric track
    cues2, skipped2 = select_cues(
        tracks,
        human_labels={},
        reference_lines=ref_lines,
        reference_source="youtube_captions",
    )
    assert len(cues2) == 0
    assert len(skipped2) == 1
    assert skipped2[0]["reason"] == "non_lyric"


def test_select_cues_reference_rescue_respects_human_and_card_rejects():
    def _track(**overrides):
        track = {
            "track_id": 10,
            "start_ms": 1000,
            "end_ms": 4000,
            "label": "non_lyric",
            "decision": "reject",
            "confidence": 0.85,
            "features": {"dominant_text": "YOUVE BEEN RUNNIN ROUND", "mean_centroid": [0.5, 0.8]},
            "segments": [
                {"segment_id": 0, "start_ms": 1000, "end_ms": 4000, "text": "YOUVE BEEN RUNNIN ROUND", "confidence": 0.85}
            ],
        }
        track.update(overrides)
        return track

    ref_lines = ["You've been runnin' round"]

    # Human reject wins over reference evidence.
    cues_human, skipped_human = select_cues(
        [_track()],
        human_labels={"10": {"human_label": "non_lyric", "human_decision": "reject"}},
        reference_lines=ref_lines,
        reference_source="lrclib_synced",
    )
    assert cues_human == []
    assert skipped_human[0]["decision"] == "reject"

    # Known card/credit subtype is not rescued.
    cues_card, _ = select_cues(
        [_track(qa_flags=["intro_bounded_card"])],
        human_labels={},
        reference_lines=ref_lines,
        reference_source="lrclib_synced",
    )
    assert cues_card == []

    # Rescue can be disabled through ReferenceConfig.
    from lyra_ocr.lyric.models import ReferenceConfig

    cues_disabled, _ = select_cues(
        [_track()],
        human_labels={},
        reference_lines=ref_lines,
        reference_source="lrclib_synced",
        reference_config=ReferenceConfig(rescue_non_lyric=False),
    )
    assert cues_disabled == []


def test_select_cues_carries_layer_id_and_lang():
    track = {
        "track_id": 42,
        "start_ms": 1000,
        "end_ms": 3000,
        "label": "lyric",
        "decision": "accept",
        "bbox": [100, 200, 300, 240],
        "features": {
            "dominant_text": "nageku uta mo kikoe nai yo",
            "layer_id": 3,
            "lang": "ja-Latn",
        },
        "segments": [
            {
                "segment_id": 0,
                "start_ms": 1000,
                "end_ms": 3000,
                "text": "nageku uta mo kikoe nai yo",
            }
        ],
    }
    cues, skipped = select_cues([track], human_labels={})
    assert len(cues) == 1
    assert cues[0].layer_id == 3
    assert cues[0].lang == "ja-Latn"
    assert cues[0].to_dict()["layer_id"] == 3
    assert cues[0].to_dict()["lang"] == "ja-Latn"


def test_select_cues_rescues_1frame_track():
    # 1-frame track at 1.0 FPS: start_ms == 5000, end_ms == 5000
    track = {
        "track_id": 77,
        "start_ms": 5000,
        "end_ms": 5000,
        "label": "non_lyric",
        "decision": "review",
        "bbox": [100, 200, 300, 240],
        "features": {
            "dominant_text": "Castle in the sky",
        },
        "segments": [
            {
                "segment_id": 0,
                "start_ms": 5000,
                "end_ms": 5000,
                "text": "Castle in the sky",
            }
        ],
    }
    from lyra_ocr.lyric.models import ReferenceConfig

    ref_lines = ["Castle in the sky"]
    cues, skipped = select_cues(
        [track],
        human_labels={},
        reference_lines=ref_lines,
        reference_source="lrclib_synced",
        reference_config=ReferenceConfig(rescue_non_lyric=True),
        sample_interval_ms=1000,
    )
    assert len(cues) == 1
    assert cues[0].text == "Castle in the sky"
    assert "reference_rescue" in cues[0].qa_flags


def test_select_cues_romaji_rescue_preserves_ocr_text():
    track = {
        "track_id": 83,
        "start_ms": 72000,
        "end_ms": 74000,
        "label": "non_lyric",
        "decision": "review",
        "bbox": [200, 330, 450, 355],
        "features": {
            "dominant_text": "nagekuluta mo kikoe nalyo",
            "lang": "ja-Latn",
            "layer_id": 2,
        },
        "segments": [
            {
                "segment_id": 0,
                "start_ms": 72000,
                "end_ms": 74000,
                "text": "nagekuluta mo kikoe nalyo",
            }
        ],
    }
    from lyra_ocr.lyric.models import ReferenceConfig

    ref_lines = ["nageku uta mo kikoe nai yo"]
    cues, skipped = select_cues(
        [track],
        human_labels={},
        reference_lines=ref_lines,
        reference_source="lrclib_romanized",
        reference_config=ReferenceConfig(rescue_non_lyric=True),
    )
    assert len(cues) == 1
    assert cues[0].decision == "accept"
    assert "reference_rescue" in cues[0].qa_flags
    assert cues[0].text == "nagekuluta mo kikoe nalyo"





