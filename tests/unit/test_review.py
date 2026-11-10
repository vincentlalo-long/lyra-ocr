"""Unit tests for review and tracks artifact generation (review.py)."""

import csv
import json
import os
import tempfile
from lyra_ocr.track import TextTrack, TextSegment, DetectionOccurrence
from lyra_ocr.classify import ClassificationResult


def test_export_tracks_json_and_review_csv():
    from lyra_ocr.review import (
        export_tracks_json,
        export_review_csv,
        append_human_label,
        load_human_labels,
    )

    tr1 = TextTrack(track_id=1)
    tr1.start_ms = 20000
    tr1.end_ms = 24500
    tr1.bbox = [274, 2, 1001, 55]
    tr1.classification = ClassificationResult(
        label="lyric",
        confidence=0.99,
        reason="Recurring text changes in stable region",
        decision="accept",
    )
    tr1.segments = [
        TextSegment(
            segment_id=0,
            start_ms=20000,
            end_ms=24500,
            text="We can be alone in the castle in the sky",
            confidence=0.99,
            num_occurrences=5,
        )
    ]
    tr1.occurrences = [
        DetectionOccurrence(
            frame_id=0,
            timestamp_ms=20000,
            polygon=[[274, 2], [1001, 2], [1001, 55], [274, 55]],
            bbox=[274, 2, 1001, 55],
            text_raw="We can be alone in the castle in the sky",
            text_norm="We can be alone in the castle in the sky",
            confidence=0.99,
            model_id="PP-OCRv6_rec_small",
            script="latin",
        )
    ]
    tr1.features = {"lifetime_ratio": 0.05, "n_text_changes": 0}

    tr2 = TextTrack(track_id=2)
    tr2.start_ms = 0
    tr2.end_ms = 140000
    tr2.bbox = [50, 50, 200, 80]
    tr2.classification = ClassificationResult(
        label="non_lyric",
        confidence=0.98,
        reason="Matched persistent artist watermark",
        decision="reject",
    )
    tr2.segments = [
        TextSegment(
            segment_id=0,
            start_ms=0,
            end_ms=140000,
            text="CLARX & HARDDOPE",
            confidence=0.98,
            num_occurrences=50,
        )
    ]
    tr2.occurrences = []
    tr2.features = {"lifetime_ratio": 1.0, "n_text_changes": 0}

    tracks = [tr1, tr2]

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "tracks.json")
        csv_path = os.path.join(tmpdir, "review.csv")

        export_tracks_json(tracks, json_path)
        export_review_csv(tracks, csv_path)

        assert os.path.exists(json_path)
        assert os.path.exists(csv_path)

        # Check tracks.json structure (including occurrences)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "total_tracks" in data
            assert data["total_tracks"] == 2
            assert len(data["tracks"]) == 2
            t0 = data["tracks"][0]
            assert t0["track_id"] == 1
            assert t0["label"] == "lyric"
            assert t0["decision"] == "accept"
            assert "reason" in t0
            assert "segments" in t0
            assert "occurrences" in t0
            assert len(t0["occurrences"]) == 1
            assert t0["occurrences"][0]["frame_id"] == 0
            assert t0["occurrences"][0]["text_norm"] == "We can be alone in the castle in the sky"

        # Check review.csv format matches doc/SPECIFICATION.md
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            # Headers: track_id,start_ms,end_ms,machine_label,machine_decision,confidence,suggested_text,human_label,human_decision,review_note
            expected_headers = {
                "track_id",
                "start_ms",
                "end_ms",
                "machine_label",
                "machine_decision",
                "confidence",
                "suggested_text",
                "human_label",
                "human_decision",
                "review_note",
            }
            assert expected_headers.issubset(set(reader.fieldnames or []))
            assert rows[0]["track_id"] == "1"
            assert rows[0]["machine_label"] == "lyric"
            assert rows[0]["machine_decision"] == "accept"
            assert rows[0]["human_label"] == ""
            assert rows[0]["human_decision"] == ""
            assert rows[0]["suggested_text"] == "We can be alone in the castle in the sky"
            assert rows[1]["machine_label"] == "non_lyric"
            assert rows[1]["machine_decision"] == "reject"

        # Test labels.jsonl human annotation workflow
        labels_path = os.path.join(tmpdir, "labels.jsonl")
        append_human_label(
            labels_path,
            track_id="1",
            human_label="lyric",
            human_decision="accept",
            annotator="tester",
            note="Verified by reviewer",
        )
        assert os.path.exists(labels_path)
        labels_map = load_human_labels(labels_path)
        assert "1" in labels_map
        assert labels_map["1"]["human_label"] == "lyric"

        # Re-export CSV with labels_jsonl linked
        export_review_csv(tracks, csv_path, labels_jsonl_path=labels_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            updated_rows = list(reader)
            assert updated_rows[0]["human_label"] == "lyric"
            assert updated_rows[0]["human_decision"] == "accept"


def test_segment_review_csv_roundtrip(tmp_path):
    from lyra_ocr.review import export_review_csv, export_tracks_json
    from lyra_ocr.labels import ingest_label_batch_csv
    from lyra_ocr.lyric.select import load_artifacts, select_cues

    tr = TextTrack(track_id=42)
    tr.start_ms = 1000
    tr.end_ms = 8000
    tr.bbox = [100, 100, 400, 200]
    tr.classification = ClassificationResult(
        label="lyric",
        confidence=0.95,
        reason="lyric",
        decision="accept",
    )
    tr.segments = [
        TextSegment(segment_id=0, start_ms=1000, end_ms=4000, text="RAW LINE ONE", confidence=0.9, num_occurrences=3),
        TextSegment(segment_id=1, start_ms=4500, end_ms=8000, text="RAW LINE TWO", confidence=0.9, num_occurrences=3),
    ]
    tr.occurrences = []
    tr.features = {"dominant_text": "RAW LINE ONE", "mean_centroid": [0.5, 0.8]}

    # 1. Export tracks and manifest
    export_tracks_json([tr], tmp_path / "tracks.json")
    (tmp_path / "manifest.json").write_text(json.dumps({"video_id": "test_vid", "probe": {"duration": 10.0}}))

    # 2. Export review CSV (should produce 42.0 and 42.1)
    csv_path = tmp_path / "review.csv"
    export_review_csv([tr], csv_path, video_id="test_vid")

    # 3. Simulate human editing the CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["track_id"] == "42.0"
    assert rows[1]["track_id"] == "42.1"

    # Annotate row 1
    rows[0]["human_label"] = "lyric"
    rows[0]["human_decision"] = "accept"
    rows[0]["corrected_text"] = "CORRECTED LINE ONE"

    # Annotate row 2
    rows[1]["human_label"] = "lyric"
    rows[1]["human_decision"] = "accept"
    rows[1]["corrected_text"] = "CORRECTED LINE TWO"

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 4. Ingest CSV
    stats = ingest_label_batch_csv(tmp_path, csv_path)
    assert stats["errors"] == 0
    assert stats["ingested"] == 2

    # 5. Load artifacts and select cues
    manifest, tracks, human_labels = load_artifacts(tmp_path)
    cues, skipped = select_cues(tracks, human_labels)
    assert len(cues) == 2
    assert cues[0].text == "CORRECTED LINE ONE"
    assert cues[1].text == "CORRECTED LINE TWO"

