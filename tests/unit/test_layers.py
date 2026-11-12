"""Unit tests for spatial text layer discovery (layers.py)."""

import pytest
from lyra_ocr.layers import TextLayer, discover_layers
from lyra_ocr.track import TextTrack, DetectionOccurrence


def _make_track(
    track_id: int,
    start_ms: int,
    end_ms: int,
    text: str,
    cx: float,
    cy: float,
    w: float = 0.4,
    h: float = 0.04,
    frame_w: int = 1280,
    frame_h: int = 720,
) -> TextTrack:
    x1 = int((cx - w / 2) * frame_w)
    x2 = int((cx + w / 2) * frame_w)
    y1 = int((cy - h / 2) * frame_h)
    y2 = int((cy + h / 2) * frame_h)
    bbox = [x1, y1, x2, y2]
    occ = DetectionOccurrence(
        frame_id=0,
        timestamp_ms=start_ms,
        polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        bbox=bbox,
        text_raw=text,
        text_norm=text,
        confidence=0.98,
        model_id="test",
        script="latin",
    )
    t = TextTrack(track_id=track_id, occurrences=[occ], start_ms=start_ms, end_ms=end_ms, bbox=bbox)
    t.features = {
        "dominant_text": text,
        "mean_centroid": [cx, cy],
        "lifetime_ms": end_ms - start_ms,
    }
    return t


def test_discover_layers_castle_pattern():
    # Castle: 10 lyric lines at top (cy ≈ 0.04, cx ≈ 0.50), plus 2 static overlay tracks at bottom (cy ≈ 0.95)
    tracks = []
    # Top dynamic lyrics
    lyric_texts = [
        "Won't you take me home",
        "Give me wings to fly",
        "Castle in the sky",
        "Higher than the sun",
        "Where the eagles cry",
        "Take me to the stars",
        "Lighting up the dark",
        "Running out of time",
        "Keep the dream alive",
        "Never look behind",
    ]
    for i, txt in enumerate(lyric_texts):
        tracks.append(_make_track(track_id=i + 1, start_ms=i * 2000, end_ms=i * 2000 + 1500, text=txt, cx=0.50, cy=0.04))

    # Bottom static watermark/banner (same text repeated)
    tracks.append(_make_track(track_id=101, start_ms=0, end_ms=5000, text="CLARX & HARDDOPE", cx=0.50, cy=0.92))
    tracks.append(_make_track(track_id=102, start_ms=5000, end_ms=10000, text="CLARX & HARDDOPE", cx=0.50, cy=0.92))
    tracks.append(_make_track(track_id=103, start_ms=10000, end_ms=15000, text="CLARX & HARDDOPE", cx=0.50, cy=0.92))
    tracks.append(_make_track(track_id=104, start_ms=15000, end_ms=20000, text="CLARX & HARDDOPE", cx=0.50, cy=0.92))

    layers = discover_layers(tracks)
    assert len(layers) >= 2

    # Verify top layer is recognized as lyric
    top_layers = [l for l in layers if 0.01 <= l.cy_range[0] <= 0.08 and l.kind == "lyric"]
    assert len(top_layers) == 1
    assert top_layers[0].lang == "en"
    assert top_layers[0].n_distinct_texts >= 8

    # Verify bottom layer is recognized as overlay
    bottom_layers = [l for l in layers if l.cy_range[0] >= 0.85]
    assert len(bottom_layers) >= 1
    assert all(l.kind == "overlay" for l in bottom_layers)

    # Verify tracks have layer features populated
    for t in tracks[:10]:
        assert t.features.get("layer_kind") == "lyric"
        assert t.features.get("lang") == "en"


def test_discover_layers_senbonzakura_pattern():
    # 3 distinct lyric bands + 1 top-right watermark
    tracks = []
    # 1. Kanji top layer (cy ≈ 0.02, cx ≈ 0.50)
    ja_texts = [
        "大胆不敵にハイカラ革命",
        "磊々落々反戦国家",
        "日の丸印の二輪車転がし",
        "悪霊退散 ICBM",
        "環状線を走り抜けて",
    ]
    for i, txt in enumerate(ja_texts):
        tracks.append(_make_track(track_id=10 + i, start_ms=i * 2000, end_ms=i * 2000 + 1500, text=txt, cx=0.50, cy=0.02))

    # 2. English lower layer (cy ≈ 0.91, cx ≈ 0.50)
    en_texts = [
        "Intrepidly raising a plush revolution",
        "Free and easy an antiwar country",
        "Peddling a bicycle marked with flag",
        "Exorcising evil spirits ICBM",
        "Running through ring roads",
    ]
    for i, txt in enumerate(en_texts):
        tracks.append(_make_track(track_id=20 + i, start_ms=i * 2000, end_ms=i * 2000 + 1500, text=txt, cx=0.50, cy=0.91))

    # 3. Romaji bottom layer (cy ≈ 0.96, cx ≈ 0.50)
    ro_texts = [
        "daitan futeki ni haikara kakumei",
        "rairairakuraku hansen kokka",
        "hinomaru jirushi no nirinsha korogashi",
        "akuryo taisan aishibiemu",
        "kanjosen o hashirinukete",
    ]
    for i, txt in enumerate(ro_texts):
        tracks.append(_make_track(track_id=30 + i, start_ms=i * 2000, end_ms=i * 2000 + 1500, text=txt, cx=0.50, cy=0.96))

    # 4. Top-right CREDIT(S) watermark (cy ≈ 0.04, cx ≈ 0.85)
    credit_texts = ["CREDIT(S)", "CREDIT (S)", "CREDITISI", "CREDIT (S)"]
    for i, txt in enumerate(credit_texts):
        tracks.append(_make_track(track_id=40 + i, start_ms=i * 2000, end_ms=i * 2000 + 1500, text=txt, cx=0.85, cy=0.04))

    layers = discover_layers(tracks)

    kinds_and_langs = {(l.kind, l.lang) for l in layers}
    assert ("lyric", "ja") in kinds_and_langs
    assert ("lyric", "en") in kinds_and_langs
    assert ("lyric", "ja-Latn") in kinds_and_langs
    assert ("overlay", "en") in kinds_and_langs or ("overlay", "und") in kinds_and_langs


def test_classify_all_tracks_with_layer_discovery():
    from lyra_ocr.classify import classify_all_tracks

    tracks = []
    # Top dynamic lyrics at cy=0.04 (normally outside old reading zone 0.20-0.95)
    lyric_texts = [
        "Won't you take me home",
        "Give me wings to fly",
        "Castle in the sky",
        "Higher than the sun",
        "Where the eagles cry",
    ]
    for i, txt in enumerate(lyric_texts):
        tracks.append(_make_track(track_id=i + 1, start_ms=i * 2000, end_ms=i * 2000 + 1500, text=txt, cx=0.50, cy=0.04))

    # Watermark
    for i in range(4):
        tracks.append(_make_track(track_id=50 + i, start_ms=i * 2000, end_ms=i * 2000 + 1500, text="CREDIT(S)", cx=0.85, cy=0.04))

    results = classify_all_tracks(tracks, sample_interval_ms=1000, video_duration_ms=60000)

    # Top lyrics MUST be accepted under Rule L1
    for r in results[:5]:
        assert r.label == "lyric"
        assert r.decision == "accept"
        assert "lyric_layer_member" in r.qa_flags

    # Watermark MUST be rejected or review as overlay
    for r in results[5:]:
        assert r.label == "non_lyric"
        assert "static_overlay_layer" in r.qa_flags


def test_discover_layers_rejects_left_aligned_staff_credits():
    # 6 staff credit tracks aligned on the left (cx ≈ 0.18, cy ≈ 0.45)
    credit_texts = [
        "DIRECTOR: NAGAI TATSUYUKI",
        "SERIES COMPOSITION: MINAKAMI SEISHI",
        "CHARACTER DESIGN: TANAKA MASAYOSHI",
        "MUSIC: I'VE SOUND / MAIKO",
        "ANIMATION PRODUCTION: J.C.STAFF",
        "PRODUCED BY: PROJECT-RAILGUN",
    ]
    tracks = []
    for i, txt in enumerate(credit_texts):
        tracks.append(_make_track(track_id=200 + i, start_ms=i * 2000, end_ms=i * 2000 + 1800, text=txt, cx=0.18, cy=0.45))

    layers = discover_layers(tracks)
    assert len(layers) == 1
    # Left-aligned cluster must be classified as overlay, NEVER lyric
    assert layers[0].kind == "overlay"


def test_discover_layers_rejects_hud_code_tracks():
    # 5 HUD animation codes centered vertically and horizontally
    hud_texts = ["00A01", "00A02", "00A03", "00A04", "00A05"]
    tracks = []
    for i, txt in enumerate(hud_texts):
        tracks.append(_make_track(track_id=300 + i, start_ms=i * 1000, end_ms=i * 1000 + 800, text=txt, cx=0.50, cy=0.50))

    layers = discover_layers(tracks)
    assert len(layers) == 1
    assert layers[0].kind == "overlay"

