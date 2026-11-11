import json
from pathlib import Path

from lyra_ocr.lyric.build import build_lrc_from_artifacts
from lyra_ocr.lyric.models import CorrectionInfo, LyricCue, ReferenceConfig
from lyra_ocr.lyric.reference import (
    AlignmentResult,
    GapFillResult,
    MatchStats,
    ReferenceBundle,
    ReferenceLine,
    ReferenceQuery,
    align_reference,
    fetch_reference_bundle,
    fill_reference_gaps,
    line_match_stats,
    load_cached_reference_bundle,
    parse_lyrics_text,
    parse_reference_query,
    parse_reference_text,
    reconcile_reference_overlaps,
    reference_cache_dir,
    reference_cache_key,
    save_cached_reference_bundle,
)


def _cue(start_ms: int, text: str) -> LyricCue:
    return LyricCue(start_ms=start_ms, end_ms=start_ms + 3000, text=text)


def test_parse_lyrics_text_strips_timestamps_and_metadata():
    text = "[ti:Song]\n[ar:Artist]\n[00:01.00]Hello world\n\n[00:05.50]Second line\n"
    assert parse_lyrics_text(text) == ["Hello world", "Second line"]


def test_parse_reference_query_castle_cleaning():
    q = parse_reference_query("Nightcore - Castle - (Lyrics)")
    assert q.track == "Castle"
    assert q.artist in ("Nightcore", "")

    q2 = parse_reference_query(
        "1080P Full Senbonzakura (One Thousand Cherry Trees) - Hatsune Miku DIVA English Romaji",
        channel="Hatsune Miku",
    )
    assert "Senbonzakura" in q2.track
    assert q2.artist == "Hatsune Miku"


def test_fetch_reference_bundle_multi_query_and_ocr_overlap(monkeypatch):
    queries_made = []

    class FakeUrlopen:
        def __init__(self, req, timeout=6.0):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            queries_made.append(url)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            # Candidate 1: same title but wrong song
            # Candidate 2: right song matching OCR
            items = [
                {
                    "id": 101,
                    "trackName": "Castle",
                    "artistName": "Wrong Artist",
                    "duration": 180.0,
                    "syncedLyrics": "[00:01.00]Completely unrelated line\n[00:05.00]No matching lyrics",
                },
                {
                    "id": 102,
                    "trackName": "Castle",
                    "artistName": "Clarx & Harddope",
                    "duration": 180.0,
                    "syncedLyrics": "[00:01.00]Won't you take me home\n[00:05.00]Give me wings to fly\n[00:09.00]Castle in the sky",
                },
            ]
            return json.dumps(items).encode("utf-8")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", FakeUrlopen)

    ocr_lines = [
        "Won't you take me home",
        "Give me wings to fly",
        "Castle in the sky",
        "Higher than the sun",
        "Where the eagles cry",
    ]
    query = ReferenceQuery(track="Castle", artist="Nightcore", duration_ms=180000)
    bundle = fetch_reference_bundle(query, ocr_lines=ocr_lines)

    assert bundle.lines
    # Must have selected candidate 102
    assert any("Give me wings to fly" in l.text for l in bundle.lines)


def test_fetch_reference_bundle_rejects_wrong_song_by_ocr_overlap(monkeypatch):
    class FakeUrlopen:
        def __init__(self, req, timeout=6.0):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            items = [
                {
                    "id": 201,
                    "trackName": "Castle",
                    "artistName": "Wrong Artist",
                    "duration": 180.0,
                    "syncedLyrics": "[00:01.00]Nothing matches here\n[00:05.00]Totally different text",
                }
            ]
            return json.dumps(items).encode("utf-8")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", FakeUrlopen)

    ocr_lines = [
        "Won't you take me home",
        "Give me wings to fly",
        "Castle in the sky",
        "Higher than the sun",
        "Where the eagles cry",
    ]
    query = ReferenceQuery(track="Castle", artist="Nightcore", duration_ms=180000)
    bundle = fetch_reference_bundle(query, ocr_lines=ocr_lines)

    assert not bundle.lines
    assert "reference_low_ocr_overlap" in bundle.warnings


def test_romanize_bundle_with_pykakasi():
    from lyra_ocr.lyric.reference import romanize_bundle, ReferenceBundle, ReferenceLine

    lines = [
        ReferenceLine(text="千本桜 夜ニ紛レ", start_ms=1000),
        ReferenceLine(text="嘆ク唄モ 聞コエナイヨ", start_ms=5000),
    ]
    bundle = ReferenceBundle(lines=lines, source="lrclib_synced", provider="lrclib", language="ja")
    rom_bundle = romanize_bundle(bundle)

    assert rom_bundle is not None
    assert rom_bundle.source == "lrclib_romanized"
    assert rom_bundle.language == "ja-Latn"
    assert len(rom_bundle.lines) == 2
    assert "senbonzakura" in rom_bundle.lines[0].text.lower()
    assert "nageku" in rom_bundle.lines[1].text.lower()




def test_align_reference_repairs_text_and_keeps_ocr_timing():
    cues = [
        _cue(1000, "I FEEL THE N1GHT"),
        _cue(5000, "WHEN WERE TOGETHER"),
    ]
    reference = ["I feel the night", "When we're together", "Unrelated outro"]

    replaced = align_reference(cues, reference)

    assert replaced == 2
    assert cues[0].text == "I feel the night"
    assert cues[0].start_ms == 1000
    assert cues[1].text == "When we're together"
    assert cues[1].start_ms == 5000
    assert cues[0].correction.method == "reference"


def test_align_reference_does_not_overwrite_human_correction():
    cue = _cue(1000, "You've been runnin' round")
    cue.correction = CorrectionInfo(method="human", changed=True, confidence=1.0)

    replaced = align_reference([cue], ["You've been running around"], source="lrclib_synced")

    assert replaced == 0
    assert cue.text == "You've been runnin' round"
    assert cue.correction.method == "human"


def test_align_reference_leaves_unrelated_line_untouched():
    cues = [_cue(0, "instrumental break")]
    replaced = align_reference(cues, ["a completely different lyric line"])
    assert replaced == 0
    assert cues[0].text == "instrumental break"


def test_align_reference_handles_empty_inputs():
    assert align_reference([], ["line"]) == 0
    cue = _cue(0, "line")
    assert align_reference([cue], []) == 0
    assert cue.text == "line"


def test_build_lrc_applies_manual_reference_text(tmp_path: Path):
    artifacts_dir = tmp_path / "ref_art"
    artifacts_dir.mkdir()
    (artifacts_dir / "manifest.json").write_text(
        json.dumps({"video_id": "ref", "probe": {"duration": 10.0}}), encoding="utf-8"
    )
    tracks = {
        "tracks": [
            {
                "track_id": 1,
                "start_ms": 1000,
                "end_ms": 4000,
                "label": "lyric",
                "decision": "accept",
                "confidence": 0.95,
                "features": {"dominant_text": "I FEEL THE N1GHT", "mean_centroid": [0.5, 0.8]},
                "segments": [
                    {
                        "segment_id": 0,
                        "start_ms": 1000,
                        "end_ms": 4000,
                        "text": "I FEEL THE N1GHT",
                        "confidence": 0.95,
                    }
                ],
            }
        ]
    }
    (artifacts_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")

    out_lrc = tmp_path / "ref.lrc"
    res = build_lrc_from_artifacts(
        artifacts_dir,
        out_lrc,
        force=True,
        reference_text="[00:01.00]I feel the night\n",
    )

    assert res["n_reference_matches"] == 1
    assert res["reference_source"] == "manual"
    assert "[00:01.00]I feel the night" in out_lrc.read_text(encoding="utf-8")

    cues_data = json.loads((tmp_path / "ref.cues.json").read_text(encoding="utf-8"))
    assert cues_data["reference_source"] == "manual"
    assert cues_data["n_reference_matches"] == 1


def test_parse_reference_query_standard_formats():
    q1 = parse_reference_query("Charlie Puth - Attention (Lyrics)", "Charlie Puth", duration_ms=212000)
    assert q1.track.lower() == "attention"
    assert q1.artist.lower() == "charlie puth"
    assert q1.duration_ms == 212000
    assert q1.raw_title == "Charlie Puth - Attention (Lyrics)"

    q2 = parse_reference_query(
        "『Lyrics AMV』 Toaru Kagaku no Railgun S OP 1 Full 「sister's noise - fripSide」",
        "AnimeSongChannel",
        duration_ms=260000,
    )
    assert q2.track.lower() == "sister's noise"
    assert q2.artist.lower() == "fripside"

    q3 = parse_reference_query("Song Name (Official Video)", "My Band - Topic", duration_ms=180000)
    assert q3.track.lower() == "song name"
    assert q3.artist.lower() == "my band"

    q4 = parse_reference_query("Simple Song [Lyrics]", "Artist VEVO")
    assert q4.track.lower() == "simple song"
    assert q4.artist.lower() == "artist"


def test_parse_reference_text_preserves_lrc_timestamps():
    text = (
        "[ti:Attention]\n"
        "[ar:Charlie Puth]\n"
        "[00:04.00] You've been runnin' round\n"
        "[00:07.50] Throwin' that dirt on my name\n"
        "[Chorus]\n"
        "Plain line without timestamp\n"
    )
    lines = parse_reference_text(text, source="manual")
    assert len(lines) == 3
    assert lines[0].text == "You've been runnin' round"
    assert lines[0].start_ms == 4000
    assert lines[1].text == "Throwin' that dirt on my name"
    assert lines[1].start_ms == 7500
    assert lines[2].text == "Plain line without timestamp"
    assert lines[2].start_ms is None


def test_parse_reference_text_multiple_timestamps():
    text = "[00:02.00][00:08.00] Repeat chorus line"
    lines = parse_reference_text(text)
    assert len(lines) == 2
    assert lines[0].start_ms == 2000
    assert lines[0].text == "Repeat chorus line"
    assert lines[1].start_ms == 8000
    assert lines[1].text == "Repeat chorus line"


def test_fetch_reference_bundle_lrclib_synced(monkeypatch):
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "reference" / "lrclib_synced.json"
    fake_body = fixture_path.read_bytes()

    class FakeResponse:
        def read(self):
            return fake_body
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=6.0: FakeResponse())

    q = ReferenceQuery(track="Attention", artist="Charlie Puth", duration_ms=212000)
    bundle = fetch_reference_bundle(q)

    assert bundle.source == "lrclib_synced"
    assert bundle.provider == "lrclib"
    assert len(bundle.lines) == 3
    assert bundle.lines[0].text == "You've been runnin' round"
    assert bundle.lines[0].start_ms == 4000


def test_fetch_reference_bundle_rejects_unrelated_candidate(monkeypatch):
    import json
    import urllib.request

    body = json.dumps([
        {
            "trackName": "Completely Different Song",
            "artistName": "Another Artist",
            "duration": 100.0,
            "syncedLyrics": "[00:01.00] unrelated lyric",
        }
    ]).encode("utf-8")

    class FakeResponse:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=6.0: FakeResponse())
    q = ReferenceQuery(track="Attention", artist="Charlie Puth", duration_ms=212000)
    bundle = fetch_reference_bundle(q)

    assert bundle.source == ""
    assert bundle.lines == []
    assert "no_suitable_lrclib_match" in bundle.warnings


def test_fetch_reference_bundle_http_error(monkeypatch):
    import urllib.request
    import urllib.error

    def fake_error(*args, **kwargs):
        raise urllib.error.URLError("Network unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", fake_error)
    q = ReferenceQuery(track="Unknown", artist="Unknown", duration_ms=100000)
    bundle = fetch_reference_bundle(q)

    assert bundle.source == ""
    assert len(bundle.lines) == 0


def test_reference_bundle_cache_roundtrip(tmp_path: Path):
    cache_dir = reference_cache_dir(tmp_path)
    q = ReferenceQuery(track="Attention", artist="Charlie Puth", duration_ms=212000)
    bundle = ReferenceBundle(
        lines=[ReferenceLine(text="You've been runnin' round", start_ms=4000, source="lrclib_synced")],
        source="lrclib_synced",
        provider="lrclib",
    )
    key = reference_cache_key(q, provider="lrclib")
    save_path = save_cached_reference_bundle(cache_dir, key, bundle, q)
    assert save_path.exists()

    loaded = load_cached_reference_bundle(cache_dir, key)
    assert loaded is not None
    assert loaded.source == "lrclib_synced"
    assert loaded.provider == "lrclib"
    assert len(loaded.lines) == 1
    assert loaded.lines[0].text == "You've been runnin' round"
    assert loaded.lines[0].start_ms == 4000


def test_fetch_reference_bundle_uses_cache(tmp_path: Path, monkeypatch):
    cache_dir = reference_cache_dir(tmp_path)
    q = ReferenceQuery(track="Attention", artist="Charlie Puth", duration_ms=212000)

    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "reference" / "lrclib_synced.json"
    fake_body = fixture_path.read_bytes()

    network_calls = [0]
    class FakeResponse:
        def read(self):
            return fake_body
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    import urllib.request
    def fake_urlopen(req, timeout=6.0):
        network_calls[0] += 1
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    # First call: hits network and writes to cache
    bundle1 = fetch_reference_bundle(q, cache_dir=cache_dir)
    assert network_calls[0] == 1
    assert bundle1.source == "lrclib_synced"

    # Second call: served from cache, no network call
    bundle2 = fetch_reference_bundle(q, cache_dir=cache_dir)
    assert network_calls[0] == 1
    assert bundle2.source == "lrclib_synced"
    assert len(bundle2.lines) == len(bundle1.lines)

    # Changing artist changes key -> cache miss, network call made
    q_diff = ReferenceQuery(track="Attention", artist="Different Artist", duration_ms=212000)
    fetch_reference_bundle(q_diff, cache_dir=cache_dir)
    assert network_calls[0] == 2


def test_line_match_stats_computation():
    stats = line_match_stats("I feel the night", "I feel the night")
    assert stats.similarity == 1.0
    assert stats.token_coverage == 1.0
    assert 0.99 <= stats.length_ratio <= 1.01

    stats_diff = line_match_stats("short", "A much longer sentence that cannot match well")
    assert stats_diff.similarity < 0.50
    assert stats_diff.length_ratio > 2.0


def test_align_reference_source_aware_thresholds():
    # OCR has slight typo giving ~0.87 similarity
    ocr_text = "Youve been roamin round"
    ref_text = "You've been runnin' round"

    # LRCLIB source: replace threshold is 0.85 -> replaces (sim 0.87 >= 0.85)
    cues_lrclib = [_cue(1000, ocr_text)]
    res1 = align_reference(cues_lrclib, [ref_text], source="lrclib_synced")
    assert res1.replaced_count == 1
    assert cues_lrclib[0].text == ref_text
    assert cues_lrclib[0].correction.method == "reference"

    # Captions source: replace threshold is 0.90 -> review only since sim < 0.90
    cues_caption = [_cue(1000, ocr_text)]
    res2 = align_reference(cues_caption, [ref_text], source="youtube_captions")
    assert res2.replaced_count == 0
    assert res2.review_count == 1
    assert cues_caption[0].text == ocr_text  # kept OCR
    assert cues_caption[0].review_required is True
    assert "reference_uncertain" in cues_caption[0].qa_flags


def test_align_reference_guards_against_short_fragment():
    cues = [_cue(1000, "I have been walking down this street for hours")]
    ref = ["street"]
    res = align_reference(cues, ref, source="lrclib_synced")
    assert res.replaced_count == 0
    assert cues[0].text == "I have been walking down this street for hours"


def test_reconcile_reference_overlaps_splits_consecutive_lines():
    cues = [
        _cue(9000, "YOUVE BEEN RUNNIN ROUND RUNNIN ROUND RUNNIN"),
        _cue(9000, "ROUND THROWIN THAT DIRT ALL ON MY NAME"),
    ]
    ref_bundle = ReferenceBundle(
        lines=[
            ReferenceLine(
                text="You've been running 'round, running 'round",
                start_ms=10720,
                end_ms=12700,
            ),
            ReferenceLine(
                text="Running 'round, throwing that dirt all on my name",
                start_ms=12790,
                end_ms=15700,
            ),
        ],
        source="lrclib_synced",
    )
    align_res = align_reference(cues, ref_bundle, source="lrclib_synced")
    assert align_res.replaced_count == 2

    reconciled, warnings = reconcile_reference_overlaps(
        cues,
        ref_bundle,
        matched_pairs=align_res.matched_pairs,
    )
    assert warnings == []
    reconciled.sort(key=lambda c: (c.start_ms, c.end_ms))
    assert reconciled[0].start_ms == 9000
    assert reconciled[0].end_ms == reconciled[1].start_ms
    assert reconciled[1].end_ms == 12000
    assert all("reference_timing_split" in c.qa_flags for c in reconciled)


def test_fill_reference_gaps_inserts_missing_line():
    cues = [
        LyricCue(start_ms=1000, end_ms=4000, text="First line here"),
        LyricCue(start_ms=10000, end_ms=13000, text="Third line here"),
    ]
    ref_bundle = ReferenceBundle(
        lines=[
            ReferenceLine(text="First line here", start_ms=1000, end_ms=4000),
            ReferenceLine(text="Recovered second line", start_ms=5500, end_ms=8500),
            ReferenceLine(text="Third line here", start_ms=10000, end_ms=13000),
        ],
        source="lrclib_synced",
    )
    merged, gap_res = fill_reference_gaps(cues, ref_bundle, duration_ms=15000)

    assert gap_res.n_filled == 1
    assert len(merged) == 3
    assert merged[1].text == "Recovered second line"
    assert merged[1].start_ms == 5500
    assert merged[1].end_ms == 8500
    assert merged[1].correction.method == "reference_gap_fill"


def test_fill_reference_gaps_skips_occupied_and_plain():
    # Gap overlap with an existing cue: skip and report a warning.
    cues = [
        LyricCue(start_ms=1000, end_ms=7000, text="First line here"),
        LyricCue(start_ms=10000, end_ms=13000, text="Third line here"),
    ]
    ref_bundle = ReferenceBundle(
        lines=[
            ReferenceLine(text="First line here", start_ms=1000, end_ms=7000),
            ReferenceLine(text="Overlapping line", start_ms=5000, end_ms=8000),
            ReferenceLine(text="Third line here", start_ms=10000, end_ms=13000),
        ],
        source="lrclib_synced",
    )
    merged1, gap_res1 = fill_reference_gaps(cues, ref_bundle)
    assert gap_res1.n_filled == 0
    assert len(merged1) == 2
    assert gap_res1.warnings

    # Plain text reference (no timestamps) -> cannot gap fill
    ref_plain = ReferenceBundle(
        lines=[ReferenceLine(text="Plain text line", start_ms=None)],
        source="lrclib_plain",
    )
    merged2, gap_res2 = fill_reference_gaps(cues, ref_plain)
    assert gap_res2.n_filled == 0


def test_fill_reference_gaps_skips_edges_by_default():
    cues = [LyricCue(start_ms=1000, end_ms=3000, text="First line here")]
    ref_bundle = ReferenceBundle(
        lines=[
            ReferenceLine(text="First line here", start_ms=1000, end_ms=3000),
            ReferenceLine(text="Edge caption line", start_ms=8000, end_ms=12000),
        ],
        source="youtube_captions",
    )

    merged, gap_res = fill_reference_gaps(cues, ref_bundle, duration_ms=10000)
    assert gap_res.n_filled == 0

    merged_edges, gap_edges = fill_reference_gaps(
        cues,
        ref_bundle,
        duration_ms=10000,
        allow_edge_gap_fill=True,
    )
    assert gap_edges.n_filled == 1
    assert merged_edges[1].end_ms == 10000  # clipped
    assert "caption_gap_fill" in merged_edges[1].qa_flags


def test_fill_reference_gaps_allows_repeated_chorus_between_anchors():
    cues = [
        LyricCue(start_ms=1000, end_ms=3000, text="First line here"),
        LyricCue(start_ms=9000, end_ms=11000, text="Third line here"),
        LyricCue(start_ms=20000, end_ms=22000, text="Fifth line here"),
    ]
    ref_bundle = ReferenceBundle(
        lines=[
            ReferenceLine(text="First line here", start_ms=1000, end_ms=3000),
            ReferenceLine(text="Repeated chorus", start_ms=5000, end_ms=7000),
            ReferenceLine(text="Third line here", start_ms=9000, end_ms=11000),
            ReferenceLine(text="Repeated chorus", start_ms=15000, end_ms=17000),
            ReferenceLine(text="Fifth line here", start_ms=20000, end_ms=22000),
        ],
        source="lrclib_synced",
    )
    merged, gap_res = fill_reference_gaps(cues, ref_bundle, duration_ms=25000)
    assert gap_res.n_filled == 2
    assert [c.text for c in gap_res.filled_cues] == ["Repeated chorus", "Repeated chorus"]







def test_line_match_stats_tokenizes_spaced_and_unspaced_cjk_consistently():
    stats = line_match_stats(
        "刻み続けていた時の中でやっと君に達えたから",
        "刻み続けていた時の中で やっと君に逢えたから",
    )

    assert stats.similarity > 0.85
    assert stats.token_coverage > 0.85
    assert 0.9 <= stats.length_ratio <= 1.1
