import json
from pathlib import Path
import pytest

from lyra_ocr.lyric.captions import (
    fetch_youtube_caption_bundle,
    parse_caption_json3,
    parse_caption_vtt,
    resolve_caption_video_ref,
)

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "reference"


def test_parse_caption_json3_dedupes_rolling_events():
    data = json.loads((FIXTURES_DIR / "caption.json3").read_text(encoding="utf-8"))
    lines = parse_caption_json3(data)

    assert len(lines) == 2
    assert lines[0].text == "You've been runnin' round"
    assert lines[0].start_ms == 4000
    assert lines[0].source == "youtube_captions"
    assert lines[1].text == "Throwin' that dirt on my name"
    assert lines[1].start_ms == 7000


def test_parse_caption_vtt_strips_tags():
    text = (FIXTURES_DIR / "attention_caption.vtt").read_text(encoding="utf-8")
    lines = parse_caption_vtt(text)

    assert len(lines) == 2
    assert lines[0].text == "You've been runnin' round"
    assert lines[0].start_ms == 4000
    assert lines[0].end_ms == 6800
    assert lines[1].text == "throwin' that dirt on my name"
    assert lines[1].start_ms == 7000


def test_fetch_youtube_caption_bundle_mocked(monkeypatch):
    json3_data = json.loads((FIXTURES_DIR / "caption.json3").read_text(encoding="utf-8"))
    fake_bytes = json.dumps(json3_data).encode("utf-8")

    class FakeResponse:
        def read(self):
            return fake_bytes
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=6.0: FakeResponse())

    fake_info = {
        "language": "en",
        "automatic_captions": {
            "en": [{"ext": "json3", "url": "https://fake.caption/en.json3"}]
        },
    }

    class FakeYDL:
        def __init__(self, *args, **kwargs):
            pass
        def extract_info(self, url, download=False):
            return fake_info

    import yt_dlp
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)

    bundle = fetch_youtube_caption_bundle("https://youtube.com/watch?v=mock123")
    assert bundle is not None
    assert bundle.source == "youtube_captions"
    assert bundle.provider == "youtube"
    assert len(bundle.lines) == 2


def test_fetch_youtube_caption_bundle_skips_translated_captions_for_non_english(monkeypatch):
    fake_info = {
        "language": "ja",
        "automatic_captions": {
            "en": [{"ext": "json3", "url": "https://fake.caption/en_trans.json3"}]
        },
    }

    class FakeYDL:
        def __init__(self, *args, **kwargs):
            pass
        def extract_info(self, url, download=False):
            return fake_info

    import yt_dlp
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)

    bundle = fetch_youtube_caption_bundle("https://youtube.com/watch?v=mock_ja")
    assert bundle is None


def test_fetch_youtube_caption_bundle_handles_failure_gracefully(monkeypatch):
    class FakeYDL:
        def __init__(self, *args, **kwargs):
            pass
        def extract_info(self, url, download=False):
            raise RuntimeError("yt-dlp extract failed")

    import yt_dlp
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)

    bundle = fetch_youtube_caption_bundle("https://youtube.com/watch?v=err")
    assert bundle is None


def test_resolve_caption_video_ref_prefers_url_and_never_local_path(tmp_path: Path):
    video_path = tmp_path / "Attention [6QYcd7RggNU].mp4"
    video_path.write_bytes(b"fake")

    assert (
        resolve_caption_video_ref(
            "https://www.youtube.com/watch?v=6QYcd7RggNU",
            "local-id",
            {"video_path": str(video_path)},
        )
        == "https://www.youtube.com/watch?v=6QYcd7RggNU"
    )
    assert (
        resolve_caption_video_ref(
            "",
            "Charlie_Puth_-_Attention_Lyrics [6QYcd7RggNU]_96b7ed52",
            {"video_path": str(video_path)},
        )
        == "6QYcd7RggNU"
    )
    assert resolve_caption_video_ref("", "local-only-id", {"video_path": str(video_path)}) == "6QYcd7RggNU"
    assert resolve_caption_video_ref("", "local-only-id", {"video_path": "/tmp/plain.mp4"}) == ""


def test_parse_caption_json3_merges_rolling_substitutions():
    data = {
        "events": [
            {
                "tStartMs": 4000,
                "dDurationMs": 2000,
                "segs": [{"utf8": "You've been runnin round"}],
            },
            {
                "tStartMs": 5000,
                "dDurationMs": 2000,
                "segs": [{"utf8": "You've been running round"}],
            },
            {
                "tStartMs": 6000,
                "dDurationMs": 2000,
                "segs": [{"utf8": "You've been running round"}],
            },
        ]
    }

    lines = parse_caption_json3(data)

    assert len(lines) == 1
    assert lines[0].text == "You've been running round"
    assert lines[0].start_ms == 5000


def test_fetch_youtube_caption_bundle_rejects_foreign_spam_by_ocr_overlap(monkeypatch):
    # Mock yt-dlp returning Polish subtitle data
    class FakeYoutubeDL:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download=False):
            return {
                "id": "test_id",
                "title": "Nightcore - Castle",
                "subtitles": {
                    "pl": [{"url": "https://fake.url/pl.json3", "ext": "json3"}]
                },
                "automatic_captions": {},
            }

    import yt_dlp
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYoutubeDL)

    # Mock urllib urlopen returning Polish text
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            data = {
                "events": [
                    {"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "cześć"}]},
                    {"tStartMs": 5000, "dDurationMs": 2000, "segs": [{"utf8": "www.pro-chem.pl"}]},
                ]
            }
            return json.dumps(data).encode("utf-8")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=6.0: FakeResponse())

    ocr_lines = [
        "Won't you take me home",
        "Give me wings to fly",
        "Castle in the sky",
        "Higher than the sun",
        "Where the eagles cry",
    ]
    bundle = fetch_youtube_caption_bundle("test_id", ocr_lines=ocr_lines)
    assert bundle is not None
    assert not bundle.lines
    assert "caption_low_ocr_overlap" in bundle.warnings

