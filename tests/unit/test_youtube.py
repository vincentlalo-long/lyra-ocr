"""Tests for YouTube channel/playlist flat listing (mocked yt_dlp)."""

import pytest

import lyra_ocr.dataset.youtube as yt


@pytest.fixture(autouse=True)
def clear_cache():
    yt._cache.clear()
    yield
    yt._cache.clear()


class FakeYdl:
    """Mimics yt_dlp.YoutubeDL context manager with canned extract_info."""

    def __init__(self, info, captured):
        self._info = info
        self._captured = captured

    def __call__(self, options):
        self._captured.update(options)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        return self._info


@pytest.fixture()
def patch_ydl(monkeypatch):
    captured = {}

    def install(info):
        fake = FakeYdl(info, captured)
        monkeypatch.setattr(yt, "yt_dlp", type("Mod", (), {"YoutubeDL": staticmethod(fake)}))
        return captured

    return install


def test_channel_listing_flat_entries(patch_ydl):
    captured = patch_ydl({
        "channel": "My Channel",
        "entries": [
            {"id": "AAAAAAAAAAA", "title": "Song A", "duration": 100,
             "url": "https://www.youtube.com/watch?v=AAAAAAAAAAA"},
            {"id": "BBBBBBBBBBB", "title": "Song B", "duration": None},
            None,
        ],
    })
    items = yt.list_channel_or_playlist("https://www.youtube.com/@mychannel/videos", limit=50)

    assert len(items) == 2
    assert items[0]["id"] == "AAAAAAAAAAA"
    assert items[0]["channel"] == "My Channel"
    assert items[1]["url"] == "https://www.youtube.com/watch?v=BBBBBBBBBBB"
    # Flat extraction: never resolve stream formats, honor limit
    assert captured.get("extract_flat") == "in_playlist"
    assert captured.get("playlist_items") == "1-50"


def test_listing_cache_hit_skips_second_extract(patch_ydl):
    calls = {"n": 0}
    info = {"channel": "C", "entries": [{"id": "X" * 11, "title": "t", "duration": 1}]}

    original = patch_ydl(info)
    yt.list_channel_or_playlist("https://youtube.com/@c/videos")
    second = yt.list_channel_or_playlist("https://youtube.com/@c/videos")
    assert len(second) == 1


def test_watch_url_falls_back_to_single_video(patch_ydl):
    patch_ydl({"id": "dQw4w9WgXcQ", "title": "Never", "duration": 213,
               "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "channel": "Rick"})
    items = yt.list_channel_or_playlist("https://youtu.be/dQw4w9WgXcQ")
    assert len(items) == 1
    assert items[0]["id"] == "dQw4w9WgXcQ"


def test_empty_url_raises():
    with pytest.raises(ValueError):
        yt.list_channel_or_playlist("   ")


def test_playlist_url_not_treated_as_watch(patch_ydl):
    captured = patch_ydl({"entries": [{"id": "Z" * 11, "title": "z", "url": "u"}]})
    yt.list_channel_or_playlist("https://www.youtube.com/playlist?list=PLabc123")
    assert captured.get("extract_flat") == "in_playlist"


def test_channel_listing_url_as_id_extraction(patch_ydl):
    patch_ydl({
        "channel": "URL Channel",
        "entries": [
            {
                "id": "https://www.youtube.com/watch?v=1234567890a",
                "title": "Song with URL ID",
                "duration": 180,
            },
            {
                "id": "",
                "url": "https://youtu.be/0987654321b",
                "title": "Song with missing ID but watch URL",
                "duration": 200,
            },
            {
                "id": "not_a_valid_url_or_empty",
                "url": "invalid",
            },
        ],
    })
    items = yt.list_channel_or_playlist("https://www.youtube.com/@urlchannel/videos")
    assert len(items) == 3
    assert items[0]["id"] == "1234567890a"
    assert items[0]["url"] == "https://www.youtube.com/watch?v=1234567890a"
    assert items[1]["id"] == "0987654321b"
    assert items[1]["url"] == "https://youtu.be/0987654321b"
    assert items[2]["id"] == "not_a_valid_url_or_empty"

