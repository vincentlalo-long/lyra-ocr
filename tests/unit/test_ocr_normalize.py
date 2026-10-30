"""Unit tests for Unicode normalization in ocr.py."""

import unicodedata
from lyra_ocr.ocr import normalize_text


def test_normalize_vietnamese_nfc():
    # Decomposed Vietnamese "Đường": Đ + u + \u031b (horn) + o + \u031b + \u0300 (grave) + n + g
    decomposed = "Đ\u01b0\u01a1\u0300ng"
    composed = "Đường"

    assert normalize_text(decomposed, form="NFC") == composed
    assert unicodedata.is_normalized("NFC", normalize_text(decomposed))


def test_strip_zero_width_chars():
    dirty = "We\u200b can\u200c be\u200d alone\ufeff"
    clean = normalize_text(dirty)
    assert clean == "We can be alone"
    assert "\u200b" not in clean
    assert "\ufeff" not in clean


def test_normalize_empty_and_whitespace():
    assert normalize_text("") == ""
    assert normalize_text("   \n\t  ") == ""


def test_normalize_preserves_nfkc():
    # Full-width Latin letters in Japanese contexts
    full_width = "ＬＹＲＡ"
    nfkc = normalize_text(full_width, form="NFKC")
    assert nfkc == "LYRA"
