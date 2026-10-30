"""Unit tests for script classification in ocr.py."""

from lyra_ocr.ocr import detect_script


def test_detect_script_latin():
    assert detect_script("Castle in the sky") == "latin"
    assert detect_script("Đường thì tối chơi vơi") == "latin"
    assert detect_script("12345!@#") == "latin"


def test_detect_script_japan():
    # Hiragana
    assert detect_script("あいうえお") == "japan"
    # Katakana
    assert detect_script("アイウエオ") == "japan"
    # Kanji only
    assert detect_script("雨降夜") == "japan"
    # Kanji + Kana
    assert detect_script("雨が降っている夜に") == "japan"


def test_detect_script_mixed():
    assert detect_script("J-POP 音楽") == "mixed"
    assert detect_script("Track 1: 桜") == "mixed"


def test_detect_script_empty():
    assert detect_script("") == "empty"
    assert detect_script("    ") == "empty"
