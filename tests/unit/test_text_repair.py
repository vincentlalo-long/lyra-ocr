"""Unit tests for reference-free Hepburn Romaji confusable repair."""

from lyra_ocr.lyric.text_repair import repair_romaji_confusables


def test_repair_romaji_confusables_fixes_rn_vv_cl_and_digits():
    assert repair_romaji_confusables("kimi rno narnida vva t0maranai") == "kimi mo namida wa tomaranai"
    assert repair_romaji_confusables("sono koclou wo k1mi he") == "sono kodou wo kimi he"
    assert repair_romaji_confusables("ovvari no nai yume") == "owari no nai yume"


def test_repair_romaji_confusables_preserves_standalone_numbers_and_normal_words():
    assert repair_romaji_confusables("Track 01 and 1000 dreams") == "Track 01 and 1000 dreams"
    assert repair_romaji_confusables("senbonzakura yoru ni magire") == "senbonzakura yoru ni magire"
