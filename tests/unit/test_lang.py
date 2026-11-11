import pytest

from lyra_ocr.lang import detect_line_lang, romaji_score, vote_layer_lang


def test_detect_line_lang_japanese_kana():
    assert detect_line_lang("嘆ク唄モ 聞コエナイヨ") == "ja"
    assert detect_line_lang("その光線銃で打ち抜いて") == "ja"
    assert detect_line_lang("千本桜 夜ニ紛レ") == "ja"
    assert detect_line_lang("安楽净土厄払い") == "ja"


def test_detect_line_lang_korean_hangul():
    assert detect_line_lang("안녕하세요 반갑습니다") == "ko"
    assert detect_line_lang("보고 싶다 이렇게 말하니까") == "ko"


def test_detect_line_lang_vietnamese():
    assert detect_line_lang("Xin chào các bạn đã đến với chúng tôi") == "vi"
    assert detect_line_lang("Đường về nhà là vào tim ta") == "vi"


def test_detect_line_lang_pure_cjk():
    assert detect_line_lang("天地玄黄 宇宙洪荒") == "zh"
    assert detect_line_lang("百戦錬磨") == "zh"


def test_romaji_score_and_detection():
    # Romaji lyrics
    r1 = "nageku uta mo kikoe nai yo"
    r2 = "seiran no sora haruka kanata"
    r3 = "daitan futeki ni haikara kakumei"
    assert romaji_score(r1) >= 0.6
    assert romaji_score(r2) >= 0.6
    assert romaji_score(r3) >= 0.6
    assert detect_line_lang(r1) == "ja-Latn"
    assert detect_line_lang(r2) == "ja-Latn"
    assert detect_line_lang(r3) == "ja-Latn"

    # English lyrics and short ambiguous words
    e1 = "A wailing song cannot even be heard"
    e2 = "Won't you take me home"
    e3 = "Give me wings to fly"
    e4 = "Castle in the sky"
    e5 = "go to me so"
    assert romaji_score(e1) < 0.4
    assert romaji_score(e2) < 0.4
    assert romaji_score(e3) < 0.4
    assert romaji_score(e4) < 0.4
    assert romaji_score(e5) < 0.5
    assert detect_line_lang(e1) == "en"
    assert detect_line_lang(e2) == "en"
    assert detect_line_lang(e3) == "en"
    assert detect_line_lang(e4) == "en"
    assert detect_line_lang(e5) == "en"


def test_detect_line_lang_empty_and_noise():
    assert detect_line_lang("") == "und"
    assert detect_line_lang("   ") == "und"
    assert detect_line_lang("12345 !@#$%") == "und"


def test_vote_layer_lang():
    assert vote_layer_lang([]) == "und"
    assert vote_layer_lang(["ja-Latn", "ja-Latn", "en"]) == "ja-Latn"
    assert vote_layer_lang(["en", "en", "ja"]) == "en"
    # Promotion: pure CJK (zh) promoted to ja when layer contains Japanese kana lines
    assert vote_layer_lang(["zh", "zh", "ja"]) == "ja"
    # Pure Chinese layer remains zh
    assert vote_layer_lang(["zh", "zh", "zh"]) == "zh"
