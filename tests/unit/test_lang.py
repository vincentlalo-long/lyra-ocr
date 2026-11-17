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


def test_detect_line_lang_spanish():
    assert detect_line_lang("Ese latido que va más allá del tiempo") == "es"
    assert detect_line_lang("por siempre con mi corazón y vida") == "es"
    assert detect_line_lang("todo el mundo bajo la luz") == "es"
    # Regular English with one stopword must not be mistaken for Spanish
    assert detect_line_lang("this is the day we wait for") == "en"


def test_detect_line_lang_english_silent_e_not_romaji():
    assert detect_line_lang("one more time") == "en"
    assert detect_line_lang("take me home") == "en"
    assert detect_line_lang("open your eyes") == "en"
    assert detect_line_lang("before I go alone") == "en"


def test_detect_line_lang_romaji_positive_indicators():
    assert detect_line_lang("kimi no kokoro wa") == "ja-Latn"
    assert detect_line_lang("sono kodou wa toki wo koete") == "ja-Latn"


def test_dare_romaji_and_english():
    # Japanese Romaji phrases containing "dare" (誰)
    assert detect_line_lang("dare ga tame ni") == "ja-Latn"
    assert detect_line_lang("dare mo") == "ja-Latn"
    assert romaji_score("dare da") >= 0.5
    # English phrase containing "dare"
    assert detect_line_lang("I dare you") == "en"


def test_romaji_score_tolerates_ocr_confusables():
    # Common Latin OCR confusables: ll->tt, vowel-l-vowel->t, trailing l->i, ck/c->o
    assert detect_line_lang("ille kcko ni nal kaze wo kanjite") == "ja-Latn"
    assert detect_line_lang("walashi no kolae wo kikoe nal yo") == "ja-Latn"
    assert detect_line_lang("fulari de tckiakasu ycake no sora") == "ja-Latn"
    assert detect_line_lang("ital hodo dckedo kokoro wa hitotsu") == "ja-Latn"
    assert romaji_score("walashi no kokoro") >= 0.6


def test_romaji_score_tolerates_single_english_loanword():
    # Mixed English loanword + strong Romaji morphology must not collapse to 0.0
    line = "Believe in starlight unmei sae uchinuite"
    assert romaji_score(line) >= 0.6
    assert detect_line_lang(line) == "ja-Latn"

    # Predominantly English lines with one non-romaji word or contractions must still stay English
    assert detect_line_lang("Running through the darkness tonight") == "en"
    assert romaji_score("Running through the darkness tonight") < 0.5
    assert detect_line_lang("You've been runnin' round") == "en"
    assert romaji_score("You've been runnin' round") < 0.5


def test_detect_line_lang_unaccented_spanish_and_romaji_disambiguation():
    # Unaccented Spanish (including common OCR typo "aumque")
    assert detect_line_lang("Aumque llore ya nada volvera a ser como antes") == "es"
    assert detect_line_lang("el futuro contigo siempre sera mejor") == "es"
    assert detect_line_lang("solo quiero olvidar este dolor") == "es"

    # Romaji lines containing particles that overlap _SPANISH_AMBIGUOUS_SHORT
    # (mi, te, se, ya, no, ni, de, o, a) must NOT be misclassified as Spanish
    assert detect_line_lang("mi te ya no ni de a kimi no sora") == "ja-Latn"
    assert detect_line_lang("kimi ni todoke yo kono omoi no mama de") == "ja-Latn"
    assert detect_line_lang("se no ni de koete yuku mirai e") == "ja-Latn"

    # Additional unaccented Spanish lines
    assert detect_line_lang("Aunque llore ya nada volvera a ser como antes") == "es"
    assert detect_line_lang("para siempre contigo") == "es"
    assert detect_line_lang("tu amor sera mi luz") == "es"


def test_detect_line_lang_english_vl_and_en_es_homographs():
    from lyra_ocr.lang import _normalize_romaji_ocr_token

    # English -Vl words must NOT rewrite trailing l->i
    for word in ("feel", "real", "soul", "heal", "deal", "goal", "fail", "sail"):
        assert _normalize_romaji_ocr_token(word) == word
    assert _normalize_romaji_ocr_token("nal") == "nai"
    assert _normalize_romaji_ocr_token("ital") == "itai"
    assert _normalize_romaji_ocr_token("ille") == "itte"
    assert _normalize_romaji_ocr_token("kcko") == "koko"

    # English -Vl lines must stay English with romaji_score == 0.0
    for en_vl in (
        "I feel my soul",
        "feel my soul",
        "you feel so real",
        "heal my soul",
        "real deal",
    ):
        assert romaji_score(en_vl) == 0.0
        assert detect_line_lang(en_vl) == "en"

    # Common English words hello/hero/zero and EN/ES homographs must stay English
    for en_line in (
        "a solo for me",
        "son of a gun",
        "hello my hero",
        "zero to hero",
        "the era of light",
        "my fatal angel",
    ):
        assert detect_line_lang(en_line) == "en"



