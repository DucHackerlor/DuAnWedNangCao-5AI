from core.labels import coco_vi, confidence_band, flower_vi, normalize_search_query, token_overlap


def test_coco_translation():
    assert coco_vi("dog") == "chó"
    assert coco_vi("cat") == "mèo"
    assert coco_vi("person") == "người"


def test_flower_translation():
    assert flower_vi("sunflowers") == "hoa hướng dương"
    assert flower_vi("roses") == "hoa hồng"


def test_query_translation_longest_phrase_first():
    assert "lobster" in normalize_search_query("tôm hùm trên đĩa")
    assert "dog" in normalize_search_query("chó và mèo")
    assert "cat" in normalize_search_query("chó và mèo")


def test_confidence_band():
    assert confidence_band(.8) == "cao"
    assert confidence_band(.6) == "trung bình"
    assert confidence_band(.3) == "thấp"


def test_token_overlap():
    assert token_overlap("yellow sunflower", "sunflower yellow") == 2
