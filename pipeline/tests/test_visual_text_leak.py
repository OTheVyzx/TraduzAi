import numpy as np

from qa.visual_text_leak import detect_visual_text_leak


def test_detects_real_english_leak_from_final_ocr():
    flags = detect_visual_text_leak(page=54, final_ocr_text="YOUNG MASTER?!", expected_layers=[{"original": "YOUNG MASTER?!"}])

    assert flags[0]["type"] == "visual_text_leak"
    assert flags[0]["severity"] == "critical"


def test_identical_page_with_text_is_page_not_processed():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    flags = detect_visual_text_leak(page=1, original_image=image, final_image=image.copy(), expected_layers=[{"original": "HELLO"}])

    assert any(flag["type"] == "page_not_processed" for flag in flags)


def test_identical_page_without_text_is_not_flagged():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    flags = detect_visual_text_leak(page=1, original_image=image, final_image=image.copy(), expected_layers=[])

    assert flags == []


def test_allowed_proper_name_is_not_leak():
    flags = detect_visual_text_leak(page=1, final_ocr_text="Ghislain Perdium", expected_layers=[{"original": "Ghislain Perdium"}], allowed_terms=["Ghislain Perdium"])

    assert flags == []


def test_preserved_sfx_is_ignored():
    flags = detect_visual_text_leak(page=1, final_ocr_text="WHAT", expected_layers=[{"tipo": "sfx", "preserve_original": True, "original": "WHAT"}])

    assert flags == []
