from vision_stack.oar_ocr_adapter import load_oar_ocr_regions, parse_oar_ocr_payload


def test_parse_oar_ocr_regions_with_word_boxes():
    payload = {
        "text_regions": [
            {
                "text": "ガシャーン",
                "bbox": [10, 20, 80, 50],
                "wordBoxes": [[10, 20, 40, 50], [42, 20, 80, 50]],
            }
        ]
    }

    regions = parse_oar_ocr_payload(payload, width=100, height=100)

    assert regions[0]["text"] == "ガシャーン"
    assert regions[0]["word_boxes"] == [[10, 20, 40, 50], [42, 20, 80, 50]]
    assert regions[0]["source"] == "oar-ocr"


def test_load_oar_ocr_regions_returns_empty_when_unconfigured(monkeypatch):
    monkeypatch.delenv("TRADUZAI_OAR_OCR_BIN", raising=False)
    monkeypatch.delenv("TRADUZAI_OAR_OCR_JSON", raising=False)

    assert load_oar_ocr_regions("missing.png", width=100, height=100) == []
