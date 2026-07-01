import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import build_project_json
from translator.translate import translate_pages


class _CapturingGoogleTranslator:
    batches: list[list[str]] = []

    def __init__(self, source="en", target="pt"):
        self._translator = self
        self.target = target

    def translate(self, text: str):
        return "ok" if text == "__traduzai_probe__" else f"pt:{text}"

    def translate_batch(self, texts: list[str]) -> list[str]:
        self.batches.append(list(texts))
        return [f"pt:{text}" for text in texts]


def test_translator_uses_confident_normalized_text_final():
    _CapturingGoogleTranslator.batches = []
    ocr_results = [
        {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "CANYOUFINDAGOOD",
                    "raw_ocr": "CANYOUFINDAGOOD",
                    "normalized_text_final": "CAN YOU FIND A GOOD",
                    "normalization": {"changed": True, "confidence_after_estimate": 0.82},
                    "confidence": 0.77,
                    "tipo": "fala",
                }
            ]
        }
    ]

    with patch("translator.translate._GoogleTranslator", _CapturingGoogleTranslator):
        with patch(
            "translator.translate._check_ollama",
            return_value={"running": False, "models": [], "has_translator": False},
        ):
            translated = translate_pages(
                ocr_results=ocr_results,
                obra="obra-teste",
                context={},
                glossario={},
                idioma_origem="en",
                idioma_destino="pt-BR",
            )

    assert _CapturingGoogleTranslator.batches == [["CAN YOU FIND A GOOD"]]
    text = translated[0]["texts"][0]
    assert text["original"] == "CANYOUFINDAGOOD"
    assert text["normalized_text_final"] == "CAN YOU FIND A GOOD"
    assert text["source_text_sent_to_translator"] == "CAN YOU FIND A GOOD"


def test_same_balloon_fragments_are_repaired_before_translation():
    _CapturingGoogleTranslator.batches = []
    ocr_results = [
        {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "THE AMOUNT IS JUST RIGHT. THIS BITCHIS AREAL",
                    "raw_ocr": "THE AMOUNT IS JUST RIGHT. THIS BITCHIS AREAL",
                    "bbox": [298, 96, 533, 203],
                    "text_pixel_bbox": [298, 96, 533, 203],
                    "balloon_bbox": [178, 0, 653, 323],
                    "bubble_mask_bbox": [178, 0, 653, 323],
                    "bubble_inner_bbox": [242, 34, 585, 286],
                    "bubble_id": "page_002_band_019_bubble_001",
                    "confidence": 0.905,
                    "tipo": "fala",
                    "band_id": "page_002_band_019",
                    "trace_id": "ocr_001@page_002_band_019",
                },
                {
                    "id": "ocr_002",
                    "text": "ACTRESS...",
                    "raw_ocr": "ACTRESS...",
                    "bbox": [314, 204, 495, 232],
                    "text_pixel_bbox": [314, 204, 495, 232],
                    "balloon_bbox": [178, 0, 653, 323],
                    "bubble_mask_bbox": [178, 0, 653, 323],
                    "bubble_inner_bbox": [242, 34, 585, 286],
                    "bubble_id": "page_002_band_019",
                    "confidence": 0.307,
                    "tipo": "fala",
                    "band_id": "page_002_band_019",
                    "trace_id": "ocr_002@page_002_band_019",
                },
            ]
        }
    ]

    with patch("translator.translate._GoogleTranslator", _CapturingGoogleTranslator):
        with patch(
            "translator.translate._check_ollama",
            return_value={"running": False, "models": [], "has_translator": False},
        ):
            translated = translate_pages(
                ocr_results=ocr_results,
                obra="obra-teste",
                context={},
                glossario={},
                idioma_origem="en",
                idioma_destino="pt-BR",
            )

    assert _CapturingGoogleTranslator.batches == [
        ["The amount is just right. this bitch is a real actress\u2026"]
    ]
    assert len(translated[0]["texts"]) == 1
    text = translated[0]["texts"][0]
    assert text["original"] == "THE AMOUNT IS JUST RIGHT. THIS BITCH IS A REAL ACTRESS..."
    assert text["normalized_text_final"] == "THE AMOUNT IS JUST RIGHT. THIS BITCH IS A REAL ACTRESS..."
    assert text["source_text_sent_to_translator"] == "The amount is just right. this bitch is a real actress\u2026"
    assert text["source_text_ids"] == ["ocr_001", "ocr_002"]
    assert text["source_trace_ids"] == ["ocr_001@page_002_band_019", "ocr_002@page_002_band_019"]
    assert "same_balloon_fragment_merged" in text["qa_flags"]
    assert "ocr_joined_repaired" in text["qa_flags"]


def test_same_band_joined_word_fragments_merge_without_bubble_evidence():
    _CapturingGoogleTranslator.batches = []
    ocr_results = [
        {
            "texts": [
                {
                    "id": "ocr_001",
                    "text": "THE AMOUNT IS JUST RIGHT. THIS BITCHIS AREAL",
                    "raw_ocr": "THE AMOUNT IS JUST RIGHT. THIS BITCHIS AREAL",
                    "bbox": [298, 96, 533, 203],
                    "text_pixel_bbox": [298, 96, 533, 203],
                    "confidence": 0.905,
                    "tipo": "fala",
                    "band_id": "page_002_band_019",
                    "trace_id": "ocr_001@page_002_band_019",
                },
                {
                    "id": "ocr_002",
                    "text": "ACTRESS...",
                    "raw_ocr": "ACTRESS...",
                    "bbox": [314, 204, 495, 232],
                    "text_pixel_bbox": [314, 204, 495, 232],
                    "confidence": 0.307,
                    "tipo": "fala",
                    "band_id": "page_002_band_019",
                    "trace_id": "ocr_002@page_002_band_019",
                },
            ]
        }
    ]

    with patch("translator.translate._GoogleTranslator", _CapturingGoogleTranslator):
        with patch(
            "translator.translate._check_ollama",
            return_value={"running": False, "models": [], "has_translator": False},
        ):
            translated = translate_pages(
                ocr_results=ocr_results,
                obra="obra-teste",
                context={},
                glossario={},
                idioma_origem="en",
                idioma_destino="pt-BR",
            )

    assert _CapturingGoogleTranslator.batches == [
        ["The amount is just right. this bitch is a real actress\u2026"]
    ]
    assert len(translated[0]["texts"]) == 1
    text = translated[0]["texts"][0]
    assert text["normalized_text_final"] == "THE AMOUNT IS JUST RIGHT. THIS BITCH IS A REAL ACTRESS..."
    assert text["source_text_ids"] == ["ocr_001", "ocr_002"]
    assert "same_band_joined_word_fragment_merged" in text["qa_flags"]
    assert "ocr_joined_repaired" in text["qa_flags"]


def test_same_balloon_financial_principal_fragment_is_merged_before_translation():
    _CapturingGoogleTranslator.batches = []
    ocr_results = [
        {
            "texts": [
                {
                    "id": "ocr_004",
                    "text": "THE INTEREST WAS ALREADY REDUCEDBY MORE THAN THREE TIMES",
                    "raw_ocr": "THE INTEREST WAS ALREADY REDUCEDBY MORE THAN THREE TIMES",
                    "bbox": [527, 7113, 688, 7221],
                    "text_pixel_bbox": [527, 7113, 688, 7221],
                    "balloon_bbox": [485, 7068, 730, 7255],
                    "bubble_mask_bbox": [461, 7044, 754, 7279],
                    "confidence": 0.91,
                    "tipo": "fala",
                    "band_id": "page_002_band_007",
                    "trace_id": "ocr_004@page_002_band_007",
                },
                {
                    "id": "ocr_003",
                    "text": "THE PRINCIPAL",
                    "raw_ocr": "THE PRINCIPAL",
                    "bbox": [555, 7209, 661, 7221],
                    "text_pixel_bbox": [555, 7209, 661, 7221],
                    "balloon_bbox": [12, 6636, 656, 7291],
                    "bubble_mask_bbox": [12, 6636, 656, 7291],
                    "confidence": 0.75,
                    "tipo": "fala",
                    "band_id": "page_002_band_007",
                    "trace_id": "ocr_003@page_002_band_007",
                },
            ]
        }
    ]

    with patch("translator.translate._GoogleTranslator", _CapturingGoogleTranslator):
        with patch(
            "translator.translate._check_ollama",
            return_value={"running": False, "models": [], "has_translator": False},
        ):
            translated = translate_pages(
                ocr_results=ocr_results,
                obra="obra-teste",
                context={},
                glossario={},
                idioma_origem="en",
                idioma_destino="pt-BR",
            )

    assert len(_CapturingGoogleTranslator.batches) == 1
    assert len(_CapturingGoogleTranslator.batches[0]) == 1
    sent = _CapturingGoogleTranslator.batches[0][0].upper()
    assert "REDUCED BY" in sent
    assert "THE PRINCIPAL" in sent
    assert all(item.upper() != "THE PRINCIPAL" for batch in _CapturingGoogleTranslator.batches for item in batch)
    assert len(translated[0]["texts"]) == 1
    assert translated[0]["texts"][0]["source_text_ids"] == ["ocr_004", "ocr_003"]


def test_project_json_preserves_raw_ocr_and_normalized_text_final(tmp_path):
    image = tmp_path / "001.jpg"
    image.write_bytes(b"fake")
    text_page = {
        "texts": [
            {
                "id": "ocr_001",
                "text": "CAN YOU FIND A GOOD",
                "original": "CANYOUFINDAGOOD",
                "translated": "Voce consegue encontrar um bom",
                "raw_ocr": "CANYOUFINDAGOOD",
                "normalized_text_final": "CAN YOU FIND A GOOD",
                "normalization": {"changed": True, "confidence_after_estimate": 0.82},
                "bbox": [10, 20, 110, 60],
                "confidence": 0.77,
                "tipo": "fala",
            }
        ]
    }

    project = build_project_json({}, {}, [{"texts": []}], [text_page], [image], 1, 1.0)

    layer = project["paginas"][0]["text_layers"][0]
    legacy = project["paginas"][0]["textos"][0]
    assert layer["raw_ocr"] == "CANYOUFINDAGOOD"
    assert layer["normalized_text_final"] == "CAN YOU FIND A GOOD"
    assert legacy["raw_ocr"] == "CANYOUFINDAGOOD"
    assert legacy["normalized_text_final"] == "CAN YOU FIND A GOOD"
