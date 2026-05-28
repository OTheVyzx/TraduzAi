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
