import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debug_tools import DebugRecorder, bind_recorder
from translator import translate as translate_module
from translator.translate import translate_pages


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_translate_pages_writes_complete_translation_debug_artifacts(tmp_path):
    class _FakeGoogleTranslator:
        def __init__(self, source="en", target="pt"):
            self._translator = self
            self._source_lang = source
            self._target_lang = target

        def translate(self, text: str):
            return "ok" if text == "__traduzai_probe__" else f"PT {text}"

        def translate_batch(self, texts: list[str]) -> list[str]:
            return [f"PT {text}" for text in texts]

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-translation-test")
    bind_recorder(recorder)
    try:
        with patch.object(translate_module, "_google", None):
            with patch.object(translate_module, "_google_health_key", None):
                with patch.object(translate_module, "_google_health_ok", False):
                    with patch.object(translate_module, "_GoogleTranslator", _FakeGoogleTranslator):
                        translated = translate_pages(
                            ocr_results=[
                                {
                                    "texts": [
                                        {
                                            "id": "ocr_001",
                                            "text": "SOPLEASE",
                                            "tipo": "fala",
                                            "band_id": "page_002_band_004",
                                            "trace_id": "ocr_001@page_002_band_004",
                                            "text_instance_id": "page_002_band_004_ocr_001",
                                        },
                                        {
                                            "id": "ocr_002",
                                            "text": "SCANLATOR",
                                            "tipo": "fala",
                                            "skip_processing": True,
                                        },
                                        {
                                            "id": "ocr_003",
                                            "text": "DO NOT TRANSLATE",
                                            "tipo": "fala",
                                            "translate_policy": "skip_translation",
                                        },
                                        {
                                            "id": "ocr_004",
                                            "text": "Mana Core",
                                            "tipo": "fala",
                                            "band_id": "page_002_band_005",
                                        },
                                    ]
                                }
                            ],
                            obra="obra-teste",
                            context={},
                            glossario={"Mana Core": "Nucleo de Mana"},
                            idioma_origem="en",
                            idioma_destino="pt-BR",
                        )
    finally:
        bind_recorder(None)

    root = tmp_path / "debug" / "e2e" / "07_translation"
    inputs = _jsonl(root / "translation_inputs.jsonl")
    outputs = _jsonl(root / "translation_outputs.jsonl")
    glossary = _jsonl(root / "glossary_application.jsonl")
    fallbacks_path = root / "translation_fallbacks.jsonl"
    summary = json.loads((root / "translation_debug_summary.json").read_text(encoding="utf-8"))

    assert len(inputs) == 2
    assert len(outputs) == 2
    assert fallbacks_path.exists()
    assert inputs[0]["page_id"] == "page_002"
    assert inputs[0]["band_id"] == "page_002_band_004"
    assert inputs[0]["trace_id"] == "ocr_001@page_002_band_004"
    assert inputs[0]["text_instance_id"] == "page_002_band_004_ocr_001"
    assert inputs[0]["audit_key"] == "ocr_001@page_002_band_004"
    assert inputs[0]["source_text_before_normalization"] == "SOPLEASE"
    assert inputs[0]["source_text_sent_to_translator"] != "SOPLEASE"
    assert outputs[0]["band_id"] == "page_002_band_004"
    assert outputs[0]["trace_id"] == "ocr_001@page_002_band_004"
    assert outputs[0]["backend"] == "google"
    assert outputs[0]["model"] == "google"
    assert outputs[0]["fallback_used"] is False
    assert isinstance(outputs[0]["duration_ms"], int)
    assert len(outputs[0]["prompt_hash"]) == 12
    assert len(outputs[0]["raw_response_preview"]) <= 256
    assert outputs[0]["final_translation_after_postprocess"]
    assert glossary
    assert glossary[0]["text_id"] == "ocr_004"
    assert summary["translation_inputs_count"] == 2
    assert summary["translation_outputs_count"] == 2
    assert summary["translation_debug_entry_count"] > 0
    assert summary["backend_distribution"]["google"] == 2
    assert summary["glossary_application_count"] >= 1
    assert summary["jsonl_counts"]["translation_inputs.jsonl"] == 2
    assert summary["identity_coverage"]["band_id"]["input_count"] == 2
    assert "page_002_band_004" in summary["identity_coverage"]["band_id"]["values"]
    assert "ocr_001@page_002_band_004" in summary["identity_coverage"]["trace_id"]["values"]
    assert translated[0]["texts"][1]["translated"] == "SCANLATOR"


def test_translation_summary_counts_jsonl_across_multiple_translate_sessions(tmp_path):
    class _FakeGoogleTranslator:
        def __init__(self, source="en", target="pt"):
            self._translator = self
            self._source_lang = source
            self._target_lang = target

        def translate(self, text: str):
            return f"PT {text}"

        def translate_batch(self, texts: list[str]) -> list[str]:
            return [f"PT {text}" for text in texts]

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-translation-test")
    bind_recorder(recorder)
    try:
        with patch.object(translate_module, "_google", None):
            with patch.object(translate_module, "_google_health_key", None):
                with patch.object(translate_module, "_google_health_ok", False):
                    with patch.object(translate_module, "_GoogleTranslator", _FakeGoogleTranslator):
                        translate_pages(
                            ocr_results=[
                                {
                                    "texts": [
                                        {
                                            "id": "ocr_001",
                                            "text": "HELLO",
                                            "tipo": "fala",
                                            "band_id": "page_001_band_001",
                                        }
                                    ]
                                }
                            ],
                            obra="obra-teste",
                            context={},
                            glossario={},
                            idioma_origem="en",
                            idioma_destino="pt-BR",
                        )
                        translate_pages(
                            ocr_results=[
                                {
                                    "texts": [
                                        {
                                            "id": "ocr_001",
                                            "text": "WAIT",
                                            "tipo": "fala",
                                            "band_id": "page_001_band_002",
                                        }
                                    ]
                                }
                            ],
                            obra="obra-teste",
                            context={},
                            glossario={},
                            idioma_origem="en",
                            idioma_destino="pt-BR",
                        )
    finally:
        bind_recorder(None)

    root = tmp_path / "debug" / "e2e" / "07_translation"
    inputs = _jsonl(root / "translation_inputs.jsonl")
    outputs = _jsonl(root / "translation_outputs.jsonl")
    summary = json.loads((root / "translation_debug_summary.json").read_text(encoding="utf-8"))

    assert len(inputs) == 2
    assert len(outputs) == 2
    assert summary["translation_inputs_count"] == 2
    assert summary["translation_outputs_count"] == 2
    assert summary["translation_debug_entry_count"] == 4
    assert sorted(entry["band_id"] for entry in inputs) == [
        "page_001_band_001",
        "page_001_band_002",
    ]


def test_translation_debug_redacts_sensitive_header_values(tmp_path):
    from debug_tools.text_diff import redact_debug_payload

    redacted = redact_debug_payload(
        {
            "headers": {
                "Authorization": "Bearer sk-test-secret",
                "cookie": "session=secret",
                "x-api-key": "abc123",
                "Content-Type": "application/json",
            },
            "raw_response_preview": "Authorization: Bearer sk-test-secret cookie=session=secret x-api-key: abc123",
        }
    )

    serialized = json.dumps(redacted, ensure_ascii=False)
    assert "sk-test-secret" not in serialized
    assert "session=secret" not in serialized
    assert "abc123" not in serialized
    assert "Authorization" not in serialized
    assert "cookie" not in serialized
    assert "x-api-key" not in serialized
    assert redacted["headers"]["redacted_sensitive_headers"] == ["[REDACTED]", "[REDACTED]", "[REDACTED]"]
