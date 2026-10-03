import json
from unittest.mock import patch

import pytest

from translator import hy_mt2_local
from translator import translate as translator


def test_local_recipe_and_prompt():
    calls = []

    def fake_post(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"prompt": payload["messages"][0]["content"]} if url.endswith("apply-template") else {"content": "Volte agora."}

    with patch.object(hy_mt2_local, "_post", fake_post):
        assert hy_mt2_local.translate("Come back now.", source_language="en", target_locale="pt-BR") == "Volte agora."
    assert [call[0].rsplit("/", 1)[-1] for call in calls] == ["apply-template", "completion"]
    assert calls[1][1]["seed"] == 0
    assert calls[1][1]["cache_prompt"] is False
    assert calls[1][1]["temperature"] == 0


def test_selected_backend_never_probes_google_or_ollama():
    row = {"texts": [{"id": "unrelated-owner", "text": "Come back now.",
                       "original": "Come back now.", "route_action": "translate_inpaint_render"}]}
    with patch.object(translator, "_probe_google_backend", side_effect=AssertionError("google")), \
         patch.object(translator, "_check_ollama", side_effect=AssertionError("ollama")), \
         patch.object(hy_mt2_local, "translate", return_value="Volte agora."):
        result = translator.translate_pages([row], obra="", context={}, glossario={},
                                            translation_context={"_translation_backend": "hy_mt2_gguf_local"})
    assert result[0]["texts"][0]["translated"] == "Volte agora."


def test_selected_backend_failure_is_error():
    row = {"texts": [{"id": "different-owner", "text": "Please wait.",
                       "original": "Please wait.", "route_action": "translate_inpaint_render"}]}
    with patch.object(hy_mt2_local, "translate", side_effect=OSError("server offline")):
        with pytest.raises(RuntimeError, match="Hy-MT2 falhou"):
            translator.translate_pages([row], obra="", context={}, glossario={},
                                       translation_context={"_translation_backend": "hy_mt2_gguf_local"})


def test_transient_provider_error_retries_once_without_legacy_fallback():
    row = {"texts": [{"id": "retry-owner", "text": "Please return.",
                       "original": "Please return.", "route_action": "translate_inpaint_render"}]}
    with patch.object(hy_mt2_local, "translate", side_effect=[OSError("busy"), "Por favor, volte."]) as local, \
         patch.object(translator, "_probe_google_backend", side_effect=AssertionError("google")):
        result = translator.translate_pages([row], obra="", context={}, glossario={},
            translation_context={"_translation_backend": "hy_mt2_gguf_local"})
    assert local.call_count == 2
    assert result[0]["texts"][0]["translated"] == "Por favor, volte."


def test_unapproved_pair_fails_without_network():
    with patch.object(hy_mt2_local, "_post", side_effect=AssertionError("network")):
        with pytest.raises(ValueError, match="sem receita aprovada"):
            hy_mt2_local.translate("Hola", source_language="es", target_locale="pt-BR")


def test_available_unchanged_translation_is_retained_with_explicit_warning():
    source="DANGER ZONE"
    row={"texts":[{"id":"new-owner","page_id":"page_007","text":source,"original":source,
                   "route_action":"translate_inpaint_render"}]}
    with patch.object(hy_mt2_local,"translate",return_value=source), \
         patch.object(translator,"_probe_google_backend",side_effect=AssertionError("google")), \
         patch.object(translator,"_check_ollama",side_effect=AssertionError("ollama")):
        result=translator.translate_pages([row],obra="",context={},glossario={},
                    translation_context={"_translation_backend":"hy_mt2_gguf_local"})
    item=result[0]["texts"][0]
    assert item["translated"]==source
    assert "translation_used_with_quality_warning" in item["qa_flags"]
    assert item["translation_delivery_policy"]=="available_translation_with_warnings_v1"
    assert translator._build_translation_debug_identity(0,0,item)["page_id"]=="page_007"


def test_empty_local_response_does_not_become_warning_success():
    row={"texts":[{"id":"empty-owner","text":"DANGER ZONE","original":"DANGER ZONE"}]}
    with patch.object(hy_mt2_local,"translate",return_value=""):
        with pytest.raises(RuntimeError,match="texto vazio"):
            translator.translate_pages([row],obra="",context={},glossario={},
                      translation_context={"_translation_backend":"hy_mt2_gguf_local"})
