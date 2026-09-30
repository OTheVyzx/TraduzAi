"""Systemic PT-BR locale and numeric-equivalence contracts."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _validate(source_text: str, target_text: str):
    from translator.locale_policy import validate_target_locale

    return validate_target_locale(
        source_text=source_text,
        target_text=target_text,
        target_locale="pt-BR",
    )


def test_pt_br_rejects_pt_pt_mil_milhoes_scale() -> None:
    result = _validate(
        "The reward is 1 billion coins.",
        "A recompensa é de mil milhões de moedas.",
    )

    assert result.status == "blocked"
    assert "pt_pt_long_scale_lexeme" in {issue.code for issue in result.issues}


def test_pt_br_accepts_bilhoes_with_decimal_comma() -> None:
    result = _validate(
        "The reward is 1.5 billion coins.",
        "A recompensa é de 1,5 bilhão de moedas.",
    )

    assert result.status == "ok"
    assert result.numeric_equivalent is True
    assert not result.issues


def test_locale_policy_preserves_numeric_meaning() -> None:
    result = _validate(
        "Only 2.5 million players remain.",
        "Restam apenas 2,5 bilhões de jogadores.",
    )

    assert result.status == "blocked"
    assert result.numeric_equivalent is False
    assert "numeric_magnitude_mismatch" in {issue.code for issue in result.issues}


def test_translate_pages_preserves_pt_br_after_backend_language_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from translator import translate

    captured: dict[str, str] = {}

    class FakeGoogle:
        def __init__(self, *, source: str, target: str):
            self._translator = object()
            self._source_lang = source
            self._target_lang = target

        def attach_persistent_cache(self, _cache) -> None:
            return None

    def fake_google(*args, **kwargs):
        captured["backend_language"] = kwargs["idioma_destino"]
        captured["target_locale"] = kwargs["target_locale"]
        record = args[0][0]["texts"][0]
        return [{"texts": [{**record, "translated": "TRADUZIDO"}]}]

    monkeypatch.setattr(translate, "_google", None)
    monkeypatch.setattr(translate, "_GoogleTranslator", FakeGoogle)
    monkeypatch.setattr(translate, "_probe_google_backend", lambda *_args: None)
    monkeypatch.setattr(translate, "_translate_with_google", fake_google)

    translate.translate_pages(
        [{"texts": [{"text": "SOURCE", "tipo": "fala"}]}],
        obra="fixture",
        context={},
        glossario={},
        idioma_origem="en",
        idioma_destino="pt-BR",
    )

    assert captured == {
        "backend_language": "pt",
        "target_locale": "pt-BR",
    }


def test_deterministic_mismatch_blocks_without_silent_rewrite() -> None:
    target = "Ele recebeu 2 milhões de moedas."
    result = _validate("He received 2 billion coins.", target)

    assert result.status == "blocked"
    assert result.target_text == target
    assert "numeric_magnitude_mismatch" in {issue.code for issue in result.issues}


def test_locale_validator_is_not_imported_by_style_modules() -> None:
    from typesetter import font_detector, renderer, style_extractor

    for module in (font_detector, renderer, style_extractor):
        assert "locale_policy" not in inspect.getsource(module)
