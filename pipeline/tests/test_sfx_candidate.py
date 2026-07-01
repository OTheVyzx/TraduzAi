import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfx.candidate import enrich_sfx_candidate
from sfx.hangul_adapter import SfxAdaptation


def test_enriches_routed_hangul_sfx_candidate():
    layer = {
        "id": "sfx-1",
        "route_action": "translate_sfx_inpaint_render",
        "translate_policy": "adapt_sfx",
        "render_policy": "sfx_style",
        "text": "\ucff5",
        "original": "\ucff5",
        "raw_ocr": "\ucff5",
        "ocr_confidence": 0.92,
    }

    enriched = enrich_sfx_candidate(layer)

    assert enriched["content_class"] == "sfx"
    assert enriched["script"] == "hangul"
    assert enriched["translated"] == "TUM"
    assert enriched["traduzido"] == "TUM"
    assert enriched["translate_policy"] == "adapt_sfx"
    assert enriched["render_policy"] == "sfx_style"
    assert enriched["sfx"]["source_text"] == "\ucff5"
    assert enriched["sfx"]["adapted_text"] == "TUM"
    assert enriched["sfx"]["confidence"] == 0.9
    assert enriched["sfx"]["kind"] == "impact"
    assert enriched["sfx"]["translation_mode"] == "onomatopoeia_adaptation"
    assert enriched["sfx"]["qa_flags"] == []
    assert enriched["sfx"]["inpaint_allowed"] is False
    assert layer.get("sfx") is None
    assert layer["raw_ocr"] == "\ucff5"


def test_preserves_user_edited_translation_and_existing_inpaint_allowed():
    layer = {
        "content_class": "sfx",
        "text": "\ucff5",
        "original": "\ucff5",
        "translated": "POW",
        "traduzido": "POW",
        "sfx": {"inpaint_allowed": True},
    }

    enriched = enrich_sfx_candidate(layer)

    assert enriched["translated"] == "POW"
    assert enriched["traduzido"] == "POW"
    assert enriched["route_action"] == "translate_sfx_inpaint_render"
    assert enriched["sfx"]["adapted_text"] == "TUM"
    assert enriched["sfx"]["inpaint_allowed"] is True


def test_sfx_enrichment_overrides_default_text_policies():
    enriched = enrich_sfx_candidate(
        {
            "route_action": "translate_sfx_inpaint_render",
            "text": "\ucff5",
            "original": "\ucff5",
            "translate_policy": "translate",
            "render_policy": "normal",
        }
    )

    assert enriched["translate_policy"] == "adapt_sfx"
    assert enriched["render_policy"] == "sfx_style"


def test_unknown_sfx_requires_review_and_does_not_fill_translation():
    enriched = enrich_sfx_candidate(
        {
            "route_action": "translate_sfx_inpaint_render",
            "text": "\ud750\uadf8\ub974",
            "original": "\ud750\uadf8\ub974",
        }
    )

    assert enriched["route_action"] == "review_required"
    assert enriched["render_policy"] == "review_required"
    assert enriched.get("translated") in (None, "")
    assert "unknown_sfx" in enriched["sfx"]["qa_flags"]
    assert "unknown_sfx" in enriched["qa_flags"]


def test_low_confidence_adaptation_requires_review(monkeypatch):
    def low_confidence_adapter(text):
        return SfxAdaptation(
            source_text=text,
            adapted_text="VUM",
            confidence=0.4,
            kind="hum",
            review_required=False,
            qa_flags=[],
        )

    monkeypatch.setattr("sfx.candidate.adapt_hangul_sfx", low_confidence_adapter)

    enriched = enrich_sfx_candidate(
        {
            "route_action": "translate_sfx_inpaint_render",
            "text": "\uc6b0\uc6b0\uc6c5",
            "original": "\uc6b0\uc6b0\uc6c5",
        }
    )

    assert enriched["route_action"] == "review_required"
    assert enriched.get("translated") in (None, "")
    assert "low_confidence" in enriched["sfx"]["qa_flags"]


def test_non_sfx_layer_is_unchanged():
    layer = {"route_action": "translate_inpaint_render", "text": "Hello"}

    assert enrich_sfx_candidate(layer) is layer
