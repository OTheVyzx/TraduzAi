from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from integration_v1.providers import ProviderUnavailable, recover_source_with_ocr, translate_complete_unit
from ownership.hash_contract import canonical_json_bytes, canonical_page_sha256, sha256_bytes, sha256_text
from ownership.ocr_contract import OCRAttempt, OCRDiagnostics, OCRInvocationResult, OCRObservationRecord, OCRRequest, OCRTransformOperation, OCRTransformSpec
from translator.translate import FrozenTranslationOCRResult, TranslationProviderAttemptResult
from consumer_operational_recovery import _accepted_or_fresh_translation, _cached_or_fresh_ocr


def _ocr_result(page, bbox):
    root_hash = canonical_page_sha256(page)
    request = OCRRequest("cold-run", "cold-execution", "page-042", "a" * 64, root_hash,
                         "cold:page-042:merge-1", "vision-paddleocr")
    transform = OCRTransformSpec.build((OCRTransformOperation(kind="crop", bbox_page=bbox),))
    crop_hash = canonical_page_sha256(transform.replay(page))
    attempt = OCRAttempt("attempt-1", request.run_id, request.origin_execution_id,
        request.page_id, request.page_source_sha256, root_hash, request.invocation_id,
        request.provider_family, "anchored_crop", crop_hash, root_hash, bbox,
        "anchored_crop", transform, bbox[2]-bbox[0], bbox[3]-bbox[1], "RGB", True, False)
    observations = tuple(OCRObservationRecord(
        f"observation-{index}", attempt.attempt_id, request.run_id, request.origin_execution_id,
        request.page_id, request.page_source_sha256, root_hash, crop_hash, request.invocation_id,
        request.provider_family, attempt.variant_id, sha256_text(text), text, confidence, box, (),
        "paddle_anchored") for index, (text, confidence, box) in enumerate((
            ("MY THOUGHTS", .91, (12, 12, 55, 28)),
            ("ARE DIFFERENT", .88, (12, 30, 70, 47))), 1))
    return OCRInvocationResult.build(request=request, observations=observations,
        full_page_lines=observations, attempts=(attempt,),
        diagnostics=OCRDiagnostics("vision-paddleocr"))


def test_cold_ocr_uses_hash_bound_physical_attempt_and_keeps_context_private():
    page = np.full((80, 100, 3), 255, dtype=np.uint8)
    calls = []
    def invoke(**kwargs):
        calls.append(kwargs)
        return _ocr_result(kwargs["page_rgb"], kwargs["bbox_page"])
    recovered = recover_source_with_ocr(page_rgb=page, bbox_page=(10, 10, 80, 60),
        page_id="page-042", page_source_sha256="a" * 64, run_id="cold-run",
        origin_execution_id="cold-execution", invocation=invoke,
        context={"characters": ["PRIVATE CONTEXT"]})
    assert len(calls) == 1
    assert recovered["source"] == "MY THOUGHTS ARE DIFFERENT"
    assert recovered["selected_observation_ids"] == ["observation-1", "observation-2"]
    assert recovered["provider_called"] is True and recovered["cache_hit"] is False
    assert recovered["source_review_sha256"]
    assert "PRIVATE CONTEXT" not in json.dumps(recovered)


def test_cold_ocr_rejects_non_physical_evidence():
    with pytest.raises(ProviderUnavailable, match="fresh physical OCR"):
        recover_source_with_ocr(page_rgb=np.zeros((20, 20, 3), dtype=np.uint8),
            bbox_page=(0, 0, 20, 20), page_id="page-042", page_source_sha256="a"*64,
            run_id="cold-run", origin_execution_id="cold-execution",
            invocation=lambda **_kwargs: SimpleNamespace(attempts=(), observations=()))


def test_operational_cache_miss_calls_vision_interface(tmp_path):
    page = np.full((80, 100, 3), 255, dtype=np.uint8)
    calls = []
    def invoke(**kwargs):
        calls.append(kwargs)
        return _ocr_result(kwargs["page_rgb"], kwargs["bbox_page"])
    job, adjudicated = _cached_or_fresh_ocr(page_rgb=page, crop=page[10:60, 10:80],
        bbox=[10, 10, 80, 60], member="042.webp", page_id="page-042",
        source_sha256="a"*64, cache_root=tmp_path, run_id="cold-run",
        origin_execution_id="cold-execution", ocr_invocation=invoke)
    assert len(calls) == 1
    assert job["provider_called"] is True and job["cache_hit"] is False
    assert adjudicated["review"]["selected_source"] == "MY THOUGHTS ARE DIFFERENT"


def _translation_result(page, target):
    translated = {"page_id": page["page_id"],
                  "texts": [{**page["texts"][0], "translated": target}]}
    encoded = canonical_json_bytes({"provider_called": True, "cache_hit": False})
    return TranslationProviderAttemptResult((FrozenTranslationOCRResult.build("owner-a", translated),),
        "google", "cold_complete_unit", "google-public", True, False, encoded, sha256_bytes(encoded))


def test_cold_translation_invokes_complete_unit_without_cache_or_context_leak():
    calls = []
    def attempt(page, _obra, context, _glossary, **kwargs):
        calls.append((page, context, kwargs))
        return _translation_result(page, "MEUS PENSAMENTOS SAO DIFERENTES")
    result = translate_complete_unit(source="MY THOUGHTS ARE DIFFERENT", owner_id="owner-a",
        page_id="page-042", context={"characters": ["PRIVATE CONTEXT"]}, invoke=attempt)
    assert result["target"] == "MEUS PENSAMENTOS SAO DIFERENTES"
    assert result["provider_called"] is True and result["cache_hit"] is False
    assert result["provenance"] == "fresh_complete_unit_translation"
    assert calls[0][2]["control"].disable_cache is True
    assert "PRIVATE CONTEXT" not in json.dumps(result)


def test_operational_translation_miss_calls_complete_unit_provider(tmp_path):
    calls = []
    def attempt(page, *_args, **kwargs):
        calls.append(kwargs)
        return _translation_result(page, "ALVO NOVO")
    binding = _accepted_or_fresh_translation(control_root=tmp_path,
        control={"complete_balloon_reconciliations": []}, member="039.webp",
        source_sha256="b"*64, source="NEW SOURCE", owners=["owner-a", "owner-b"],
        page_id="page-039", context={}, glossary={}, translation_invocation=attempt)
    assert len(calls) == 1
    assert binding["target"] == "ALVO NOVO"
    assert binding["group_path"] is None
    assert binding["provenance"] == "fresh_complete_unit_translation"


def test_cold_translation_ignores_prepared_binding(tmp_path):
    calls = []
    def attempt(page, *_args, **kwargs):
        calls.append(kwargs)
        return _translation_result(page, "ALVO FRIO")
    binding = _accepted_or_fresh_translation(control_root=tmp_path,
        control={"complete_balloon_reconciliations": ["forbidden-answer.json"]},
        member="042.webp", source_sha256="a"*64, source="VISIBLE SOURCE",
        owners=["owner-a", "owner-b"], page_id="page-042", context={}, glossary={},
        translation_invocation=attempt, allow_accepted=False)
    assert len(calls) == 1
    assert binding["target"] == "ALVO FRIO"


def test_cold_translation_fails_closed_when_provider_returns_source():
    def attempt(page, *_args, **_kwargs):
        return _translation_result(page, page["texts"][0]["text"])
    with pytest.raises(ProviderUnavailable, match="unchanged_source_dialogue"):
        translate_complete_unit(source="THIS IS STILL ENGLISH", owner_id="owner-a",
            page_id="page-042", invoke=attempt)
