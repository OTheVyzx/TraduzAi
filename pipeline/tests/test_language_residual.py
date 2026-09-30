from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from ownership.hash_contract import canonical_json_sha256, sha256_text
from ownership.hash_contract import canonical_page_sha256
from qa.final_pixel_observer import (
    DetectorOcrFinalPixelObserver,
    FinalPixelObservation,
    TerminalVerificationIdentityError,
)
from qa.final_pixel_qa import classify_final_observation_language
from qa.language_residual import (
    ResidualRegion,
    classify_language_residual,
    classify_unowned_text,
    deduplicate_language_issues,
    verify_replacement_pixels,
)
from test_owner_source_replacement_enforce import _binding


def _region(**overrides):
    binding = _binding()
    values = dict(
        run_id=binding.run_id,
        execution_id="execution-final-qa",
        page_id=binding.page_id,
        page_source_sha256=binding.page_source_sha256,
        page_output_pixel_sha256="f" * 64,
        bbox_page=(10, 12, 110, 58),
        owner_id=binding.owner_id,
        component_id=binding.component_ids[0],
        container_id="container-dialogue",
        invocation_ids=("ocr-final",),
    )
    values.update(overrides)
    return ResidualRegion(**values)


def _binding_with_text(source: str, target: str):
    updated = replace(
        _binding(),
        source_text=source,
        target_text=target,
        source_payload_sha256=sha256_text(source),
        target_payload_sha256=sha256_text(target),
    )
    return replace(
        updated,
        translation_binding_sha256=canonical_json_sha256(updated.canonical_payload()),
    )


def test_visible_english_dialogue_is_source_residual_by_binding():
    source = "THE ARENA WILL BEGIN"
    target = "A ARENA COMEÇARÁ"
    binding = _binding_with_text(source, target)

    issues = classify_language_residual(
        observed=source,
        binding=binding,
        region=_region(),
    )

    assert [issue.kind for issue in issues] == ["source_language_visible"]
    assert issues[0].repair_required is True


def test_shared_name_and_number_are_not_false_source_residuals():
    binding = _binding_with_text(
        "KIM SIMUN HAS 10 KILLS", "KIM SIMUN TEM 10 ABATES"
    )

    assert classify_language_residual(
        observed="KIM SIMUN TEM 10 ABATES", binding=binding, region=_region()
    ) == ()


def test_text_inside_translatable_container_without_owner_is_coverage_issue():
    issues = classify_unowned_text(
        {
            "text": "WAIT...",
            "bbox": [12, 20, 58, 40],
            "glyph_support": True,
            "invocation_id": "ocr-final",
            "run_id": _region().run_id,
            "execution_id": _region().execution_id,
            "page_id": _region().page_id,
            "page_source_sha256": _region().page_source_sha256,
            "page_output_pixel_sha256": _region().page_output_pixel_sha256,
        },
        {"container_id": "dialogue", "translatable": True},
    )

    assert [issue.kind for issue in issues] == [
        "independently_detected_text_without_owner"
    ]
    assert issues[0].owner_id is None


def test_material_line_without_component_or_container_is_coverage_issue():
    issues = classify_unowned_text(
        {
            "text": "SOMETHING MOVED",
            "bbox": [12, 20, 98, 40],
            "glyph_support": True,
            "invocation_id": "ocr-final",
            "run_id": _region().run_id,
            "execution_id": _region().execution_id,
            "page_id": _region().page_id,
            "page_source_sha256": _region().page_source_sha256,
            "page_output_pixel_sha256": _region().page_output_pixel_sha256,
        },
        None,
    )

    assert len(issues) == 1
    assert issues[0].component_id is None
    assert issues[0].container_id is None


def test_two_invocations_for_same_defect_deduplicate_and_accumulate_evidence():
    binding = _binding()
    first = classify_language_residual(
        observed=binding.source_text,
        binding=binding,
        region=_region(invocation_ids=("ocr-final",)),
    )[0]
    second = classify_language_residual(
        observed=binding.source_text,
        binding=binding,
        region=_region(invocation_ids=("detector-challenge",)),
    )[0]

    issues = deduplicate_language_issues((first, second))

    assert len(issues) == 1
    assert set(issues[0].invocation_ids) == {"ocr-final", "detector-challenge"}


def test_uncovered_source_support_emits_cleanup_issue_without_ocr_guess():
    source = np.zeros((32, 64), dtype=np.uint8)
    source[10:14, 12:48] = 255
    cleanup = np.zeros_like(source)
    cleanup[10:14, 12:30] = 255

    issue = verify_replacement_pixels(
        source_support_mask=source,
        cleanup_mask=cleanup,
        target_glyph_mask=np.ones_like(source),
        binding=_binding(),
        region=_region(),
    )

    assert issue.kind == "cleanup_incomplete"
    assert issue.repair_required is True


def test_missing_target_glyph_mask_emits_rerender_issue():
    source = np.zeros((32, 64), dtype=np.uint8)
    source[10:14, 12:48] = 255

    issue = verify_replacement_pixels(
        source_support_mask=source,
        cleanup_mask=source,
        target_glyph_mask=np.zeros_like(source),
        binding=_binding(),
        region=_region(),
    )

    assert issue.kind == "target_glyphs_missing"
    assert issue.repair_required is True


def _write_candidate(path, pixels: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    assert ok
    path.write_bytes(encoded.tobytes())


def test_final_observer_rejects_root_hash_from_other_persisted_candidate(tmp_path):
    pixels = np.arange(12 * 18 * 3, dtype=np.uint8).reshape(12, 18, 3)
    candidate = tmp_path / "candidate.png"
    _write_candidate(candidate, pixels)

    class EmptyDetector:
        def detect(self, _image_rgb):
            return []

    class StaleRuntime:
        def run_final_pixel_ocr_probe(self, _image_rgb, **_kwargs):
            return {
                "raw_ocr_records": [],
                "ocr_attempts": [],
                "coverage_complete": True,
                "coverage_failures": [],
                "request_scoped": True,
                "root_input_pixel_sha256": canonical_page_sha256(255 - pixels),
            }

    with pytest.raises(TerminalVerificationIdentityError):
        DetectorOcrFinalPixelObserver(
            detector=EmptyDetector(), runtime=StaleRuntime()
        ).observe(candidate, source_language="en")


def test_final_probe_records_exact_native_full_page_input_without_full_page_variants():
    from ownership.ocr_contract import OCRRequest
    from vision_stack import runtime
    from vision_stack.ocr import OCREngine

    pixels = np.arange(13 * 19 * 3, dtype=np.uint8).reshape(13, 19, 3)
    captured: list[np.ndarray] = []

    class Model:
        def ocr(self, image, det=True, rec=True, cls=False):
            del det, rec, cls
            captured.append(np.asarray(image).copy())
            return [[]]

    engine = OCREngine.__new__(OCREngine)
    engine._backend = "paddleocr"
    engine._model = Model()
    root_sha256 = canonical_page_sha256(pixels)
    request = OCRRequest(
        run_id="run-final",
        origin_execution_id="execution-final",
        page_id="page-final",
        page_source_sha256=root_sha256,
        root_input_pixel_sha256=root_sha256,
        invocation_id="final-qa:page-final",
        provider_family="final-pixel-ocr",
    )

    with patch.object(runtime, "_get_ocr_engine", return_value=engine):
        probe = runtime.run_final_pixel_ocr_probe(
            pixels,
            detected_blocks=[],
            source_challenges=[],
            page_id="page-final",
            page_number=1,
            source_language="en",
            request_scoped=True,
            root_input_pixel_sha256=root_sha256,
            ocr_request=request,
        )

    attempts = {attempt["variant_id"]: attempt for attempt in probe.ocr_attempts}
    assert set(attempts) == {"full_page"}
    assert len(captured) == 1
    assert attempts["full_page"]["input_pixel_sha256"] == canonical_page_sha256(captured[0])
    assert attempts["full_page"]["provider_called"] is True
    assert attempts["full_page"]["cache_hit"] is False


def test_request_scoped_probe_physically_reads_empty_detector_page():
    from ownership.ocr_contract import OCRRequest
    from vision_stack import runtime
    from vision_stack.ocr import OCREngine

    pixels = np.full((14, 20, 3), 231, dtype=np.uint8)
    calls: list[int] = []

    class Model:
        def ocr(self, image, det=True, rec=True, cls=False):
            del image, det, rec, cls
            calls.append(1)
            return [[]]

    engine = OCREngine.__new__(OCREngine)
    engine._backend = "paddleocr"
    engine._model = Model()
    root_sha256 = canonical_page_sha256(pixels)
    request = OCRRequest(
        run_id="run-empty",
        origin_execution_id="execution-empty",
        page_id="page-empty",
        page_source_sha256=root_sha256,
        root_input_pixel_sha256=root_sha256,
        invocation_id="final-qa:page-empty",
        provider_family="final-pixel-ocr",
    )

    with patch.object(runtime, "_get_ocr_engine", return_value=engine):
        probe = runtime.run_final_pixel_ocr_probe(
            pixels,
            detected_blocks=[],
            source_challenges=[],
            page_id="page-empty",
            page_number=2,
            source_language="en",
            request_scoped=True,
            root_input_pixel_sha256=root_sha256,
            ocr_request=request,
        )

    full_page = next(
        attempt for attempt in probe.ocr_attempts if attempt["variant_id"] == "full_page"
    )
    assert calls == [1]
    assert full_page["provider_called"] is True
    assert full_page["cache_hit"] is False
    assert full_page["root_input_pixel_sha256"] == canonical_page_sha256(pixels)
    assert probe.coverage_complete is True


def test_final_observation_maps_alias_variants_to_one_owner_issue():
    binding = _binding_with_text("THE ARENA WILL BEGIN", "A ARENA COMEÃ‡ARÃ")
    pixels = np.zeros((80, 140, 3), dtype=np.uint8)
    observation = FinalPixelObservation(
        image_path="final.png",
        persisted_sha256="a" * 64,
        image_rgb=pixels,
        detected_blocks=(),
        ocr_records=(
            {
                "text": "THE ARENA WILL BEGIN",
                "bbox": [12, 14, 108, 54],
                "invocation_id": "ocr-final-native",
            },
            {
                "text": "THE ARENA WILL BEGIN",
                "bbox": [12, 14, 108, 54],
                "invocation_id": "ocr-final-gray",
            },
        ),
        source_language="en",
        page_id=binding.page_id,
    )

    issues = classify_final_observation_language(
        observation=observation,
        bindings=(binding,),
        regions_by_owner={binding.owner_id: _region()},
        run_id=binding.run_id,
        execution_id=_region().execution_id,
        page_source_sha256=binding.page_source_sha256,
        page_output_pixel_sha256=_region().page_output_pixel_sha256,
    )

    assert len(issues) == 1
    assert issues[0].owner_id == binding.owner_id
    assert set(issues[0].invocation_ids) == {
        "ocr-final",
        "ocr-final-native",
        "ocr-final-gray",
    }
