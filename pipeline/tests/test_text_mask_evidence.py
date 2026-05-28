import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.export_gate import evaluate_export_gate
from qa.translation_qa import severity_for_flag, summarize_flags
from inpainter import (
    _apply_fast_solid_balloon_fill,
    _fast_fill_mask_evidence_rejection_reason,
    _prime_mask_evidence_for_fast_fill,
)
from inpainter.mask_builder import build_inpaint_mask, consolidate_mask_evidence
from main import _ensure_project_mask_evidence, _ensure_project_render_contract
from vision_stack.text_mask_evidence import measure_mask_coverage, normalize_text_evidence


def test_normalize_text_evidence_keeps_bbox_and_text():
    page = {
        "texts": [{"text": "ガシャーン", "bbox": [10, 20, 80, 50], "confidence": 0.91}],
        "_vision_blocks": [{"bbox": [8, 18, 84, 54]}],
    }

    evidence = normalize_text_evidence(page, width=100, height=100)

    assert evidence[0].text == "ガシャーン"
    assert evidence[0].bbox == [10, 20, 80, 50]
    assert evidence[0].source == "ocr"


def test_measure_mask_coverage_counts_dark_pixels_inside_evidence():
    image = np.full((80, 120, 3), 240, dtype=np.uint8)
    image[30:40, 50:60] = 20
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[30:35, 50:60] = 255
    evidence = normalize_text_evidence({"texts": [{"bbox": [45, 25, 70, 50]}]}, width=120, height=80)

    coverage = measure_mask_coverage(mask, image, evidence)

    assert coverage["dark_pixels"] == 100
    assert coverage["dark_inside_mask"] == 50
    assert coverage["dark_outside_mask"] == 50
    assert coverage["coverage_ratio"] == 0.5


def test_dialogue_fast_fill_without_raw_glyph_evidence_is_blocked():
    region = {"content_class": "dialogue", "qa_flags": []}

    evidence = consolidate_mask_evidence(
        region,
        kind="ocr_pixels",
        raw_mask_pixels=0,
        expanded_mask_pixels=42,
        evidence_score=0.8,
    )

    assert evidence["fast_fill_allowed"] is False
    assert evidence["fast_fill_reject_reasons"] == ["raw_mask_pixels_zero"]
    assert region["mask_evidence"] == evidence
    assert "fast_fill_no_glyph_evidence" in region["qa_flags"]
    assert severity_for_flag("fast_fill_no_glyph_evidence") == "critical"

    project = {"paginas": [{"numero": 1, "text_layers": [region]}]}
    project["qa"] = {"summary": summarize_flags([region])}

    assert project["qa"]["summary"]["highest_severity"] == "critical"
    assert evaluate_export_gate(project)["status"] == "BLOCK"


def test_valid_dialogue_mask_evidence_allows_fast_fill_without_flag():
    region = {"content_class": "dialogue", "qa_flags": []}

    evidence = consolidate_mask_evidence(
        region,
        kind="glyph_segmentation",
        raw_mask_pixels=120,
        expanded_mask_pixels=180,
        evidence_score=0.72,
    )

    assert evidence == {
        "kind": "glyph_segmentation",
        "raw_mask_pixels": 120,
        "expanded_mask_pixels": 180,
        "evidence_score": 0.72,
        "fast_fill_allowed": True,
        "fast_fill_reject_reasons": [],
    }
    assert region["qa_flags"] == []


def test_reconsolidating_valid_evidence_removes_stale_automatic_reject_reasons():
    region = {
        "content_class": "dialogue",
        "qa_flags": ["fast_fill_no_glyph_evidence"],
        "mask_evidence": {
            "kind": "none",
            "raw_mask_pixels": 0,
            "expanded_mask_pixels": 0,
            "evidence_score": 0.0,
            "fast_fill_allowed": False,
            "fast_fill_reject_reasons": [
                "raw_mask_pixels_zero",
                "coverage_too_low",
                "mask_kind_not_fast_fill_allowed",
            ],
        },
    }

    evidence = consolidate_mask_evidence(
        region,
        kind="ocr_pixels",
        raw_mask_pixels=64,
        expanded_mask_pixels=96,
        evidence_score=0.9,
    )

    assert evidence["fast_fill_allowed"] is True
    assert evidence["fast_fill_reject_reasons"] == []
    assert "fast_fill_no_glyph_evidence" not in region["qa_flags"]


def test_fast_solid_missing_mask_evidence_rejects_before_local_evidence(monkeypatch):
    monkeypatch.delenv("TRADUZAI_STRIP_FAST_LOCAL_INPAINT", raising=False)
    monkeypatch.delenv("TRADUZAI_STRIP_FAST_WHITE_INPAINT", raising=False)
    monkeypatch.setenv("TRADUZAI_STRIP_FAST_SOLID_INPAINT", "1")
    band_rgb = np.full((80, 120, 3), 255, dtype=np.uint8)
    text = {
        "id": "t1",
        "content_class": "dialogue",
        "tipo": "fala",
        "bbox": [30, 20, 70, 40],
        "balloon_bbox": [20, 10, 90, 60],
    }
    ocr_page = {"texts": [text]}
    vision_blocks = [{"id": "t1", "bbox": [30, 20, 70, 40], "content_class": "dialogue"}]

    _result, remaining, stats = _apply_fast_solid_balloon_fill(band_rgb, ocr_page, vision_blocks)

    assert remaining == vision_blocks
    assert stats["solid_balloon_count"] == 0
    assert "mask_evidence" not in text
    assert ocr_page["_strip_fast_solid_rejection_reasons"] == {"mask_evidence:missing": 1}
    assert ocr_page.get("_strip_inpaint_decision_flags", []) == []


def test_prime_mask_evidence_for_fast_fill_persists_block_evidence_on_text(monkeypatch):
    band_rgb = np.full((80, 120, 3), 255, dtype=np.uint8)
    text = {
        "id": "t1",
        "trace_id": "t1@band_001",
        "content_class": "dialogue",
        "tipo": "fala",
        "bbox": [30, 20, 70, 40],
        "balloon_bbox": [20, 10, 90, 60],
        "qa_flags": [],
    }
    ocr_page = {"texts": [text]}
    vision_blocks = [
        {
            "id": "t1",
            "trace_id": "t1@band_001",
            "content_class": "dialogue",
            "tipo": "fala",
            "bbox": [30, 20, 70, 40],
            "balloon_bbox": [20, 10, 90, 60],
            "qa_flags": [],
        }
    ]

    def fake_build_inpaint_mask(block, image_shape, image_rgb=None):
        block["mask_evidence"] = {
            "kind": "glyph_segmentation",
            "raw_mask_pixels": 64,
            "expanded_mask_pixels": 80,
            "evidence_score": 1.0,
            "fast_fill_allowed": True,
            "fast_fill_reject_reasons": [],
        }
        return np.ones(image_shape[:2], dtype=np.uint8)

    monkeypatch.setattr("inpainter.build_inpaint_mask", fake_build_inpaint_mask)

    _prime_mask_evidence_for_fast_fill(ocr_page, vision_blocks, band_rgb)

    assert text["mask_evidence"]["fast_fill_allowed"] is True
    assert vision_blocks[0]["mask_evidence"]["fast_fill_allowed"] is True
    assert _fast_fill_mask_evidence_rejection_reason(text) == ""


def test_build_inpaint_mask_persists_mask_evidence_for_dialogue_without_raw_pixels(monkeypatch):
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    block = {
        "content_class": "dialogue",
        "tipo": "fala",
        "bbox": [30, 20, 70, 40],
        "balloon_bbox": [20, 10, 90, 60],
        "qa_flags": [],
    }
    monkeypatch.setattr(
        "inpainter.mask_builder.build_raw_text_mask_from_image",
        lambda *_args, **_kwargs: np.zeros((80, 120), dtype=np.uint8),
    )

    mask = build_inpaint_mask(block, image.shape, image)

    assert mask is None
    assert block["mask_evidence"]["kind"] == "none"
    assert block["mask_evidence"]["raw_mask_pixels"] == 0
    assert block["mask_evidence"]["expanded_mask_pixels"] == 0


def test_final_project_mask_evidence_is_filled_for_every_text_layer():
    project = {
        "paginas": [
            {
                "numero": 1,
                "text_layers": [
                    {
                        "id": "dialogue_missing",
                        "content_class": "dialogue",
                        "route_action": "translate_inpaint_render",
                        "qa_flags": [],
                    },
                    {
                        "id": "preserved_logo",
                        "content_class": "logo",
                        "route_action": "preserve",
                        "qa_flags": [],
                    },
                    {
                        "id": "existing",
                        "content_class": "dialogue",
                        "route_action": "translate_inpaint_render",
                        "mask_evidence": {
                            "kind": "glyph_segmentation",
                            "raw_mask_pixels": 7,
                            "expanded_mask_pixels": 9,
                            "evidence_score": 0.9,
                            "fast_fill_allowed": True,
                            "fast_fill_reject_reasons": [],
                        },
                        "qa_flags": [],
                    },
                ],
            }
        ]
    }

    filled = _ensure_project_mask_evidence(project)

    layers = project["paginas"][0]["text_layers"]
    assert filled == 2
    assert all(isinstance(layer.get("mask_evidence"), dict) for layer in layers)
    assert layers[0]["mask_evidence"]["kind"] == "none"
    assert layers[0]["mask_evidence"]["fast_fill_allowed"] is False
    assert "fast_fill_no_glyph_evidence" in layers[0]["qa_flags"]
    assert layers[1]["mask_evidence"]["kind"] == "none"
    assert "fast_fill_no_glyph_evidence" not in layers[1]["qa_flags"]
    assert layers[2]["mask_evidence"]["raw_mask_pixels"] == 7


def test_final_project_mask_evidence_does_not_fast_fill_block_review_required_layer():
    project = {
        "paginas": [
            {
                "numero": 1,
                "text_layers": [
                    {
                        "id": "joined_ocr",
                        "content_class": "dialogue",
                        "route_action": "review_required",
                        "qa_flags": ["ocr_truncated_or_joined"],
                    }
                ],
            }
        ]
    }

    filled = _ensure_project_mask_evidence(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert filled == 1
    assert layer["mask_evidence"]["fast_fill_allowed"] is False
    assert "fast_fill_no_glyph_evidence" not in layer["qa_flags"]
    assert "ocr_truncated_or_joined" in layer["qa_flags"]


def test_final_project_render_contract_flags_translate_layer_without_render_bbox():
    project = {
        "paginas": [
            {
                "numero": 1,
                "text_layers": [
                    {
                        "id": "needs_render",
                        "route_action": "translate_inpaint_render",
                        "skip_processing": False,
                        "bbox": [10, 20, 80, 60],
                        "qa_flags": [],
                    },
                    {
                        "id": "skipped_noise",
                        "route_action": "skip",
                        "skip_processing": True,
                        "bbox": [90, 20, 110, 60],
                        "qa_flags": [],
                    },
                ],
            }
        ]
    }

    audit = _ensure_project_render_contract(project)

    layer = project["paginas"][0]["text_layers"][0]
    skipped = project["paginas"][0]["text_layers"][1]
    assert audit["checked_layers"] == 1
    assert audit["missing_render_bbox_count"] == 1
    assert "missing_render_bbox" in layer["qa_flags"]
    assert layer["fit_status"] == "below_minimum_legible"
    assert layer["fit_attempts"][-1]["status"] == "missing_render_bbox"
    assert "missing_render_bbox" not in skipped["qa_flags"]
