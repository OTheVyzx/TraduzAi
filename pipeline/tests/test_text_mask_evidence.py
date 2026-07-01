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
from main import (
    _copy_group_sibling_render_metadata,
    _clear_non_bubble_panel_mask_flags,
    _clear_stale_panel_weak_residual_flags,
    _clear_stale_valid_image_bubble_mask_flags,
    _ensure_project_mask_evidence,
    _ensure_project_render_contract,
    _hide_merged_candidate_sibling_layers,
    _mark_final_layer_as_page_space,
    _merge_same_balloon_fragment_layers,
    _primary_layer_for_merged_candidate,
    _suppress_broad_fallback_merge_layers,
    _visual_target_bbox_for_merge,
    _visual_targets_share_merge_region,
    _visible_render_texts,
    neutralize_removed_decision_fields,
)
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


def test_project_mask_evidence_clears_peer_covered_fragment_no_glyph_flag():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "band_id": "page_003_band_035",
                        "bbox": [148, 235, 310, 255],
                        "line_polygons": [
                            [[148, 7248], [310, 7248], [310, 7268], [148, 7268]],
                            [[345, 7665], [537, 7665], [537, 7686], [345, 7686]],
                        ],
                        "mask_evidence": {
                            "kind": "component_bubble_cleaner",
                            "raw_mask_pixels": 1629,
                            "expanded_mask_pixels": 2850,
                            "evidence_score": 1.0,
                            "fast_fill_allowed": True,
                            "fast_fill_reject_reasons": [],
                        },
                        "qa_flags": [],
                    },
                    {
                        "id": "ocr_001_fragment_2",
                        "band_id": "page_003_band_035",
                        "bbox": [345, 7665, 537, 7686],
                        "text_pixel_bbox": [345, 7665, 537, 7686],
                        "mask_evidence": {
                            "kind": "none",
                            "raw_mask_pixels": 0,
                            "expanded_mask_pixels": 0,
                            "evidence_score": 0.0,
                            "fast_fill_allowed": False,
                            "fast_fill_reject_reasons": ["raw_mask_pixels_zero"],
                        },
                        "qa_flags": ["safe_text_box_recomputed", "fast_fill_no_glyph_evidence"],
                    },
                ],
            }
        ]
    }

    changed = _ensure_project_mask_evidence(project)

    fragment = project["paginas"][0]["text_layers"][1]
    assert changed == 1
    assert "fast_fill_no_glyph_evidence" not in fragment["qa_flags"]
    assert fragment["mask_evidence"]["kind"] == "covered_by_peer_mask"


def test_project_mask_evidence_clears_missing_bubble_for_valid_non_white_panel():
    project = {
        "paginas": [
            {
                "numero": 6,
                "text_layers": [
                    {
                        "id": "panel_text",
                        "band_id": "page_006_band_107",
                        "translated": "Classe de nível atual: espírito flutuante",
                        "route_action": "review_required",
                        "route_reason": "mask_outside_balloon_critical",
                        "background_rgb": [118, 98, 49],
                        "bubble_mask_source": "derived_white_crop_rejected",
                        "bubble_mask_error": "derived_mask_not_anchored_to_text",
                        "mask_evidence": {
                            "kind": "ocr_pixels",
                            "raw_mask_pixels": 2564,
                            "expanded_mask_pixels": 7091,
                            "evidence_score": 1.0,
                            "fast_fill_allowed": True,
                            "fast_fill_reject_reasons": [],
                        },
                        "qa_flags": [
                            "rejected_derived_bubble_mask",
                            "missing_real_bubble_mask",
                            "mask_outside_balloon_critical",
                            "debug_derived_bubble_mask_rejected",
                        ],
                    }
                ],
            }
        ]
    }

    changed = _ensure_project_mask_evidence(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert changed == 1
    assert "missing_real_bubble_mask" not in layer["qa_flags"]
    assert "mask_outside_balloon_critical" not in layer["qa_flags"]
    assert "debug_derived_bubble_mask_rejected" in layer["qa_flags"]
    assert layer["route_action"] == "translate_inpaint_render"
    assert "route_reason" not in layer


def test_non_bubble_annotation_clears_balloon_mask_flags_even_on_white_background():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "translator_note",
                        "text": "T/N: HYUNGNIM IS A TERM USED FOR CALLING ONE'S MOB BOSS",
                        "translated": "T/N: HYUNGNIM É UM TERMO USADO PARA CHAMAR O CHEFE",
                        "route_action": "review_required",
                        "route_reason": "mask_outside_balloon_critical",
                        "background_rgb": [251, 251, 251],
                        "render_bbox": [571, 14304, 761, 14387],
                        "safe_text_box": [570, 14294, 762, 14397],
                        "mask_evidence": {
                            "kind": "component_bubble_cleaner",
                            "raw_mask_pixels": 1796,
                            "expanded_mask_pixels": 4291,
                            "evidence_score": 1.0,
                            "fast_fill_allowed": True,
                            "fast_fill_reject_reasons": [],
                        },
                        "qa_flags": [
                            "safe_text_box_recomputed",
                            "bbox_fallback_bubble_mask",
                            "fit_below_minimum_legible",
                            "mask_outside_balloon",
                            "mask_outside_balloon_critical",
                            "debug_derived_bubble_mask_rejected",
                        ],
                    },
                    {
                        "id": "credits",
                        "text": "TL Kiki Pr Mars Shadow CI Ts Erian Qc Shadow Rp Shadow",
                        "context_before": "READFIRSTAT: NTERM SECRETSCANS. CO Discord.gg/xzeKn8V",
                        "translated": "Tl kiki pr mars sombra ci ts erian qc sombra rp sombra",
                        "route_action": "review_required",
                        "route_reason": "mask_outside_balloon_critical",
                        "background_rgb": [226, 224, 227],
                        "mask_evidence": {
                            "kind": "component_bubble_cleaner",
                            "raw_mask_pixels": 7560,
                            "expanded_mask_pixels": 22359,
                            "evidence_score": 1.0,
                            "fast_fill_allowed": True,
                            "fast_fill_reject_reasons": [],
                        },
                        "qa_flags": [
                            "ocr_run_on_suspect",
                            "bbox_fallback_bubble_mask",
                            "mask_outside_balloon_critical",
                        ],
                    },
                ],
            }
        ]
    }

    changed = _clear_non_bubble_panel_mask_flags(project)

    note, credits = project["paginas"][0]["text_layers"]
    assert changed == 2
    assert "mask_outside_balloon_critical" not in note["qa_flags"]
    assert "bbox_fallback_bubble_mask" not in note["qa_flags"]
    assert "fit_below_minimum_legible" not in note["qa_flags"]
    assert "safe_text_box_recomputed" in note["qa_flags"]
    assert note["route_action"] == "translate_inpaint_render"
    assert "route_reason" not in note
    assert "mask_outside_balloon_critical" not in credits["qa_flags"]
    assert "ocr_run_on_suspect" in credits["qa_flags"]


def test_project_mask_evidence_clears_stale_weak_residual_for_valid_card_panel():
    project = {
        "paginas": [
            {
                "numero": 6,
                "text_layers": [
                    {
                        "id": "panel_text",
                        "band_id": "page_006_band_107",
                        "translated": "Classe de nível atual: espírito flutuante",
                        "route_action": "translate_inpaint_render",
                        "background_rgb": [118, 98, 49],
                        "bubble_mask_source": "derived_card_panel_mask",
                        "mask_evidence": {
                            "kind": "component_bubble_cleaner",
                            "raw_mask_pixels": 2564,
                            "expanded_mask_pixels": 11745,
                            "evidence_score": 1.0,
                            "fast_fill_allowed": True,
                            "fast_fill_reject_reasons": [],
                        },
                        "qa_flags": [
                            "bubble_clip_preserved_raw_text",
                            "weak_text_residual_after_inpaint",
                        ],
                    }
                ],
            }
        ]
    }

    changed = _ensure_project_mask_evidence(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert changed == 1
    assert "weak_text_residual_after_inpaint" not in layer["qa_flags"]
    assert "bubble_clip_preserved_raw_text" in layer["qa_flags"]


def test_final_panel_cleanup_clears_late_debug_weak_residual_for_valid_card_panel():
    project = {
        "paginas": [
            {
                "numero": 6,
                "text_layers": [
                    {
                        "id": "panel_text",
                        "band_id": "page_006_band_107",
                        "route_action": "translate_inpaint_render",
                        "background_rgb": [118, 98, 49],
                        "bubble_mask_source": "derived_card_panel_mask",
                        "mask_evidence": {
                            "kind": "component_bubble_cleaner",
                            "raw_mask_pixels": 2564,
                            "expanded_mask_pixels": 11745,
                            "evidence_score": 1.0,
                            "fast_fill_allowed": True,
                            "fast_fill_reject_reasons": [],
                        },
                        "qa_flags": [
                            "weak_text_residual_after_inpaint",
                            "bubble_clip_preserved_raw_text",
                        ],
                    }
                ],
            }
        ]
    }

    changed = _clear_stale_panel_weak_residual_flags(project)

    layer = project["paginas"][0]["text_layers"][0]
    assert changed == 1
    assert layer["qa_flags"] == ["bubble_clip_preserved_raw_text"]


def test_final_panel_cleanup_clears_late_debug_weak_residual_for_valid_image_bubble():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "bubble_text",
                        "band_id": "page_003_band_035",
                        "route_action": "translate_inpaint_render",
                        "bubble_mask_source": "image_white_bubble_mask",
                        "mask_evidence": {
                            "kind": "component_bubble_cleaner",
                            "raw_mask_pixels": 1633,
                            "expanded_mask_pixels": 5979,
                            "evidence_score": 1.0,
                            "fast_fill_allowed": True,
                            "fast_fill_reject_reasons": [],
                        },
                        "qa_flags": [
                            "bbox_fallback_bubble_mask",
                            "weak_text_residual_after_inpaint",
                        ],
                    },
                    {
                        "id": "real_residual",
                        "band_id": "page_003_band_036",
                        "route_action": "translate_inpaint_render",
                        "bubble_mask_source": "image_white_bubble_mask",
                        "residual_text": {"has_residual": True},
                        "mask_evidence": {
                            "kind": "component_bubble_cleaner",
                            "raw_mask_pixels": 120,
                            "expanded_mask_pixels": 240,
                            "evidence_score": 1.0,
                            "fast_fill_allowed": True,
                            "fast_fill_reject_reasons": [],
                        },
                        "qa_flags": ["weak_text_residual_after_inpaint"],
                    },
                ],
            }
        ]
    }

    changed = _clear_stale_panel_weak_residual_flags(project)

    cleaned, retained = project["paginas"][0]["text_layers"]
    assert changed == 1
    assert "weak_text_residual_after_inpaint" not in cleaned["qa_flags"]
    assert "bbox_fallback_bubble_mask" in cleaned["qa_flags"]
    assert "weak_text_residual_after_inpaint" in retained["qa_flags"]


def test_final_cleanup_clears_stale_fallback_flags_for_valid_image_bubble_mask():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "valid_image_bubble",
                        "route_action": "translate_inpaint_render",
                        "bubble_mask_source": "image_contour_bubble_mask",
                        "bubble_mask_error": None,
                        "translated": "Por favor!",
                        "mask_evidence": {
                            "kind": "component_bubble_cleaner",
                            "raw_mask_pixels": 1775,
                            "expanded_mask_pixels": 3985,
                            "evidence_score": 1.0,
                            "fast_fill_allowed": True,
                            "fast_fill_reject_reasons": [],
                        },
                        "qa_flags": [
                            "bbox_fallback_bubble_mask",
                            "debug_derived_bubble_mask_rejected",
                            "safe_text_box_recomputed",
                        ],
                    },
                    {
                        "id": "real_residual_keeps_flags",
                        "route_action": "translate_inpaint_render",
                        "bubble_mask_source": "image_white_bubble_mask",
                        "residual_text": {"has_residual": True},
                        "mask_evidence": {
                            "kind": "component_bubble_cleaner",
                            "raw_mask_pixels": 100,
                            "expanded_mask_pixels": 200,
                            "evidence_score": 1.0,
                        },
                        "qa_flags": [
                            "bbox_fallback_bubble_mask",
                            "debug_derived_bubble_mask_rejected",
                        ],
                    },
                ],
            }
        ]
    }

    changed = _clear_stale_valid_image_bubble_mask_flags(project)

    cleaned, retained = project["paginas"][0]["text_layers"]
    assert changed == 1
    assert cleaned["qa_flags"] == ["safe_text_box_recomputed"]
    assert "bbox_fallback_bubble_mask" in retained["qa_flags"]
    assert "debug_derived_bubble_mask_rejected" in retained["qa_flags"]


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


def test_group_sibling_render_metadata_hydrates_hidden_translated_fragment():
    project = {
        "paginas": [
            {
                "numero": 4,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_004_band_054",
                        "band_id": "page_004_band_054",
                        "route_action": "translate_inpaint_render",
                        "visible": False,
                        "qa_flags": ["missing_render_bbox"],
                    },
                    {
                        "id": "ocr_002",
                        "text_id": "ocr_002",
                        "trace_id": "ocr_002@page_004_band_054",
                        "band_id": "page_004_band_054",
                        "route_action": "translate_inpaint_render",
                        "render_bbox": [114, 4485, 355, 4536],
                        "safe_text_box": [92, 4454, 377, 4567],
                        "target_bbox": [0, 4383, 425, 4714],
                        "fit_status": "ok",
                        "fit_attempts": [{"status": "ok"}],
                        "qa_flags": [],
                    },
                ],
            }
        ]
    }
    candidate = {
        "band_id": "page_004_band_054",
        "source_text_ids": [
            "ocr_001",
            "ocr_002",
            "ocr_001@page_004_band_054",
            "ocr_002@page_004_band_054",
        ],
        "source_trace_ids": ["ocr_001@page_004_band_054", "ocr_002@page_004_band_054"],
        "render_bbox": [114, 4485, 355, 4536],
        "safe_text_box": [92, 4454, 377, 4567],
        "target_bbox": [0, 4383, 425, 4714],
        "fit_status": "ok",
        "fit_attempts": [{"status": "ok"}],
    }

    hydrated = _copy_group_sibling_render_metadata(project, [candidate])

    hidden = project["paginas"][0]["text_layers"][0]
    assert hydrated == 1
    assert hidden["render_bbox"] == [114, 4485, 355, 4536]
    assert hidden["safe_text_box"] == [92, 4454, 377, 4567]
    assert hidden["fit_status"] == "ok"
    assert "missing_render_bbox" not in hidden["qa_flags"]
    assert hidden["_render_metadata_group_sibling_geometry"] is True


def test_group_sibling_render_metadata_prefers_real_peer_over_rejected_local_fragment():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_004",
                        "text_id": "ocr_004",
                        "trace_id": "ocr_004@page_002_band_007",
                        "band_id": "page_002_band_007",
                        "route_action": "translate_inpaint_render",
                        "translated": "OS JUROS JÁ FORAM REDUZIDOS EM MAIS DE TRÊS VEZES O DIRETOR",
                        "bbox": [527, 7113, 688, 7221],
                        "text_pixel_bbox": [527, 7113, 688, 7221],
                        "balloon_bbox": [485, 7068, 730, 7255],
                        "bubble_mask_source": "image_contour_bubble_mask",
                        "qa_flags": ["same_balloon_fragment_merged"],
                    },
                    {
                        "id": "ocr_003",
                        "text_id": "ocr_003",
                        "trace_id": "ocr_003@page_002_band_007",
                        "band_id": "page_002_band_007",
                        "route_action": "translate_inpaint_render",
                        "translated": "OS JUROS JÁ FORAM REDUZIDOS EM MAIS DE TRÊS VEZES O DIRETOR",
                        "bbox": [555, 7209, 661, 7221],
                        "text_pixel_bbox": [555, 7209, 661, 7221],
                        "target_bbox": [461, 618, 754, 853],
                        "safe_text_box": [550, 673, 668, 806],
                        "render_bbox": [559, 678, 660, 800],
                        "bubble_mask_source": "derived_white_crop_rejected",
                        "qa_flags": ["debug_derived_bubble_mask_rejected", "same_balloon_fragment_merged"],
                    },
                ],
            }
        ]
    }
    candidate = {
        "band_id": "page_002_band_007",
        "source_text_ids": ["ocr_004", "ocr_003"],
        "source_trace_ids": ["ocr_004@page_002_band_007", "ocr_003@page_002_band_007"],
        "translated": "OS JUROS JÁ FORAM REDUZIDOS EM MAIS DE TRÊS VEZES O DIRETOR",
        "target_bbox": [461, 7044, 754, 7279],
        "safe_text_box": [550, 7099, 668, 7232],
        "render_bbox": [559, 7104, 660, 7226],
        "fit_status": "ok",
    }

    hydrated = _copy_group_sibling_render_metadata(project, [candidate])

    real_peer, rejected = project["paginas"][0]["text_layers"]
    assert hydrated == 1
    assert real_peer.get("visible") is not False
    assert real_peer["render_bbox"] == [559, 7104, 660, 7226]
    assert real_peer["safe_text_box"] == [550, 7099, 668, 7232]
    assert rejected["visible"] is False
    assert rejected["render_policy"] == "merged_into_primary"
    assert rejected["merged_into_text_id"] == "ocr_004"


def test_same_balloon_merge_keeps_real_peer_visible_over_rejected_local_fragment():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_004",
                        "text_id": "ocr_004",
                        "trace_id": "ocr_004@page_002_band_007",
                        "band_id": "page_002_band_007",
                        "route_action": "translate_inpaint_render",
                        "translated": "OS JUROS JÁ FORAM REDUZIDOS EM MAIS DE TRÊS VEZES O DIRETOR",
                        "bbox": [527, 7113, 688, 7221],
                        "text_pixel_bbox": [527, 7113, 688, 7221],
                        "balloon_bbox": [485, 7068, 730, 7255],
                        "render_bbox": [559, 7104, 660, 7226],
                        "safe_text_box": [550, 7099, 668, 7232],
                        "bubble_mask_source": "image_contour_bubble_mask",
                        "source_text_ids": ["ocr_004", "ocr_003"],
                        "source_trace_ids": ["ocr_004@page_002_band_007", "ocr_003@page_002_band_007"],
                        "qa_flags": ["same_balloon_fragment_merged", "safe_text_box_recomputed"],
                    },
                    {
                        "id": "ocr_003",
                        "text_id": "ocr_003",
                        "trace_id": "ocr_003@page_002_band_007",
                        "band_id": "page_002_band_007",
                        "route_action": "translate_inpaint_render",
                        "translated": "OS JUROS JÁ FORAM REDUZIDOS EM MAIS DE TRÊS VEZES O DIRETOR",
                        "bbox": [555, 7209, 661, 7221],
                        "text_pixel_bbox": [555, 7209, 661, 7221],
                        "target_bbox": [461, 618, 754, 853],
                        "safe_text_box": [550, 673, 668, 806],
                        "render_bbox": [559, 678, 660, 800],
                        "bubble_mask_source": "derived_white_crop_rejected",
                        "source_text_ids": ["ocr_004", "ocr_003"],
                        "source_trace_ids": ["ocr_004@page_002_band_007", "ocr_003@page_002_band_007"],
                        "qa_flags": ["debug_derived_bubble_mask_rejected", "same_balloon_fragment_merged"],
                    },
                ],
            }
        ]
    }

    merged = _merge_same_balloon_fragment_layers(project)

    real_peer, rejected = project["paginas"][0]["text_layers"]
    assert merged == 1
    assert real_peer.get("visible") is not False
    assert real_peer["render_bbox"] == [559, 7104, 660, 7226]
    assert rejected["visible"] is False
    assert rejected["render_policy"] == "merged_into_primary"
    assert rejected["merged_into_text_id"] == "ocr_004"


def test_same_balloon_merge_suppresses_rejected_fragment_from_different_bubble_id():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_004",
                        "text_id": "ocr_004",
                        "trace_id": "ocr_004@page_002_band_007",
                        "band_id": "page_002_band_007",
                        "route_action": "translate_inpaint_render",
                        "visible": True,
                        "translated": "OS JUROS JA FORAM REDUZIDOS EM MAIS DE TRES VEZES O DIRETOR",
                        "bbox": [527, 7113, 688, 7221],
                        "text_pixel_bbox": [527, 7113, 688, 7221],
                        "balloon_bbox": [485, 7068, 730, 7255],
                        "bubble_id": "page_002_band_007_bubble_004",
                        "source_text_ids": ["ocr_004", "ocr_003"],
                        "source_trace_ids": [
                            "ocr_004@page_002_band_007",
                            "ocr_003@page_002_band_007",
                        ],
                        "qa_flags": ["same_balloon_fragment_merged"],
                    },
                    {
                        "id": "ocr_003",
                        "text_id": "ocr_003",
                        "trace_id": "ocr_003@page_002_band_007",
                        "band_id": "page_002_band_007",
                        "route_action": "translate_inpaint_render",
                        "visible": True,
                        "translated": "O DIRETOR",
                        "bbox": [555, 7209, 661, 7221],
                        "text_pixel_bbox": [555, 7209, 661, 7221],
                        "target_bbox": [461, 618, 754, 853],
                        "safe_text_box": [550, 673, 668, 806],
                        "render_bbox": [559, 678, 660, 800],
                        "bubble_id": "page_002_band_007_bubble_003",
                        "qa_flags": ["debug_derived_bubble_mask_rejected"],
                    },
                ],
            }
        ]
    }

    merged = _merge_same_balloon_fragment_layers(project)

    primary, other_bubble = project["paginas"][0]["text_layers"]
    assert merged == 1
    assert primary["translated"] == "OS JUROS JA FORAM REDUZIDOS EM MAIS DE TRES VEZES"
    assert other_bubble.get("visible") is False
    assert other_bubble.get("render_policy") == "merged_into_primary"
    assert other_bubble.get("route_action") == "merged_into_primary"


def test_same_balloon_fragment_is_merged_into_visible_peer_before_final_render():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001_fragment_2",
                        "band_id": "page_003_band_035",
                        "route_action": "translate_inpaint_render",
                        "translated": "ESTOU MORRENDO DE FOME",
                        "bbox": [345, 7665, 537, 7686],
                        "render_bbox": [193, 7408, 477, 7423],
                        "safe_text_box": [98, 7214, 574, 7617],
                        "qa_flags": ["missing_render_bbox"],
                    },
                    {
                        "id": "ocr_003",
                        "band_id": "page_003_band_035",
                        "route_action": "translate_inpaint_render",
                        "visible": True,
                        "translated": "QUEM ESTÁ PAGANDO HOJE?",
                        "bbox": [344, 7702, 540, 7761],
                        "target_bbox": [276, 7612, 598, 7799],
                        "safe_text_box": [344, 7645, 505, 7766],
                        "render_bbox": [361, 7670, 488, 7740],
                        "qa_flags": [],
                    },
                ],
            }
        ]
    }

    merged = _merge_same_balloon_fragment_layers(project)
    audit = _ensure_project_render_contract(project)

    fragment = project["paginas"][0]["text_layers"][0]
    primary = project["paginas"][0]["text_layers"][1]
    assert merged == 1
    assert primary["translated"] == "ESTOU MORRENDO DE FOME\nQUEM ESTÁ PAGANDO HOJE?"
    assert "same_balloon_fragment_merged" in primary["qa_flags"]
    assert fragment["visible"] is False
    assert fragment["route_action"] == "merged_into_primary"
    assert fragment["render_policy"] == "merged_into_primary"
    assert "missing_render_bbox" not in fragment["qa_flags"]
    assert audit["checked_layers"] == 1


def test_same_balloon_fragment_merge_keeps_distinct_dark_connected_lobes_separate():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_001_002",
                        "text_id": "ocr_001_002",
                        "trace_id": "ocr_001_002@page_002_band_023",
                        "band_id": "page_002_band_023",
                        "route_action": "translate_inpaint_render",
                        "visible": True,
                        "translated": "VOCÊ ERA LEAL AOS OUTROS",
                        "bbox": [132, 5078, 419, 5193],
                        "text_pixel_bbox": [132, 5078, 419, 5193],
                        "balloon_bbox": [63, 4962, 519, 5383],
                        "bubble_mask_bbox": [63, 4962, 519, 5383],
                        "bubble_mask_source": "image_dark_bubble_mask",
                        "qa_flags": ["dark_bubble_connected_lobes_promoted"],
                    },
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "trace_id": "ocr_001@page_002_band_023",
                        "band_id": "page_002_band_023",
                        "route_action": "translate_inpaint_render",
                        "visible": True,
                        "translated": "VOCÊ ERA O REI DE SER UMA TAREFA SIMPLES",
                        "bbox": [476, 5251, 649, 5324],
                        "text_pixel_bbox": [386, 5125, 649, 5322],
                        "balloon_bbox": [294, 4962, 712, 5393],
                        "bubble_mask_bbox": [294, 4962, 712, 5393],
                        "bubble_mask_source": "image_dark_bubble_mask",
                        "qa_flags": ["dark_bubble_connected_lobes_promoted"],
                    },
                    {
                        "id": "ocr_001_fragment_2",
                        "text_id": "ocr_001_fragment_2",
                        "trace_id": "ocr_001@page_002_band_023#fragment_2",
                        "band_id": "page_002_band_023",
                        "route_action": "translate_inpaint_render",
                        "translated": "VOCÊ ERA LEAL AOS OUTROS VOCÊ ERA O REI DE SER UMA TAREFA SIMPLES",
                        "bbox": [160, 5054, 613, 5290],
                        "balloon_bbox": [63, 4962, 712, 5393],
                        "bubble_mask_source": "image_dark_bubble_mask",
                        "qa_flags": ["dark_bubble_connected_lobes_promoted"],
                    },
                ],
            }
        ]
    }

    merged = _merge_same_balloon_fragment_layers(project)

    left, right, fragment = project["paginas"][0]["text_layers"]
    assert merged == 0
    assert left["translated"] == "VOCÊ ERA LEAL AOS OUTROS"
    assert right["translated"] == "VOCÊ ERA O REI DE SER UMA TAREFA SIMPLES"
    assert fragment.get("visible") is False
    assert fragment.get("render_policy") == "suppressed_dark_connected_combined_fragment"
    assert fragment.get("route_action") == "suppressed_dark_connected_combined_fragment"
    assert "dark_connected_combined_fragment_suppressed" in fragment.get("qa_flags", [])
    assert "same_balloon_fragment_merged" not in left.get("qa_flags", [])
    assert "same_balloon_fragment_merged" not in right.get("qa_flags", [])


def test_same_balloon_fragment_prefers_smaller_non_fallback_peer():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "band_id": "page_003_band_035",
                        "route_action": "translate_inpaint_render",
                        "visible": True,
                        "translated": "EI, VAMOS!",
                        "target_bbox": [29, 7109, 642, 7722],
                        "safe_text_box": [85, 7232, 393, 7284],
                        "qa_flags": ["bbox_fallback_bubble_mask", "debug_derived_bubble_mask_rejected"],
                    },
                    {
                        "id": "ocr_001_fragment_2",
                        "band_id": "page_003_band_035",
                        "route_action": "translate_inpaint_render",
                        "translated": "ESTOU MORRENDO DE FOME",
                        "bbox": [345, 7665, 537, 7686],
                        "qa_flags": ["missing_render_bbox"],
                    },
                    {
                        "id": "ocr_003",
                        "band_id": "page_003_band_035",
                        "route_action": "translate_inpaint_render",
                        "visible": True,
                        "translated": "QUEM ESTÁ PAGANDO HOJE?",
                        "bbox": [344, 7702, 540, 7761],
                        "target_bbox": [276, 7612, 598, 7799],
                        "safe_text_box": [344, 7645, 505, 7766],
                        "qa_flags": [],
                    },
                ],
            }
        ]
    }

    assert _merge_same_balloon_fragment_layers(project) == 1

    fallback, fragment, primary = project["paginas"][0]["text_layers"]
    assert fallback["translated"] == "EI, VAMOS!"
    assert primary["translated"] == "ESTOU MORRENDO DE FOME\nQUEM ESTÁ PAGANDO HOJE?"
    assert fragment["visible"] is False
    assert fragment["route_action"] == "merged_into_primary"


def test_broad_fallback_layer_is_suppressed_and_text_moves_to_smaller_peer():
    project = {
        "paginas": [
            {
                "numero": 2,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "band_id": "page_002_band_005",
                        "route_action": "translate_inpaint_render",
                        "visible": True,
                        "translated": "POR FAVOR, POR\nPELO BEM DA CRIANÇA.",
                        "target_bbox": [466, 4529, 696, 4700],
                        "bubble_mask_bbox": [25, 4357, 667, 4629],
                        "qa_flags": ["bbox_fallback_bubble_mask", "debug_derived_bubble_mask_rejected"],
                    },
                    {
                        "id": "ocr_002",
                        "text_id": "ocr_002",
                        "band_id": "page_002_band_005",
                        "route_action": "translate_inpaint_render",
                        "visible": True,
                        "translated": "PELO BEM DA CRIANÇA.",
                        "target_bbox": [466, 4527, 696, 4698],
                        "bubble_mask_bbox": [418, 4479, 744, 4746],
                        "qa_flags": ["rejected_derived_bubble_mask"],
                    },
                ],
            }
        ]
    }

    assert _suppress_broad_fallback_merge_layers(project) == 1

    broad, peer = project["paginas"][0]["text_layers"]
    assert broad["visible"] is False
    assert broad["route_action"] == "merged_into_primary"
    assert broad["render_policy"] == "merged_into_primary"
    assert broad["merged_into_text_id"] == "ocr_002"
    assert peer["translated"] == "POR FAVOR, POR PELO BEM DA CRIANÇA."
    assert "broad_fallback_text_merged" in peer["qa_flags"]


def test_broad_fallback_layer_does_not_suppress_different_overlapping_text():
    project = {
        "paginas": [
            {
                "numero": 3,
                "text_layers": [
                    {
                        "id": "ocr_001",
                        "text_id": "ocr_001",
                        "band_id": "page_003_band_035",
                        "route_action": "translate_inpaint_render",
                        "visible": True,
                        "translated": "EI, VAMOS!",
                        "target_bbox": [29, 7109, 642, 7722],
                        "bubble_mask_bbox": [29, 7109, 642, 7722],
                        "qa_flags": ["bbox_fallback_bubble_mask", "debug_derived_bubble_mask_rejected"],
                    },
                    {
                        "id": "ocr_003",
                        "text_id": "ocr_003",
                        "band_id": "page_003_band_035",
                        "route_action": "translate_inpaint_render",
                        "visible": True,
                        "translated": "ESTOU MORRENDO DE FOME QUEM ESTA PAGANDO HOJE?",
                        "target_bbox": [276, 7612, 598, 7799],
                        "bubble_mask_bbox": [276, 7612, 598, 7799],
                        "qa_flags": ["bbox_fallback_bubble_mask"],
                    },
                ],
            }
        ]
    }

    assert _suppress_broad_fallback_merge_layers(project) == 0

    broad, peer = project["paginas"][0]["text_layers"]
    assert broad["visible"] is True
    assert broad["route_action"] == "translate_inpaint_render"
    assert peer["visible"] is True


def test_neutralize_preserves_merged_into_primary_as_non_renderable():
    layer = {
        "id": "fragment",
        "route_action": "merged_into_primary",
        "render_policy": "merged_into_primary",
        "visible": False,
        "translated": "ESTOU MORRENDO DE FOME",
        "content_class": "dialogue",
        "tipo": "fala",
    }

    normalized = neutralize_removed_decision_fields(layer)

    assert normalized["route_action"] == "merged_into_primary"
    assert normalized["render_policy"] == "merged_into_primary"
    assert normalized["visible"] is False
    assert _visible_render_texts([normalized]) == []


def test_hidden_translated_layer_is_not_rendered_without_explicit_force_flag():
    hidden = {
        "id": "hidden",
        "visible": False,
        "render_policy": "normal",
        "route_action": "translate_inpaint_render",
        "translated": "EI, VAMOS!",
    }
    forced = {**hidden, "id": "forced", "_force_render_hidden": True}

    assert _visible_render_texts([hidden]) == []
    assert _visible_render_texts([forced]) == [forced]


def test_visual_merge_ignores_rejected_broad_bubble_mask_bbox():
    broad_fallback = {
        "band_id": "page_002_band_005",
        "target_bbox": [466, 4529, 696, 4700],
        "bubble_mask_bbox": [25, 4357, 667, 4629],
        "qa_flags": ["bbox_fallback_bubble_mask", "debug_derived_bubble_mask_rejected"],
    }
    correct_peer = {
        "band_id": "page_002_band_005",
        "target_bbox": [466, 4527, 696, 4698],
        "bubble_mask_bbox": [418, 4479, 744, 4746],
        "qa_flags": ["rejected_derived_bubble_mask"],
    }

    assert _visual_target_bbox_for_merge(broad_fallback) == [466, 4529, 696, 4700]
    assert _visual_targets_share_merge_region(broad_fallback, correct_peer) is True


def test_merged_candidate_prefers_non_fallback_peer_as_primary_layer():
    broad_fallback = {
        "id": "ocr_001",
        "text_id": "ocr_001",
        "trace_id": "ocr_001@page_002_band_005",
        "band_id": "page_002_band_005",
        "bbox": [499, 4578, 656, 4663],
        "target_bbox": [466, 4529, 696, 4700],
        "bubble_mask_bbox": [25, 4357, 667, 4629],
        "qa_flags": ["bbox_fallback_bubble_mask", "debug_derived_bubble_mask_rejected"],
    }
    correct_peer = {
        "id": "ocr_002",
        "text_id": "ocr_002",
        "trace_id": "ocr_002@page_002_band_005",
        "band_id": "page_002_band_005",
        "bbox": [509, 4607, 647, 4661],
        "target_bbox": [466, 4527, 696, 4698],
        "bubble_mask_bbox": [418, 4479, 744, 4746],
        "qa_flags": ["rejected_derived_bubble_mask"],
    }
    candidate = {
        "band_id": "page_002_band_005",
        "source_text_ids": ["ocr_001", "ocr_002"],
        "source_trace_ids": ["ocr_001@page_002_band_005", "ocr_002@page_002_band_005"],
    }
    identities = {
        "ocr_001": [broad_fallback],
        "ocr_001@page_002_band_005": [broad_fallback],
        "ocr_002": [correct_peer],
        "ocr_002@page_002_band_005": [correct_peer],
    }

    assert _primary_layer_for_merged_candidate(broad_fallback, candidate, identities) is correct_peer
    assert _primary_layer_for_merged_candidate(correct_peer, candidate, identities) is correct_peer


def test_hide_merged_candidate_siblings_only_runs_for_resolved_primary():
    broad_fallback = {
        "id": "ocr_001",
        "text_id": "ocr_001",
        "trace_id": "ocr_001@page_002_band_005",
        "band_id": "page_002_band_005",
        "route_action": "translate_inpaint_render",
        "target_bbox": [466, 4529, 696, 4700],
        "bubble_mask_bbox": [25, 4357, 667, 4629],
        "qa_flags": ["bbox_fallback_bubble_mask", "debug_derived_bubble_mask_rejected"],
    }
    correct_peer = {
        "id": "ocr_002",
        "text_id": "ocr_002",
        "trace_id": "ocr_002@page_002_band_005",
        "band_id": "page_002_band_005",
        "route_action": "translate_inpaint_render",
        "target_bbox": [466, 4527, 696, 4698],
        "bubble_mask_bbox": [418, 4479, 744, 4746],
        "qa_flags": ["rejected_derived_bubble_mask"],
    }
    candidate = {
        "band_id": "page_002_band_005",
        "source_text_ids": ["ocr_001", "ocr_002"],
        "source_trace_ids": ["ocr_001@page_002_band_005", "ocr_002@page_002_band_005"],
    }
    identities = {
        "ocr_001": [broad_fallback],
        "ocr_001@page_002_band_005": [broad_fallback],
        "ocr_002": [correct_peer],
        "ocr_002@page_002_band_005": [correct_peer],
    }

    assert _hide_merged_candidate_sibling_layers(broad_fallback, candidate, identities) == 0
    assert correct_peer.get("visible") is not False
    assert _hide_merged_candidate_sibling_layers(correct_peer, candidate, identities) == 1
    assert broad_fallback["visible"] is False
    assert broad_fallback["merged_into_text_id"] == "ocr_002"


def test_mark_final_layer_as_page_space_infers_shifted_target_from_bubble_mask():
    layer = {
        "coordinate_space": "page",
        "source_coordinate_space": "page",
        "target_bbox": [29, 96, 642, 709],
        "safe_text_box": [85, 219, 393, 271],
        "render_bbox": [148, 235, 300, 260],
        "bubble_mask_bbox": [29, 7109, 642, 7722],
        "bbox": [148, 235, 310, 255],
        "text_pixel_bbox": [148, 235, 310, 255],
    }

    fixed = _mark_final_layer_as_page_space(layer)

    assert fixed["target_bbox"] == [29, 7109, 642, 7722]
    assert fixed["safe_text_box"] == [85, 7232, 393, 7284]
    assert fixed["render_bbox"] == [148, 7248, 300, 7273]
    assert fixed["bbox"] == [148, 7248, 310, 7268]
    assert fixed["text_pixel_bbox"] == [148, 7248, 310, 7268]


def test_mark_final_layer_as_page_space_infers_primary_bbox_shift_from_line_polygons():
    layer = {
        "coordinate_space": "page",
        "source_coordinate_space": "page",
        "target_bbox": [29, 7109, 642, 7722],
        "safe_text_box": [85, 7232, 393, 7284],
        "render_bbox": [148, 7248, 300, 7273],
        "bubble_mask_bbox": [29, 7109, 642, 7722],
        "bbox": [148, 235, 310, 255],
        "text_pixel_bbox": [148, 235, 310, 255],
        "line_polygons": [[[148, 7248], [310, 7248], [310, 7268], [148, 7268]]],
    }

    fixed = _mark_final_layer_as_page_space(layer)

    assert fixed["target_bbox"] == [29, 7109, 642, 7722]
    assert fixed["bbox"] == [148, 7248, 310, 7268]
    assert fixed["text_pixel_bbox"] == [148, 7248, 310, 7268]


def test_mark_final_layer_as_page_space_drops_foreign_line_polygons():
    layer = {
        "coordinate_space": "page",
        "source_coordinate_space": "page",
        "bbox": [148, 7248, 310, 7268],
        "text_pixel_bbox": [148, 7248, 310, 7268],
        "line_polygons": [
            [[148, 7248], [310, 7248], [310, 7268], [148, 7268]],
            [[345, 7665], [537, 7665], [537, 7686], [345, 7686]],
        ],
    }

    fixed = _mark_final_layer_as_page_space(layer)

    assert fixed["line_polygons"] == [
        [[148, 7248], [310, 7248], [310, 7268], [148, 7268]]
    ]
