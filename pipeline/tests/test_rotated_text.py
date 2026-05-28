from __future__ import annotations

import numpy as np

from inpainter.mask_builder import mask_from_text_geometry
from ocr.ocr_normalizer import normalize_ocr_record
from runtime_profiles import ROTATED_TEXT_POLICY, resolve_runtime_profile
from qa.translation_qa import severity_for_flag
from typesetter.renderer import plan_text_layout
from vision_stack.detector import TextDetector
from vision_stack.ocr import _attach_rotated_text_metadata, infer_rotation_deg_from_line_polygons


def _style() -> dict:
    return {
        "fonte": "ComicNeue-Bold.ttf",
        "tamanho": 28,
        "cor": "#111111",
        "contorno": "",
        "contorno_px": 0,
        "alinhamento": "center",
        "sombra": False,
        "glow": False,
        "cor_gradiente": [],
        "rotacao": 0,
    }


def test_ocr_normalizer_preserves_trace_id_and_adds_rotation_aliases():
    polygon = [[10, 10], [72, 22], [68, 42], [6, 30]]
    normalized = normalize_ocr_record(
        {
            "trace_id": "page_001_band_001_t1",
            "text": "Tilted line",
            "rotation_deg": 12.345,
            "rotation_source": "ocr",
            "line_polygons": [polygon],
        }
    )

    assert normalized["trace_id"] == "page_001_band_001_t1"
    assert normalized["text_angle_degrees"] == 12.35
    assert normalized["text_orientation"] == "rotated"
    assert normalized["rotated_polygon"] == polygon


def test_ocr_normalizer_uses_rotation_deg_when_text_angle_alias_is_none():
    normalized = normalize_ocr_record(
        {
            "trace_id": "page_001_band_001_t2",
            "text": "Tilted",
            "text_angle_degrees": None,
            "rotation_deg": 18,
            "line_polygons": [[[20, 20], [86, 40], [80, 60], [14, 40]]],
        }
    )

    assert normalized["trace_id"] == "page_001_band_001_t2"
    assert normalized["text_angle_degrees"] == 18
    assert normalized["text_orientation"] == "rotated"


def test_vision_stack_ocr_records_rotation_aliases_when_angle_is_available():
    record = {
        "text": "Vertical sign",
        "rotation_deg": 90,
        "line_polygons": [[[30, 5], [50, 5], [50, 95], [30, 95]]],
    }

    updated = _attach_rotated_text_metadata(record)

    assert updated is record
    assert updated["text_angle_degrees"] == 90
    assert updated["text_orientation"] == "vertical"
    assert updated["rotated_polygon"] == [[30, 5], [50, 5], [50, 95], [30, 95]]


def test_vision_stack_ocr_inferrs_oblique_angle_from_single_line_polygon():
    rotation = infer_rotation_deg_from_line_polygons(
        [
            [[20, 20], [86, 40], [80, 60], [14, 40]],
        ]
    )

    assert rotation == 16.86


def test_renderer_inferrs_oblique_angle_from_single_line_polygon():
    text_data = {
        "style_origin": "auto",
        "translated": "INCLINADO",
        "bbox": [10, 10, 110, 90],
        "source_bbox": [10, 10, 110, 90],
        "text_pixel_bbox": [14, 20, 86, 60],
        "balloon_bbox": [10, 10, 110, 90],
        "line_polygons": [[[20, 20], [86, 40], [80, 60], [14, 40]]],
        "tipo": "fala",
        "estilo": _style(),
        "layout_shape": "wide",
        "layout_align": "center",
    }

    plan = plan_text_layout(text_data)

    assert plan["rotation_deg"] == 16.86
    assert plan["rotation_source"] == "line_polygons"


def test_mask_builder_prefers_rotated_polygon_over_axis_aligned_bbox_for_angled_text():
    block = {
        "rotation_deg": 18,
        "rotated_polygon": [[30, 40], [92, 54], [87, 76], [25, 62]],
        "text_pixel_bbox": [4, 4, 18, 18],
        "bbox": [0, 0, 118, 118],
        "font_size_px": 4,
    }

    mask = mask_from_text_geometry(block, (120, 120, 3))

    assert mask is not None
    assert int(mask[58, 50]) > 0
    assert int(mask[10, 10]) == 0


def test_detector_preserves_paddle_polygon_rotation_metadata():
    detector = object.__new__(TextDetector)
    results = [
        [
            [[20, 20], [86, 40], [80, 60], [14, 40]],
        ]
    ]

    blocks = detector._parse_paddle_detection(results, orig_h=120, orig_w=120, infer_size=(120, 120))

    assert len(blocks) == 1
    assert blocks[0].line_polygons == [[[20, 20], [86, 40], [80, 60], [14, 40]]]
    assert blocks[0].rotation_deg == 16.86
    assert blocks[0].rotation_source == "detector_polygon"
    assert blocks[0].text_angle_degrees == 16.86
    assert blocks[0].text_orientation == "rotated"
    assert blocks[0].rotated_polygon == [[20, 20], [86, 40], [80, 60], [14, 40]]


def test_runtime_profile_exposes_rotated_text_policy_and_flag_default_off(monkeypatch):
    monkeypatch.delenv("TRADUZAI_FLAG_ROTATED_TEXT_V2", raising=False)

    decision = resolve_runtime_profile({})

    assert ROTATED_TEXT_POLICY["dialogue"]["render_same_angle_if_abs_angle_in"] == (5, 35)
    assert decision.visual_pipeline_flags["rotated_text_v2"] is False
    assert decision.to_dict()["rotated_text_policy"]["dialogue"]["block_if_vertical"] is True


def test_runtime_profile_allows_rotated_text_flag_env_override(monkeypatch):
    monkeypatch.setenv("TRADUZAI_FLAG_ROTATED_TEXT_V2", "true")

    decision = resolve_runtime_profile({})

    assert decision.visual_pipeline_flags["rotated_text_v2"] is True


def test_renderer_flags_vertical_dialogue_as_rotated_text_policy_unmet():
    text_data = {
        "style_origin": "auto",
        "translated": "SIDEWAYS",
        "bbox": [90, 40, 150, 220],
        "source_bbox": [90, 40, 150, 220],
        "text_pixel_bbox": [105, 52, 135, 208],
        "balloon_bbox": [90, 40, 150, 220],
        "rotation_deg": 90,
        "tipo": "fala",
        "estilo": _style(),
        "layout_shape": "tall",
        "layout_align": "center",
    }

    plan = plan_text_layout(text_data)

    assert plan["rotation_deg"] == 90
    assert "rotated_text_policy_unmet" in text_data["qa_flags"]
    assert text_data["needs_review"] is True
    assert severity_for_flag("rotated_text_policy_unmet") == "high"


def test_renderer_keeps_signage_at_source_angle_without_policy_flag():
    text_data = {
        "style_origin": "auto",
        "translated": "EXIT",
        "bbox": [20, 20, 150, 90],
        "source_bbox": [20, 20, 150, 90],
        "text_pixel_bbox": [28, 32, 142, 78],
        "balloon_bbox": [20, 20, 150, 90],
        "rotation_deg": -22,
        "tipo": "sign",
        "estilo": _style(),
        "layout_shape": "wide",
        "layout_align": "center",
    }

    plan = plan_text_layout(text_data)

    assert plan["rotation_deg"] == -22
    assert "rotated_text_policy_unmet" not in text_data.get("qa_flags", [])
