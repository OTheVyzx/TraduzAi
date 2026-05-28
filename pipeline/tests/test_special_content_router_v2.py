import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr.text_router import route_text
from typesetter.renderer import build_render_blocks


def test_text_marker_routes_as_sign_with_confident_sign_bbox():
    result = route_text(
        "TEXT: DARLING KARAOKE",
        text_id="ocr_sign",
        tipo="narracao",
        bbox=[120, 220, 390, 280],
        sign_bbox=[100, 200, 410, 300],
    )

    assert result["route"] == "sign"
    assert result["content_class"] == "sign"
    assert result["tipo"] == "sign"
    assert result["route_action"] == "translate_render_only"
    assert result["route_reason"] == "sign_bbox_available"
    assert result["skip_processing"] is False
    assert result["render_policy"] == "render_in_sign_bbox"
    assert result["sign_bbox"] == [100, 200, 410, 300]
    assert result["needs_review"] is False


def test_text_marker_without_sign_bbox_is_preserved_for_review_not_narration():
    result = route_text("TEXT: DARLING KARAOKE", text_id="ocr_sign", tipo="narracao")

    assert result["route"] == "sign"
    assert result["content_class"] == "sign"
    assert result["tipo"] == "sign"
    assert result["route_action"] == "review_required"
    assert result["route_reason"] == "missing_reliable_sign_bbox"
    assert result["skip_processing"] is False
    assert result["render_policy"] == "preserve_original"
    assert result["needs_review"] is True


def test_renderer_does_not_render_sign_as_narration_without_sign_bbox():
    blocks = build_render_blocks(
        [
            {
                "id": "ocr_sign",
                "text": "TEXT: DARLING KARAOKE",
                "translated": "DARLING KARAOKE",
                "bbox": [100, 200, 410, 300],
                "tipo": "narracao",
                "content_class": "sign",
                "skip_processing": False,
            }
        ]
    )

    assert blocks == []


def test_renderer_updates_route_action_for_sign_missing_sign_bbox():
    text = {
        "id": "ocr_sign",
        "text": "TEXT: DARLING KARAOKE",
        "translated": "DARLING KARAOKE",
        "bbox": [100, 200, 410, 300],
        "tipo": "sign",
        "content_class": "sign",
        "route_action": "translate_render_only",
        "route_reason": "sign_bbox_available",
        "skip_processing": False,
    }

    blocks = build_render_blocks([text])

    assert blocks == []
    assert text["route_action"] == "review_required"
    assert text["route_reason"] == "missing_reliable_sign_bbox"
    assert text["skip_processing"] is False
    assert text["preserve_original"] is True
    assert text["render_policy"] == "preserve_original"


def test_renderer_respects_non_render_route_action_without_mutating_skip():
    text = {
        "id": "ocr_watermark",
        "text": "Read at ASURACOMIC.NET",
        "translated": "Read at ASURACOMIC.NET",
        "bbox": [100, 200, 410, 300],
        "tipo": "watermark",
        "content_class": "url_watermark",
        "route_action": "inpaint_only",
        "route_reason": "watermark_detected",
        "skip_processing": False,
    }

    blocks = build_render_blocks([text])

    assert blocks == []
    assert text["skip_processing"] is False


def test_renderer_clamps_sign_layout_to_sign_bbox():
    blocks = build_render_blocks(
        [
            {
                "id": "ocr_sign",
                "text": "TEXT: DARLING KARAOKE",
                "translated": "DARLING KARAOKE",
                "bbox": [80, 180, 470, 340],
                "balloon_bbox": [60, 160, 500, 360],
                "sign_bbox": [120, 210, 390, 290],
                "tipo": "narracao",
                "content_class": "sign",
                "skip_processing": False,
            }
        ]
    )

    assert len(blocks) == 1
    block = blocks[0]
    assert block["tipo"] == "sign"
    assert block["bbox"] == [120, 210, 390, 290]
    assert block["balloon_bbox"] == [120, 210, 390, 290]
    assert block["render_policy"] == "render_in_sign_bbox"
