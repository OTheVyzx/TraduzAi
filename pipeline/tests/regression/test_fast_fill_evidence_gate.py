from unittest.mock import patch

import cv2
import numpy as np


def _allowed_mask_evidence(kind: str = "glyph_segmentation") -> dict:
    return {
        "kind": kind,
        "raw_mask_pixels": 120,
        "expanded_mask_pixels": 160,
        "evidence_score": 1.0,
        "fast_fill_allowed": True,
        "fast_fill_reject_reasons": [],
    }


def _blocked_mask_evidence(reason: str) -> dict:
    return {
        "kind": "clipped_line_polygon",
        "raw_mask_pixels": 0,
        "expanded_mask_pixels": 160,
        "evidence_score": 0.0,
        "fast_fill_allowed": False,
        "fast_fill_reject_reasons": [reason],
    }


def _white_balloon_fixture(*, with_evidence: bool = True, blocked_reason: str | None = None):
    image = np.full((120, 220, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (110, 60), (76, 40), 0, 0, 360, (255, 255, 255), -1)
    image[54:66, 76:144] = 10
    text = {
        "id": "white_001",
        "text_id": "white_001",
        "trace_id": "white_001@band",
        "bbox": [70, 48, 150, 72],
        "text_pixel_bbox": [76, 54, 144, 66],
        "line_polygons": [[[76, 54], [144, 54], [144, 66], [76, 66]]],
        "balloon_bbox": [34, 20, 186, 100],
        "balloon_type": "white",
        "layout_profile": "white_balloon",
        "content_class": "dialogue",
        "tipo": "fala",
        "skip_processing": False,
    }
    if blocked_reason:
        text["mask_evidence"] = _blocked_mask_evidence(blocked_reason)
    elif with_evidence:
        text["mask_evidence"] = _allowed_mask_evidence()
    return image, text, {"texts": [text], "_vision_blocks": [dict(text)]}


def test_fast_white_allows_simple_white_balloon_with_mask_evidence():
    from inpainter import _apply_fast_white_balloon_fill

    image, _text, page = _white_balloon_fixture(with_evidence=True)

    with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1"}, clear=False):
        result, remaining, stats = _apply_fast_white_balloon_fill(image, page, list(page["_vision_blocks"]))

    assert stats["white_balloon_count"] == 1
    assert remaining == []
    assert np.any(result != image)


def test_fast_white_rejects_missing_mask_evidence_before_local_geometry_fill():
    from inpainter import _apply_fast_white_balloon_fill

    image, text, page = _white_balloon_fixture(with_evidence=False)

    with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1"}, clear=False):
        result, remaining, stats = _apply_fast_white_balloon_fill(image, page, list(page["_vision_blocks"]))

    assert stats["white_balloon_count"] == 0
    assert remaining == [dict(text)]
    assert page["_strip_fast_white_rejection_reasons"] == {"mask_evidence:missing": 1}
    assert np.array_equal(result, image)


def test_fast_white_rejects_translucent_mask_evidence_before_visual_heuristics():
    from inpainter import _apply_fast_white_balloon_fill

    image, text, page = _white_balloon_fixture(blocked_reason="translucent_background")

    with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1"}, clear=False):
        result, remaining, stats = _apply_fast_white_balloon_fill(image, page, list(page["_vision_blocks"]))

    assert stats["white_balloon_count"] == 0
    assert remaining == [dict(text)]
    assert page["_strip_fast_white_rejection_reasons"] == {"mask_evidence:translucent_background": 1}
    assert np.array_equal(result, image)


def test_fast_white_mask_density_flag_cannot_bypass_missing_evidence_gate():
    from inpainter import _apply_fast_white_balloon_fill

    image, text, page = _white_balloon_fixture(with_evidence=False)
    text["qa_flags"] = ["mask_density_high"]
    page["_vision_blocks"] = [dict(text)]

    with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1"}, clear=False):
        result, remaining, stats = _apply_fast_white_balloon_fill(image, page, list(page["_vision_blocks"]))

    assert stats["white_balloon_count"] == 0
    assert remaining == [dict(text)]
    assert page["_strip_fast_white_rejection_reasons"] == {"mask_evidence:missing": 1}
    assert np.array_equal(result, image)


def test_connected_white_geometry_fill_requires_mask_evidence():
    from inpainter import _apply_connected_white_geometry_fill

    image = np.full((140, 240, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (78, 68), (62, 44), 0, 0, 360, (0, 0, 0), 2)
    cv2.ellipse(image, (164, 76), (66, 48), 0, 0, 360, (0, 0, 0), 2)
    image[48:58, 50:112] = 12
    image[66:76, 48:118] = 12
    image[68:78, 144:208] = 12
    image[88:98, 138:214] = 12
    text = {
        "id": "connected_001",
        "text_id": "connected_001",
        "trace_id": "connected_001@band",
        "bbox": [46, 46, 216, 100],
        "text_pixel_bbox": [48, 48, 214, 98],
        "line_polygons": [
            [[50, 48], [112, 48], [112, 58], [50, 58]],
            [[48, 66], [118, 66], [118, 76], [48, 76]],
            [[144, 68], [208, 68], [208, 78], [144, 78]],
            [[138, 88], [214, 88], [214, 98], [138, 98]],
        ],
        "balloon_bbox": [8, 22, 232, 126],
        "balloon_subregions": [[8, 22, 124, 112], [108, 28, 232, 126]],
        "layout_profile": "connected_balloon",
        "balloon_type": "white",
        "content_class": "dialogue",
        "tipo": "fala",
        "skip_processing": False,
    }
    page = {"texts": [text], "_vision_blocks": [dict(text)]}

    with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1"}, clear=False):
        result, remaining, stats = _apply_connected_white_geometry_fill(image, page, list(page["_vision_blocks"]))

    assert stats["connected_white_count"] == 0
    assert remaining == [dict(text)]
    assert page["_strip_connected_white_rejection_reasons"] == {"mask_evidence:missing": 1}
    assert np.array_equal(result, image)


def test_connected_white_geometry_fill_allows_explicit_mask_evidence():
    from inpainter import _apply_connected_white_geometry_fill

    image = np.full((140, 240, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (78, 68), (62, 44), 0, 0, 360, (0, 0, 0), 2)
    cv2.ellipse(image, (164, 76), (66, 48), 0, 0, 360, (0, 0, 0), 2)
    image[48:58, 50:112] = 12
    image[66:76, 48:118] = 12
    image[68:78, 144:208] = 12
    image[88:98, 138:214] = 12
    text = {
        "id": "connected_001",
        "text_id": "connected_001",
        "trace_id": "connected_001@band",
        "bbox": [46, 46, 216, 100],
        "text_pixel_bbox": [48, 48, 214, 98],
        "line_polygons": [
            [[50, 48], [112, 48], [112, 58], [50, 58]],
            [[48, 66], [118, 66], [118, 76], [48, 76]],
            [[144, 68], [208, 68], [208, 78], [144, 78]],
            [[138, 88], [214, 88], [214, 98], [138, 98]],
        ],
        "balloon_bbox": [8, 22, 232, 126],
        "balloon_subregions": [[8, 22, 124, 112], [108, 28, 232, 126]],
        "layout_profile": "connected_balloon",
        "balloon_type": "white",
        "content_class": "dialogue",
        "tipo": "fala",
        "skip_processing": False,
        "mask_evidence": _allowed_mask_evidence(),
    }
    page = {"texts": [text], "_vision_blocks": [dict(text)]}

    with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_WHITE_INPAINT": "1"}, clear=False):
        result, remaining, stats = _apply_connected_white_geometry_fill(image, page, list(page["_vision_blocks"]))

    assert stats["connected_white_count"] == 1
    assert remaining == []
    assert np.any(result != image)


def test_fast_local_fill_requires_mask_evidence():
    from inpainter import _apply_fast_local_balloon_fill

    image = np.full((90, 180, 3), 255, dtype=np.uint8)
    image[38:52, 62:118] = 15
    text = {
        "id": "local_001",
        "text_id": "local_001",
        "bbox": [56, 32, 124, 58],
        "text_pixel_bbox": [62, 38, 118, 52],
        "line_polygons": [[[62, 38], [118, 38], [118, 52], [62, 52]]],
        "balloon_bbox": [26, 18, 154, 74],
        "background_rgb": [255, 255, 255],
        "balloon_type": "white",
        "content_class": "dialogue",
        "tipo": "fala",
        "skip_processing": False,
    }
    page = {"texts": [text], "_vision_blocks": [dict(text)]}

    with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "1"}, clear=False):
        result, remaining, stats = _apply_fast_local_balloon_fill(image, page, list(page["_vision_blocks"]))

    assert stats["local_balloon_count"] == 0
    assert remaining == [dict(text)]
    assert page["_strip_fast_local_rejection_reasons"] == {"mask_evidence:missing": 1}
    assert np.array_equal(result, image)


def test_fast_local_fill_rejects_disallowed_mask_evidence():
    from inpainter import _apply_fast_local_balloon_fill

    image = np.full((90, 180, 3), 255, dtype=np.uint8)
    image[38:52, 62:118] = 15
    text = {
        "id": "local_001",
        "text_id": "local_001",
        "bbox": [56, 32, 124, 58],
        "text_pixel_bbox": [62, 38, 118, 52],
        "line_polygons": [[[62, 38], [118, 38], [118, 52], [62, 52]]],
        "balloon_bbox": [26, 18, 154, 74],
        "background_rgb": [255, 255, 255],
        "balloon_type": "white",
        "content_class": "dialogue",
        "tipo": "fala",
        "skip_processing": False,
        "mask_evidence": _blocked_mask_evidence("translucent_background"),
    }
    page = {"texts": [text], "_vision_blocks": [dict(text)]}

    with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "1"}, clear=False):
        result, remaining, stats = _apply_fast_local_balloon_fill(image, page, list(page["_vision_blocks"]))

    assert stats["local_balloon_count"] == 0
    assert remaining == [dict(text)]
    assert page["_strip_fast_local_rejection_reasons"] == {"mask_evidence:translucent_background": 1}
    assert np.array_equal(result, image)


def test_fast_dark_fill_requires_mask_evidence():
    from inpainter import _apply_fast_dark_panel_text_fill

    image = np.full((90, 180, 3), 7, dtype=np.uint8)
    image[34:50, 48:132] = [210, 208, 184]
    text = {
        "id": "dark_001",
        "text_id": "dark_001",
        "bbox": [48, 34, 132, 50],
        "text_pixel_bbox": [48, 34, 132, 50],
        "line_polygons": [[[48, 34], [132, 34], [132, 50], [48, 50]]],
        "balloon_bbox": [28, 20, 152, 66],
        "balloon_type": "textured",
        "layout_profile": "standard",
        "content_class": "dialogue",
        "tipo": "narracao",
        "skip_processing": False,
    }
    page = {"texts": [text], "_vision_blocks": [dict(text)]}

    with patch.dict("os.environ", {"TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "1"}, clear=False):
        result, remaining, stats = _apply_fast_dark_panel_text_fill(image, page, list(page["_vision_blocks"]))

    assert stats["dark_panel_fill_count"] == 0
    assert remaining == [dict(text)]
    assert page["_strip_fast_dark_rejection_reasons"] == {"mask_evidence:missing": 1}
    assert np.array_equal(result, image)


def test_fast_solid_allows_black_and_colored_solid_regions_with_mask_evidence():
    from inpainter import _apply_fast_solid_balloon_fill

    fixtures = [
        ("black", np.asarray([3, 3, 3], dtype=np.uint8), np.asarray([245, 245, 245], dtype=np.uint8), "solid_dark"),
        ("colored", np.asarray([221, 238, 246], dtype=np.uint8), np.asarray([8, 8, 8], dtype=np.uint8), "solid_color"),
    ]

    for name, fill, text_color, profile in fixtures:
        image = np.full((100, 220, 3), 255, dtype=np.uint8)
        image[22:78, 36:184] = fill
        image[44:56, 76:144] = text_color
        text = {
            "id": f"{name}_001",
            "text_id": f"{name}_001",
            "bbox": [68, 38, 152, 62],
            "text_pixel_bbox": [76, 44, 144, 56],
            "line_polygons": [[[76, 44], [144, 44], [144, 56], [76, 56]]],
            "balloon_bbox": [36, 22, 184, 78],
            "balloon_type": "dark" if name == "black" else "colored",
            "layout_profile": profile,
            "content_class": "dialogue",
            "tipo": "fala",
            "skip_processing": False,
            "mask_evidence": _allowed_mask_evidence(),
        }
        page = {"texts": [text], "_vision_blocks": [dict(text)]}

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1",
                "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
                "TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0",
            },
            clear=False,
        ):
            result, remaining, stats = _apply_fast_solid_balloon_fill(image, page, list(page["_vision_blocks"]))

        assert stats["solid_balloon_count"] == 1
        assert remaining == []
        assert page["_strip_fast_solid_fill_samples"][0]["mask_evidence"]["fast_fill_allowed"] is True
        assert np.any(result != image)


def test_fast_solid_rejects_disallowed_mask_evidence():
    from inpainter import _apply_fast_solid_balloon_fill

    image = np.full((100, 220, 3), 255, dtype=np.uint8)
    image[22:78, 36:184] = np.asarray([221, 238, 246], dtype=np.uint8)
    image[44:56, 76:144] = 8
    text = {
        "id": "colored_001",
        "text_id": "colored_001",
        "bbox": [68, 38, 152, 62],
        "text_pixel_bbox": [76, 44, 144, 56],
        "line_polygons": [[[76, 44], [144, 44], [144, 56], [76, 56]]],
        "balloon_bbox": [36, 22, 184, 78],
        "balloon_type": "colored",
        "layout_profile": "solid_color",
        "content_class": "dialogue",
        "tipo": "fala",
        "skip_processing": False,
        "mask_evidence": _blocked_mask_evidence("translucent_background"),
    }
    page = {"texts": [text], "_vision_blocks": [dict(text)]}

    with patch.dict(
        "os.environ",
        {
            "TRADUZAI_STRIP_FAST_SOLID_INPAINT": "1",
            "TRADUZAI_STRIP_FAST_WHITE_INPAINT": "0",
            "TRADUZAI_STRIP_FAST_LOCAL_INPAINT": "0",
            "TRADUZAI_STRIP_FAST_DARK_PANEL_FILL": "0",
        },
        clear=False,
    ):
        result, remaining, stats = _apply_fast_solid_balloon_fill(image, page, list(page["_vision_blocks"]))

    assert stats["solid_balloon_count"] == 0
    assert remaining == [dict(text)]
    assert page["_strip_fast_solid_rejection_reasons"] == {"mask_evidence:translucent_background": 1}
    assert np.array_equal(result, image)
