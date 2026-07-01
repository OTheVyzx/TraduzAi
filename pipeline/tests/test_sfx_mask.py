import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfx.mask import build_sfx_glyph_mask


def _draw_sfx_image() -> tuple[np.ndarray, dict]:
    image = np.full((120, 220, 3), 236, dtype=np.uint8)
    layer = {
        "bbox": [20, 20, 200, 100],
        "text": "쿵",
        "content_class": "sfx",
        "script": "hangul",
        "route_action": "translate_sfx_inpaint_render",
    }
    cv2.rectangle(image, (42, 34), (54, 78), (12, 12, 12), -1)
    cv2.rectangle(image, (42, 68), (78, 80), (12, 12, 12), -1)
    cv2.rectangle(image, (132, 34), (144, 82), (12, 12, 12), -1)
    cv2.rectangle(image, (132, 34), (166, 46), (12, 12, 12), -1)
    cv2.rectangle(image, (152, 58), (166, 82), (12, 12, 12), -1)
    return image, layer


def _component_count(mask: np.ndarray) -> int:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return sum(1 for label in range(1, count) if int(stats[label, cv2.CC_STAT_AREA]) > 0)


def test_sfx_mask_does_not_fill_bbox():
    image, layer = _draw_sfx_image()

    result = build_sfx_glyph_mask(image, layer)

    assert result.mask is not None
    bbox_area = (layer["bbox"][2] - layer["bbox"][0]) * (layer["bbox"][3] - layer["bbox"][1])
    mask_area = int(np.count_nonzero(result.mask))
    assert mask_area < bbox_area * 0.45
    assert result.evidence["kind"] == "sfx_glyph_mask"
    assert result.evidence["bbox_fill_ratio"] < 0.45


def test_sfx_mask_preserves_component_shape():
    image, layer = _draw_sfx_image()

    result = build_sfx_glyph_mask(image, layer)

    assert result.mask is not None
    assert result.evidence["component_count"] >= 2
    assert _component_count(result.mask) >= 2


def test_sfx_mask_rejects_high_density_full_bbox_masks():
    image, layer = _draw_sfx_image()
    layer["mask"] = np.full((80, 120), 255, dtype=np.uint8)

    result = build_sfx_glyph_mask(image, layer)

    assert result.mask is None
    assert result.evidence["reject_reason"] == "density_too_high"
    assert result.evidence["bbox_fill_ratio"] >= 0.52


def test_sfx_mask_rejects_non_hangul_non_sfx_layers():
    image, layer = _draw_sfx_image()
    layer.update(
        {
            "text": "BOOM",
            "content_class": "text",
            "script": "latin",
            "route_action": "translate_inpaint_render",
            "tipo": "fala",
        }
    )

    result = build_sfx_glyph_mask(image, layer)

    assert result.mask is None
    assert result.evidence["reject_reason"] == "missing_hangul_sfx_evidence"


def test_sfx_mask_rejects_invalid_bbox():
    image, layer = _draw_sfx_image()
    layer["bbox"] = [20, 20, 20, 80]

    result = build_sfx_glyph_mask(image, layer)

    assert result.mask is None
    assert result.evidence["reject_reason"] == "invalid_bbox"


def test_sfx_mask_rejects_empty_image():
    _image, layer = _draw_sfx_image()

    result = build_sfx_glyph_mask(np.zeros((0, 0, 3), dtype=np.uint8), layer)

    assert result.mask is None
    assert result.evidence["reject_reason"] == "empty_image"


def test_sfx_mask_rejects_border_heavy_mask():
    image, layer = _draw_sfx_image()
    layer["mask"] = np.zeros((80, 180), dtype=np.uint8)
    layer["mask"][0, :] = 255
    layer["mask"][-1, :] = 255
    layer["mask"][:, 0] = 255
    layer["mask"][:, -1] = 255

    result = build_sfx_glyph_mask(image, layer)

    assert result.mask is None
    assert result.evidence["reject_reason"] == "touches_most_crop_border"


def test_sfx_mask_consumes_existing_segmentation_mask_key():
    image, layer = _draw_sfx_image()
    segmentation_mask = np.zeros((80, 180), dtype=np.uint8)
    segmentation_mask[10:26, 20:28] = 255
    segmentation_mask[34:50, 92:100] = 255
    layer["segmentation_mask"] = segmentation_mask

    result = build_sfx_glyph_mask(image, layer)

    assert result.mask is not None
    assert result.evidence["component_count"] >= 2


def test_sfx_mask_uses_color_chroma_when_speed_lines_are_dense():
    image = np.full((140, 220, 3), 235, dtype=np.uint8)
    layer = {
        "bbox": [20, 20, 200, 120],
        "text": "\uc73d",
        "content_class": "sfx",
        "script": "hangul",
        "route_action": "translate_sfx_inpaint_render",
    }
    for x in range(22, 200, 6):
        cv2.line(image, (x, 20), (max(20, x - 80), 120), (24, 24, 24), 1)
    cv2.rectangle(image, (45, 34), (62, 98), (104, 28, 32), -1)
    cv2.rectangle(image, (45, 84), (120, 102), (104, 28, 32), -1)
    cv2.rectangle(image, (145, 34), (164, 102), (104, 28, 32), -1)
    cv2.rectangle(image, (145, 34), (190, 52), (104, 28, 32), -1)

    result = build_sfx_glyph_mask(image, layer)

    assert result.mask is not None
    assert result.evidence["mask_source"] == "color_chroma"
    assert result.evidence["bbox_fill_ratio"] < 0.42
