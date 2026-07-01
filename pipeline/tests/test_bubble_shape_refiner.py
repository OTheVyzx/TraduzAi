import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask > 0))


def test_refiner_removes_connected_block_from_oval_body():
    from vision_stack.bubble_shape_refiner import refine_bubble_shape_mask

    mask = np.zeros((160, 240), dtype=np.uint8)
    cv2.ellipse(mask, (130, 98), (66, 38), 0, 0, 360, 255, -1)
    mask[52:84, 70:126] = 255

    result = refine_bubble_shape_mask(mask)

    assert result.accepted is True
    assert result.shape_kind == "oval"
    assert result.mask[60, 82] == 0
    assert result.mask[98, 130] > 0
    assert result.removed_pixels > 200


def test_refiner_preserves_pointed_balloon_extensions_and_tail():
    from vision_stack.bubble_shape_refiner import refine_bubble_shape_mask

    mask = np.zeros((140, 260), dtype=np.uint8)
    cv2.ellipse(mask, (120, 58), (72, 30), 0, 0, 360, 255, -1)
    points = np.array(
        [
            [[43, 52], [4, 35], [50, 65]],
            [[196, 52], [248, 38], [202, 66]],
            [[106, 85], [126, 132], [136, 83]],
        ],
        dtype=np.int32,
    )
    for poly in points:
        cv2.fillPoly(mask, [poly], 255)

    result = refine_bubble_shape_mask(mask)

    assert result.accepted is True
    assert result.shape_kind == "irregular"
    assert result.mask[45, 20] > 0
    assert result.mask[49, 230] > 0
    assert result.mask[118, 123] > 0
    assert result.removed_pixels < 120


def test_refiner_removes_rectangular_side_protrusion_without_rounding_body():
    from vision_stack.bubble_shape_refiner import refine_bubble_shape_mask

    mask = np.zeros((140, 260), dtype=np.uint8)
    mask[42:104, 76:210] = 255
    mask[70:96, 52:80] = 255

    result = refine_bubble_shape_mask(mask)

    assert result.accepted is True
    assert result.shape_kind == "rectangle"
    assert result.mask[78, 58] == 0
    assert result.mask[78, 120] > 0
    ys, xs = np.where(result.mask > 0)
    assert int(xs.min()) >= 74
    assert int(xs.max()) <= 211


def test_refiner_removes_irregular_white_attachment_from_rectangular_card():
    from vision_stack.bubble_shape_refiner import refine_bubble_shape_mask

    mask = np.zeros((180, 300), dtype=np.uint8)
    mask[72:132, 128:246] = 255
    mask[58:86, 108:142] = 255
    mask[88:112, 104:130] = 255

    result = refine_bubble_shape_mask(mask)

    assert result.accepted is True
    assert result.shape_kind == "rectangle"
    assert result.mask[66, 116] == 0
    assert result.mask[94, 112] == 0
    assert result.mask[96, 180] > 0


def test_refiner_keeps_simple_oval_close_to_input():
    from vision_stack.bubble_shape_refiner import refine_bubble_shape_mask

    mask = np.zeros((120, 220), dtype=np.uint8)
    cv2.ellipse(mask, (110, 60), (64, 34), 0, 0, 360, 255, -1)

    result = refine_bubble_shape_mask(mask)
    delta = cv2.absdiff(mask, result.mask)

    assert result.accepted is True
    assert result.shape_kind == "oval"
    assert _count(delta) < int(_count(mask) * 0.12)


def test_refiner_models_edge_clipped_oval_body_instead_of_preserving_flat_crop():
    from vision_stack.bubble_shape_refiner import refine_bubble_shape_mask

    mask = np.zeros((120, 240), dtype=np.uint8)
    cv2.ellipse(mask, (120, 120), (80, 50), 0, 0, 360, 255, -1)

    result = refine_bubble_shape_mask(mask)

    assert result.accepted is True
    assert result.shape_kind == "oval"
    assert result.added_pixels > 0
    assert result.mask[118, 120] > 0


def test_refiner_models_edge_clipped_oval_body_without_tail_or_blob():
    from vision_stack.bubble_shape_refiner import refine_bubble_shape_mask

    mask = np.zeros((240, 360), dtype=np.uint8)
    cv2.ellipse(mask, (150, 92), (112, 64), 0, 0, 360, 255, -1)
    cv2.fillPoly(mask, [np.array([[[122, 146], [146, 198], [160, 142]]], dtype=np.int32)], 255)
    mask[132:176, 210:230] = 255
    mask[176:240, 220:246] = 255

    result = refine_bubble_shape_mask(mask)

    assert result.accepted is True
    assert result.shape_kind == "oval"
    assert result.mask[182, 146] == 0
    assert result.mask[210, 232] == 0
    assert result.mask[92, 150] > 0


def test_refiner_preserves_edge_clipped_pointed_balloon():
    from vision_stack.bubble_shape_refiner import refine_bubble_shape_mask

    mask = np.zeros((150, 320), dtype=np.uint8)
    mask[0:72, 46:236] = 255
    cv2.circle(mask, (46, 36), 34, 255, -1)
    cv2.circle(mask, (236, 36), 34, 255, -1)
    cv2.fillPoly(mask, [np.array([[[96, 68], [132, 128], [152, 66]]], dtype=np.int32)], 255)
    cv2.fillPoly(mask, [np.array([[[44, 10], [2, 0], [42, 38]]], dtype=np.int32)], 255)
    cv2.fillPoly(mask, [np.array([[[236, 10], [306, 0], [238, 38]]], dtype=np.int32)], 255)

    result = refine_bubble_shape_mask(mask)

    assert result.accepted is True
    assert result.shape_kind == "irregular"
    assert result.mask[112, 132] > 0
    assert result.mask[20, 18] > 0
    assert result.removed_pixels == 0


def test_refiner_refines_disconnected_components_independently():
    from vision_stack.bubble_shape_refiner import refine_bubble_shape_mask

    mask = np.zeros((260, 360), dtype=np.uint8)
    mask[0:62, 34:180] = 255
    cv2.circle(mask, (34, 31), 28, 255, -1)
    cv2.circle(mask, (180, 31), 28, 255, -1)
    cv2.fillPoly(mask, [np.array([[[74, 58], [104, 118], [122, 57]]], dtype=np.int32)], 255)
    cv2.ellipse(mask, (265, 190), (58, 32), 0, 0, 360, 255, -1)
    mask[154:178, 210:252] = 255

    result = refine_bubble_shape_mask(mask)

    assert result.accepted is True
    assert result.shape_kind == "mixed"
    assert result.mask[98, 104] > 0
    assert result.mask[162, 218] == 0
    assert result.mask[190, 265] > 0
