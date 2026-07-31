from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.style_masks import (
    build_mask_backed_typographic_layers,
    build_typographic_mask_layers,
    write_typographic_mask_debug,
)


def _outlined_text() -> np.ndarray:
    image = np.full((120, 300, 3), (238, 238, 238), dtype=np.uint8)
    cv2.putText(image, "TEST", (34, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (25, 25, 25), 7, cv2.LINE_AA)
    cv2.putText(image, "TEST", (34, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (250, 250, 250), 3, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def test_typographic_masks_separate_foreground_core_ring_and_background():
    layers = build_typographic_mask_layers(_outlined_text())

    assert layers["foreground_mask"].dtype == np.uint8
    assert int(np.count_nonzero(layers["foreground_mask"])) > 300
    assert int(np.count_nonzero(layers["core_mask"])) > 30
    assert int(np.count_nonzero(layers["stroke_ring_mask"])) > 30
    assert not np.any((layers["core_mask"] > 0) & (layers["background_mask"] > 0))
    assert layers["metrics"]["normalized_stroke_width"] > 0.0


def test_background_mask_excludes_text_even_on_colored_art_like_background():
    image = np.full((100, 240, 3), (60, 100, 150), dtype=np.uint8)
    cv2.circle(image, (185, 50), 32, (80, 160, 210), -1)
    cv2.putText(image, "SFX", (22, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 230, 255), 3, cv2.LINE_AA)

    layers = build_typographic_mask_layers(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    assert int(np.count_nonzero(layers["foreground_mask"])) > 100
    assert layers["metrics"]["background_rgb"] != [20, 230, 255]
    assert int(np.count_nonzero(layers["background_mask"])) > int(np.count_nonzero(layers["foreground_mask"]))


def test_mask_debug_writes_only_explicit_debug_artifacts(tmp_path: Path):
    root = tmp_path / "debug"
    output = write_typographic_mask_debug(root, "style-case", build_typographic_mask_layers(_outlined_text()))

    assert output["metrics"].is_file()
    assert output["foreground_mask"].is_file()
    assert output["core_mask"].is_file()
    assert output["stroke_ring_mask"].is_file()
    assert output["background_mask"].is_file()


def test_authoritative_glyph_mask_excludes_colored_card_art():
    image = np.full((100, 220, 3), (60, 150, 210), dtype=np.uint8)
    image[25:75, 150:205] = (220, 80, 45)
    glyph = np.zeros(image.shape[:2], dtype=np.uint8)
    glyph[38:62, 25:125] = 255
    image[glyph > 0] = (248, 248, 248)
    context = np.full(image.shape[:2], 255, dtype=np.uint8)

    layers = build_mask_backed_typographic_layers(image, glyph, context)

    assert np.array_equal(layers["core_mask"], glyph)
    assert not np.any(layers["core_mask"][:, 150:205])
    assert layers["metrics"]["normalization_unit"] == "source_x_height"
