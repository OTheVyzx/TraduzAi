import unittest

import numpy as np
from PIL import Image

from inpainter.classical import (
    apply_fill,
    build_corpus_inpainting_profile,
    classify_background,
    clean_image,
)


class InpaintingProfileTests(unittest.TestCase):
    def test_corpus_profile_tightens_naturality_on_dark_mid_pages(self):
        profile = build_corpus_inpainting_profile(
            {
                "page_geometry": {"median_width": 800},
                "luminance_profile": {"light_pages": 10, "mid_pages": 460, "dark_pages": 30},
            }
        )

        self.assertGreaterEqual(profile["ring_width"], 16)
        self.assertLess(profile["naturality_threshold"], 20.0)
        self.assertGreaterEqual(profile["feather_radius"], 2)

    def test_classify_background_detects_vertical_textured_balloon(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        for x in range(120):
            color = 80 + (x % 6) * 18
            image[:, x] = [140 + color // 4, 20 + color // 8, 30 + color // 8]

        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[20:60, 35:85] = 255

        bg_type, _ = classify_background(image, [35, 20, 85, 60], mask)

        self.assertEqual(bg_type, "textured_vertical")

    def test_directional_fill_preserves_column_variation(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        for x in range(120):
            color = 100 + (x % 8) * 16
            image[:, x] = [150 + color // 4, 18 + color // 10, 32 + color // 10]

        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[20:60, 35:85] = 255

        filled = apply_fill(image, mask, [35, 20, 85, 60], "textured_vertical", {})
        center_band = filled[30:50, 40:80]
        column_means = center_band.mean(axis=(0, 2))

        self.assertGreater(np.std(column_means), 2.0)

    def test_clean_image_overlays_text_inside_white_balloon_without_erasing_outline(self):
        image = np.full((180, 220, 3), 235, dtype=np.uint8)
        cy, cx = 70, 110
        ry, rx = 38, 75

        yy, xx = np.ogrid[:180, :220]
        ellipse = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
        inner = (((yy - cy) / (ry - 3)) ** 2 + ((xx - cx) / (rx - 3)) ** 2) <= 1.0

        image[ellipse] = [245, 245, 245]
        outline = ellipse & ~inner
        image[outline] = [20, 20, 20]

        image[52:62, 82:142] = [15, 15, 15]
        image[70:80, 76:150] = [15, 15, 15]
        image[88:98, 90:135] = [15, 15, 15]

        texts = [
            {"bbox": [82, 52, 142, 62], "tipo": "fala", "confidence": 0.9},
            {"bbox": [76, 70, 150, 80], "tipo": "fala", "confidence": 0.9},
            {"bbox": [90, 88, 135, 98], "tipo": "fala", "confidence": 0.9},
        ]

        cleaned = np.array(clean_image(Image.fromarray(image), texts))

        self.assertGreaterEqual(float(cleaned[57:60, 95:130].mean()), 235.0)
        self.assertLess(float(cleaned[70, 36].mean()), 60.0)

    def test_clean_image_overlays_textured_balloon_with_soft_edge(self):
        image = np.full((180, 220, 3), 230, dtype=np.uint8)
        image[40:125, 55:175] = [130, 24, 38]
        for x in range(55, 175):
            image[40:125, x, 0] = np.clip(120 + (x - 55) // 3, 0, 255)
            image[40:125, x, 1] = np.clip(20 + (x - 55) // 10, 0, 255)
            image[40:125, x, 2] = np.clip(35 + (x - 55) // 9, 0, 255)

        image[65:76, 78:152] = [245, 245, 245]
        image[82:94, 70:160] = [245, 245, 245]

        texts = [
            {"bbox": [78, 65, 152, 76], "tipo": "fala", "confidence": 0.9},
            {"bbox": [70, 82, 160, 94], "tipo": "fala", "confidence": 0.9},
        ]

        cleaned = np.array(clean_image(Image.fromarray(image), texts))

        self.assertLess(float(cleaned[70:90, 85:145, 0].std()), 10.0)
        self.assertLess(float(np.abs(cleaned[64, 79].astype(float) - cleaned[63, 79].astype(float)).mean()), 60.0)

    def test_clean_image_textured_balloon_uses_cardinal_gradient_without_bleeding_outside(self):
        image = np.full((220, 260, 3), 232, dtype=np.uint8)
        cy, cx = 110, 130
        ry, rx = 52, 82

        yy, xx = np.ogrid[:220, :260]
        ellipse = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
        image[ellipse] = [90, 15, 24]

        top_weight = np.clip((yy - (cy - ry)) / (2 * ry), 0, 1)
        left_weight = np.clip((xx - (cx - rx)) / (2 * rx), 0, 1)
        image[..., 0] = np.where(ellipse, 88 + top_weight * 70 + left_weight * 30, image[..., 0])
        image[..., 1] = np.where(ellipse, 10 + top_weight * 18 + left_weight * 10, image[..., 1])
        image[..., 2] = np.where(ellipse, 20 + top_weight * 22 + left_weight * 12, image[..., 2])

        image[92:104, 94:166] = [245, 245, 245]
        image[111:124, 84:176] = [245, 245, 245]

        texts = [
            {"bbox": [94, 92, 166, 104], "tipo": "fala", "confidence": 0.9},
            {"bbox": [84, 111, 176, 124], "tipo": "fala", "confidence": 0.9},
        ]

        cleaned = np.array(clean_image(Image.fromarray(image), texts))

        top_overlay = cleaned[97:101, 108:152].mean(axis=(0, 1))
        bottom_overlay = cleaned[116:120, 104:156].mean(axis=(0, 1))
        outside_balloon = cleaned[110, 34].astype(float)

        self.assertGreater(float(bottom_overlay[0] - top_overlay[0]), 4.0)
        self.assertGreater(float(bottom_overlay[1] - top_overlay[1]), 1.0)
        self.assertLess(float(np.abs(outside_balloon - np.array([232, 232, 232])).mean()), 3.0)


if __name__ == "__main__":
    unittest.main()
