import unittest

from inpainter.mask_builder import build_mask_regions, build_region_pixel_mask


class MaskBuilderTests(unittest.TestCase):
    def test_merges_nearby_text_boxes_from_same_balloon(self):
        texts = [
            {"bbox": [100, 100, 180, 140], "tipo": "fala", "confidence": 0.91},
            {"bbox": [110, 148, 176, 186], "tipo": "fala", "confidence": 0.88},
        ]

        regions = build_mask_regions(texts=texts, image_shape=(400, 300, 3))

        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertLessEqual(region["bbox"][0], 96)
        self.assertLessEqual(region["bbox"][1], 96)
        self.assertGreaterEqual(region["bbox"][2], 180)
        self.assertGreaterEqual(region["bbox"][3], 186)
        self.assertEqual(region["kind"], "cluster")

    def test_keeps_far_apart_regions_separate(self):
        texts = [
            {"bbox": [10, 20, 60, 55], "tipo": "fala", "confidence": 0.82},
            {"bbox": [220, 260, 280, 320], "tipo": "fala", "confidence": 0.79},
        ]

        regions = build_mask_regions(texts=texts, image_shape=(400, 300, 3))

        self.assertEqual(len(regions), 2)

    def test_pixel_mask_is_smaller_than_full_union_box(self):
        texts = [
            {"bbox": [100, 100, 180, 140], "tipo": "fala", "confidence": 0.91},
            {"bbox": [110, 150, 176, 186], "tipo": "fala", "confidence": 0.88},
        ]

        regions = build_mask_regions(texts=texts, image_shape=(400, 300, 3))
        region = regions[0]
        pixel_mask = build_region_pixel_mask((400, 300), region)

        x1, y1, x2, y2 = region["bbox"]
        full_union_area = (x2 - x1) * (y2 - y1)
        masked_pixels = int(pixel_mask.sum() // 255)

        self.assertGreater(masked_pixels, 0)
        self.assertLess(masked_pixels, full_union_area)


if __name__ == "__main__":
    unittest.main()
