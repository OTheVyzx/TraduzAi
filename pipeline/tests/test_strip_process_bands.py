"""Testes de process_bands.py."""

import unittest


class BandToPageDictTests(unittest.TestCase):
    def test_band_to_page_dict_remaps_balloon_coords_to_local(self):
        from strip.process_bands import _band_to_page_dict
        from strip.types import Band, Balloon, BBox
        import numpy as np

        slice_img = np.zeros((100, 300, 3), dtype=np.uint8)
        band = Band(
            y_top=500,
            y_bottom=600,
            balloons=[
                Balloon(strip_bbox=BBox(50, 510, 150, 590), confidence=0.9),
            ],
            strip_slice=slice_img,
            original_slice=slice_img.copy(),
        )

        page_dict = _band_to_page_dict(band, page_idx=0)

        self.assertEqual(page_dict["width"], 300)
        self.assertEqual(page_dict["height"], 100)
        block = page_dict["_vision_blocks"][0]
        self.assertEqual(block["bbox"], [50, 10, 150, 90])
