"""Testes de bands.py — agrupamento de balões em bandas horizontais."""

import unittest


class GroupBalloonsIntoBandsTests(unittest.TestCase):
    def _balloon(self, y_top, y_bottom, x1=10, x2=200, conf=0.9):
        from strip.types import Balloon, BBox
        return Balloon(strip_bbox=BBox(x1, y_top, x2, y_bottom), confidence=conf)

    def test_close_balloons_merge_into_one_band(self):
        from strip.bands import group_balloons_into_bands
        balloons = [
            self._balloon(100, 200),
            self._balloon(220, 280),  # gap de 20 < threshold(64)
        ]
        bands = group_balloons_into_bands(balloons, gap_threshold=64, margin=16)
        self.assertEqual(len(bands), 1)
        self.assertEqual(len(bands[0].balloons), 2)
        self.assertEqual(bands[0].y_top, 100 - 16)
        self.assertEqual(bands[0].y_bottom, 280 + 16)

    def test_distant_balloons_form_separate_bands(self):
        from strip.bands import group_balloons_into_bands
        balloons = [
            self._balloon(100, 200),
            self._balloon(500, 600),  # gap de 300 >> threshold
        ]
        bands = group_balloons_into_bands(balloons, gap_threshold=64, margin=16)
        self.assertEqual(len(bands), 2)

    def test_unsorted_balloons_are_sorted_by_y(self):
        from strip.bands import group_balloons_into_bands
        balloons = [
            self._balloon(500, 600),
            self._balloon(100, 200),
            self._balloon(300, 400),
        ]
        bands = group_balloons_into_bands(balloons, gap_threshold=50, margin=16)
        self.assertEqual(len(bands), 3)
        self.assertLess(bands[0].y_top, bands[1].y_top)
        self.assertLess(bands[1].y_top, bands[2].y_top)

    def test_overlapping_balloons_merge_into_band(self):
        from strip.bands import group_balloons_into_bands
        balloons = [
            self._balloon(100, 250, x1=10, x2=200),
            self._balloon(200, 300, x1=300, x2=500),
        ]
        bands = group_balloons_into_bands(balloons, gap_threshold=64, margin=16)
        self.assertEqual(len(bands), 1)
        self.assertEqual(len(bands[0].balloons), 2)


class BandSliceFullWidthTests(unittest.TestCase):
    def test_attach_band_slices_uses_full_width(self):
        from strip.bands import attach_band_slices
        from strip.types import VerticalStrip, Band, Balloon, BBox
        import numpy as np

        strip = VerticalStrip(
            image=np.random.randint(0, 256, (500, 300, 3), dtype=np.uint8),
            width=300, height=500, source_page_breaks=[0, 500],
        )
        bands = [
            Band(y_top=100, y_bottom=200, balloons=[
                Balloon(strip_bbox=BBox(50, 110, 150, 190), confidence=0.9),
            ]),
        ]
        attach_band_slices(strip, bands)
        self.assertEqual(bands[0].strip_slice.shape, (100, 300, 3))
        self.assertEqual(bands[0].original_slice.shape, (100, 300, 3))
        bands[0].strip_slice[0, 0, 0] = 99
        self.assertNotEqual(bands[0].original_slice[0, 0, 0], 99)
