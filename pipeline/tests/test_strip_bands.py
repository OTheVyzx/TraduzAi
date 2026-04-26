"""Testes de strip/bands.py — agrupamento e cap de bandas."""

import unittest
import numpy as np


class BandSplitTooLargeTests(unittest.TestCase):
    """Bandas acima de max_band_height devem ser quebradas."""

    def test_band_taller_than_max_is_split_into_two(self):
        """2 balões a 6000px de distância com max_band_height=4000 -> 2 bandas."""
        from strip.bands import group_balloons_into_bands
        from strip.types import Balloon, BBox

        balloons = [
            Balloon(strip_bbox=BBox(0, 100, 100, 200), confidence=0.9),
            Balloon(strip_bbox=BBox(0, 6000, 100, 6100), confidence=0.9),
        ]
        # gap entre os 2 baloes = 6000 - 200 = 5800 > gap_threshold=64 -> separados
        bands = group_balloons_into_bands(balloons, gap_threshold=64, margin=16, max_band_height=4000)
        self.assertEqual(len(bands), 2)

    def test_band_within_max_height_not_split(self):
        """Banda abaixo do limite nao deve ser quebrada."""
        from strip.bands import group_balloons_into_bands
        from strip.types import Balloon, BBox

        # 2 baloes proximos -> 1 banda de ~200px (< 4000 -> ok)
        balloons = [
            Balloon(strip_bbox=BBox(50, 100, 200, 200), confidence=0.9),
            Balloon(strip_bbox=BBox(50, 210, 200, 290), confidence=0.8),
        ]
        bands = group_balloons_into_bands(balloons, gap_threshold=64, margin=16, max_band_height=4000)
        self.assertEqual(len(bands), 1)

    def test_close_balloons_in_one_huge_band_get_capped(self):
        """3 baloes com gaps de 100px > gap_threshold=64 -> 3 bandas separadas."""
        from strip.bands import group_balloons_into_bands
        from strip.types import Balloon, BBox

        # 3 baloes com gaps de 100px (> gap_threshold=64) -> 3 bandas
        balloons = [
            Balloon(strip_bbox=BBox(0, 0, 100, 100), confidence=0.9),
            Balloon(strip_bbox=BBox(0, 200, 100, 300), confidence=0.9),   # gap 100 > 64
            Balloon(strip_bbox=BBox(0, 400, 100, 500), confidence=0.9),   # gap 100 > 64
        ]
        bands = group_balloons_into_bands(balloons, gap_threshold=64, margin=16, max_band_height=4000)
        self.assertEqual(len(bands), 3)


class BandGroupingBasicTests(unittest.TestCase):
    def test_empty_balloons_returns_no_bands(self):
        from strip.bands import group_balloons_into_bands
        bands = group_balloons_into_bands([])
        self.assertEqual(bands, [])

    def test_single_balloon_produces_one_band(self):
        from strip.bands import group_balloons_into_bands
        from strip.types import Balloon, BBox

        balloons = [Balloon(strip_bbox=BBox(10, 50, 200, 150), confidence=0.9)]
        bands = group_balloons_into_bands(balloons, margin=16)
        self.assertEqual(len(bands), 1)
        # Band y_top deve ser balloon.y1 - margin
        self.assertEqual(bands[0].y_top, max(0, 50 - 16))
        self.assertEqual(bands[0].y_bottom, 150 + 16)

    def test_close_balloons_grouped_into_one_band(self):
        from strip.bands import group_balloons_into_bands
        from strip.types import Balloon, BBox

        # Gap = 200 - 150 = 50 < gap_threshold=64 -> same band
        b1 = Balloon(strip_bbox=BBox(0, 100, 200, 150), confidence=0.9)
        b2 = Balloon(strip_bbox=BBox(0, 200, 200, 250), confidence=0.9)
        bands = group_balloons_into_bands([b1, b2], gap_threshold=64)
        self.assertEqual(len(bands), 1)
        self.assertEqual(len(bands[0].balloons), 2)

    def test_far_balloons_split_into_separate_bands(self):
        from strip.bands import group_balloons_into_bands
        from strip.types import Balloon, BBox

        # Gap = 500 - 150 = 350 >> gap_threshold=64 -> different bands
        b1 = Balloon(strip_bbox=BBox(0, 100, 200, 150), confidence=0.9)
        b2 = Balloon(strip_bbox=BBox(0, 500, 200, 600), confidence=0.9)
        bands = group_balloons_into_bands([b1, b2], gap_threshold=64)
        self.assertEqual(len(bands), 2)
