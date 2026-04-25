"""Testes de concat.py — concatenação vertical de páginas em strip."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


class BuildStripTests(unittest.TestCase):
    def _write_png(self, dir_path: Path, name: str, image: np.ndarray) -> Path:
        p = dir_path / name
        cv2.imwrite(str(p), image)
        return p

    def test_build_strip_concatenates_without_separator_pixels(self):
        from strip.concat import build_strip
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            page_a = np.full((100, 200, 3), 50, dtype=np.uint8)
            page_b = np.full((80, 200, 3), 200, dtype=np.uint8)
            page_a[-1, :] = 50
            page_b[0, :] = 200
            paths = [
                self._write_png(tmp_path, "a.png", page_a),
                self._write_png(tmp_path, "b.png", page_b),
            ]

            strip = build_strip(paths)

            self.assertEqual(strip.image.shape[0], 180)
            self.assertTrue(np.all(strip.image[99, :] == 50))
            self.assertTrue(np.all(strip.image[100, :] == 200))

    def test_build_strip_letterboxes_narrow_pages_with_white(self):
        from strip.concat import build_strip
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wide = np.full((50, 300, 3), 0, dtype=np.uint8)
            narrow = np.full((50, 200, 3), 0, dtype=np.uint8)
            paths = [
                self._write_png(tmp_path, "wide.png", wide),
                self._write_png(tmp_path, "narrow.png", narrow),
            ]

            strip = build_strip(paths)

            self.assertEqual(strip.width, 300)
            # Narrow page is centered: 50px letterbox on each side
            self.assertTrue(np.all(strip.image[60, 0:49] == 255))
            self.assertTrue(np.all(strip.image[60, 251:300] == 255))
            self.assertTrue(np.all(strip.image[60, 50:250] == 0))

    def test_build_strip_records_page_breaks(self):
        from strip.concat import build_strip
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = [
                self._write_png(tmp_path, f"p{i}.png", np.zeros((100, 200, 3), dtype=np.uint8))
                for i in range(3)
            ]

            strip = build_strip(paths)

            self.assertEqual(strip.source_page_breaks, [0, 100, 200, 300])
