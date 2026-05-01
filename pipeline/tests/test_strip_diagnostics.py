"""Testes do helper de diagnóstico do strip."""

import tempfile
import unittest
from pathlib import Path

import numpy as np


class DumpStripDebugTests(unittest.TestCase):
    def test_dump_writes_strip_overlay_and_bands_txt(self):
        from strip._diagnostics import dump_strip_debug
        from strip.types import VerticalStrip, Band, Balloon, BBox

        strip = VerticalStrip(
            image=np.full((400, 300, 3), 200, dtype=np.uint8),
            width=300, height=400, source_page_breaks=[0, 200, 400],
        )
        bands = [
            Band(y_top=10, y_bottom=90, balloons=[
                Balloon(strip_bbox=BBox(50, 20, 150, 80), confidence=0.9),
            ]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            dump_strip_debug(strip, bands, out)
            self.assertTrue((out / "strip_overlay.png").exists())
            self.assertTrue((out / "bands_overview.txt").exists())

    def test_bands_overview_contains_band_info(self):
        from strip._diagnostics import dump_strip_debug
        from strip.types import VerticalStrip, Band, Balloon, BBox

        strip = VerticalStrip(
            image=np.full((200, 100, 3), 128, dtype=np.uint8),
            width=100, height=200, source_page_breaks=[0, 200],
        )
        bands = [
            Band(y_top=0, y_bottom=80, balloons=[
                Balloon(strip_bbox=BBox(10, 5, 90, 75), confidence=0.85),
            ]),
            Band(y_top=100, y_bottom=190, balloons=[]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            dump_strip_debug(strip, bands, out)
            text = (out / "bands_overview.txt").read_text(encoding="utf-8")
            self.assertIn("band[000]", text)
            self.assertIn("band[001]", text)
            self.assertIn("balloons=1", text)
            self.assertIn("balloons=0", text)

    def test_is_debug_enabled_reads_env(self):
        import os
        from strip._diagnostics import is_debug_enabled

        original = os.environ.pop("STRIP_DEBUG", None)
        try:
            self.assertFalse(is_debug_enabled())
            os.environ["STRIP_DEBUG"] = "1"
            self.assertTrue(is_debug_enabled())
            os.environ["STRIP_DEBUG"] = "true"
            self.assertTrue(is_debug_enabled())
            os.environ["STRIP_DEBUG"] = "0"
            self.assertFalse(is_debug_enabled())
        finally:
            if original is None:
                os.environ.pop("STRIP_DEBUG", None)
            else:
                os.environ["STRIP_DEBUG"] = original
