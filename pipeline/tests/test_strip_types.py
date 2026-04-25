"""Smoke test: garante que os tipos do strip importam sem erro."""

import unittest


class StripTypesImportTests(unittest.TestCase):
    def test_types_module_imports_without_error(self):
        from strip.types import BBox, Page, VerticalStrip, Balloon, Band, OutputPage
        # Smoke: só importar é suficiente. Validação real virá nas tasks seguintes.
        self.assertTrue(BBox is not None)
        self.assertTrue(VerticalStrip is not None)

    def test_bbox_width_height_handle_inverted_coords(self):
        from strip.types import BBox
        bbox = BBox(x1=10, y1=20, x2=5, y2=15)  # invertido
        self.assertEqual(bbox.width, 0)
        self.assertEqual(bbox.height, 0)
