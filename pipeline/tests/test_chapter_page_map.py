import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from strip.page_map import ChapterPageMap


class ChapterPageMapTests(unittest.TestCase):
    def test_map_preserves_order_geometry_hashes_and_has_no_public_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for name, shape, value in (
                ("002.png", (7, 4, 3), 20),
                ("010.png", (3, 8, 3), 90),
            ):
                path = root / name
                cv2.imwrite(str(path), np.full(shape, value, dtype=np.uint8))
                paths.append(path)

            page_map = ChapterPageMap.from_paths(paths)
            public = page_map.to_public_dict()

            self.assertEqual([item.page_id for item in page_map.pages], ["page_001", "page_002"])
            self.assertEqual([item.virtual_y_offset for item in page_map.pages], [0, 7])
            self.assertEqual(page_map.frame_width, 8)
            self.assertEqual(page_map.virtual_height, 10)
            self.assertEqual(public["chapter_raster_mode"], "page_map_v1")
            self.assertNotIn(str(root), str(public))

            first = page_map.pages[0].load_as_local_strip()
            self.assertEqual(first.image.shape, (7, 8, 3))
            self.assertEqual(first.page_number_offset, 0)
            self.assertEqual(first.page_x_offsets, [2])
            self.assertTrue(np.all(first.image[:, :2] == 255))
            second = page_map.pages[1].load_as_local_strip()
            self.assertEqual(second.page_number_offset, 1)

    def test_load_rejects_pixels_changed_after_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            cv2.imwrite(str(path), np.zeros((4, 5, 3), dtype=np.uint8))
            page_map = ChapterPageMap.from_paths([path])
            cv2.imwrite(str(path), np.full((4, 5, 3), 255, dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "pixels changed"):
                page_map.pages[0].load_framed_rgb()


if __name__ == "__main__":
    unittest.main()
