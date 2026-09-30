import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import cv2

from strip.detect_balloons import _nms_balloons, detect_strip_balloons
from strip.run import _source_page_bounds, _source_page_number_for_band
from strip.types import BBox, Balloon, Band, VerticalStrip


class _Block:
    x1, y1, x2, y2 = 1, 2, 8, 9
    confidence = 0.9


class _Detector:
    def detect(self, _pixels, conf_threshold=None):
        del conf_threshold
        return [_Block()]


class PageNativeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.surface = VerticalStrip(
            image=np.full((20, 10, 3), 255, dtype=np.uint8),
            width=10,
            height=20,
            source_page_breaks=[0, 20],
            page_x_offsets=[0],
            source_page_widths=[10],
            page_number_offset=7,
        )

    def test_local_surface_resolves_chapter_global_page_number_and_bounds(self):
        band = Band(0, 20)
        self.assertEqual(_source_page_number_for_band(self.surface, band), 8)
        self.assertEqual(_source_page_bounds(self.surface, 8), (0, 20))

    def test_detector_assigns_chapter_global_page_identity(self):
        balloons = detect_strip_balloons(
            self.surface,
            _Detector(),
            confidence_threshold=0.1,
            max_height_fraction=1.0,
            max_width_fraction=1.0,
        )
        self.assertEqual(len(balloons), 1)
        self.assertEqual(balloons[0].metadata["page_id"], "page_008")
        self.assertTrue(balloons[0].metadata["region_id"].startswith("region_p008_"))

    def test_page_native_detector_dedupe_requires_class_and_accepts_containment(self):
        candidates = [
            Balloon(BBox(0, 0, 100, 100), 0.9, metadata={"candidate_kind": "text"}),
            Balloon(BBox(10, 10, 30, 30), 0.8, metadata={"candidate_kind": "text"}),
            Balloon(BBox(10, 10, 30, 30), 0.7, metadata={"candidate_kind": "sfx"}),
        ]
        kept = _nms_balloons(
            candidates,
            iou_threshold=0.5,
            intersection_smaller_threshold=0.8,
            require_same_class=True,
        )
        self.assertEqual(len(kept), 2)
        self.assertEqual(
            {item.metadata["candidate_kind"] for item in kept}, {"text", "sfx"}
        )

    def test_page_native_detector_windows_cover_top_overlap_and_short_tail(self):
        heights = []

        class RecordingDetector:
            def detect(self, pixels, conf_threshold=None):
                del conf_threshold
                heights.append(int(pixels.shape[0]))
                return []

        surface = VerticalStrip(
            image=np.full((9000, 16, 3), 255, dtype=np.uint8),
            width=16,
            height=9000,
            source_page_breaks=[0, 9000],
            page_x_offsets=[0],
            source_page_widths=[16],
            raster_mode="page_map_v1",
        )
        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_NEGATIVE_DETECT_MERGE": "0",
                "TRADUZAI_STRIP_WHITE_BALLOON_BAND_SCAN": "0",
                "TRADUZAI_STRIP_DARK_BALLOON_BAND_SCAN": "0",
                "TRADUZAI_STRIP_UI_LAYOUT_BAND_SCAN": "0",
            },
        ):
            detect_strip_balloons(surface, RecordingDetector())
        self.assertEqual(heights, [4096, 4096, 1832])
        self.assertTrue(all(height <= 4096 for height in heights))

    def test_enforce_multi_page_dispatches_before_physical_strip_builder(self):
        from strip import run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index in range(2):
                path = root / f"{index + 1:03d}.png"
                cv2.imwrite(str(path), np.full((12 + index, 10, 3), 30, dtype=np.uint8))
                paths.append(path)
            expected = [object(), object()]
            with patch.object(
                run, "_run_page_native_chapter", return_value=expected, create=True
            ) as page_native, patch.object(
                run, "build_strip", side_effect=AssertionError("physical strip called")
            ):
                actual = run.run_chapter(
                    paths,
                    root / "out",
                    detector=MagicMock(),
                    runtime=MagicMock(),
                    translator=MagicMock(),
                    inpainter=MagicMock(),
                    typesetter=MagicMock(),
                    owner_graph_mode="enforce",
                    run_id="run-page-map",
                    execution_id="execution-page-map",
                )

            self.assertIs(actual, expected)
            page_native.assert_called_once()


if __name__ == "__main__":
    unittest.main()
