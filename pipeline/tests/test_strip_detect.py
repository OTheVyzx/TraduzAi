"""Testes de detect_balloons.py — detecção de balões sobre o strip."""

import unittest


class IouTests(unittest.TestCase):
    def test_iou_identical_boxes_returns_one(self):
        from strip.detect_balloons import _iou
        from strip.types import BBox
        b = BBox(0, 0, 100, 100)
        self.assertAlmostEqual(_iou(b, b), 1.0)

    def test_iou_disjoint_boxes_returns_zero(self):
        from strip.detect_balloons import _iou
        from strip.types import BBox
        a = BBox(0, 0, 50, 50)
        b = BBox(100, 100, 150, 150)
        self.assertAlmostEqual(_iou(a, b), 0.0)

    def test_iou_partial_overlap(self):
        from strip.detect_balloons import _iou
        from strip.types import BBox
        a = BBox(0, 0, 100, 100)
        b = BBox(50, 50, 150, 150)
        # intersection: 50x50=2500, union=10000+10000-2500=17500
        self.assertAlmostEqual(_iou(a, b), 2500 / 17500, places=4)


class NmsBalloonsTests(unittest.TestCase):
    def test_nms_removes_duplicate_high_iou(self):
        from strip.detect_balloons import _nms_balloons
        from strip.types import Balloon, BBox
        balloons = [
            Balloon(strip_bbox=BBox(0, 0, 100, 100), confidence=0.9),
            Balloon(strip_bbox=BBox(2, 2, 102, 102), confidence=0.7),
        ]
        kept = _nms_balloons(balloons, iou_threshold=0.5)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0].confidence, 0.9)

    def test_nms_keeps_distant_balloons(self):
        from strip.detect_balloons import _nms_balloons
        from strip.types import Balloon, BBox
        balloons = [
            Balloon(strip_bbox=BBox(0, 0, 50, 50), confidence=0.9),
            Balloon(strip_bbox=BBox(200, 200, 250, 250), confidence=0.8),
        ]
        kept = _nms_balloons(balloons, iou_threshold=0.5)
        self.assertEqual(len(kept), 2)


class SplitIntoChunksTests(unittest.TestCase):
    def test_short_strip_returns_single_chunk(self):
        from strip.detect_balloons import _split_into_chunks
        chunks = _split_into_chunks(strip_height=2000, chunk_height=4096, overlap=512)
        self.assertEqual(chunks, [(0, 2000)])

    def test_long_strip_returns_overlapping_chunks(self):
        from strip.detect_balloons import _split_into_chunks
        chunks = _split_into_chunks(strip_height=10000, chunk_height=4096, overlap=512)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 10000)
        for i in range(1, len(chunks)):
            self.assertLess(chunks[i][0], chunks[i - 1][1])

    def test_chunks_cover_entire_strip(self):
        from strip.detect_balloons import _split_into_chunks
        chunks = _split_into_chunks(strip_height=15000, chunk_height=4096, overlap=512)
        covered = [False] * 15000
        for y0, y1 in chunks:
            for y in range(y0, y1):
                covered[y] = True
        self.assertTrue(all(covered))

    def test_full_page_detect_chunks_follow_source_page_breaks(self):
        from strip.detect_balloons import _detect_chunks_for_strip
        from strip.types import VerticalStrip
        import numpy as np
        from unittest.mock import patch

        strip = VerticalStrip(
            image=np.zeros((9000, 720, 3), dtype=np.uint8),
            width=720,
            height=9000,
            source_page_breaks=[0, 3100, 6200, 9000],
        )

        with patch.dict("os.environ", {"TRADUZAI_STRIP_DETECT_FULL_PAGE": "1"}):
            chunks, mode = _detect_chunks_for_strip(strip, chunk_height=4096, overlap=512)

        self.assertEqual(mode, "source_page")
        self.assertEqual(chunks, [(0, 3100), (3100, 6200), (6200, 9000)])


class DetectStripBalloonsTests(unittest.TestCase):
    def _make_negative_detector_for(self, bbox, confidence=0.91):
        from unittest.mock import MagicMock

        block = type("FakeBlock", (), {})()
        block.x1, block.y1, block.x2, block.y2 = bbox
        block.confidence = confidence
        detector = MagicMock()
        detector.detect.side_effect = [[], [block]]
        return detector

    def _assert_dark_candidate_contract(self, candidate):
        metadata = getattr(candidate, "metadata", None)
        if metadata is None:
            metadata = candidate if isinstance(candidate, dict) else {}

        self.assertNotEqual(metadata.get("balloon_type"), "white")
        self.assertEqual(metadata.get("background_polarity"), "dark")
        self.assertTrue(metadata.get("dark_light_text_evidence", {}).get("useful"))

    def test_dark_bubble_negative_geometry_reports_dark_profile(self):
        from unittest.mock import patch
        from strip.detect_balloons import detect_strip_balloons
        from strip.types import VerticalStrip
        import cv2
        import numpy as np

        image = np.zeros((260, 360, 3), dtype=np.uint8)
        cv2.ellipse(image, (180, 130), (112, 58), 0, 0, 360, (4, 8, 12), -1)
        cv2.ellipse(image, (180, 130), (112, 58), 0, 0, 360, (40, 145, 190), 4)
        cv2.putText(image, "WHITE", (116, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "WORDS", (112, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 248, 255), 2, cv2.LINE_AA)

        strip = VerticalStrip(image=image, width=360, height=260, source_page_breaks=[0, 260])
        detector = self._make_negative_detector_for((96, 92, 264, 166))

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_NEGATIVE_DETECT_MERGE": "1",
                "TRADUZAI_STRIP_WHITE_BALLOON_BAND_SCAN": "0",
                "TRADUZAI_STRIP_UI_LAYOUT_BAND_SCAN": "0",
            },
        ):
            candidates = detect_strip_balloons(strip, detector=detector)

        self.assertEqual(len(candidates), 1)
        self._assert_dark_candidate_contract(candidates[0])

    def test_dark_panel_negative_geometry_reports_dark_profile(self):
        from unittest.mock import patch
        from strip.detect_balloons import detect_strip_balloons
        from strip.types import VerticalStrip
        import cv2
        import numpy as np

        image = np.full((220, 420, 3), 235, dtype=np.uint8)
        cv2.rectangle(image, (72, 62), (348, 154), (8, 8, 10), -1)
        cv2.rectangle(image, (72, 62), (348, 154), (95, 190, 220), 3)
        cv2.putText(image, "QUEST", (124, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (248, 248, 255), 2, cv2.LINE_AA)

        strip = VerticalStrip(image=image, width=420, height=220, source_page_breaks=[0, 220])
        detector = self._make_negative_detector_for((106, 82, 302, 126))

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_NEGATIVE_DETECT_MERGE": "1",
                "TRADUZAI_STRIP_WHITE_BALLOON_BAND_SCAN": "0",
                "TRADUZAI_STRIP_UI_LAYOUT_BAND_SCAN": "0",
            },
        ):
            candidates = detect_strip_balloons(strip, detector=detector)

        self.assertEqual(len(candidates), 1)
        self._assert_dark_candidate_contract(candidates[0])

    def test_negative_white_text_without_balloon_reports_dark_profile_not_white_balloon(self):
        from unittest.mock import patch
        from strip.detect_balloons import detect_strip_balloons
        from strip.types import VerticalStrip
        import cv2
        import numpy as np

        image = np.full((180, 360, 3), 5, dtype=np.uint8)
        cv2.putText(image, "SYSTEM", (82, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (246, 246, 250), 2, cv2.LINE_AA)
        cv2.putText(image, "ONLINE", (86, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (246, 246, 250), 2, cv2.LINE_AA)

        strip = VerticalStrip(image=image, width=360, height=180, source_page_breaks=[0, 180])
        detector = self._make_negative_detector_for((76, 52, 240, 126))

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_NEGATIVE_DETECT_MERGE": "1",
                "TRADUZAI_STRIP_WHITE_BALLOON_BAND_SCAN": "0",
                "TRADUZAI_STRIP_UI_LAYOUT_BAND_SCAN": "0",
            },
        ):
            candidates = detect_strip_balloons(strip, detector=detector)

        self.assertEqual(len(candidates), 1)
        self._assert_dark_candidate_contract(candidates[0])

    def test_detect_strip_balloons_full_page_env_runs_one_detection_per_source_page(self):
        from unittest.mock import MagicMock, patch
        from strip.detect_balloons import detect_strip_balloons
        from strip.types import VerticalStrip
        import numpy as np

        def make_block(_x1, _y1, _x2, _y2):
            block = type("FakeBlock", (), {})()
            block.x1, block.y1, block.x2, block.y2 = _x1, _y1, _x2, _y2
            block.confidence = 0.9
            return block

        fake_detector = MagicMock()
        fake_detector.detect.side_effect = [
            [make_block(40, 100, 140, 180)],
            [make_block(50, 200, 180, 290)],
            [make_block(60, 300, 200, 390)],
        ]

        strip = VerticalStrip(
            image=np.zeros((9000, 720, 3), dtype=np.uint8),
            width=720,
            height=9000,
            source_page_breaks=[0, 3000, 6000, 9000],
        )

        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_DETECT_FULL_PAGE": "1",
                "TRADUZAI_STRIP_NEGATIVE_DETECT_MERGE": "0",
            },
        ):
            balloons = detect_strip_balloons(strip, detector=fake_detector)

        self.assertEqual(fake_detector.detect.call_count, 3)
        self.assertEqual([balloon.strip_bbox.y1 for balloon in balloons], [100, 3200, 6300])

    def test_detect_strip_balloons_dedupes_overlap_zone(self):
        from unittest.mock import MagicMock
        from strip.detect_balloons import detect_strip_balloons
        from strip.types import VerticalStrip
        import numpy as np

        fake_block = type("FakeBlock", (), {})()
        fake_block.x1, fake_block.y1, fake_block.x2, fake_block.y2 = 100, 50, 200, 150
        fake_block.confidence = 0.9

        fake_detector = MagicMock()
        fake_detector.detect.return_value = [fake_block]

        strip = VerticalStrip(
            image=np.zeros((10000, 800, 3), dtype=np.uint8),
            width=800,
            height=10000,
            source_page_breaks=[0, 5000, 10000],
        )

        balloons = detect_strip_balloons(strip, detector=fake_detector)

        self.assertGreaterEqual(len(balloons), 1)
        self.assertLessEqual(len(balloons), 4)

    def test_detect_strip_balloons_remaps_to_strip_coords(self):
        from unittest.mock import MagicMock
        from strip.detect_balloons import detect_strip_balloons
        from strip.types import VerticalStrip
        import numpy as np

        fake_block = type("FakeBlock", (), {})()
        fake_block.x1, fake_block.y1, fake_block.x2, fake_block.y2 = 100, 50, 200, 150
        fake_block.confidence = 0.9

        fake_detector = MagicMock()
        fake_detector.detect.return_value = [fake_block]

        strip = VerticalStrip(
            image=np.zeros((1000, 400, 3), dtype=np.uint8),
            width=400, height=1000, source_page_breaks=[0, 1000],
        )
        balloons = detect_strip_balloons(strip, detector=fake_detector)
        self.assertEqual(len(balloons), 1)
        self.assertEqual(balloons[0].strip_bbox.y1, 50)
        self.assertEqual(balloons[0].strip_bbox.y2, 150)

    def test_detect_strip_balloons_adds_missed_white_balloon_band(self):
        from unittest.mock import MagicMock
        from strip.detect_balloons import detect_strip_balloons
        from strip.types import VerticalStrip
        import cv2
        import numpy as np

        image = np.full((500, 320, 3), 28, dtype=np.uint8)
        cv2.ellipse(image, (160, 250), (72, 42), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (160, 250), (72, 42), 0, 0, 360, (0, 0, 0), 3)
        cv2.putText(
            image,
            "NONE",
            (125, 257),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        fake_detector = MagicMock()
        fake_detector.detect.return_value = []

        strip = VerticalStrip(
            image=image,
            width=320,
            height=500,
            source_page_breaks=[0, 500],
        )

        balloons = detect_strip_balloons(strip, detector=fake_detector)

        self.assertEqual(len(balloons), 1)
        balloon = balloons[0].strip_bbox
        self.assertLessEqual(balloon.x1, 125)
        self.assertGreaterEqual(balloon.x2, 190)
        self.assertLessEqual(balloon.y1, 245)
        self.assertGreaterEqual(balloon.y2, 268)

    def test_detect_strip_balloons_adds_perspective_ui_panel_band_candidate(self):
        from unittest.mock import MagicMock
        from strip.bands import group_balloons_into_bands
        from strip.detect_balloons import detect_strip_balloons
        from strip.types import VerticalStrip
        import cv2
        import numpy as np

        image = np.full((500, 420, 3), 255, dtype=np.uint8)
        panel = np.array([[90, 232], [360, 252], [338, 326], [70, 308]], dtype=np.int32)
        cv2.fillPoly(image, [panel], [86, 82, 80])
        cv2.putText(
            image,
            "Search",
            (118, 292),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        fake_block = type("FakeBlock", (), {})()
        fake_block.x1, fake_block.y1, fake_block.x2, fake_block.y2 = 50, 160, 150, 220
        fake_block.confidence = 0.9
        fake_detector = MagicMock()
        fake_detector.detect.return_value = [fake_block]

        strip = VerticalStrip(
            image=image,
            width=420,
            height=500,
            source_page_breaks=[0, 500],
        )

        balloons = detect_strip_balloons(strip, detector=fake_detector)
        bands = group_balloons_into_bands(balloons, gap_threshold=64, margin=16)

        self.assertTrue(any(balloon.strip_bbox.y2 >= 326 for balloon in balloons), balloons)
        self.assertEqual(len(bands), 1)
        self.assertGreaterEqual(bands[0].y_bottom, 342)

    def test_negative_detect_merge_adds_dark_balloon_text_candidate(self):
        from unittest.mock import MagicMock, patch
        from strip.detect_balloons import detect_strip_balloons
        from strip.types import VerticalStrip
        import cv2
        import numpy as np

        image = np.zeros((360, 420, 3), dtype=np.uint8)
        cv2.ellipse(image, (210, 180), (130, 70), 0, 0, 360, (4, 8, 12), -1)
        cv2.ellipse(image, (210, 180), (130, 70), 0, 0, 360, (35, 145, 190), 4)
        cv2.putText(image, "DARK", (155, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "TEXT", (158, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 248, 255), 2, cv2.LINE_AA)

        block = type("FakeBlock", (), {})()
        block.x1, block.y1, block.x2, block.y2 = 145, 145, 280, 215
        block.confidence = 0.91
        fake_detector = MagicMock()
        fake_detector.detect.side_effect = [[], [block]]

        strip = VerticalStrip(image=image, width=420, height=360, source_page_breaks=[0, 360])
        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_NEGATIVE_DETECT_MERGE": "1",
                "TRADUZAI_STRIP_WHITE_BALLOON_BAND_SCAN": "0",
                "TRADUZAI_STRIP_UI_LAYOUT_BAND_SCAN": "0",
            },
        ):
            balloons = detect_strip_balloons(strip, detector=fake_detector)

        self.assertEqual(len(balloons), 1)
        self.assertLessEqual(balloons[0].strip_bbox.x1, 145)
        self.assertGreaterEqual(balloons[0].strip_bbox.x2, 280)

    def test_dark_balloon_scan_adds_sparse_light_text_candidate_without_detector(self):
        from unittest.mock import MagicMock, patch
        from strip.detect_balloons import detect_strip_balloons
        from strip.types import VerticalStrip
        import cv2
        import numpy as np

        image = np.zeros((420, 420, 3), dtype=np.uint8)
        cv2.ellipse(image, (210, 190), (120, 70), 0, 0, 360, (2, 5, 8), -1)
        cv2.ellipse(image, (210, 190), (120, 70), 0, 0, 360, (35, 145, 190), 3)
        cv2.putText(image, "1,000 points..", (112, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 248, 255), 2, cv2.LINE_AA)

        fake_detector = MagicMock()
        fake_detector.detect.return_value = []

        strip = VerticalStrip(image=image, width=420, height=420, source_page_breaks=[0, 420])
        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_DARK_BALLOON_BAND_SCAN": "1",
                "TRADUZAI_STRIP_NEGATIVE_DETECT_MERGE": "0",
                "TRADUZAI_STRIP_WHITE_BALLOON_BAND_SCAN": "0",
                "TRADUZAI_STRIP_UI_LAYOUT_BAND_SCAN": "0",
            },
        ):
            balloons = detect_strip_balloons(strip, detector=fake_detector)

        self.assertEqual(len(balloons), 1)
        balloon = balloons[0].strip_bbox
        self.assertLessEqual(balloon.x1, 115)
        self.assertGreaterEqual(balloon.x2, 290)
        self.assertLessEqual(balloon.y1, 175)
        self.assertGreaterEqual(balloon.y2, 214)
        self.assertTrue(balloons[0].metadata.get("dark_band_scan_candidate"))

    def test_sparse_negative_dark_candidate_expands_to_bubble_body(self):
        from strip.detect_balloons import BBox, _expand_sparse_dark_negative_candidate
        import cv2
        import numpy as np

        image = np.zeros((520, 420, 3), dtype=np.uint8)
        cv2.ellipse(image, (165, 300), (145, 120), 0, 0, 360, (4, 8, 12), -1)
        cv2.ellipse(image, (165, 300), (145, 120), 0, 0, 360, (35, 145, 190), 4)
        cv2.putText(image, "1,000 points..", (82, 382), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
        tight_text = BBox(84, 356, 266, 400)

        expanded = _expand_sparse_dark_negative_candidate(image, tight_text, confidence=0.82)

        self.assertLessEqual(expanded.x1, 10)
        self.assertLessEqual(expanded.y1, 100)
        self.assertGreaterEqual(expanded.x2, 330)
        self.assertGreaterEqual(expanded.y2, 470)

    def test_negative_detect_merge_rejects_thin_artifact_line(self):
        from unittest.mock import MagicMock, patch
        from strip.detect_balloons import detect_strip_balloons
        from strip.types import VerticalStrip
        import cv2
        import numpy as np

        image = np.zeros((260, 420, 3), dtype=np.uint8)
        cv2.line(image, (80, 120), (340, 120), (245, 245, 245), 3, cv2.LINE_AA)

        block = type("FakeBlock", (), {})()
        block.x1, block.y1, block.x2, block.y2 = 80, 114, 340, 126
        block.confidence = 0.88
        fake_detector = MagicMock()
        fake_detector.detect.side_effect = [[], [block]]

        strip = VerticalStrip(image=image, width=420, height=260, source_page_breaks=[0, 260])
        with patch.dict(
            "os.environ",
            {
                "TRADUZAI_STRIP_NEGATIVE_DETECT_MERGE": "1",
                "TRADUZAI_STRIP_WHITE_BALLOON_BAND_SCAN": "0",
                "TRADUZAI_STRIP_UI_LAYOUT_BAND_SCAN": "0",
            },
        ):
            balloons = detect_strip_balloons(strip, detector=fake_detector)

        self.assertEqual(balloons, [])

    def test_white_balloon_scan_adds_connected_component_clipped_by_chunk_edge(self):
        from strip.detect_balloons import _scan_white_balloon_band_candidates
        from strip.types import Balloon, BBox
        import cv2
        import numpy as np

        image = np.full((600, 360, 3), 18, dtype=np.uint8)
        cv2.ellipse(image, (135, 4), (90, 54), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (135, 4), (90, 54), 0, 0, 360, (0, 0, 0), 3)
        cv2.ellipse(image, (210, 96), (104, 62), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(image, (210, 96), (104, 62), 0, 0, 360, (0, 0, 0), 3)
        cv2.rectangle(image, (150, 45), (195, 80), (255, 255, 255), -1)
        cv2.putText(image, "OF COURSE", (82, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, "TRYING TO", (148, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "TRACK YOU", (143, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)

        existing = [Balloon(BBox(82, 22, 168, 43), confidence=0.9)]

        added = _scan_white_balloon_band_candidates(image, existing)

        self.assertTrue(added)
        union = added[0].strip_bbox
        self.assertLessEqual(union.x1, 140)
        self.assertGreaterEqual(union.x2, 250)
        self.assertLessEqual(union.y1, 80)
        self.assertGreaterEqual(union.y2, 130)

    def test_white_balloon_scan_adds_text_inside_large_white_panel(self):
        from strip.detect_balloons import _scan_white_balloon_band_candidates
        from strip.types import Balloon, BBox
        import cv2
        import numpy as np

        image = np.full((520, 360, 3), 24, dtype=np.uint8)
        image[0:330, :] = 255
        cv2.putText(image, "KNOWN", (118, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "MISSING", (84, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 2, cv2.LINE_AA)
        existing = [Balloon(BBox(105, 42, 235, 86), confidence=0.94)]

        added = _scan_white_balloon_band_candidates(image, existing)

        self.assertTrue(added)
        union = added[0].strip_bbox
        self.assertLessEqual(union.x1, 90)
        self.assertGreaterEqual(union.x2, 200)
        self.assertLessEqual(union.y1, 205)
        self.assertGreaterEqual(union.y2, 230)


class FalsePositiveFilterTests(unittest.TestCase):
    """Bboxes absurdamente grandes são false-positives do detector e devem ser descartados."""

    def _make_detector_with(self, *blocks):
        from unittest.mock import MagicMock
        det = MagicMock()
        det.detect.return_value = list(blocks)
        return det

    def _make_block(self, x1, y1, x2, y2, conf=0.9):
        b = type("B", (), {})()
        b.x1, b.y1, b.x2, b.y2, b.confidence = x1, y1, x2, y2, conf
        return b

    def test_oversized_balloon_taller_than_25pct_is_dropped(self):
        """Balão com altura > 25% do strip deve ser descartado."""
        from strip.detect_balloons import detect_strip_balloons
        from strip.types import VerticalStrip
        import numpy as np

        # Strip de 1000px (< chunk_height=4096) → 1 chunk → mock chamado 1x
        small = self._make_block(50, 50, 150, 150)       # 100px = 10% de 1000px → ok
        huge = self._make_block(0, 0, 800, 600, conf=0.7)  # 600px = 60% → false-positive

        strip = VerticalStrip(
            image=np.zeros((1000, 800, 3), dtype=np.uint8),
            width=800, height=1000, source_page_breaks=[0, 1000],
        )
        det = self._make_detector_with(small, huge)
        balloons = detect_strip_balloons(strip, detector=det)

        # Só o pequeno deve passar (huge: 600/1000 = 60% > 25%)
        self.assertEqual(len(balloons), 1, f"Esperado 1 balão, got {len(balloons)}")
        self.assertEqual(balloons[0].strip_bbox.x2, 150)

    def test_oversized_balloon_wider_than_95pct_is_dropped(self):
        """Balão com largura > 95% da largura do strip deve ser descartado."""
        from strip.detect_balloons import detect_strip_balloons
        from strip.types import VerticalStrip
        import numpy as np

        # Strip de 500px (< chunk_height) → 1 chunk
        normal = self._make_block(50, 100, 200, 200)
        wide = self._make_block(5, 50, 795, 200, conf=0.6)  # 790px de 800px = 99%

        strip = VerticalStrip(
            image=np.zeros((500, 800, 3), dtype=np.uint8),
            width=800, height=500, source_page_breaks=[0, 500],
        )
        det = self._make_detector_with(normal, wide)
        balloons = detect_strip_balloons(strip, detector=det)

        self.assertEqual(len(balloons), 1, f"Esperado 1 balão, got {len(balloons)}")
        self.assertLess(balloons[0].strip_bbox.width, 200)

    def test_normal_balloon_passes_filter(self):
        """Balão com tamanho normal deve passar."""
        from strip.detect_balloons import detect_strip_balloons
        from strip.types import VerticalStrip
        import numpy as np

        b = self._make_block(100, 100, 300, 300)  # 200x200 em strip 1000x600 = 20%

        strip = VerticalStrip(
            image=np.zeros((1000, 600, 3), dtype=np.uint8),
            width=600, height=1000, source_page_breaks=[0, 1000],
        )
        det = self._make_detector_with(b)
        balloons = detect_strip_balloons(strip, detector=det)

        self.assertEqual(len(balloons), 1)
        self.assertEqual(balloons[0].strip_bbox.y1, 100)
