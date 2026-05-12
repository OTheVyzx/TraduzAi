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


class DetectStripBalloonsTests(unittest.TestCase):
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
