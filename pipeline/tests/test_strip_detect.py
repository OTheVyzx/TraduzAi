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
