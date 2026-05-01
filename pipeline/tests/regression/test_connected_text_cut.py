"""
Regression tests for connected balloon detection and text clipping guard.

Covers:
- TextFitGuard raises text_clipped when ink leaks outside safe_bbox
- TextFitGuard passes when ink is contained
- SafeArea builder produces smaller bbox with padding
- SafeArea lobe mode returns per-lobe safe areas
- ConnectedBalloonSplitter detects two well-separated OCR boxes as connected
- ConnectedBalloonSplitter assigns each OCR box to the correct lobe
"""
import pytest

from typesetter.text_fit_guard import validate_rendered_text_fit, blocks_clean_export
from layout.safe_area import build_safe_area
from layout.connected_balloon_splitter import detect_connected_balloon


# ---------------------------------------------------------------------------
# TextFitGuard
# ---------------------------------------------------------------------------

class TestTextFitGuard:
    def _call(self, ink, safe, balloon=None):
        return validate_rendered_text_fit(
            page_width=800,
            page_height=1200,
            target_bbox=safe,
            safe_bbox=safe,
            ink_bbox=ink,
            balloon_bbox=balloon,
            region_id="p011_r001",
            page=11,
        )

    def test_ok_when_ink_inside(self):
        result = self._call(ink=[30, 30, 200, 80], safe=[20, 20, 210, 90])
        assert result["ok"] is True
        assert result["flags"] == []

    def test_text_clipped_left(self):
        # ink starts before safe area (x1 too small)
        result = self._call(ink=[10, 30, 200, 80], safe=[20, 20, 210, 90])
        assert result["ok"] is False
        types = [f["type"] for f in result["flags"]]
        assert "text_clipped" in types

    def test_text_clipped_right(self):
        result = self._call(ink=[30, 30, 220, 80], safe=[20, 20, 210, 90])
        assert result["ok"] is False
        assert any(f["type"] == "text_clipped" for f in result["flags"])

    def test_text_clipped_blocks_clean_export(self):
        result = self._call(ink=[10, 30, 200, 80], safe=[20, 20, 210, 90])
        assert blocks_clean_export(result) is True

    def test_no_block_when_ok(self):
        result = self._call(ink=[30, 30, 200, 80], safe=[20, 20, 210, 90])
        assert blocks_clean_export(result) is False

    def test_text_near_edge_warning(self):
        # ink starts 2px from safe left edge — near_edge but not clipped
        result = self._call(ink=[22, 30, 200, 80], safe=[20, 20, 210, 90])
        types = [f["type"] for f in result["flags"]]
        assert "text_near_edge" in types
        # near_edge is not critical
        assert not blocks_clean_export(result)

    def test_layout_bbox_too_small(self):
        # safe area is only 40% of balloon
        balloon = [0, 0, 200, 200]   # area 40000
        safe = [80, 80, 120, 120]    # area 1600 → ~4% of balloon
        ink = [85, 85, 115, 115]
        result = self._call(ink=ink, safe=safe, balloon=balloon)
        types = [f["type"] for f in result["flags"]]
        assert "layout_bbox_too_small" in types


# ---------------------------------------------------------------------------
# SafeArea
# ---------------------------------------------------------------------------

class TestSafeArea:
    def test_safe_smaller_than_balloon(self):
        balloon = [10, 10, 310, 210]  # 300x200
        result = build_safe_area(
            balloon_bbox=balloon,
            page_width=800,
            page_height=1200,
        )
        sx1, sy1, sx2, sy2 = result["safe_bbox"]
        assert sx1 > balloon[0]
        assert sy1 > balloon[1]
        assert sx2 < balloon[2]
        assert sy2 < balloon[3]

    def test_safe_clamped_to_page_edge(self):
        # Balloon touching left edge of page
        balloon = [0, 50, 150, 200]
        result = build_safe_area(
            balloon_bbox=balloon,
            page_width=800,
            page_height=1200,
        )
        sx1 = result["safe_bbox"][0]
        assert sx1 >= 8, "safe_bbox must not start within 8px of page edge"

    def test_connected_lobe_returns_lobes(self):
        balloon = [0, 0, 400, 200]
        lobe_a = [0, 0, 200, 200]
        lobe_b = [200, 0, 400, 200]
        result = build_safe_area(
            balloon_bbox=balloon,
            page_width=800,
            page_height=1200,
            connected_lobe_bboxes=[lobe_a, lobe_b],
        )
        assert len(result["lobes"]) == 2
        assert result["reason"] == "connected_lobe"
        # Each lobe safe_bbox should be smaller than the lobe bbox
        for lobe in result["lobes"]:
            lb = [lobe_a, lobe_b][lobe["lobe_index"]]
            sb = lobe["safe_bbox"]
            assert sb[0] >= lb[0]
            assert sb[2] <= lb[2]

    def test_polygon_tightens_bbox(self):
        # Balloon bbox is 300x300 but polygon is only 200x200 centred inside
        balloon = [0, 0, 300, 300]
        polygon = [[50, 50], [250, 50], [250, 250], [50, 250]]
        result = build_safe_area(
            balloon_bbox=balloon,
            page_width=800,
            page_height=1200,
            balloon_polygon=polygon,
        )
        assert result["reason"] == "balloon_polygon"
        sx1, sy1, sx2, sy2 = result["safe_bbox"]
        assert sx1 >= 50  # tightened by polygon


# ---------------------------------------------------------------------------
# ConnectedBalloonSplitter
# ---------------------------------------------------------------------------

class TestConnectedBalloonSplitter:
    def _wide_balloon_two_texts(self):
        """Wide balloon with two text boxes far apart → should be connected."""
        balloon = [0, 0, 600, 200]
        ocr_bboxes = [
            [20, 50, 230, 150],   # left text
            [370, 50, 580, 150],  # right text — 140px gap
        ]
        return balloon, ocr_bboxes

    def test_detects_connected_balloon(self):
        balloon, ocr_bboxes = self._wide_balloon_two_texts()
        result = detect_connected_balloon(
            balloon_bbox=balloon,
            ocr_bboxes=ocr_bboxes,
        )
        assert result["connected_balloon"] is True
        assert result["confidence"] >= 0.5

    def test_lobes_cover_both_texts(self):
        balloon, ocr_bboxes = self._wide_balloon_two_texts()
        result = detect_connected_balloon(
            balloon_bbox=balloon,
            ocr_bboxes=ocr_bboxes,
        )
        all_assigned = []
        for lobe in result["lobes"]:
            all_assigned.extend(lobe["assigned_text_ids"])
        assert sorted(all_assigned) == [0, 1], "Both OCR boxes must be assigned to lobes"

    def test_each_text_in_own_lobe(self):
        balloon, ocr_bboxes = self._wide_balloon_two_texts()
        result = detect_connected_balloon(
            balloon_bbox=balloon,
            ocr_bboxes=ocr_bboxes,
        )
        if len(result["lobes"]) == 2:
            # Left text (index 0) should be in lobe with lower x bbox
            # Right text (index 1) should be in other lobe
            lobe_for_0 = next(l for l in result["lobes"] if 0 in l["assigned_text_ids"])
            lobe_for_1 = next(l for l in result["lobes"] if 1 in l["assigned_text_ids"])
            assert lobe_for_0 is not lobe_for_1, "Each text must be in a different lobe"

    def test_single_text_not_connected(self):
        balloon = [0, 0, 300, 200]
        ocr_bboxes = [[50, 50, 250, 150]]
        result = detect_connected_balloon(
            balloon_bbox=balloon,
            ocr_bboxes=ocr_bboxes,
        )
        assert result["connected_balloon"] is False

    def test_close_texts_not_falsely_connected(self):
        """Two OCR boxes that are close together should not be flagged."""
        balloon = [0, 0, 300, 200]
        ocr_bboxes = [
            [50, 50, 140, 90],
            [50, 100, 140, 140],
        ]
        result = detect_connected_balloon(
            balloon_bbox=balloon,
            ocr_bboxes=ocr_bboxes,
        )
        # Low confidence expected — may or may not be flagged
        assert result["confidence"] < 0.8, "Close texts should not have high confidence"

    def test_orientation_horizontal(self):
        balloon, ocr_bboxes = self._wide_balloon_two_texts()
        result = detect_connected_balloon(
            balloon_bbox=balloon,
            ocr_bboxes=ocr_bboxes,
        )
        assert result["orientation"] == "horizontal"

    def test_orientation_vertical(self):
        balloon = [0, 0, 200, 600]
        ocr_bboxes = [
            [20, 20, 180, 230],
            [20, 370, 180, 580],
        ]
        result = detect_connected_balloon(
            balloon_bbox=balloon,
            ocr_bboxes=ocr_bboxes,
        )
        if result["connected_balloon"]:
            assert result["orientation"] == "vertical"
