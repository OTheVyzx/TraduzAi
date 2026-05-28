import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debug_tools import DebugRecorder, bind_recorder
from ocr.contextual_reviewer import contextual_review_page


def test_contextual_review_keeps_tight_bbox_over_giant_duplicate(tmp_path):
    recorder = DebugRecorder(tmp_path, enabled=True, run_id="test")
    bind_recorder(recorder)
    try:
        page = {
            "page": 1,
            "texts": [
                {
                    "id": "ocr_017_bad",
                    "text": "PLEASE!",
                    "bbox": [88, 16, 775, 823],
                    "source_bbox": [88, 16, 775, 823],
                    "text_pixel_bbox": [468, 57, 644, 100],
                    "balloon_bbox": [88, 16, 775, 823],
                    "line_polygons": [],
                    "confidence": 0.72,
                    "balloon_type": "white",
                    "background_rgb": [245, 238, 220],
                },
                {
                    "id": "ocr_017_good",
                    "text": "PLEASE!",
                    "bbox": [468, 57, 644, 100],
                    "source_bbox": [468, 57, 644, 100],
                    "text_pixel_bbox": [468, 57, 644, 100],
                    "balloon_bbox": [88, 16, 775, 823],
                    "line_polygons": [[[468, 57], [644, 57], [644, 100], [468, 100]]],
                    "confidence": 0.91,
                    "balloon_type": "white",
                    "background_rgb": [255, 255, 255],
                },
            ],
        }

        reviewed = contextual_review_page(page, [], [])

        assert len(reviewed["texts"]) == 1
        kept = reviewed["texts"][0]
        assert kept["id"] == "ocr_017_good"
        assert kept["bbox"] == [468, 57, 644, 100]
        assert "bbox_overreach_critical" not in kept.get("qa_flags", [])

        decisions_path = (
            tmp_path / "debug" / "e2e" / "03_ocr" / "ocr_dedupe_decisions.jsonl"
        )
        rows = [
            json.loads(line)
            for line in decisions_path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 1
        decision = rows[0]
        assert decision["action"] == "dedupe_blocks"
        assert decision["reason"] == "geometry_quality_score"
        assert decision["kept"]["text_id"] == "ocr_017_good"
        assert decision["dropped"]["text_id"] == "ocr_017_bad"
        assert decision["kept_score"] > decision["dropped_score"]
    finally:
        bind_recorder(None)
