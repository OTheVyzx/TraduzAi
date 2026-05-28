import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_shift_text_geometry_y_shifts_layout_bbox_with_page_metadata():
    from strip.run import _shift_text_geometry_y

    shifted = _shift_text_geometry_y(
        {
            "bbox": [10, 20, 50, 60],
            "source_bbox": [11, 21, 51, 61],
            "text_pixel_bbox": [12, 22, 52, 62],
            "balloon_bbox": [8, 18, 58, 68],
            "layout_bbox": [14, 24, 54, 64],
        },
        100,
    )

    assert shifted["layout_bbox"] == [14, 124, 54, 164]


def test_layout_bbox_debug_artifacts_report_page_space_and_mismatch(tmp_path):
    from debug_tools import DebugRecorder, bind_recorder
    from debug_tools.bbox import write_layout_geometry_debug_artifacts

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        pages = [
            {
                "page_id": "page_001",
                "height": 6200,
                "width": 760,
                "texts": [
                    {
                        "id": "ocr_001",
                        "band_id": "page_001_band_002",
                        "band_y_top": 2700,
                        "band_height": 500,
                        "bbox": [473, 65, 641, 96],
                        "source_bbox": [88, 2716, 775, 3523],
                        "text_pixel_bbox": [473, 2765, 641, 2796],
                        "balloon_bbox": [88, 2716, 775, 3523],
                        "layout_bbox": [473, 65, 641, 96],
                    }
                ],
            }
        ]

        write_layout_geometry_debug_artifacts(pages)
    finally:
        bind_recorder(None)

    root = tmp_path / "debug" / "e2e" / "05_layout_geometry"
    block = json.loads((root / "layout_blocks.jsonl").read_text(encoding="utf-8").splitlines()[0])
    audit = json.loads((root / "bbox_coordinate_audit.json").read_text(encoding="utf-8"))

    assert block["coordinate_space"] == "page"
    assert block["bboxes"]["layout_bbox"] == {"value": [473, 65, 641, 96], "space": "page"}
    assert audit["summary"]["mixed_coordinate_space_count"] == 1
    assert any(finding["blocker"] == "layout_bbox_coordinate_mismatch" for finding in audit["findings"])


def test_layout_bbox_debug_artifacts_report_source_bbox_balloon_copy(tmp_path):
    from debug_tools import DebugRecorder, bind_recorder
    from debug_tools.bbox import write_layout_geometry_debug_artifacts

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        write_layout_geometry_debug_artifacts(
            [
                {
                    "page_id": "page_001",
                    "height": 2400,
                    "width": 760,
                    "texts": [
                        {
                            "id": "ocr_001",
                            "source_bbox": [88, 716, 775, 1523],
                            "balloon_bbox": [88, 716, 775, 1523],
                            "text_pixel_bbox": [473, 765, 641, 796],
                            "decision_trace_reason": "refined_same_as_cluster",
                        }
                    ],
                }
            ]
        )
    finally:
        bind_recorder(None)

    overreach = (
        tmp_path
        / "debug"
        / "e2e"
        / "05_layout_geometry"
        / "source_bbox_balloon_overreach.jsonl"
    ).read_text(encoding="utf-8")

    record = json.loads(overreach.splitlines()[0])
    assert record["blocker"] == "source_bbox_assigned_from_balloon"
    assert record["area_ratio"] > 10


def test_source_bbox_equals_balloon_bbox_records_small_equal_bbox(tmp_path):
    from debug_tools import DebugRecorder, bind_recorder
    from debug_tools.bbox import write_layout_geometry_debug_artifacts

    recorder = DebugRecorder(tmp_path, enabled=True, run_id="run-test")
    bind_recorder(recorder)
    try:
        write_layout_geometry_debug_artifacts(
            [
                {
                    "page_id": "page_001",
                    "height": 500,
                    "width": 500,
                    "texts": [
                        {
                            "id": "ocr_003",
                            "source_bbox": [441, 91, 499, 441],
                            "balloon_bbox": [441, 91, 499, 441],
                            "text_pixel_bbox": [441, 91, 499, 441],
                            "decision_trace_reason": "refined_same_as_cluster",
                        }
                    ],
                }
            ]
        )
    finally:
        bind_recorder(None)

    overreach = (
        tmp_path
        / "debug"
        / "e2e"
        / "05_layout_geometry"
        / "source_bbox_balloon_overreach.jsonl"
    ).read_text(encoding="utf-8")

    record = json.loads(overreach.splitlines()[0])
    assert record["issue"] == "source_bbox_equals_balloon_bbox"
    assert record["area_ratio"] == 1.0
    assert record["severity"] == "warning"


def test_assign_balloon_bbox_decision_does_not_copy_balloon_to_source_bbox():
    import layout.balloon_layout as balloon_layout

    recorded = []

    def fake_regions(*, texts, image_shape):
        return [{"bbox": [80, 70, 300, 240], "texts": texts}]

    def fake_record_decision(**payload):
        if payload.get("action") == "assign_balloon_bbox":
            recorded.append(payload)

    page = {
        "width": 400,
        "height": 300,
        "texts": [
            {
                "id": "ocr_001",
                "text": "HELLO",
                "bbox": [120, 100, 180, 130],
                "text_pixel_bbox": [122, 102, 178, 128],
                "tipo": "fala",
            }
        ],
        "_cached_image_bgr": np.full((300, 400, 3), 255, dtype=np.uint8),
    }

    with patch.object(balloon_layout, "build_mask_regions", side_effect=fake_regions), patch.object(
        balloon_layout, "refine_balloon_bbox_from_image", return_value=[80, 70, 300, 240]
    ), patch.object(balloon_layout, "record_decision", side_effect=fake_record_decision):
        enriched = balloon_layout.enrich_page_layout(page)

    assert enriched["texts"][0].get("source_bbox") in (None, [])
    assert recorded
    assert recorded[0]["bbox"] == [80, 70, 300, 240]
    assert recorded[0]["details"]["source_bbox"] == []
