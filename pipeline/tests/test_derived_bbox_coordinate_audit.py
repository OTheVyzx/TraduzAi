import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_shift_text_geometry_y_shifts_all_derived_bbox_keys():
    from strip.run import _shift_text_geometry_y

    shifted = _shift_text_geometry_y(
        {
            "bbox": [1, 2, 3, 4],
            "render_bbox": [10, 20, 30, 40],
            "safe_text_box": [11, 21, 31, 41],
            "_debug_safe_text_box": [12, 22, 32, 42],
            "layout_safe_bbox": [13, 23, 33, 43],
            "position_bbox": [14, 24, 34, 44],
            "capacity_bbox": [15, 25, 35, 45],
            "target_bbox": [16, 26, 36, 46],
            "bubble_mask_bbox": [17, 27, 37, 47],
            "bubble_inner_bbox": [18, 28, 38, 48],
            "balloon_inner_bbox": [19, 29, 39, 49],
            "connected_position_bboxes": [[17, 27, 37, 47]],
            "qa_metrics": {
                "render_bbox": [18, 28, 38, 48],
                "nested": {"safe_text_box_bbox": [19, 29, 39, 49]},
            },
            "_render_debug": {
                "target_bbox": [20, 30, 40, 50],
                "layout_safe_bbox": [21, 31, 41, 51],
            },
        },
        1000,
    )

    assert shifted["render_bbox"] == [10, 1020, 30, 1040]
    assert shifted["safe_text_box"] == [11, 1021, 31, 1041]
    assert shifted["_debug_safe_text_box"] == [12, 1022, 32, 1042]
    assert shifted["layout_safe_bbox"] == [13, 1023, 33, 1043]
    assert shifted["position_bbox"] == [14, 1024, 34, 1044]
    assert shifted["capacity_bbox"] == [15, 1025, 35, 1045]
    assert shifted["target_bbox"] == [16, 1026, 36, 1046]
    assert shifted["bubble_mask_bbox"] == [17, 1027, 37, 1047]
    assert shifted["bubble_inner_bbox"] == [18, 1028, 38, 1048]
    assert shifted["balloon_inner_bbox"] == [19, 1029, 39, 1049]
    assert shifted["connected_position_bboxes"] == [[17, 1027, 37, 1047]]
    assert shifted["qa_metrics"]["render_bbox"] == [18, 1028, 38, 1048]
    assert shifted["qa_metrics"]["nested"]["safe_text_box_bbox"] == [19, 1029, 39, 1049]
    assert shifted["_render_debug"]["target_bbox"] == [20, 1030, 40, 1050]
    assert shifted["_render_debug"]["layout_safe_bbox"] == [21, 1031, 41, 1051]


def test_bbox_coordinate_audit_reports_derived_bbox_mismatches_by_key():
    from debug_tools.bbox import audit_bbox_coordinate_space, layout_block_records

    records = layout_block_records(
        [
            {
                "page_id": "page_001",
                "height": 6200,
                "width": 760,
                "texts": [
                    {
                        "id": "ocr_001",
                        "band_id": "page_001_band_003",
                        "band_y_top": 2700,
                        "band_height": 900,
                        "bbox": [80, 2720, 200, 2760],
                        "source_bbox": [70, 2710, 220, 2780],
                        "render_bbox": [90, 20, 190, 50],
                        "safe_text_box": [85, 18, 205, 60],
                        "position_bbox": [80, 16, 210, 70],
                        "capacity_bbox": [82, 17, 208, 68],
                        "target_bbox": [75, 10, 215, 80],
                        "layout_safe_bbox": [78, 12, 212, 78],
                        "_render_debug": {
                            "layout_safe_bbox": [78, 12, 212, 78],
                        },
                        "qa_metrics": {
                            "render_bbox": [90, 20, 190, 50],
                        },
                    }
                ],
            }
        ]
    )

    audit = audit_bbox_coordinate_space(records)

    assert audit["summary"]["derived_bbox_coordinate_mismatch_count"] == 8
    assert audit["summary"]["by_key"]["render_bbox"]["mismatch"] == 1
    assert audit["summary"]["by_key"]["safe_text_box"]["mismatch"] == 1
    assert audit["summary"]["by_key"]["position_bbox"]["mismatch"] == 1
    assert audit["summary"]["by_key"]["capacity_bbox"]["mismatch"] == 1
    assert audit["summary"]["by_key"]["target_bbox"]["mismatch"] == 1
    assert audit["summary"]["by_key"]["layout_safe_bbox"]["mismatch"] == 1
    assert audit["summary"]["by_key"]["_render_debug.layout_safe_bbox"]["mismatch"] == 1
    assert audit["summary"]["by_key"]["qa_metrics.render_bbox"]["mismatch"] == 1
    assert audit["summary"]["by_key"]["bbox"]["mismatch"] == 0
    assert {finding["key"] for finding in audit["findings"]} >= {
        "render_bbox",
        "safe_text_box",
        "position_bbox",
        "capacity_bbox",
        "target_bbox",
        "layout_safe_bbox",
        "_render_debug.layout_safe_bbox",
        "qa_metrics.render_bbox",
    }


def test_bbox_coordinate_audit_accepts_page_local_band_y_top():
    from debug_tools.bbox import audit_bbox_coordinate_space, layout_block_records

    records = layout_block_records(
        [
            {
                "page_id": "page_002",
                "height": 13820,
                "width": 800,
                "texts": [
                    {
                        "id": "ocr_001",
                        "band_id": "page_002_band_019",
                        "band_y_top": 48,
                        "band_height": 900,
                        "bbox": [217, 64, 406, 170],
                        "text_pixel_bbox": [218, 70, 401, 166],
                        "balloon_bbox": [151, 48, 462, 260],
                        "render_bbox": [213, 79, 406, 160],
                        "safe_text_box": [180, 79, 438, 229],
                    }
                ],
            }
        ]
    )

    audit = audit_bbox_coordinate_space(records)

    assert audit["summary"]["derived_bbox_coordinate_mismatch_count"] == 0
    assert audit["summary"]["mixed_coordinate_space_count"] == 0


def test_audit_flags_bubble_and_safe_boxes_that_remain_band_local():
    from debug_tools.bbox import audit_bbox_coordinate_space, layout_block_records

    page = {
        "page_id": "page_001",
        "height": 13832,
        "width": 800,
        "texts": [
            {
                "id": "ocr_002",
                "band_id": "page_002_band_005",
                "band_y_top": 5420,
                "band_height": 895,
                "bbox": [25, 5436, 667, 5745],
                "text_pixel_bbox": [498, 5655, 656, 5740],
                "balloon_bbox": [466, 5606, 696, 5777],
                "bubble_mask_bbox": [501, 218, 661, 325],
                "bubble_inner_bbox": [513, 230, 649, 313],
                "safe_text_box": [525, 242, 637, 301],
                "render_bbox": [542, 246, 620, 296],
            }
        ],
    }

    records = layout_block_records([page])
    audit = audit_bbox_coordinate_space(records)

    assert audit["summary"]["all_consistent"] is False
    assert audit["summary"]["by_key"]["bubble_mask_bbox"]["mismatch"] == 1
    assert audit["summary"]["by_key"]["bubble_inner_bbox"]["mismatch"] == 1
    assert audit["summary"]["by_key"]["safe_text_box"]["mismatch"] == 1
    assert audit["summary"]["by_key"]["render_bbox"]["mismatch"] == 1
    assert any(
        finding["blocker"] == "derived_bbox_coordinate_mismatch"
        and finding["severity"] == "critical"
        for finding in audit["findings"]
    )


def test_coordinate_audit_flags_promote_page_space_mismatch():
    from debug_tools.bbox import audit_bbox_coordinate_space, coordinate_audit_flags, layout_block_records

    records = layout_block_records(
        [
            {
                "height": 13832,
                "width": 800,
                "texts": [
                    {
                        "id": "ocr_002",
                        "band_id": "page_002_band_005",
                        "band_y_top": 5420,
                        "band_height": 895,
                        "bbox": [25, 5436, 667, 5745],
                        "balloon_bbox": [466, 5606, 696, 5777],
                        "bubble_inner_bbox": [513, 230, 649, 313],
                        "safe_text_box": [525, 242, 637, 301],
                    }
                ],
            }
        ]
    )

    flags = coordinate_audit_flags(audit_bbox_coordinate_space(records))

    assert "layout_bbox_coordinate_mismatch" in flags
    assert "bubble_inner_bbox_coordinate_mismatch" in flags
    assert "page_space_rerender_mixed_coordinates" in flags
