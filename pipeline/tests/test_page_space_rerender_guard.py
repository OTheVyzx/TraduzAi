from __future__ import annotations

import main


def test_page_text_coordinate_audit_flags_detects_local_safe_box():
    texts = [
        {
            "id": "ocr_002",
            "band_id": "page_002_band_005",
            "band_y_top": 5420,
            "band_height": 895,
            "bbox": [25, 5436, 667, 5745],
            "text_pixel_bbox": [498, 5655, 656, 5740],
            "balloon_bbox": [466, 5606, 696, 5777],
            "bubble_inner_bbox": [513, 230, 649, 313],
            "safe_text_box": [525, 242, 637, 301],
            "render_bbox": [542, 246, 620, 296],
            "translated": "POR FAVOR, PELO BEM DA CRIANCA.",
        }
    ]

    flags = main._page_text_coordinate_audit_flags(texts, height=13832, width=800)

    assert "layout_bbox_coordinate_mismatch" in flags
    assert "bubble_inner_bbox_coordinate_mismatch" in flags
    assert "page_space_rerender_mixed_coordinates" in flags


def test_append_page_text_flags_marks_all_processable_texts():
    texts = [{"id": "ocr_1", "qa_flags": ["TEXT_OVERFLOW"]}, {"id": "ocr_2", "skip_processing": True}]

    main._append_page_text_flags(texts, ["page_space_rerender_mixed_coordinates"])

    assert texts[0]["qa_flags"] == ["TEXT_OVERFLOW", "page_space_rerender_mixed_coordinates"]
    assert texts[1]["qa_flags"] == ["page_space_rerender_mixed_coordinates"]
