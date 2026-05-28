from layout.simple_text_geometry import normalize_text_geometry, sanitize_for_simple_text_only


def test_normalize_preserves_connected_metadata():
    text = {
        "bbox": [10, 20, 110, 80],
        "text_pixel_bbox": [20, 30, 90, 70],
        "balloon_bbox": [8, 18, 130, 96],
        "balloon_subregions": [[8, 18, 60, 96], [62, 18, 130, 96]],
        "connected_lobe_bboxes": [[8, 18, 60, 96], [62, 18, 130, 96]],
        "connected_group_confidence": 0.91,
        "connected_balloon_orientation": "left-right",
        "layout_group_size": 2,
        "layout_profile": "connected_balloon",
    }

    normalized = normalize_text_geometry(text)

    assert normalized["balloon_bbox"] == [8, 18, 130, 96]
    assert normalized["text_pixel_bbox"] == [20, 30, 90, 70]
    assert normalized["balloon_subregions"] == [[8, 18, 60, 96], [62, 18, 130, 96]]
    assert normalized["connected_lobe_bboxes"] == [[8, 18, 60, 96], [62, 18, 130, 96]]
    assert normalized["connected_group_confidence"] == 0.91
    assert normalized["connected_balloon_orientation"] == "left-right"
    assert normalized["layout_group_size"] == 2
    assert normalized["layout_profile"] == "connected_balloon"


def test_normalize_fills_missing_layout_aliases_without_collapsing_balloon():
    normalized = normalize_text_geometry(
        {
            "bbox": [10, 20, 110, 80],
            "source_bbox": [8, 18, 130, 96],
            "text_pixel_bbox": [20, 30, 90, 70],
        }
    )

    assert normalized["layout_bbox"] == [20, 30, 90, 70]
    assert normalized["balloon_bbox"] == [20, 30, 90, 70]
    assert normalized["ocr_text_bbox"] == [10, 20, 110, 80]
    assert normalized["layout_group_size"] == 1


def test_sanitize_for_simple_text_only_still_strips_connected_metadata():
    sanitized = sanitize_for_simple_text_only(
        {
            "bbox": [10, 20, 110, 80],
            "text_pixel_bbox": [20, 30, 90, 70],
            "balloon_bbox": [8, 18, 130, 96],
            "balloon_subregions": [[8, 18, 60, 96], [62, 18, 130, 96]],
            "connected_lobe_bboxes": [[8, 18, 60, 96], [62, 18, 130, 96]],
            "connected_group_confidence": 0.91,
            "connected_balloon_orientation": "left-right",
            "layout_group_size": 2,
            "layout_profile": "connected_balloon",
        }
    )

    assert sanitized["balloon_bbox"] == [20, 30, 90, 70]
    assert sanitized["balloon_subregions"] == []
    assert sanitized["connected_lobe_bboxes"] == []
    assert sanitized["connected_group_confidence"] == 0.0
    assert sanitized["connected_balloon_orientation"] == ""
    assert sanitized["layout_group_size"] == 1
    assert sanitized["layout_profile"] != "connected_balloon"
