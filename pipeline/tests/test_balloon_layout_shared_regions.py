from layout.balloon_layout import _region_supports_shared_layout


def test_shared_layout_rejects_distinct_bubble_mask_regions():
    region = {
        "texts": [
            {
                "bbox": [140, 120, 300, 150],
                "bubble_mask_bbox": [80, 80, 340, 190],
            },
            {
                "bbox": [360, 310, 520, 360],
                "bubble_mask_bbox": [300, 260, 580, 430],
            },
        ]
    }

    assert _region_supports_shared_layout(region, "text") is False


def test_shared_layout_allows_same_bubble_stack():
    region = {
        "texts": [
            {
                "bbox": [180, 120, 300, 150],
                "bubble_mask_bbox": [100, 80, 380, 240],
            },
            {
                "bbox": [184, 156, 298, 190],
                "bubble_mask_bbox": [100, 80, 380, 240],
            },
        ]
    }

    assert _region_supports_shared_layout(region, "text") is True
